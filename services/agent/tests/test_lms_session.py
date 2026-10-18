from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from atlas_agent.browser.session import LOCK_NAME, ProfileLock, browser_session
from atlas_agent.tools.context import ToolError

LMS = "lms.example.edu"


def _chromium_missing() -> bool:
    with sync_playwright() as pw:
        return not Path(pw.chromium.executable_path).exists()


needs_chromium = pytest.mark.skipif(
    _chromium_missing(), reason="Chromium not installed: run `playwright install chromium`"
)


def test_profile_lock_is_exclusive_and_releasable(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    first = ProfileLock(profile)
    first.acquire()
    assert (profile / LOCK_NAME).is_file()
    second = ProfileLock(profile)
    with pytest.raises(ToolError, match="^browser profile is busy$") as exc_info:
        second.acquire()
    assert str(tmp_path) not in str(exc_info.value)
    first.release()
    second.acquire()
    second.release()
    second.release()  # idempotent


def test_other_profile_is_not_blocked(tmp_path: Path) -> None:
    one = ProfileLock(tmp_path / "one")
    two = ProfileLock(tmp_path / "two")
    one.acquire()
    try:
        two.acquire()
        two.release()
    finally:
        one.release()


def test_busy_profile_fails_before_any_browser_starts(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    held = ProfileLock(profile)
    held.acquire()

    async def _call() -> None:
        async with browser_session(profile, headless=True, allowed_host=LMS):
            raise AssertionError("session must not start while the profile is busy")

    try:
        with pytest.raises(ToolError, match="^browser profile is busy$"):
            asyncio.run(_call())
    finally:
        held.release()


@needs_chromium
def test_session_opens_closes_and_releases_lock(tmp_path: Path) -> None:
    profile = tmp_path / "profile"

    async def _call() -> None:
        async with browser_session(profile, headless=True, allowed_host=LMS) as context:
            page = context.pages[0] if context.pages else await context.new_page()
            await page.set_content("<p>hello</p>")
            with pytest.raises(ToolError, match="busy"):
                ProfileLock(profile).acquire()
        lock = ProfileLock(profile)
        lock.acquire()
        lock.release()

    asyncio.run(_call())


@needs_chromium
def test_session_closes_cleanly_on_cancel(tmp_path: Path) -> None:
    profile = tmp_path / "profile"

    async def _call() -> None:
        started = asyncio.Event()

        async def _hold() -> None:
            async with browser_session(profile, headless=True, allowed_host=LMS):
                started.set()
                await asyncio.sleep(3600)

        task = asyncio.create_task(_hold())
        await asyncio.wait_for(started.wait(), timeout=60)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        lock = ProfileLock(profile)
        lock.acquire()
        lock.release()

    asyncio.run(_call())
