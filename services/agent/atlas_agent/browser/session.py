from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import BinaryIO

from playwright.async_api import BrowserContext, Playwright, async_playwright
from playwright.async_api import Error as PlaywrightError

from atlas_agent.browser.allowlist import install_navigation_guard
from atlas_agent.tools.context import ToolError

logger = logging.getLogger("atlas_agent")

LOCK_NAME = ".atlas-browser.lock"
BUSY_MESSAGE = "browser profile is busy"
NOT_INSTALLED_MESSAGE = "browser is not installed, run: playwright install chromium"
START_FAILED_MESSAGE = "browser could not start"


def _lock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ProfileLock:
    """OS-level lock held on a file inside the profile folder.

    The OS drops the lock if the process dies, so a crash never leaves a stale lock.
    """

    def __init__(self, profile_dir: Path) -> None:
        self._profile_dir = profile_dir
        self._handle: BinaryIO | None = None

    def acquire(self) -> None:
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        handle = (self._profile_dir / LOCK_NAME).open("a+b")
        try:
            _lock_file(handle)
        except OSError:
            handle.close()
            raise ToolError(BUSY_MESSAGE) from None
        self._handle = handle

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        with suppress(OSError):
            _unlock_file(handle)
        handle.close()


async def _launch(
    playwright: Playwright, profile_dir: Path, headless: bool
) -> BrowserContext:
    try:
        return await playwright.chromium.launch_persistent_context(
            profile_dir,
            headless=headless,
            accept_downloads=False,
            service_workers="block",
        )
    except PlaywrightError as exc:
        logger.warning("browser launch failed: %s", type(exc).__name__)
        if "Executable doesn't exist" in str(exc):
            raise ToolError(NOT_INSTALLED_MESSAGE) from None
        raise ToolError(START_FAILED_MESSAGE) from None


async def _close_quietly(context: BrowserContext | None, playwright: Playwright) -> None:
    with suppress(PlaywrightError, OSError):
        if context is not None:
            await context.close()
    with suppress(PlaywrightError, OSError):
        await playwright.stop()


async def _finish_close(context: BrowserContext | None, playwright: Playwright) -> None:
    """Close even when the surrounding task is being cancelled."""
    closing = asyncio.ensure_future(_close_quietly(context, playwright))
    try:
        await asyncio.shield(closing)
    except asyncio.CancelledError:
        await closing
        raise


@asynccontextmanager
async def browser_session(
    profile_dir: Path, *, headless: bool, allowed_host: str
) -> AsyncIterator[BrowserContext]:
    """One persistent Chromium per profile folder. Closed cleanly, also on cancel.

    No tracing, video, HAR or screenshots are ever enabled here.
    """
    lock = ProfileLock(profile_dir)
    lock.acquire()
    try:
        playwright = await async_playwright().start()
        context: BrowserContext | None = None
        try:
            context = await _launch(playwright, profile_dir, headless)
            await install_navigation_guard(context, allowed_host)
            yield context
        finally:
            await _finish_close(context, playwright)
    finally:
        lock.release()
