from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

from atlas_agent.browser.allowlist import (
    install_navigation_guard,
    is_allowed_url,
    is_lms_url,
    make_route_handler,
    safe_label,
)

LMS = "lms.example.edu"


def _chromium_missing() -> bool:
    with sync_playwright() as pw:
        return not Path(pw.chromium.executable_path).exists()


ALLOWED = [
    "https://lms.example.edu/my/",
    "https://LMS.example.edu/login/index.php?x=1#frag",
    "https://lms.example.edu:443/my/",
    "https://login.microsoftonline.com/common/oauth2/authorize?client_id=x",
    "https://login.live.com/login.srf",
    "https://tenant.microsoftonline.com/path",
    "https://a.b.microsoftonline.com/",
]

REJECTED = [
    "https://lms.example.edu.evil.com/my/",
    "https://evil.com/lms.example.edu",
    "https://lms.example.edu@evil.com/",
    "https://evil.com@lms.example.edu/",
    "https://user:pw@lms.example.edu/",
    "https://lms.example.edu\\@evil.com/",
    "https://lms.example.edu:8443/",
    "http://lms.example.edu/my/",
    "http://login.microsoftonline.com/",
    "file:///C:/Windows/win.ini",
    "javascript:alert(1)",
    "data:text/html,<p>x</p>",
    "about:blank",
    "//lms.example.edu/my/",
    "https://microsoftonline.com.evil.com/",
    "https://evilmicrosoftonline.com/",
    "https://microsoftonline.com/",
    "https://login.live.com.evil.com/",
    "https://lms.example.edu./my/",
    "https://lms.example.edu/\n@evil.com",
    "https://lms.example.edu /x",
    "",
]


@pytest.mark.parametrize("url", ALLOWED)
def test_allowed_urls(url: str) -> None:
    assert is_allowed_url(url, LMS) is True


@pytest.mark.parametrize("url", REJECTED)
def test_rejected_urls(url: str) -> None:
    assert is_allowed_url(url, LMS) is False


def test_is_lms_url_excludes_microsoft_hosts() -> None:
    assert is_lms_url("https://lms.example.edu/x", LMS) is True
    assert is_lms_url("https://login.microsoftonline.com/x", LMS) is False
    assert is_lms_url("https://lms.example.edu.evil.com/x", LMS) is False


def test_safe_label_is_host_and_path_only() -> None:
    label = safe_label("https://login.microsoftonline.com/a/b?code=SECRET#token=SECRET")
    assert label == "login.microsoftonline.com/a/b"


class _FakeRequest:
    def __init__(self, url: str, *, navigation: bool) -> None:
        self.url = url
        self._navigation = navigation

    def is_navigation_request(self) -> bool:
        return self._navigation


class _FakeRoute:
    def __init__(self, request: _FakeRequest) -> None:
        self.request = request
        self.calls: list[str] = []

    async def abort(self) -> None:
        self.calls.append("abort")

    async def continue_(self) -> None:
        self.calls.append("continue")


def _route_calls(url: str, *, navigation: bool) -> list[str]:
    route = _FakeRoute(_FakeRequest(url, navigation=navigation))
    handler = make_route_handler(LMS)
    asyncio.run(handler(route))  # type: ignore[arg-type]
    return route.calls


def test_route_handler_aborts_offsite_navigation_only() -> None:
    assert _route_calls("https://evil.com/x", navigation=True) == ["abort"]
    assert _route_calls("http://lms.example.edu/my/", navigation=True) == ["abort"]
    assert _route_calls("https://lms.example.edu/my/", navigation=True) == ["continue"]
    assert _route_calls("https://login.live.com/x", navigation=True) == ["continue"]
    # Subresources (images, scripts, XHR) are not top-level navigations.
    assert _route_calls("https://cdn.evil.com/a.js", navigation=False) == ["continue"]


@pytest.mark.skipif(
    _chromium_missing(), reason="Chromium not installed: run `playwright install chromium`"
)
def test_route_guard_blocks_offsite_goto_in_real_chromium() -> None:
    async def _call() -> None:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                context = await browser.new_context()
                await install_navigation_guard(context, LMS)
                page = await context.new_page()
                with pytest.raises(PlaywrightError):
                    await page.goto("https://evil.example.com/steal?token=1", timeout=10_000)
                assert "evil.example.com" not in page.url
            finally:
                await browser.close()

    asyncio.run(_call())
