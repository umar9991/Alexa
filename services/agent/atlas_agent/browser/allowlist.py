from __future__ import annotations

import logging
import unicodedata
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from playwright.async_api import BrowserContext, Route

logger = logging.getLogger("atlas_agent")

MICROSOFT_LOGIN_HOSTS = frozenset({"login.microsoftonline.com", "login.live.com"})
MICROSOFT_SUFFIX = ".microsoftonline.com"


def _parse_https_host(url: str) -> str | None:
    """Return the lowercase host of a plain https URL, or None if it is unsafe."""
    if not isinstance(url, str) or not url:
        return None
    if "\\" in url or any(unicodedata.category(ch) in {"Cc", "Cf", "Zs"} for ch in url):
        return None
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme != "https" or not parts.hostname:
        return None
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        return None
    if port not in (None, 443):
        return None
    return parts.hostname.lower()


def is_microsoft_host(host: str) -> bool:
    return host in MICROSOFT_LOGIN_HOSTS or host.endswith(MICROSOFT_SUFFIX)


def is_allowed_url(url: str, lms_host: str) -> bool:
    host = _parse_https_host(url)
    if host is None:
        return False
    return host == lms_host.lower() or is_microsoft_host(host)


def is_lms_url(url: str, lms_host: str) -> bool:
    host = _parse_https_host(url)
    return host is not None and host == lms_host.lower()


def safe_label(url: str) -> str:
    """Host and path only. Never query strings or fragments (SSO URLs carry tokens)."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
    except ValueError:
        return "invalid-url"
    return f"{host}{parts.path}"[:200]


def make_route_handler(lms_host: str) -> Callable[[Route], Awaitable[None]]:
    async def handle(route: Route) -> None:
        request = route.request
        if request.is_navigation_request() and not is_allowed_url(request.url, lms_host):
            logger.warning("blocked navigation to %s", safe_label(request.url))
            await route.abort()
            return
        await route.continue_()

    return handle


async def install_navigation_guard(context: BrowserContext, lms_host: str) -> None:
    """Abort document navigations to hosts outside the allowlist.

    The route runs once per request chain, so server redirects are not seen
    here. Callers must also check the final page URL with `is_allowed_url`.
    """
    await context.route("**/*", make_route_handler(lms_host))
