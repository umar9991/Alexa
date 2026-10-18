from __future__ import annotations

import asyncio
import logging
import unicodedata
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from playwright.async_api import BrowserContext, Page
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from pydantic import BaseModel

from atlas_agent.browser.allowlist import is_allowed_url, is_lms_url, safe_label
from atlas_agent.browser.session import browser_session
from atlas_agent.config import LmsSettings
from atlas_agent.tools.context import ToolError

logger = logging.getLogger("atlas_agent")

MAX_COURSES = 40
MAX_ITEMS = 300
MAX_TEXT = 200
MAX_URL = 500
NAV_TIMEOUT_MS = 30_000
COURSES_WAIT_MS = 20_000
SCAN_TOTAL_TIMEOUT_S = 180
POLL_INTERVAL_S = 2.0

COURSES_SELECTOR = 'a[href*="/course/view.php"]'

# Selectors and structure follow the owner's reference scraper (lms_scraper.py).
COURSES_JS = """() => {
  const links = document.querySelectorAll('a[href*="/course/view.php"]');
  const seen = new Set();
  const unique = [];
  links.forEach((a) => {
    const href = a.href;
    const text = (a.innerText || '').trim();
    if (!text || seen.has(href)) return;
    seen.add(href);
    unique.push({ name: text.split('\\n')[0], url: href });
  });
  return unique;
}"""

# The clean name comes from .instancename minus the hidden .accesshide span.
ACTIVITIES_JS = """() => {
  const cleanName = (el) => {
    if (!el) return '';
    const copy = el.cloneNode(true);
    copy.querySelectorAll('.accesshide').forEach((node) => node.remove());
    return (copy.textContent || '').replace(/\\s+/g, ' ').trim();
  };
  const results = [];
  const seen = new Set();
  document.querySelectorAll('.activity').forEach((item) => {
    const isAssign = item.classList.contains('assign');
    const isQuiz = item.classList.contains('quiz');
    if (!isAssign && !isQuiz) return;
    const nameEl = item.querySelector('.activityname a');
    if (!nameEl) return;
    const url = nameEl.href;
    if (seen.has(url)) return;
    seen.add(url);
    const name = cleanName(item.querySelector('.instancename')) || cleanName(nameEl);
    const dueEl = item.querySelector('.duedate, .due-date, [data-region="due-date"]');
    const due = dueEl ? (dueEl.innerText || '').trim() : '';
    results.push({ name, url, due: due || null, type: isAssign ? 'assignment' : 'quiz' });
  });
  return results;
}"""

EXPIRED_MESSAGE = "LMS session expired, run lms.login"


class ScanItem(BaseModel):
    course: str
    name: str
    url: str
    type: Literal["assignment", "quiz"]
    due: str | None


class ScanResult(BaseModel):
    courses_scanned: int
    items: list[ScanItem]


class LoginResult(BaseModel):
    status: Literal["logged_in", "already_logged_in"]


class Course(BaseModel):
    name: str
    url: str


def clean_text(value: object, limit: int = MAX_TEXT) -> str:
    """Drop control and format characters, collapse whitespace, cap length."""
    if not isinstance(value, str):
        return ""
    spaced = "".join(" " if ch in "\n\r\t" else ch for ch in value)
    kept = "".join(ch for ch in spaced if unicodedata.category(ch) not in {"Cc", "Cf"})
    return " ".join(kept.split())[:limit]


def clean_url(value: object, settings: LmsSettings) -> str | None:
    """Same-host https URL without fragment, or None."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if len(candidate) > MAX_URL or not is_lms_url(candidate, settings.host):
        return None
    parts = urlsplit(candidate)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def parse_courses(raw: object, settings: LmsSettings) -> list[Course]:
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    courses: list[Course] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        url = clean_url(entry.get("url"), settings)
        name = clean_text(entry.get("name"))
        if url is None or not name or url in seen:
            continue
        seen.add(url)
        courses.append(Course(name=name, url=url))
        if len(courses) >= MAX_COURSES:
            break
    return courses


def parse_items(
    raw: object, course_name: str, settings: LmsSettings, seen: set[str]
) -> list[ScanItem]:
    if not isinstance(raw, list):
        return []
    items: list[ScanItem] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        kind = entry.get("type")
        url = clean_url(entry.get("url"), settings)
        name = clean_text(entry.get("name"))
        if kind not in ("assignment", "quiz") or url is None or not name or url in seen:
            continue
        seen.add(url)
        due = clean_text(entry.get("due")) or None
        items.append(ScanItem(course=course_name, name=name, url=url, type=kind, due=due))
    return items


def _path(url: str) -> str:
    try:
        return urlsplit(url).path
    except ValueError:
        return ""


def _is_login_page(url: str, settings: LmsSettings) -> bool:
    if is_lms_url(url, settings.host):
        return _path(url).startswith(f"{settings.base_path}/login/")
    return is_allowed_url(url, settings.host)


def _is_my_page(url: str, settings: LmsSettings) -> bool:
    if not is_lms_url(url, settings.host):
        return False
    path = _path(url)
    prefix = f"{settings.base_path}/my"
    return path == prefix or path.startswith(f"{prefix}/")


def _require_lms_page(page: Page, settings: LmsSettings) -> None:
    url = page.url
    if _is_login_page(url, settings):
        raise ToolError(EXPIRED_MESSAGE)
    if not is_lms_url(url, settings.host):
        logger.warning("unexpected page host %s", safe_label(url))
        raise ToolError("unexpected page host")


async def _goto(page: Page, url: str, *, wait_until: Literal["domcontentloaded", "load"]) -> None:
    try:
        await page.goto(url, wait_until=wait_until, timeout=NAV_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        logger.warning("page load timed out: %s", safe_label(url))
        raise ToolError("LMS page timed out") from None
    except PlaywrightError as exc:
        logger.warning("page load failed (%s): %s", type(exc).__name__, safe_label(url))
        raise ToolError("LMS page could not be loaded") from None


async def _evaluate(page: Page, script: str) -> object:
    try:
        return await page.evaluate(script)
    except PlaywrightError as exc:
        logger.warning("page read failed: %s", type(exc).__name__)
        raise ToolError("LMS page could not be read") from None


async def _first_page(context: BrowserContext) -> Page:
    if context.pages:
        return context.pages[0]
    return await context.new_page()


async def _collect_courses(page: Page, settings: LmsSettings) -> list[Course]:
    await _goto(page, f"{settings.base_url}/my/courses.php", wait_until="domcontentloaded")
    _require_lms_page(page, settings)
    try:
        await page.wait_for_selector(
            COURSES_SELECTOR, state="attached", timeout=COURSES_WAIT_MS
        )
    except PlaywrightTimeoutError:
        _require_lms_page(page, settings)
        raise ToolError("LMS courses page did not load") from None
    _require_lms_page(page, settings)
    courses = parse_courses(await _evaluate(page, COURSES_JS), settings)
    if not courses:
        raise ToolError("no courses found on the LMS page")
    return courses


async def scan_with_page(page: Page, settings: LmsSettings) -> ScanResult:
    courses = await _collect_courses(page, settings)
    items: list[ScanItem] = []
    seen: set[str] = set()
    scanned = 0
    for course in courses:
        await _goto(page, course.url, wait_until="load")
        _require_lms_page(page, settings)
        raw = await _evaluate(page, ACTIVITIES_JS)
        scanned += 1
        items.extend(parse_items(raw, course.name, settings, seen))
        if len(items) >= MAX_ITEMS:
            items = items[:MAX_ITEMS]
            break
    return ScanResult(courses_scanned=scanned, items=items)


async def run_scan(settings: LmsSettings) -> dict[str, object]:
    try:
        async with asyncio.timeout(SCAN_TOTAL_TIMEOUT_S):
            async with browser_session(
                settings.profile_dir, headless=settings.headless, allowed_host=settings.host
            ) as context:
                result = await scan_with_page(await _first_page(context), settings)
    except TimeoutError:
        raise ToolError("LMS scan timed out") from None
    except PlaywrightError as exc:
        logger.warning("browser operation failed: %s", type(exc).__name__)
        raise ToolError("browser operation failed") from None
    return result.model_dump(mode="json")


async def login_with_page(
    page: Page,
    settings: LmsSettings,
    timeout_s: int,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> Literal["logged_in", "already_logged_in"]:
    login_url = f"{settings.base_url}/login/index.php"
    await _goto(page, login_url, wait_until="domcontentloaded")
    await _goto(page, f"{settings.base_url}/my/", wait_until="domcontentloaded")
    if _is_my_page(page.url, settings):
        return "already_logged_in"
    await _goto(page, login_url, wait_until="domcontentloaded")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        if page.is_closed():
            raise ToolError("login window was closed")
        url = page.url
        if url != "about:blank" and not is_allowed_url(url, settings.host):
            logger.warning("unexpected page host %s", safe_label(url))
            raise ToolError("unexpected page host")
        if _is_my_page(url, settings):
            return "logged_in"
        if loop.time() >= deadline:
            raise ToolError("login not completed in time")
        await asyncio.sleep(poll_interval_s)


async def run_login(
    settings: LmsSettings, timeout_s: int, poll_interval_s: float = POLL_INTERVAL_S
) -> dict[str, object]:
    try:
        async with browser_session(
            settings.profile_dir, headless=False, allowed_host=settings.host
        ) as context:
            status = await login_with_page(
                await _first_page(context), settings, timeout_s, poll_interval_s
            )
    except PlaywrightError as exc:
        logger.warning("browser operation failed: %s", type(exc).__name__)
        raise ToolError("browser operation failed") from None
    return LoginResult(status=status).model_dump(mode="json")
