from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from jsonschema import ValidationError
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

from atlas_agent.browser import lms
from atlas_agent.browser.lms import (
    DETAIL_JS,
    MAX_DETAIL_CHARS,
    MAX_INTRO,
    parse_detail,
    validate_detail_url,
)
from atlas_agent.config import LmsSettings
from atlas_agent.core.policy import Tier, decide
from atlas_agent.tools.context import ToolContext, ToolError
from atlas_agent.tools.registry import ToolRegistry
from tests.test_lms_flow import FakePage, FakeSession

BASE = "https://lms.example.edu"
SENTINEL = "s3cret-sentinel"
ASSIGN_URL = f"{BASE}/mod/assign/view.php?id=11"
QUIZ_URL = f"{BASE}/mod/quiz/view.php?id=12"
UNSUPPORTED = "unsupported LMS url"


def _chromium_missing() -> bool:
    with sync_playwright() as pw:
        return not Path(pw.chromium.executable_path).exists()


needs_chromium = pytest.mark.skipif(
    _chromium_missing(), reason="Chromium not installed: run `playwright install chromium`"
)


def _settings(base_path: str = "") -> LmsSettings:
    return LmsSettings(
        base_url=f"{BASE}{base_path}",
        host="lms.example.edu",
        base_path=base_path,
        profile_dir=Path("unused"),
        headless=True,
    )


ASSIGNMENT_HTML = f"""
<div id="page-header"><div class="page-header-headings"><h1 class="h2">Homework 1</h1></div></div>
<div id="region-main">
  <div data-region="activity-information">
    <div data-region="activity-dates" class="activity-dates"><div class="description-inner">
      <div><strong>Opened:</strong> Monday, 1 January 2100, 9:00 AM</div>
      <div><strong>Due:</strong> Friday, 1 February 2100, 11:59 PM</div>
    </div></div>
  </div>
  <div id="intro" class="box generalbox"><div class="no-overflow">
    <p>Write a short report.</p>
    <p>Use <a href="{BASE}/pluginfile.php/1/mod_assign/intro/spec.pdf">spec.pdf</a> as a guide.</p>
  </div></div>
  <div class="submissionstatustable"><h3>Submission status</h3>
    <div class="box submissionsummarytable"><table class="generaltable"><tbody>
      <tr><th class="cell c0">Attempt number</th><td class="cell c1">This is attempt 1.</td></tr>
      <tr><th class="cell c0">Submission status</th><td class="cell c1">No attempt</td></tr>
      <tr><th class="cell c0">Grading status</th><td class="cell c1">Not graded</td></tr>
    </tbody></table></div>
  </div>
  <a href="{BASE}/pluginfile.php/1/mod_assign/intro/notes.txt#page=2">notes.txt</a>
  <a href="https://evil.example.com/pluginfile.php/1/x/evil.pdf">evil.pdf</a>
  <a href="{BASE}/mod/assign/view.php?id=11&action=editsubmission">Edit submission</a>
</div>
"""

QUIZ_HTML = """
<div class="page-header-headings"><h1>Quiz 1</h1></div>
<div id="region-main">
  <div data-region="activity-dates"><div>
    <div><strong>Opened:</strong> Monday, 1 January 2100, 9:00 AM</div>
    <div><strong>Closes:</strong> Tuesday, 2 January 2100, 9:00 AM</div>
  </div></div>
  <div id="intro"><div class="no-overflow"><p>Answer all questions.</p></div></div>
  <div class="quizinfo">
    <p>Attempts allowed: 2</p>
    <p>Time limit: 1 hour</p>
  </div>
</div>
"""

PARTIAL_HTML = """
<h1>Only A Title</h1>
<div id="region-main">
  <div data-region="activity-dates"><div><div><strong>Due:</strong> Someday</div></div></div>
</div>
"""

NOTHING_HTML = """
<div id="region-main"></div>
<div class="footer">SECRET-PAGE-TEXT</div>
"""


def _extract(html: str) -> object:
    async def _call() -> object:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.set_content(html)
                return await page.evaluate(DETAIL_JS)
            finally:
                await browser.close()

    return asyncio.run(_call())


@needs_chromium
def test_assignment_fixture_in_real_chromium() -> None:
    detail = parse_detail(_extract(ASSIGNMENT_HTML), "assignment", _settings())
    assert detail.type == "assignment"
    assert detail.title == "Homework 1"
    assert detail.opened == "Monday, 1 January 2100, 9:00 AM"
    assert detail.due == "Friday, 1 February 2100, 11:59 PM"
    assert detail.submission_status == "No attempt"
    assert detail.grading_status == "Not graded"
    assert detail.intro_text == "Write a short report. Use spec.pdf as a guide."
    assert [(a.name, a.url) for a in detail.attachments] == [
        ("spec.pdf", f"{BASE}/pluginfile.php/1/mod_assign/intro/spec.pdf"),
        ("notes.txt", f"{BASE}/pluginfile.php/1/mod_assign/intro/notes.txt"),
    ]
    assert detail.time_limit is None and detail.attempts_allowed is None
    assert detail.missing == []


@needs_chromium
def test_quiz_fixture_in_real_chromium() -> None:
    detail = parse_detail(_extract(QUIZ_HTML), "quiz", _settings())
    assert detail.type == "quiz"
    assert detail.title == "Quiz 1"
    assert detail.opened == "Monday, 1 January 2100, 9:00 AM"
    assert detail.due == "Tuesday, 2 January 2100, 9:00 AM"
    assert detail.intro_text == "Answer all questions."
    assert detail.time_limit == "1 hour"
    assert detail.attempts_allowed == "2"
    assert detail.submission_status is None
    assert detail.missing == ["submission_status", "grading_status", "attachments"]


@needs_chromium
def test_partial_page_has_nulls_and_missing_list() -> None:
    detail = parse_detail(_extract(PARTIAL_HTML), "assignment", _settings())
    assert detail.title == "Only A Title"
    assert detail.due == "Someday"
    assert detail.intro_text is None
    assert detail.opened is None
    assert detail.attachments == []
    assert detail.missing == [
        "intro_text",
        "opened",
        "submission_status",
        "grading_status",
        "attachments",
    ]


@needs_chromium
def test_page_with_nothing_useful_is_unexpected_layout() -> None:
    with pytest.raises(ToolError) as exc_info:
        parse_detail(_extract(NOTHING_HTML), "assignment", _settings())
    message = str(exc_info.value)
    assert message.startswith("unexpected page layout")
    assert "title" in message and "intro_text" in message
    assert "SECRET-PAGE-TEXT" not in message


@needs_chromium
def test_huge_intro_with_control_characters_in_real_chromium() -> None:
    html = (
        "<h1>T</h1><div id='region-main'></div>"
        f"<div id='intro'>a\x07b\u200bc\n\n{'word ' * 100_000}</div>"
    )
    detail = parse_detail(_extract(html), "assignment", _settings())
    assert detail.intro_text is not None
    assert len(detail.intro_text) <= MAX_INTRO
    assert detail.intro_text.startswith("abc word")
    assert "\x07" not in detail.intro_text and "\u200b" not in detail.intro_text
    assert len(detail.model_dump_json()) < MAX_DETAIL_CHARS


def test_parse_detail_cleans_caps_and_limits_output_size() -> None:
    settings = _settings()
    long = "x" * 1000
    raw = {
        "title": f"T\x00{long}",
        "intro_text": "i\x01 " + "y" * 50_000,
        "opened": long,
        "due": long,
        "submission_status": long,
        "grading_status": long,
        "time_limit": long,
        "attempts_allowed": long,
        "attachments": [
            {"name": long, "url": f"{BASE}/pluginfile.php/1/f{i}.pdf?forcedownload=1"}
            for i in range(60)
        ],
    }
    detail = parse_detail(raw, "quiz", settings)
    assert detail.title is not None and len(detail.title) == 300 and "\x00" not in detail.title
    assert detail.intro_text is not None and "\x01" not in detail.intro_text
    for field in (detail.opened, detail.due, detail.submission_status, detail.time_limit):
        assert field is not None and len(field) == 300
    assert len(detail.model_dump_json()) <= MAX_DETAIL_CHARS
    assert len(detail.attachments) <= 20
    assert len(detail.model_dump_json()) < 8000


def test_parse_detail_attachment_rules() -> None:
    raw = {
        "title": "T",
        "attachments": [
            {"name": "", "url": f"{BASE}/pluginfile.php/1/my%20file.pdf#frag"},
            {"name": "dup", "url": f"{BASE}/pluginfile.php/1/my%20file.pdf"},
            {"name": "off", "url": "https://evil.example.com/pluginfile.php/1/a.pdf"},
            {"name": "http", "url": "http://lms.example.edu/pluginfile.php/1/a.pdf"},
            {"name": "page", "url": f"{BASE}/mod/assign/view.php?id=1"},
            "junk",
        ],
    }
    detail = parse_detail(raw, "assignment", _settings())
    assert [(a.name, a.url) for a in detail.attachments] == [
        ("my file.pdf", f"{BASE}/pluginfile.php/1/my%20file.pdf")
    ]
    with pytest.raises(ToolError, match="unexpected page layout"):
        parse_detail("not a dict", "assignment", _settings())


VALID_URLS = [
    (ASSIGN_URL, "assignment", ""),
    (QUIZ_URL, "quiz", ""),
    ("https://LMS.example.edu/mod/quiz/view.php?id=007", "quiz", ""),
    (f"{BASE}/moodle/mod/assign/view.php?id=5", "assignment", "/moodle"),
]

INVALID_URLS = [
    "https://evil.example.com/mod/assign/view.php?id=11",
    "https://lms.example.edu.evil.com/mod/assign/view.php?id=11",
    "http://lms.example.edu/mod/assign/view.php?id=11",
    "https://user@lms.example.edu/mod/assign/view.php?id=11",
    "https://user:pw@lms.example.edu/mod/assign/view.php?id=11",
    "https://lms.example.edu@evil.com/mod/assign/view.php?id=11",
    "https://lms.example.edu:443/mod/assign/view.php?id=11",
    "https://lms.example.edu:8443/mod/assign/view.php?id=11",
    f"{BASE}/mod/assign/view.php?id=11&sesskey=x",
    f"{BASE}/mod/assign/view.php?id=11&action=grading",
    f"{BASE}/mod/assign/view.php?id=1&id=2",
    f"{BASE}/mod/assign/view.php?action=editsubmission&id=11",
    f"{BASE}/mod/assign/view.php",
    f"{BASE}/mod/assign/view.php?",
    f"{BASE}/mod/assign/view.php?id=",
    f"{BASE}/mod/assign/view.php?id=abc",
    f"{BASE}/mod/assign/view.php?id=1x",
    f"{BASE}/mod/assign/view.php?id=-1",
    f"{BASE}/mod/assign/view.php?id=\uff11\uff12",
    f"{BASE}/mod/assign/view.php?id=12345678901",
    f"{BASE}/mod/assign/view.php?id=11#section",
    f"{BASE}/mod/assign/view.php?id=11#",
    f"{BASE}/login/logout.php?sesskey=x",
    f"{BASE}/login/logout.php?id=1",
    f"{BASE}/admin/index.php",
    f"{BASE}/mod/forum/view.php?id=1",
    f"{BASE}/mod/assign/view.php/extra?id=1",
    f"{BASE}/mod/assign/%76iew.php?id=1",
    f"{BASE}/mod/assign/view.php;x=1?id=1",
    f"{BASE}/mod/assign/../../login/logout.php?id=1",
    f"{BASE}/moodle/mod/assign/view.php?id=11",
    f"{BASE}/mod/assign/view.php?id=11 ",
    " " + ASSIGN_URL,
    "https://lms.example.edu\\@evil.com/mod/assign/view.php?id=11",
    "https://login.microsoftonline.com/mod/assign/view.php?id=11",
    "javascript:alert(1)",
    "file:///C:/mod/assign/view.php?id=1",
    "/mod/assign/view.php?id=11",
    "",
    f"{BASE}/mod/assign/view.php?id=1" + "1" * 600,
]


@pytest.mark.parametrize(("url", "kind", "base_path"), VALID_URLS)
def test_valid_detail_urls(url: str, kind: str, base_path: str) -> None:
    settings = _settings(base_path)
    got_kind, canonical = validate_detail_url(url, settings)
    assert got_kind == kind
    assert canonical.startswith(f"{BASE}{base_path}/mod/")
    assert "#" not in canonical and canonical.count("?") == 1


@pytest.mark.parametrize("url", INVALID_URLS)
def test_invalid_detail_urls(url: str) -> None:
    with pytest.raises(ToolError) as exc_info:
        validate_detail_url(url, _settings())
    assert str(exc_info.value) == UNSUPPORTED


def test_non_string_url_is_unsupported() -> None:
    for value in (None, 5, ["x"], {"url": ASSIGN_URL}):
        with pytest.raises(ToolError, match="^unsupported LMS url$"):
            validate_detail_url(value, _settings())


class DetailPage(FakePage):
    def __init__(self, raw: object = None, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.raw = raw

    async def evaluate(self, script: str) -> object:
        if self.evaluate_error is not None:
            raise self.evaluate_error
        assert script == DETAIL_JS
        return self.raw


RAW_OK = {"title": "Homework 1", "intro_text": "Do it.", "due": "Soon"}


def _ctx() -> ToolContext:
    return ToolContext(workspace=None, run_id="run-detail")


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("LMS_BASE_URL", "ATLAS_BROWSER_PROFILE", "LMS_HEADLESS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LMS_BASE_URL", BASE)


def _install(monkeypatch: pytest.MonkeyPatch, page: FakePage) -> FakeSession:
    session = FakeSession(page)
    monkeypatch.setattr(lms, "browser_session", session)
    return session


def _run_tool(url: object) -> dict[str, object]:
    return asyncio.run(ToolRegistry.load().run("lms.assignment_detail", {"url": url}, _ctx()))


def test_tool_runs_one_get_navigation_and_never_interacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, tmp_path)
    page = DetailPage(RAW_OK)
    session = _install(monkeypatch, page)
    result = _run_tool(ASSIGN_URL)
    assert result["title"] == "Homework 1"
    assert result["type"] == "assignment"
    assert result["due"] == "Soon"
    assert result["opened"] is None
    assert "opened" in result["missing"]  # type: ignore[operator]
    assert set(result) == {
        "type",
        "title",
        "due",
        "opened",
        "submission_status",
        "grading_status",
        "intro_text",
        "attachments",
        "time_limit",
        "attempts_allowed",
        "missing",
    }
    assert page.visited == [ASSIGN_URL]
    assert page.typed == []
    assert session.closed is True


def test_unsupported_url_never_starts_a_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, tmp_path)

    @asynccontextmanager
    async def forbidden(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        raise AssertionError("browser must not start for an unsupported URL")
        yield  # pragma: no cover

    monkeypatch.setattr(lms, "browser_session", forbidden)
    for url in INVALID_URLS[:12] + [f"{BASE}/login/logout.php?sesskey=x"]:
        with pytest.raises(ToolError, match="^unsupported LMS url$"):
            _run_tool(url)


def test_tool_requires_base_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, tmp_path)
    monkeypatch.delenv("LMS_BASE_URL")
    with pytest.raises(ToolError, match="^LMS_BASE_URL is not configured$"):
        _run_tool(ASSIGN_URL)


DETAIL_ERRORS: list[tuple[str, FakePage, str]] = [
    (
        "expired",
        DetailPage(RAW_OK, redirects={ASSIGN_URL: f"{BASE}/login/index.php?token={SENTINEL}"}),
        "LMS session expired, run lms.login",
    ),
    (
        "expired-sso",
        DetailPage(
            RAW_OK,
            redirects={ASSIGN_URL: f"https://login.microsoftonline.com/x?code={SENTINEL}#t"},
        ),
        "LMS session expired, run lms.login",
    ),
    (
        "goto-timeout",
        DetailPage(RAW_OK, goto_error=PlaywrightTimeoutError(f"{ASSIGN_URL}&t={SENTINEL}")),
        "LMS page timed out",
    ),
    (
        "goto-error",
        DetailPage(RAW_OK, goto_error=PlaywrightError(f"net::ERR {ASSIGN_URL}&t={SENTINEL}")),
        "LMS page could not be loaded",
    ),
    (
        "selector-timeout",
        DetailPage(RAW_OK, selector_error=PlaywrightTimeoutError(f"x?t={SENTINEL}")),
        "LMS page did not load",
    ),
    (
        "selector-timeout-then-login",
        DetailPage(
            RAW_OK,
            selector_error=PlaywrightTimeoutError("x"),
            url_after_selector_error=f"{BASE}/login/index.php?t={SENTINEL}",
        ),
        "LMS session expired, run lms.login",
    ),
    (
        "other-lms-page",
        DetailPage(RAW_OK, redirects={ASSIGN_URL: f"{BASE}/enrol/index.php?id=11&t={SENTINEL}"}),
        "unexpected page layout",
    ),
    (
        "other-host",
        DetailPage(RAW_OK, redirects={ASSIGN_URL: f"https://evil.example.com/x?t={SENTINEL}"}),
        "unexpected page host",
    ),
    (
        "evaluate-error",
        DetailPage(RAW_OK, evaluate_error=PlaywrightError(f"boom {ASSIGN_URL}&t={SENTINEL}")),
        "LMS page could not be read",
    ),
]


@pytest.mark.parametrize(
    ("name", "page", "message"), DETAIL_ERRORS, ids=[e[0] for e in DETAIL_ERRORS]
)
def test_detail_error_mapping_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    name: str,
    page: FakePage,
    message: str,
) -> None:
    del name
    caplog.set_level(logging.DEBUG)
    _env(monkeypatch, tmp_path)
    page.url = "about:blank"
    page.visited.clear()
    session = _install(monkeypatch, page)
    with pytest.raises(ToolError) as exc_info:
        _run_tool(ASSIGN_URL)
    assert str(exc_info.value) == message
    for blob in (str(exc_info.value), caplog.text):
        assert SENTINEL not in blob
        assert "?" not in blob
        assert "#" not in blob
    assert session.closed is True


def test_unexpected_layout_error_lists_section_names_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, tmp_path)
    _install(monkeypatch, DetailPage({"title": None, "footer": "SECRET-PAGE-TEXT"}))
    with pytest.raises(ToolError) as exc_info:
        _run_tool(QUIZ_URL)
    message = str(exc_info.value)
    assert message.startswith("unexpected page layout; sections not found: title, intro_text")
    assert "time_limit" in message
    assert "SECRET-PAGE-TEXT" not in message


def test_detail_total_timeout_closes_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, tmp_path)
    monkeypatch.setattr(lms, "DETAIL_TOTAL_TIMEOUT_S", 0.05)
    session = _install(monkeypatch, DetailPage(RAW_OK, goto_delay=5))
    with pytest.raises(ToolError, match="^LMS page timed out$"):
        _run_tool(ASSIGN_URL)
    assert session.closed is True


def test_registry_tier_policy_and_schema() -> None:
    registry = ToolRegistry.load()
    tool = registry.get("lms.assignment_detail")
    assert tool is not None
    assert tool.spec.tier is Tier.T0
    assert decide(tool.spec, {"url": ASSIGN_URL}).action == "allow"
    registry.validate_args("lms.assignment_detail", {"url": ASSIGN_URL})
    for args in ({}, {"url": 5}, {"url": ASSIGN_URL, "extra": 1}, {"url": "x" * 501}):
        with pytest.raises(ValidationError):
            registry.validate_args("lms.assignment_detail", args)
