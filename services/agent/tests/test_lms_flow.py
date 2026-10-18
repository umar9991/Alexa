from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from jsonschema import ValidationError
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from atlas_agent.browser import lms
from atlas_agent.browser.session import ProfileLock
from atlas_agent.config import LmsSettings
from atlas_agent.core.policy import Tier, decide
from atlas_agent.tools.context import ToolContext, ToolError
from atlas_agent.tools.registry import ToolRegistry

BASE = "https://lms.example.edu"
SENTINEL = "s3cret-sentinel"
LOGIN_URL = f"{BASE}/login/index.php"
MY_URL = f"{BASE}/my/"
COURSES_URL = f"{BASE}/my/courses.php"
SSO_URL = f"https://login.microsoftonline.com/common/oauth2/authorize?code={SENTINEL}#t={SENTINEL}"
TOKEN_LOGIN_URL = f"{LOGIN_URL}?token={SENTINEL}#frag={SENTINEL}"


def _settings(tmp_path: Path) -> LmsSettings:
    return LmsSettings(
        base_url=BASE,
        host="lms.example.edu",
        base_path="",
        profile_dir=tmp_path / "profile",
        headless=True,
    )


def _course(index: int) -> dict[str, object]:
    return {"name": f"Course {index}", "url": f"{BASE}/course/view.php?id={index}"}


def _item(course: int, index: int, kind: str = "assignment") -> dict[str, object]:
    return {
        "name": f"Item {course}-{index}",
        "url": f"{BASE}/mod/assign/view.php?id={course * 1000 + index}",
        "type": kind,
        "due": None,
    }


class FakePage:
    def __init__(
        self,
        *,
        courses: list[dict[str, object]] | None = None,
        items_by_url: dict[str, list[dict[str, object]]] | None = None,
        redirects: dict[str, str] | None = None,
        goto_error: Exception | None = None,
        selector_error: Exception | None = None,
        url_after_selector_error: str | None = None,
        evaluate_error: Exception | None = None,
        goto_delay: float = 0.0,
        poll_urls: list[str] | None = None,
    ) -> None:
        self.url = "about:blank"
        self.courses = courses if courses is not None else []
        self.items_by_url = items_by_url or {}
        self.redirects = redirects or {}
        self.goto_error = goto_error
        self.selector_error = selector_error
        self.url_after_selector_error = url_after_selector_error
        self.evaluate_error = evaluate_error
        self.goto_delay = goto_delay
        self.poll_urls = list(poll_urls or [])
        self.closed = False
        self.visited: list[str] = []
        self.typed: list[str] = []

    async def goto(self, url: str, *, wait_until: str | None = None, timeout: float | None = None) -> None:
        del wait_until, timeout
        self.visited.append(url)
        if self.goto_delay:
            await asyncio.sleep(self.goto_delay)
        if self.goto_error is not None:
            raise self.goto_error
        self.url = self.redirects.get(url, url)

    async def wait_for_selector(
        self, selector: str, *, state: str | None = None, timeout: float | None = None
    ) -> None:
        del selector, state, timeout
        if self.selector_error is not None:
            if self.url_after_selector_error:
                self.url = self.url_after_selector_error
            raise self.selector_error

    async def evaluate(self, script: str) -> object:
        if self.evaluate_error is not None:
            raise self.evaluate_error
        if script == lms.COURSES_JS:
            return self.courses
        return self.items_by_url.get(self.url, [])

    def is_closed(self) -> bool:
        if self.poll_urls:
            self.url = self.poll_urls.pop(0)
        return self.closed

    async def fill(self, *args: object, **kwargs: object) -> None:
        self.typed.append("fill")

    async def click(self, *args: object, **kwargs: object) -> None:
        self.typed.append("click")

    async def type(self, *args: object, **kwargs: object) -> None:
        self.typed.append("type")


class FakeContext:
    def __init__(self, page: FakePage) -> None:
        self.pages = [page]

    async def new_page(self) -> FakePage:
        return self.pages[0]


class FakeSession:
    def __init__(self, page: FakePage) -> None:
        self.page = page
        self.headless_calls: list[bool] = []
        self.closed = False

    @asynccontextmanager
    async def __call__(
        self, profile_dir: Path, *, headless: bool, allowed_host: str
    ) -> AsyncIterator[FakeContext]:
        del profile_dir, allowed_host
        self.headless_calls.append(headless)
        try:
            yield FakeContext(self.page)
        finally:
            self.closed = True


def _install(monkeypatch: pytest.MonkeyPatch, page: FakePage) -> FakeSession:
    session = FakeSession(page)
    monkeypatch.setattr(lms, "browser_session", session)
    return session


def _scan(settings: LmsSettings) -> dict[str, object]:
    return asyncio.run(lms.run_scan(settings))


def _assert_clean(text: str) -> None:
    assert SENTINEL not in text
    assert "?" not in text
    assert "#" not in text


def test_scan_success_shape_dedupe_and_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = _item(1, 1)
    page = FakePage(
        courses=[_course(1), _course(2)],
        items_by_url={
            _course(1)["url"]: [shared, _item(1, 2, "quiz")],  # type: ignore[index]
            _course(2)["url"]: [shared, _item(2, 1)],  # type: ignore[index]
        },
    )
    session = _install(monkeypatch, page)
    result = _scan(_settings(tmp_path))
    assert session.headless_calls == [True]
    assert session.closed is True
    assert result["courses_scanned"] == 2
    items = result["items"]
    assert isinstance(items, list)
    assert [item["name"] for item in items] == ["Item 1-1", "Item 1-2", "Item 2-1"]
    assert set(items[0]) == {"course", "name", "url", "type", "due"}
    assert items[1]["type"] == "quiz"
    assert items[0]["course"] == "Course 1"
    assert page.visited[0] == COURSES_URL
    assert page.typed == []


def test_scan_zero_items_on_healthy_scan_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, FakePage(courses=[_course(1)]))
    assert _scan(_settings(tmp_path)) == {"courses_scanned": 1, "items": []}


def test_scan_caps_courses_and_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    many = [_course(i) for i in range(1, 46)]
    page = FakePage(courses=many, items_by_url={c["url"]: [_item(1, 1)] for c in many})  # type: ignore[misc]
    _install(monkeypatch, page)
    assert _scan(_settings(tmp_path))["courses_scanned"] == 40

    big_page = FakePage(
        courses=[_course(i) for i in range(1, 21)],
        items_by_url={
            _course(i)["url"]: [_item(i, n) for n in range(20)]  # type: ignore[index]
            for i in range(1, 21)
        },
    )
    _install(monkeypatch, big_page)
    result = _scan(_settings(tmp_path))
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 300
    assert result["courses_scanned"] == 15


SCENARIOS: list[tuple[str, Callable[[], FakePage], str]] = [
    (
        "expired-lms-login",
        lambda: FakePage(courses=[_course(1)], redirects={COURSES_URL: TOKEN_LOGIN_URL}),
        "LMS session expired, run lms.login",
    ),
    (
        "expired-microsoft",
        lambda: FakePage(courses=[_course(1)], redirects={COURSES_URL: SSO_URL}),
        "LMS session expired, run lms.login",
    ),
    (
        "expired-after-selector-timeout",
        lambda: FakePage(
            selector_error=PlaywrightTimeoutError(f"timeout {BASE}/x?t={SENTINEL}"),
            url_after_selector_error=TOKEN_LOGIN_URL,
        ),
        "LMS session expired, run lms.login",
    ),
    (
        "expired-mid-scan",
        lambda: FakePage(
            courses=[_course(1), _course(2)],
            redirects={_course(2)["url"]: TOKEN_LOGIN_URL},  # type: ignore[dict-item]
        ),
        "LMS session expired, run lms.login",
    ),
    (
        "goto-timeout",
        lambda: FakePage(
            goto_error=PlaywrightTimeoutError(f"Timeout navigating to {COURSES_URL}?t={SENTINEL}")
        ),
        "LMS page timed out",
    ),
    (
        "goto-error",
        lambda: FakePage(goto_error=PlaywrightError(f"net::ERR_FAILED at {SSO_URL}")),
        "LMS page could not be loaded",
    ),
    (
        "selector-timeout",
        lambda: FakePage(selector_error=PlaywrightTimeoutError(f"{BASE}/?t={SENTINEL}")),
        "LMS courses page did not load",
    ),
    (
        "no-courses",
        lambda: FakePage(courses=[]),
        "no courses found on the LMS page",
    ),
    (
        "unexpected-host",
        lambda: FakePage(redirects={COURSES_URL: f"https://evil.example.com/x?t={SENTINEL}"}),
        "unexpected page host",
    ),
    (
        "evaluate-error",
        lambda: FakePage(evaluate_error=PlaywrightError(f"boom {SSO_URL}")),
        "LMS page could not be read",
    ),
]


@pytest.mark.parametrize(("name", "make_page", "message"), SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_scan_error_mapping_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    name: str,
    make_page: Callable[[], FakePage],
    message: str,
) -> None:
    del name
    caplog.set_level(logging.DEBUG)
    session = _install(monkeypatch, make_page())
    with pytest.raises(ToolError) as exc_info:
        _scan(_settings(tmp_path))
    assert str(exc_info.value) == message
    _assert_clean(str(exc_info.value))
    _assert_clean(caplog.text)
    assert session.closed is True


def test_scan_total_timeout_closes_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lms, "SCAN_TOTAL_TIMEOUT_S", 0.05)
    session = _install(monkeypatch, FakePage(courses=[_course(1)], goto_delay=5))
    with pytest.raises(ToolError, match="^LMS scan timed out$"):
        _scan(_settings(tmp_path))
    assert session.closed is True


def test_scan_busy_profile_and_browser_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    held = ProfileLock(settings.profile_dir)
    held.acquire()
    try:
        with pytest.raises(ToolError, match="^browser profile is busy$") as exc_info:
            _scan(settings)
        assert str(tmp_path) not in str(exc_info.value)
    finally:
        held.release()

    @asynccontextmanager
    async def broken(*args: object, **kwargs: object) -> AsyncIterator[FakeContext]:
        raise PlaywrightError(f"crash at {SSO_URL}")
        yield FakeContext(FakePage())  # pragma: no cover

    monkeypatch.setattr(lms, "browser_session", broken)
    with pytest.raises(ToolError, match="^browser operation failed$") as failure:
        _scan(settings)
    _assert_clean(str(failure.value))


def _login(settings: LmsSettings, timeout_s: int = 5) -> dict[str, object]:
    return asyncio.run(lms.run_login(settings, timeout_s, poll_interval_s=0))


def test_login_already_logged_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    page = FakePage()
    session = _install(monkeypatch, page)
    assert _login(_settings(tmp_path)) == {"status": "already_logged_in"}
    assert session.headless_calls == [False]
    assert page.visited == [LOGIN_URL, MY_URL]
    assert page.typed == []


def test_login_waits_for_human_and_never_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    page = FakePage(
        redirects={MY_URL: TOKEN_LOGIN_URL},
        poll_urls=[SSO_URL, SSO_URL, TOKEN_LOGIN_URL, f"{BASE}/my/?x={SENTINEL}#y"],
    )
    session = _install(monkeypatch, page)
    result = _login(_settings(tmp_path))
    assert result == {"status": "logged_in"}
    assert session.headless_calls == [False]
    assert session.closed is True
    assert page.typed == []
    _assert_clean(json.dumps(result))
    _assert_clean(caplog.text)


def test_login_timeout_window_closed_and_unexpected_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    settings = _settings(tmp_path)
    redirects = {MY_URL: TOKEN_LOGIN_URL}
    _install(monkeypatch, FakePage(redirects=redirects))
    with pytest.raises(ToolError, match="^login not completed in time$") as timed_out:
        _login(settings, timeout_s=0)
    _assert_clean(str(timed_out.value))

    closed_page = FakePage(redirects=redirects)
    closed_page.closed = True
    _install(monkeypatch, closed_page)
    with pytest.raises(ToolError, match="^login window was closed$"):
        _login(settings)

    _install(
        monkeypatch,
        FakePage(redirects=redirects, poll_urls=[f"https://evil.example.com/x?t={SENTINEL}"]),
    )
    with pytest.raises(ToolError, match="^unexpected page host$") as bad_host:
        _login(settings)
    _assert_clean(str(bad_host.value))
    _assert_clean(caplog.text)


def _ctx() -> ToolContext:
    return ToolContext(workspace=None, run_id="run-lms")


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **values: str) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("LMS_BASE_URL", "ATLAS_BROWSER_PROFILE", "LMS_HEADLESS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_tools_require_base_url_and_run_through_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ToolRegistry.load()
    _env(monkeypatch, tmp_path)
    for name in ("lms.scan", "lms.login"):
        with pytest.raises(ToolError, match="^LMS_BASE_URL is not configured$"):
            asyncio.run(registry.run(name, {}, _ctx()))
    _env(monkeypatch, tmp_path, LMS_BASE_URL="http://lms.example.edu")
    with pytest.raises(ToolError, match="https"):
        asyncio.run(registry.run("lms.scan", {}, _ctx()))

    _env(monkeypatch, tmp_path, LMS_BASE_URL=BASE, LMS_HEADLESS="false")
    _install(monkeypatch, FakePage(courses=[_course(1)], items_by_url={}))
    scanned = asyncio.run(registry.run("lms.scan", {}, _ctx()))
    assert scanned == {"courses_scanned": 1, "items": []}

    seen: list[int] = []

    async def fake_login(settings: LmsSettings, timeout_s: int) -> dict[str, object]:
        assert settings.host == "lms.example.edu"
        seen.append(timeout_s)
        return {"status": "logged_in"}

    monkeypatch.setattr(lms, "run_login", fake_login)
    assert asyncio.run(registry.run("lms.login", {}, _ctx())) == {"status": "logged_in"}
    assert asyncio.run(registry.run("lms.login", {"timeout_s": 42}, _ctx())) == {
        "status": "logged_in"
    }
    assert seen == [180, 42]


def test_registry_tiers_policy_and_schema_validation() -> None:
    registry = ToolRegistry.load()
    login = registry.get("lms.login")
    scan = registry.get("lms.scan")
    assert login is not None and scan is not None
    assert login.spec.tier is Tier.T1
    assert scan.spec.tier is Tier.T0
    assert decide(login.spec, {}).action == "allow"
    assert decide(scan.spec, {}).action == "allow"
    registry.validate_args("lms.login", {})
    registry.validate_args("lms.login", {"timeout_s": 300})
    registry.validate_args("lms.scan", {})
    for name, args in (
        ("lms.login", {"timeout_s": 301}),
        ("lms.login", {"timeout_s": 0}),
        ("lms.login", {"timeout_s": "5"}),
        ("lms.login", {"username": "x"}),
        ("lms.scan", {"url": "https://evil.example.com"}),
    ):
        with pytest.raises(ValidationError):
            registry.validate_args(name, args)
