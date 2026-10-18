from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

from atlas_agent.browser.lms import (
    ACTIVITIES_JS,
    COURSES_JS,
    MAX_COURSES,
    MAX_TEXT,
    clean_text,
    clean_url,
    parse_courses,
    parse_items,
)
from atlas_agent.config import LmsSettings

BASE = "https://lms.example.edu"


def _chromium_missing() -> bool:
    with sync_playwright() as pw:
        return not Path(pw.chromium.executable_path).exists()


needs_chromium = pytest.mark.skipif(
    _chromium_missing(), reason="Chromium not installed: run `playwright install chromium`"
)

COURSES_HTML = f"""
<div class="courses">
  <a href="{BASE}/course/view.php?id=7"><img alt=""></a>
  <a href="{BASE}/course/view.php?id=7">Databases 101<div>Category: Computing</div></a>
  <a href="{BASE}/course/view.php?id=7">Databases 101</a>
  <a href="{BASE}/course/view.php?id=8">Networks</a>
  <a href="https://evil.example.com/course/view.php?id=9">Offsite Course</a>
  <a href="{BASE}/mod/forum/view.php?id=1">Not a course</a>
</div>
"""

ACTIVITIES_HTML = f"""
<ul>
  <li class="activity assign modtype_assign">
    <div class="activityname"><a href="{BASE}/mod/assign/view.php?id=11">
      <span class="instancename">Homework 1 <span class="accesshide"> Assignment</span></span>
    </a></div>
    <div class="duedate">Due: Friday, 1 January 2100, 11:59 PM</div>
  </li>
  <li class="activity quiz modtype_quiz">
    <div class="activityname"><a href="{BASE}/mod/quiz/view.php?id=12">
      <span class="instancename">Quiz 1 <span class="accesshide"> Quiz</span></span>
    </a></div>
  </li>
  <li class="activity forum modtype_forum">
    <div class="activityname"><a href="{BASE}/mod/forum/view.php?id=13">
      <span class="instancename">Announcements <span class="accesshide"> Forum</span></span>
    </a></div>
  </li>
  <li class="activity assign modtype_assign">
    <div class="activityname"><a href="{BASE}/mod/assign/view.php?id=11">
      <span class="instancename">Homework 1 duplicate <span class="accesshide"> Assignment</span></span>
    </a></div>
  </li>
  <li class="activity assign modtype_assign">
    <div class="activityname"><a href="https://evil.example.com/mod/assign/view.php?id=99">
      <span class="instancename">Offsite Task <span class="accesshide"> Assignment</span></span>
    </a></div>
  </li>
  <li class="activity assign modtype_assign">
    <div class="activityname"><a href="{BASE}/mod/assign/view.php?id=14#section-2">
      <span class="instancename">Lab Report <span class="accesshide"> Assignment</span></span>
    </a></div>
    <div data-region="due-date">   </div>
  </li>
  <li class="activity assign modtype_assign">
    <div class="activityname"><a href="{BASE}/mod/assign/view.php?id=15">Plain Link Name</a></div>
  </li>
  <li class="activity page modtype_page">
    <div class="activityname"><a href="{BASE}/mod/page/view.php?id=16">
      <span class="instancename">Reading</span></a></div>
  </li>
</ul>
"""


def _settings() -> LmsSettings:
    return LmsSettings(
        base_url=BASE,
        host="lms.example.edu",
        base_path="",
        profile_dir=Path("unused"),
        headless=True,
    )


def _evaluate(html: str, script: str) -> object:
    async def _call() -> object:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.set_content(html)
                return await page.evaluate(script)
            finally:
                await browser.close()

    return asyncio.run(_call())


@needs_chromium
def test_courses_js_in_real_chromium() -> None:
    raw = _evaluate(COURSES_HTML, COURSES_JS)
    assert isinstance(raw, list)
    urls = [entry["url"] for entry in raw]
    assert urls.count(f"{BASE}/course/view.php?id=7") == 1
    assert f"{BASE}/mod/forum/view.php?id=1" not in urls
    assert raw[0]["name"] == "Databases 101"  # first line only; empty-text link skipped
    courses = parse_courses(raw, _settings())
    assert [(c.name, c.url) for c in courses] == [
        ("Databases 101", f"{BASE}/course/view.php?id=7"),
        ("Networks", f"{BASE}/course/view.php?id=8"),
    ]


@needs_chromium
def test_activities_js_in_real_chromium() -> None:
    raw = _evaluate(ACTIVITIES_HTML, ACTIVITIES_JS)
    assert isinstance(raw, list)
    by_url = {entry["url"]: entry for entry in raw}
    assert f"{BASE}/mod/forum/view.php?id=13" not in by_url
    assert f"{BASE}/mod/page/view.php?id=16" not in by_url
    assert [entry["url"] for entry in raw].count(f"{BASE}/mod/assign/view.php?id=11") == 1
    first = by_url[f"{BASE}/mod/assign/view.php?id=11"]
    assert first["name"] == "Homework 1"
    assert first["type"] == "assignment"
    assert first["due"] == "Due: Friday, 1 January 2100, 11:59 PM"
    quiz = by_url[f"{BASE}/mod/quiz/view.php?id=12"]
    assert quiz["name"] == "Quiz 1"
    assert quiz["type"] == "quiz"
    assert quiz["due"] is None
    assert by_url[f"{BASE}/mod/assign/view.php?id=14#section-2"]["due"] is None
    assert by_url[f"{BASE}/mod/assign/view.php?id=15"]["name"] == "Plain Link Name"

    items = parse_items(raw, "Databases 101", _settings(), set())
    assert [(i.name, i.type, i.due) for i in items] == [
        ("Homework 1", "assignment", "Due: Friday, 1 January 2100, 11:59 PM"),
        ("Quiz 1", "quiz", None),
        ("Lab Report", "assignment", None),
        ("Plain Link Name", "assignment", None),
    ]
    assert all("evil.example.com" not in i.url for i in items)
    assert all("#" not in i.url for i in items)
    assert items[2].url == f"{BASE}/mod/assign/view.php?id=14"


def test_clean_text_strips_controls_and_caps() -> None:
    assert clean_text("a\x00b\x07c\u202ed") == "abcd"
    assert clean_text("line1\nline2\t x") == "line1 line2 x"
    assert len(clean_text("x" * 1000)) == MAX_TEXT
    assert clean_text(None) == ""
    assert clean_text(5) == ""


def test_clean_url_keeps_same_host_https_without_fragment() -> None:
    settings = _settings()
    assert clean_url(f"{BASE}/mod/quiz/view.php?id=1#top", settings) == (
        f"{BASE}/mod/quiz/view.php?id=1"
    )
    assert clean_url("https://evil.example.com/x", settings) is None
    assert clean_url("http://lms.example.edu/x", settings) is None
    assert clean_url("javascript:alert(1)", settings) is None
    assert clean_url(f"{BASE}/" + "a" * 600, settings) is None
    assert clean_url("https://login.microsoftonline.com/x", settings) is None
    assert clean_url(None, settings) is None


def test_parse_items_dedupes_across_calls_and_skips_bad_entries() -> None:
    settings = _settings()
    seen: set[str] = set()
    entry = {"name": "A\x00", "url": f"{BASE}/mod/assign/view.php?id=1", "type": "assignment"}
    first = parse_items([entry, entry, "junk", {"type": "forum"}], "C", settings, seen)
    assert [i.name for i in first] == ["A"]
    assert parse_items([entry], "D", settings, seen) == []


def test_parse_courses_caps_count() -> None:
    raw = [{"name": f"C{i}", "url": f"{BASE}/course/view.php?id={i}"} for i in range(100)]
    assert len(parse_courses(raw, _settings())) == MAX_COURSES
    assert parse_courses("nope", _settings()) == []
