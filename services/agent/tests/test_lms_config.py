from __future__ import annotations

from pathlib import Path

import pytest

from atlas_agent.config import ConfigError, load_lms_settings

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    for name in ("LMS_BASE_URL", "ATLAS_BROWSER_PROFILE", "LMS_HEADLESS"):
        monkeypatch.delenv(name, raising=False)
    return root


def test_data_folder_is_gitignored() -> None:
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/" in [line.strip() for line in lines]


def test_missing_and_empty_base_url_are_unset(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ConfigError, match="not configured"):
        load_lms_settings(repo)
    monkeypatch.setenv("LMS_BASE_URL", "   ")
    with pytest.raises(ConfigError, match="not configured"):
        load_lms_settings(repo)


@pytest.mark.parametrize(
    "value",
    [
        "http://lms.example.edu",
        "ftp://lms.example.edu",
        "lms.example.edu",
        "https://user:pw@lms.example.edu",
        "https://lms.example.edu@evil.com",
        "https://lms.example.edu:8443",
        "https://lms.example.edu/?a=b",
        "https://lms.example.edu/#frag",
        "https://",
    ],
)
def test_base_url_must_be_plain_https(
    repo: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("LMS_BASE_URL", value)
    with pytest.raises(ConfigError) as exc_info:
        load_lms_settings(repo)
    assert "pw" not in str(exc_info.value)


def test_valid_settings_and_defaults(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMS_BASE_URL", "https://LMS.example.edu/moodle/")
    settings = load_lms_settings(repo)
    assert settings.base_url == "https://lms.example.edu/moodle"
    assert settings.host == "lms.example.edu"
    assert settings.base_path == "/moodle"
    assert settings.profile_dir == (repo / "data" / "browser-profile").resolve()
    assert settings.headless is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("false", False), ("0", False), ("No", False), ("true", True), ("on", True), ("junk", True)],
)
def test_headless_parsing(
    repo: Path, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    monkeypatch.setenv("LMS_BASE_URL", "https://lms.example.edu")
    monkeypatch.setenv("LMS_HEADLESS", raw)
    assert load_lms_settings(repo).headless is expected


def test_empty_headless_and_profile_are_unset(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LMS_BASE_URL", "https://lms.example.edu")
    monkeypatch.setenv("LMS_HEADLESS", "")
    monkeypatch.setenv("ATLAS_BROWSER_PROFILE", "")
    settings = load_lms_settings(repo)
    assert settings.headless is True
    assert settings.profile_dir == (repo / "data" / "browser-profile").resolve()


def test_profile_must_stay_inside_data(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMS_BASE_URL", "https://lms.example.edu")
    ok = repo / "data" / "other-profile"
    monkeypatch.setenv("ATLAS_BROWSER_PROFILE", str(ok))
    assert load_lms_settings(repo).profile_dir == ok.resolve()
    monkeypatch.setenv("ATLAS_BROWSER_PROFILE", "data/relative-profile")
    assert load_lms_settings(repo).profile_dir == (repo / "data" / "relative-profile").resolve()
    for bad in (
        str(repo / "elsewhere"),
        str(repo / "data"),
        "data/../outside",
        str(repo.parent / "data" / "x"),
        "..",
    ):
        monkeypatch.setenv("ATLAS_BROWSER_PROFILE", bad)
        with pytest.raises(ConfigError, match="inside the data folder") as exc_info:
            load_lms_settings(repo)
        assert str(repo) not in str(exc_info.value)
