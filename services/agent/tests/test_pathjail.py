from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from atlas_agent.core.pathjail import display_path, resolve_in_workspace
from atlas_agent.tools.context import ToolError


def _assert_no_abs(workspace: Path, payload: object) -> None:
    blob = json.dumps(payload, default=str, ensure_ascii=False)
    if isinstance(payload, Exception):
        blob += str(payload)
    resolved = str(workspace.resolve())
    assert resolved not in blob
    assert resolved.replace("\\", "/") not in blob
    assert resolved.replace("/", "\\") not in blob


@pytest.mark.parametrize(
    "rel",
    [
        "a",
        "a/b/c.txt",
        "a\\b",
        "Courses/DBMS/Assignments/Lab-06",
        "Café/笔记.txt",
    ],
)
def test_jail_accepts_relative_paths(tmp_path: Path, rel: str) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    resolved = resolve_in_workspace(workspace, rel)
    shown = display_path(workspace, resolved)
    assert ".." not in shown.split("/")
    assert "\\" not in shown
    _assert_no_abs(workspace, shown)
    assert _is_inside(workspace, resolved)


def test_jail_accepts_root_when_allowed(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    resolved = resolve_in_workspace(workspace, ".", allow_root=True)
    assert display_path(workspace, resolved) == "."
    assert resolved == workspace.resolve()


@pytest.mark.parametrize(
    "rel",
    [
        "..",
        "a/../../x",
        "a/..",
        "/etc/passwd",
        "C:\\Windows",
        "C:foo",
        "//server/share/x",
        "~/x",
        "x\x00y",
        "file.txt:stream",
        "CON",
        "con.txt",
        "nul",
        "COM1.log",
        "foo.",
        "foo ",
        "a<b",
        "",
        "a" * 201,
    ],
)
def test_jail_rejects_unsafe_paths(tmp_path: Path, rel: str) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises(ToolError) as exc_info:
        resolve_in_workspace(workspace, rel)
    _assert_no_abs(workspace, str(exc_info.value))


def test_jail_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope", encoding="utf-8")
    link = workspace / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"cannot create symlink: {exc}")
    with pytest.raises(ToolError) as direct:
        resolve_in_workspace(workspace, "escape")
    _assert_no_abs(workspace, str(direct.value))
    mid_dir = workspace / "mid"
    mid_dir.mkdir()
    nested = mid_dir / "link"
    try:
        nested.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"cannot create symlink: {exc}")
    with pytest.raises(ToolError) as middle:
        resolve_in_workspace(workspace, "mid/link/secret.txt")
    _assert_no_abs(workspace, str(middle.value))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows case-folding only")
def test_jail_windows_case_fold_stays_inside(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "Notes").mkdir()
    resolved = resolve_in_workspace(workspace, "notes")
    assert _is_inside(workspace, resolved)
    shown = display_path(workspace, resolved)
    _assert_no_abs(workspace, shown)


def test_forbid_trash_rejects_writes_allows_resolve(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    trash = workspace / "_trash"
    trash.mkdir()
    listed = resolve_in_workspace(workspace, "_trash", allow_root=False)
    assert display_path(workspace, listed) == "_trash"
    with pytest.raises(ToolError, match="_trash") as exc_info:
        resolve_in_workspace(workspace, "_trash/x.txt", forbid_trash=True)
    _assert_no_abs(workspace, str(exc_info.value))


def _is_inside(workspace: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(workspace.resolve())
        return True
    except ValueError:
        return False
