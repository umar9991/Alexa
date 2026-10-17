from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from atlas_agent.tools.context import ToolContext, ToolError
from atlas_agent.tools.registry import ToolRegistry


def _ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, run_id="run-del")


def assert_no_abs(workspace: Path, payload: object) -> None:
    blob = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    resolved = str(workspace.resolve())
    assert resolved not in blob
    assert resolved.replace("\\", "/") not in blob


def test_delete_file_and_directory_to_trash(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "gone.txt").write_text("payload", encoding="utf-8")
    tree = workspace / "tree" / "sub"
    tree.mkdir(parents=True)
    (tree / "a.txt").write_text("a", encoding="utf-8")
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)

    async def _call() -> None:
        file_out = await registry.run("fs.delete", {"path": "gone.txt"}, ctx)
        assert not (workspace / "gone.txt").exists()
        trashed = workspace / Path(str(file_out["trashed_to"]))
        assert trashed.read_text(encoding="utf-8") == "payload"
        assert file_out["kind"] == "file"
        assert_no_abs(workspace, file_out)
        dir_out = await registry.run("fs.delete", {"path": "tree"}, ctx)
        assert not (workspace / "tree").exists()
        restored = workspace / Path(str(dir_out["trashed_to"])) / "sub" / "a.txt"
        assert restored.read_text(encoding="utf-8") == "a"
        assert_no_abs(workspace, dir_out)

    asyncio.run(_call())


def test_delete_ids_differ_same_second_and_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)

    class FrozenDateTime:
        @classmethod
        def now(cls, tz: object | None = None) -> datetime:
            del tz
            return datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)

    monkeypatch.setattr("atlas_agent.core.trash.datetime", FrozenDateTime)

    async def _call() -> None:
        (workspace / "same.txt").write_text("1", encoding="utf-8")
        first = await registry.run("fs.delete", {"path": "same.txt"}, ctx)
        (workspace / "same.txt").write_text("2", encoding="utf-8")
        second = await registry.run("fs.delete", {"path": "same.txt"}, ctx)
        assert first["trashed_to"] != second["trashed_to"]
        assert str(first["trashed_to"]).startswith("_trash/20260501-120000-")
        index = (workspace / "_trash" / ".index.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(index) == 2
        for raw in index:
            row = json.loads(raw)
            assert set(row) >= {"id", "ts", "kind", "original_path", "trash_path", "run_id"}
            assert row["kind"] == "delete"
            assert row["original_path"] == "same.txt"
            assert row["run_id"] == "run-del"
            assert_no_abs(workspace, row)
        assert_no_abs(workspace, first)
        assert_no_abs(workspace, second)

    asyncio.run(_call())


def test_delete_rejects_root_trash_and_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "_trash" / "id" / "x.txt").parent.mkdir(parents=True)
    (workspace / "_trash" / "id" / "x.txt").write_text("x", encoding="utf-8")
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)

    async def _call() -> None:
        for path in (".", "_trash", "_trash/id/x.txt"):
            with pytest.raises(ToolError) as exc_info:
                await registry.run("fs.delete", {"path": path}, ctx)
            assert_no_abs(workspace, str(exc_info.value))
        assert (workspace / "_trash" / "id" / "x.txt").exists()
        with pytest.raises(ToolError) as missing:
            await registry.run("fs.delete", {"path": "nope.txt"}, ctx)
        assert_no_abs(workspace, str(missing.value))

    asyncio.run(_call())


def test_delete_describe_count_and_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    folder = workspace / "Courses" / "DBMS" / "Old"
    folder.mkdir(parents=True)
    (folder / "a.txt").write_text("aa", encoding="utf-8")
    (folder / "b.txt").write_text("bb", encoding="utf-8")
    registry = ToolRegistry.load()
    tool = registry.get("fs.delete")
    assert tool is not None and tool.describe is not None
    ctx = _ctx(workspace)

    async def _call() -> None:
        text = await tool.describe({"path": "Courses/DBMS/Old"}, ctx)
        assert "Courses/DBMS/Old" in text
        assert "items" in text
        assert "_trash" in text
        assert "\n" not in text
        assert len(text) <= 300
        assert (folder / "a.txt").exists()
        assert_no_abs(workspace, text)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret" * 100, encoding="utf-8")
        link = workspace / "linkdir"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"cannot create symlink: {exc}")
        linked = await tool.describe({"path": "linkdir"}, ctx)
        assert "10000+" not in linked
        assert "1 items" in linked or "1 item" in linked
        assert_no_abs(workspace, linked)

    asyncio.run(_call())
