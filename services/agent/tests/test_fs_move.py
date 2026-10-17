from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from atlas_agent.tools.context import ToolContext, ToolError
from atlas_agent.tools.registry import ToolRegistry
from tests.test_pathjail import UNSAFE_REL_PATHS


def _ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, run_id="run-move")


def assert_no_abs(workspace: Path, payload: object) -> None:
    blob = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    resolved = str(workspace.resolve())
    assert resolved not in blob
    assert resolved.replace("\\", "/") not in blob


def test_move_file_and_dir_creates_parents(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "src.txt").write_text("hello", encoding="utf-8")
    nested = workspace / "folder" / "inner"
    nested.mkdir(parents=True)
    (nested / "a.txt").write_text("a", encoding="utf-8")
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)

    async def _call() -> None:
        file_out = await registry.run(
            "fs.move", {"source": "src.txt", "destination": "out/new.txt"}, ctx
        )
        assert file_out["source"] == "src.txt"
        assert file_out["destination"] == "out/new.txt"
        assert file_out["kind"] == "file"
        assert not (workspace / "src.txt").exists()
        assert (workspace / "out" / "new.txt").read_text(encoding="utf-8") == "hello"
        assert_no_abs(workspace, file_out)
        dir_out = await registry.run(
            "fs.move", {"source": "folder", "destination": "moved/folder"}, ctx
        )
        assert (workspace / "moved" / "folder" / "inner" / "a.txt").read_text(
            encoding="utf-8"
        ) == "a"
        assert not (workspace / "folder").exists()
        assert_no_abs(workspace, dir_out)

    asyncio.run(_call())


def test_move_dest_exists_and_missing_source(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.txt").write_text("keep", encoding="utf-8")
    (workspace / "b.txt").write_text("other", encoding="utf-8")
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)

    async def _call() -> None:
        with pytest.raises(ToolError) as exists:
            await registry.run(
                "fs.move", {"source": "a.txt", "destination": "b.txt"}, ctx
            )
        assert_no_abs(workspace, str(exists.value))
        assert (workspace / "a.txt").read_text(encoding="utf-8") == "keep"
        assert (workspace / "b.txt").read_text(encoding="utf-8") == "other"
        with pytest.raises(ToolError) as missing:
            await registry.run(
                "fs.move", {"source": "gone.txt", "destination": "x.txt"}, ctx
            )
        assert_no_abs(workspace, str(missing.value))

    asyncio.run(_call())


def test_move_trash_rules(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    trash_item = workspace / "_trash" / "id1" / "notes.md"
    trash_item.parent.mkdir(parents=True)
    trash_item.write_text("restored", encoding="utf-8")
    (workspace / "keep.txt").write_text("x", encoding="utf-8")
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)

    async def _call() -> None:
        with pytest.raises(ToolError) as into_trash:
            await registry.run(
                "fs.move",
                {"source": "keep.txt", "destination": "_trash/keep.txt"},
                ctx,
            )
        assert_no_abs(workspace, str(into_trash.value))
        assert (workspace / "keep.txt").exists()
        out = await registry.run(
            "fs.move",
            {"source": "_trash/id1/notes.md", "destination": "Lab-07/notes.md"},
            ctx,
        )
        assert (workspace / "Lab-07" / "notes.md").read_text(encoding="utf-8") == "restored"
        assert not trash_item.exists()
        assert_no_abs(workspace, out)

    asyncio.run(_call())


def test_move_rejects_root_trash_self_and_descendant(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "_trash").mkdir()
    nested = workspace / "dir" / "sub"
    nested.mkdir(parents=True)
    (nested / "f.txt").write_text("f", encoding="utf-8")
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)

    async def _call() -> None:
        with pytest.raises(ToolError) as root:
            await registry.run("fs.move", {"source": ".", "destination": "x"}, ctx)
        assert_no_abs(workspace, str(root.value))
        with pytest.raises(ToolError) as trash:
            await registry.run("fs.move", {"source": "_trash", "destination": "x"}, ctx)
        assert_no_abs(workspace, str(trash.value))
        with pytest.raises(ToolError) as into:
            await registry.run(
                "fs.move", {"source": "dir", "destination": "dir/sub/out"}, ctx
            )
        assert_no_abs(workspace, str(into.value))
        assert (workspace / "dir" / "sub" / "f.txt").exists()
        with pytest.raises(ToolError) as same:
            await registry.run("fs.move", {"source": "dir", "destination": "dir"}, ctx)
        assert_no_abs(workspace, str(same.value))

    asyncio.run(_call())


@pytest.mark.parametrize("bad", UNSAFE_REL_PATHS)
def test_move_rejects_jail_escapes(tmp_path: Path, bad: str) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "ok.txt").write_text("ok", encoding="utf-8")
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)

    async def _call() -> None:
        with pytest.raises(ToolError) as src:
            await registry.run("fs.move", {"source": bad, "destination": "out.txt"}, ctx)
        assert_no_abs(workspace, str(src.value))
        with pytest.raises(ToolError) as dst:
            await registry.run("fs.move", {"source": "ok.txt", "destination": bad}, ctx)
        assert_no_abs(workspace, str(dst.value))
        assert (workspace / "ok.txt").exists()

    asyncio.run(_call())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows case-only rename")
def test_move_case_only_rename(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "lab-06").mkdir()
    (workspace / "lab-06" / "n.txt").write_text("n", encoding="utf-8")
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)

    async def _call() -> None:
        out = await registry.run(
            "fs.move", {"source": "lab-06", "destination": "Lab-06"}, ctx
        )
        assert out["destination"] == "Lab-06"
        assert (workspace / "Lab-06" / "n.txt").read_text(encoding="utf-8") == "n"
        assert_no_abs(workspace, out)

    asyncio.run(_call())
