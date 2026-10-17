from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from atlas_agent.core.fsutil import WRITE_LIMIT
from atlas_agent.tools.context import ToolContext, ToolError
from atlas_agent.tools.registry import ToolRegistry


def _ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, run_id="run-ow")


def assert_no_abs(workspace: Path, payload: object) -> None:
    blob = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    resolved = str(workspace.resolve())
    assert resolved not in blob
    assert resolved.replace("\\", "/") not in blob


def test_overwrite_backs_up_and_missing_points_to_write_text(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = workspace / "Courses" / "x" / "notes.md"
    target.parent.mkdir(parents=True)
    target.write_text("hello", encoding="utf-8")
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)

    async def _call() -> None:
        out = await registry.run(
            "fs.overwrite_text",
            {"path": "Courses/x/notes.md", "content": "updated\ntext"},
            ctx,
        )
        assert target.read_text(encoding="utf-8") == "updated\ntext"
        backup = workspace / Path(str(out["backup"]))
        assert backup.read_text(encoding="utf-8") == "hello"
        assert out["old_bytes"] == len(b"hello")
        assert out["new_bytes"] == len(b"updated\ntext")
        assert_no_abs(workspace, out)
        with pytest.raises(ToolError, match="fs.write_text") as missing:
            await registry.run(
                "fs.overwrite_text",
                {"path": "missing.md", "content": "x"},
                ctx,
            )
        assert_no_abs(workspace, str(missing.value))

    asyncio.run(_call())


def test_overwrite_atomic_replace_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = workspace / "keep.md"
    target.write_text("original", encoding="utf-8")
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)

    def boom(_src: object, _dst: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("atlas_agent.tools.fs_overwrite_text.handler.os.replace", boom)

    async def _call() -> None:
        with pytest.raises(ToolError, match="internal tool error"):
            await registry.run(
                "fs.overwrite_text", {"path": "keep.md", "content": "new"}, ctx
            )
        assert target.read_text(encoding="utf-8") == "original"
        leftovers = list(workspace.glob(".keep.md.atlas_tmp_*"))
        assert leftovers == []

    asyncio.run(_call())


def test_overwrite_extension_size_and_describe(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "ok.md").write_text("old", encoding="utf-8")
    (workspace / "bad.exe").write_text("x", encoding="utf-8")
    registry = ToolRegistry.load()
    tool = registry.get("fs.overwrite_text")
    assert tool is not None and tool.describe is not None
    ctx = _ctx(workspace)

    async def _call() -> None:
        with pytest.raises(ToolError) as ext:
            await registry.run(
                "fs.overwrite_text", {"path": "bad.exe", "content": "x"}, ctx
            )
        assert_no_abs(workspace, str(ext.value))
        with pytest.raises(ToolError) as huge:
            await registry.run(
                "fs.overwrite_text",
                {"path": "ok.md", "content": "a" * (WRITE_LIMIT + 1)},
                ctx,
            )
        assert_no_abs(workspace, str(huge.value))
        text = await tool.describe(
            {"path": "ok.md", "content": "hello\nworld\x00next"},
            ctx,
        )
        assert "ok.md" in text
        assert "\n" not in text
        assert "\x00" not in text
        assert len(text) <= 300
        assert "KB" in text or "B" in text
        assert (workspace / "ok.md").read_text(encoding="utf-8") == "old"
        assert_no_abs(workspace, text)

    asyncio.run(_call())
