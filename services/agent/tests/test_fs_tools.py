from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from atlas_agent.tools.context import ToolContext, ToolError
from atlas_agent.tools.registry import ToolRegistry


def _ctx(workspace: Path | None) -> ToolContext:
    return ToolContext(workspace=workspace, run_id="run-fs")


def _blob(payload: object) -> str:
    if isinstance(payload, BaseException):
        return str(payload)
    return json.dumps(payload, default=str, ensure_ascii=False)


def assert_no_workspace_path(workspace: Path | None, payload: object) -> None:
    blob = _blob(payload)
    if workspace is None:
        return
    resolved = str(workspace.resolve())
    assert resolved not in blob
    assert resolved.replace("\\", "/") not in blob
    assert resolved.replace("/", "\\") not in blob


async def _run(
    registry: ToolRegistry, name: str, args: dict[str, object], ctx: ToolContext
) -> dict[str, object]:
    return await registry.run(name, args, ctx)


def test_fs_tools_require_workspace() -> None:
    registry = ToolRegistry.load()
    ctx = _ctx(None)

    async def _call() -> None:
        for name, args in (
            ("fs.list", {"path": "."}),
            ("fs.read_text", {"path": "a.txt"}),
            ("fs.make_dir", {"path": "a"}),
            ("fs.write_text", {"path": "a.txt", "content": "x"}),
        ):
            with pytest.raises(ToolError, match="workspace is not configured") as exc_info:
                await _run(registry, name, args, ctx)
            assert_no_workspace_path(None, str(exc_info.value))

    asyncio.run(_call())


def test_make_dir_nested_idempotent_and_file_conflict(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)

    async def _call() -> None:
        first = await _run(
            registry, "fs.make_dir", {"path": "Courses/DBMS/Assignments/Lab-06"}, ctx
        )
        assert first == {"path": "Courses/DBMS/Assignments/Lab-06", "created": True}
        assert_no_workspace_path(workspace, first)
        assert (workspace / "Courses" / "DBMS" / "Assignments" / "Lab-06").is_dir()
        second = await _run(
            registry, "fs.make_dir", {"path": "Courses/DBMS/Assignments/Lab-06"}, ctx
        )
        assert second == {"path": "Courses/DBMS/Assignments/Lab-06", "created": False}
        assert_no_workspace_path(workspace, second)
        (workspace / "a_file").write_text("x", encoding="utf-8")
        with pytest.raises(ToolError) as exc_info:
            await _run(registry, "fs.make_dir", {"path": "a_file"}, ctx)
        assert_no_workspace_path(workspace, str(exc_info.value))

    asyncio.run(_call())


def test_write_text_create_only_and_limits(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)

    async def _call() -> None:
        created = await _run(
            registry,
            "fs.write_text",
            {"path": "nested/notes.md", "content": "hello\nworld"},
            ctx,
        )
        assert created["path"] == "nested/notes.md"
        assert created["bytes"] == len(b"hello\nworld")
        assert_no_workspace_path(workspace, created)
        assert (workspace / "nested" / "notes.md").read_text(encoding="utf-8") == "hello\nworld"
        with pytest.raises(ToolError, match="overwriting is not available") as exists:
            await _run(
                registry,
                "fs.write_text",
                {"path": "nested/notes.md", "content": "changed"},
                ctx,
            )
        assert_no_workspace_path(workspace, str(exists.value))
        assert (workspace / "nested" / "notes.md").read_text(encoding="utf-8") == "hello\nworld"
        with pytest.raises(ToolError) as ext:
            await _run(registry, "fs.write_text", {"path": "notes.exe", "content": "x"}, ctx)
        assert_no_workspace_path(workspace, str(ext.value))
        with pytest.raises(ToolError) as missing:
            await _run(registry, "fs.write_text", {"path": "notes", "content": "x"}, ctx)
        assert_no_workspace_path(workspace, str(missing.value))
        with pytest.raises(ToolError, match="too large") as big:
            await _run(
                registry,
                "fs.write_text",
                {"path": "big.txt", "content": "a" * (200 * 1024 + 1)},
                ctx,
            )
        assert_no_workspace_path(workspace, str(big.value))

    asyncio.run(_call())


def test_read_text_utf8_truncation_and_binary(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)
    (workspace / "ok.txt").write_text("hi", encoding="utf-8")
    (workspace / "wide.txt").write_bytes(b"a" * (100 * 1024 + 20))
    (workspace / "bin.dat").write_bytes(b"hello\x00world")
    (workspace / "bad.txt").write_bytes(b"\xff\xfe not utf8")

    async def _call() -> None:
        ok = await _run(registry, "fs.read_text", {"path": "ok.txt"}, ctx)
        assert ok["content"] == "hi"
        assert ok["truncated"] is False
        assert_no_workspace_path(workspace, ok)
        wide = await _run(registry, "fs.read_text", {"path": "wide.txt"}, ctx)
        assert wide["truncated"] is True
        assert len(wide["content"].encode("utf-8")) == 100 * 1024
        assert_no_workspace_path(workspace, wide)
        with pytest.raises(ToolError) as binary:
            await _run(registry, "fs.read_text", {"path": "bin.dat"}, ctx)
        assert_no_workspace_path(workspace, str(binary.value))
        with pytest.raises(ToolError) as encoding:
            await _run(registry, "fs.read_text", {"path": "bad.txt"}, ctx)
        assert_no_workspace_path(workspace, str(encoding.value))

    asyncio.run(_call())


def test_list_order_truncation_and_root(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)
    (workspace / "B_dir").mkdir()
    (workspace / "a_dir").mkdir()
    (workspace / "C.txt").write_text("c", encoding="utf-8")
    (workspace / "b.txt").write_text("b", encoding="utf-8")

    async def _call() -> None:
        listed = await _run(registry, "fs.list", {"path": "."}, ctx)
        names = [item["name"] for item in listed["entries"]]
        assert names == ["a_dir", "B_dir", "b.txt", "C.txt"]
        assert listed["path"] == "."
        assert listed["truncated"] is False
        assert_no_workspace_path(workspace, listed)
        many = workspace / "many"
        many.mkdir()
        for index in range(210):
            (many / f"f{index:03d}.txt").write_text("x", encoding="utf-8")
        truncated = await _run(registry, "fs.list", {"path": "many"}, ctx)
        assert truncated["truncated"] is True
        assert len(truncated["entries"]) == 200
        assert_no_workspace_path(workspace, truncated)

    asyncio.run(_call())


def test_oserror_becomes_internal_tool_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)

    def boom(self: Path) -> list[Path]:
        raise OSError(f"cannot list {workspace.resolve() / 'secret'}")

    monkeypatch.setattr(Path, "iterdir", boom)

    async def _call() -> None:
        with pytest.raises(ToolError, match="internal tool error") as exc_info:
            await _run(registry, "fs.list", {"path": "."}, ctx)
        assert str(exc_info.value) == "internal tool error"
        assert_no_workspace_path(workspace, str(exc_info.value))

    asyncio.run(_call())
