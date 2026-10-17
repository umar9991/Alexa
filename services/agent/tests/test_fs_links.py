from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path

import pytest

from atlas_agent.config import ensure_workspace
from atlas_agent.core.pathjail import resolve_in_workspace
from atlas_agent.tools.context import ToolContext, ToolError
from atlas_agent.tools.registry import ToolRegistry
from tests.conftest import (
    make_junction,
    make_symlink,
    outside_layout,
    remove_reparse,
)

LINK_ERROR = "links are not supported inside the workspace"


def _fingerprint(root: Path) -> dict[str, object]:
    listing: list[str] = []
    hashes: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [
            name
            for name in dirnames
            if not os.path.isjunction(os.path.join(dirpath, name))
            and not os.path.islink(os.path.join(dirpath, name))
        ]
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        if rel_dir != ".":
            listing.append(rel_dir)
        for name in filenames:
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            listing.append(rel)
            if full.is_file() and not os.path.islink(full):
                hashes[rel] = hashlib.sha256(full.read_bytes()).hexdigest()
    return {"listing": sorted(listing), "hashes": hashes}


def _ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, run_id="run-link")


def _assert_link_error(exc: pytest.ExceptionInfo[ToolError]) -> None:
    assert str(exc.value) == LINK_ERROR


async def _reject_all_tools(workspace: Path, outside: Path, link_rel: str) -> None:
    before = _fingerprint(outside)
    registry = ToolRegistry.load()
    ctx = _ctx(workspace)
    (workspace / "plain.txt").write_text("plain", encoding="utf-8")
    calls: list[tuple[str, dict[str, object]]] = [
        ("fs.list", {"path": link_rel}),
        ("fs.read_text", {"path": f"{link_rel}/secret.txt"}),
        ("fs.make_dir", {"path": f"{link_rel}/newdir"}),
        ("fs.write_text", {"path": f"{link_rel}/new.txt", "content": "x"}),
        ("fs.move", {"source": f"{link_rel}/secret.txt", "destination": "out.txt"}),
        ("fs.move", {"source": "plain.txt", "destination": f"{link_rel}/moved.txt"}),
        ("fs.delete", {"path": link_rel}),
        ("fs.delete", {"path": f"{link_rel}/secret.txt"}),
        ("fs.overwrite_text", {"path": f"{link_rel}/secret.txt", "content": "hacked"}),
    ]
    for name, args in calls:
        with pytest.raises(ToolError) as exc_info:
            await registry.run(name, args, ctx)
        _assert_link_error(exc_info)
    assert _fingerprint(outside) == before
    assert (outside / "secret.txt").read_text(encoding="utf-8") == "secret-bytes"


def test_jail_rejects_junction_escape(tmp_path: Path) -> None:
    workspace, outside = outside_layout(tmp_path)
    link = workspace / "link"
    nested = workspace / "a" / "link"
    try:
        make_junction(link, outside)
        make_junction(nested, outside)
        for rel in ("link", "link/secret.txt", "link/new.txt", "a/link/x"):
            with pytest.raises(ToolError) as exc_info:
                resolve_in_workspace(workspace, rel, allow_root=True)
            _assert_link_error(exc_info)
        assert (outside / "secret.txt").read_text(encoding="utf-8") == "secret-bytes"
    finally:
        remove_reparse(nested)
        remove_reparse(link)


def test_all_tools_reject_outside_junction(tmp_path: Path) -> None:
    workspace, outside = outside_layout(tmp_path)
    link = workspace / "link"
    try:
        make_junction(link, outside)
        asyncio.run(_reject_all_tools(workspace, outside, "link"))
    finally:
        remove_reparse(link)


def test_all_tools_reject_inside_junction(tmp_path: Path) -> None:
    workspace, outside = outside_layout(tmp_path)
    inner = workspace / "inside"
    inner.mkdir()
    (inner / "secret.txt").write_text("inside", encoding="utf-8")
    link = workspace / "link"
    try:
        make_junction(link, inner)
        asyncio.run(_reject_all_tools(workspace, outside, "link"))
        assert (inner / "secret.txt").read_text(encoding="utf-8") == "inside"
    finally:
        remove_reparse(link)


def test_delete_describe_counts_junction_as_one(tmp_path: Path) -> None:
    workspace, outside = outside_layout(tmp_path)
    folder = workspace / "box"
    folder.mkdir()
    (folder / "a.txt").write_text("aa", encoding="utf-8")
    (folder / "b.txt").write_text("bb", encoding="utf-8")
    link = folder / "out"
    registry = ToolRegistry.load()
    tool = registry.get("fs.delete")
    assert tool is not None and tool.describe is not None
    before = _fingerprint(outside)
    try:
        make_junction(link, outside)

        async def _call() -> None:
            text = await tool.describe({"path": "box"}, _ctx(workspace))
            assert "10000+" not in text
            assert "(4 items," in text
            assert (outside / "secret.txt").read_text(encoding="utf-8") == "secret-bytes"
            assert _fingerprint(outside) == before

        asyncio.run(_call())
    finally:
        remove_reparse(link)


def test_list_reports_junction_as_link(tmp_path: Path) -> None:
    workspace, outside = outside_layout(tmp_path)
    folder = workspace / "box"
    folder.mkdir()
    (folder / "a.txt").write_text("aa", encoding="utf-8")
    link = folder / "out"
    try:
        make_junction(link, outside)

        async def _call() -> None:
            listed = await ToolRegistry.load().run(
                "fs.list", {"path": "box"}, _ctx(workspace)
            )
            by_name = {item["name"]: item for item in listed["entries"]}
            assert by_name["out"]["type"] == "link"
            assert by_name["out"]["size"] is None
            assert by_name["a.txt"]["type"] == "file"

        asyncio.run(_call())
    finally:
        remove_reparse(link)


def test_workspace_via_junction_resolves(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    workspace, outside = outside_layout(tmp_path)
    alias = tmp_path / "alias_ws"
    escape = workspace / "escape"
    caplog.set_level(logging.INFO, logger="atlas_agent")
    try:
        make_junction(alias, workspace)
        make_junction(escape, outside)
        resolved = ensure_workspace(alias)
        assert resolved == workspace.resolve()
        assert str(workspace.resolve()) in caplog.text

        async def _call() -> None:
            ctx = _ctx(resolved)
            registry = ToolRegistry.load()
            created = await registry.run(
                "fs.write_text", {"path": "ok.txt", "content": "yes"}, ctx
            )
            assert created["path"] == "ok.txt"
            assert (workspace / "ok.txt").read_text(encoding="utf-8") == "yes"
            with pytest.raises(ToolError):
                await registry.run("fs.list", {"path": ".."}, ctx)
            with pytest.raises(ToolError) as exc_info:
                await registry.run("fs.list", {"path": "escape"}, ctx)
            _assert_link_error(exc_info)

        asyncio.run(_call())
        assert (outside / "secret.txt").read_text(encoding="utf-8") == "secret-bytes"
    finally:
        remove_reparse(escape)
        remove_reparse(alias)


def _symlink_setup(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    workspace, outside = outside_layout(tmp_path)
    link = workspace / "link"
    nested = workspace / "a" / "link"
    make_symlink(link, outside, target_is_directory=True)
    make_symlink(nested, outside, target_is_directory=True)
    return workspace, outside, link, nested


def test_jail_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace, outside, link, nested = _symlink_setup(tmp_path)
    try:
        for rel in ("link", "link/secret.txt", "link/new.txt", "a/link/x"):
            with pytest.raises(ToolError) as exc_info:
                resolve_in_workspace(workspace, rel, allow_root=True)
            _assert_link_error(exc_info)
    finally:
        remove_reparse(nested)
        remove_reparse(link)
        del outside


def test_all_tools_reject_outside_symlink(tmp_path: Path) -> None:
    workspace, outside, link, nested = _symlink_setup(tmp_path)
    try:
        asyncio.run(_reject_all_tools(workspace, outside, "link"))
    finally:
        remove_reparse(nested)
        remove_reparse(link)


def test_delete_describe_counts_symlink_as_one(tmp_path: Path) -> None:
    workspace, outside = outside_layout(tmp_path)
    folder = workspace / "box"
    folder.mkdir()
    (folder / "a.txt").write_text("aa", encoding="utf-8")
    (folder / "b.txt").write_text("bb", encoding="utf-8")
    link = folder / "out"
    registry = ToolRegistry.load()
    tool = registry.get("fs.delete")
    assert tool is not None and tool.describe is not None
    before = _fingerprint(outside)
    make_symlink(link, outside, target_is_directory=True)
    try:

        async def _call() -> None:
            text = await tool.describe({"path": "box"}, _ctx(workspace))
            assert "10000+" not in text
            assert "(4 items," in text
            assert _fingerprint(outside) == before

        asyncio.run(_call())
    finally:
        remove_reparse(link)


def test_list_reports_symlink_as_link(tmp_path: Path) -> None:
    workspace, outside = outside_layout(tmp_path)
    folder = workspace / "box"
    folder.mkdir()
    (folder / "a.txt").write_text("aa", encoding="utf-8")
    link = folder / "out"
    make_symlink(link, outside, target_is_directory=True)
    try:

        async def _call() -> None:
            listed = await ToolRegistry.load().run(
                "fs.list", {"path": "box"}, _ctx(workspace)
            )
            by_name = {item["name"]: item for item in listed["entries"]}
            assert by_name["out"]["type"] == "link"
            assert by_name["out"]["size"] is None

        asyncio.run(_call())
    finally:
        remove_reparse(link)
