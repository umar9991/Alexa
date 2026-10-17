from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from atlas_agent.core.fsutil import entry_kind
from atlas_agent.core.pathjail import (
    display_path,
    is_trash_display,
    resolve_in_workspace,
)
from atlas_agent.tools.context import ToolContext, ToolError, require_workspace


async def run(args: dict[str, object], ctx: ToolContext) -> dict[str, object]:
    workspace = require_workspace(ctx)
    source = args.get("source", "")
    destination = args.get("destination", "")
    if not isinstance(source, str) or not isinstance(destination, str):
        raise ToolError("source and destination must be strings")
    return await asyncio.to_thread(_move_sync, workspace, source, destination)


def _move_sync(workspace: Path, source_rel: str, dest_rel: str) -> dict[str, object]:
    src = resolve_in_workspace(workspace, source_rel, allow_root=True, forbid_trash=False)
    dst = resolve_in_workspace(workspace, dest_rel, allow_root=False, forbid_trash=True)
    src_display = display_path(workspace, src)
    dst_display = display_path(workspace, dst)
    src_name = Path(source_rel.replace("\\", "/")).name
    dst_name = Path(dest_rel.replace("\\", "/")).name
    if src_display == ".":
        raise ToolError("cannot move the workspace root")
    if src_display.casefold() == "_trash":
        raise ToolError("cannot move _trash")
    if is_trash_display(dst_display):
        raise ToolError("destination may not be inside _trash")
    case_only = _is_case_only_rename(src, dst, src_name, dst_name)
    if src_display == dst_display and not case_only:
        raise ToolError("source and destination are the same")
    if not src.exists():
        raise ToolError("source not found")
    if (
        src.is_dir()
        and not os.path.islink(src)
        and not os.path.isjunction(src)
        and dst_display.startswith(src_display + "/")
    ):
        raise ToolError("cannot move a directory into itself")
    if dst.exists() and not case_only:
        raise ToolError("destination already exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    kind = entry_kind(src)
    if case_only:
        _case_rename(src, dst)
        parent = display_path(workspace, dst.parent)
        dst_display = dst_name if parent == "." else f"{parent}/{dst_name}"
    else:
        src.rename(dst)
        dst_display = display_path(workspace, dst)
    return {"source": src_display, "destination": dst_display, "kind": kind}


def _is_case_only_rename(src: Path, dst: Path, src_name: str, dst_name: str) -> bool:
    if src_name.casefold() != dst_name.casefold() or src_name == dst_name:
        return False
    try:
        if src.parent.resolve() != dst.parent.resolve():
            return False
    except OSError:
        return False
    if not src.exists():
        return False
    if not dst.exists():
        return True
    try:
        return os.path.samefile(src, dst)
    except OSError:
        return False


def _case_rename(src: Path, dst: Path) -> None:
    temp = src.with_name(f"{src.name}.atlas_case_{uuid.uuid4().hex[:8]}")
    src.rename(temp)
    try:
        temp.rename(dst)
    except OSError:
        temp.rename(src)
        raise
