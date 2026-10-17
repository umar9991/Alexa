from __future__ import annotations

import asyncio
from pathlib import Path

from atlas_agent.core.fsutil import count_tree, entry_kind, format_size
from atlas_agent.core.pathjail import (
    display_path,
    is_trash_display,
    resolve_in_workspace,
)
from atlas_agent.core.trash import new_trash_id, record_trash_entry, trash_file_path
from atlas_agent.tools.context import ToolContext, ToolError, require_workspace


async def describe(args: dict[str, object], ctx: ToolContext) -> str:
    workspace = require_workspace(ctx)
    raw = args.get("path", "")
    if not isinstance(raw, str):
        raise ToolError("path must be a string")
    return await asyncio.to_thread(_describe_sync, workspace, raw)


async def run(args: dict[str, object], ctx: ToolContext) -> dict[str, object]:
    workspace = require_workspace(ctx)
    raw = args.get("path", "")
    if not isinstance(raw, str):
        raise ToolError("path must be a string")
    return await asyncio.to_thread(_delete_sync, workspace, raw, ctx.run_id)


def _describe_sync(workspace: Path, rel: str) -> str:
    target = _checked_target(workspace, rel)
    shown = display_path(workspace, target)
    items, size, capped = count_tree(target)
    count = "10000+" if capped else str(items)
    label = "folder" if entry_kind(target) == "dir" else "file"
    return (
        f"Delete {label} '{shown}' ({count} items, {format_size(size)}), "
        "moved to _trash, recoverable"
    )


def _delete_sync(workspace: Path, rel: str, run_id: str) -> dict[str, object]:
    target = _checked_target(workspace, rel)
    shown = display_path(workspace, target)
    kind = entry_kind(target)
    items, _size, capped = count_tree(target)
    trash_id = new_trash_id()
    dest = trash_file_path(workspace, trash_id, shown)
    dest.parent.mkdir(parents=True, exist_ok=True)
    target.rename(dest)
    trash_display = f"_trash/{trash_id}/{shown}"
    record_trash_entry(
        workspace,
        trash_id=trash_id,
        kind="delete",
        original_path=shown,
        trash_path=trash_display,
        run_id=run_id,
    )
    return {
        "path": shown,
        "trashed_to": trash_display,
        "kind": kind,
        "items": "10000+" if capped else items,
    }


def _checked_target(workspace: Path, rel: str) -> Path:
    target = resolve_in_workspace(workspace, rel, allow_root=True, forbid_trash=False)
    shown = display_path(workspace, target)
    if shown == ".":
        raise ToolError("cannot delete the workspace root")
    if is_trash_display(shown):
        raise ToolError("cannot delete _trash")
    if not target.exists():
        raise ToolError("no longer exists")
    return target
