from __future__ import annotations

import asyncio
from pathlib import Path

from atlas_agent.core.pathjail import display_path, resolve_in_workspace
from atlas_agent.tools.context import ToolContext, ToolError, require_workspace


async def run(args: dict[str, object], ctx: ToolContext) -> dict[str, object]:
    workspace = require_workspace(ctx)
    raw = args.get("path", "")
    if not isinstance(raw, str):
        raise ToolError("path must be a string")
    return await asyncio.to_thread(_make_dir_sync, workspace, raw)


def _make_dir_sync(workspace: Path, rel: str) -> dict[str, object]:
    target = resolve_in_workspace(workspace, rel, allow_root=False, forbid_trash=True)
    shown = display_path(workspace, target)
    if target.exists():
        if target.is_dir():
            return {"path": shown, "created": False}
        raise ToolError("a file already exists at that path")
    target.mkdir(parents=True, exist_ok=True)
    return {"path": shown, "created": True}
