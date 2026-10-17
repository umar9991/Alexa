from __future__ import annotations

import asyncio
import os
from pathlib import Path

from atlas_agent.core.pathjail import display_path, resolve_in_workspace
from atlas_agent.tools.context import ToolContext, ToolError, require_workspace

LIST_LIMIT = 200


async def run(args: dict[str, object], ctx: ToolContext) -> dict[str, object]:
    workspace = require_workspace(ctx)
    raw = args.get("path", ".")
    if not isinstance(raw, str):
        raise ToolError("path must be a string")
    return await asyncio.to_thread(_list_sync, workspace, raw)


def _list_sync(workspace: Path, rel: str) -> dict[str, object]:
    target = resolve_in_workspace(workspace, rel, allow_root=True)
    shown = display_path(workspace, target)
    if not target.exists():
        raise ToolError("path not found")
    if not target.is_dir():
        raise ToolError("not a directory")
    dirs: list[dict[str, object]] = []
    links: list[dict[str, object]] = []
    files: list[dict[str, object]] = []
    for child in target.iterdir():
        if os.path.islink(child) or os.path.isjunction(child):
            links.append({"name": child.name, "type": "link", "size": None})
        elif child.is_dir():
            dirs.append({"name": child.name, "type": "dir", "size": None})
        elif child.is_file():
            files.append({"name": child.name, "type": "file", "size": child.stat().st_size})
    dirs.sort(key=lambda item: str(item["name"]).casefold())
    links.sort(key=lambda item: str(item["name"]).casefold())
    files.sort(key=lambda item: str(item["name"]).casefold())
    combined = dirs + links + files
    truncated = len(combined) > LIST_LIMIT
    return {
        "path": shown,
        "entries": combined[:LIST_LIMIT],
        "truncated": truncated,
    }
