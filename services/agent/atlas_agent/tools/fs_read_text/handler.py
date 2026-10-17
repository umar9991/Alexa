from __future__ import annotations

import asyncio
from pathlib import Path

from atlas_agent.core.pathjail import display_path, resolve_in_workspace
from atlas_agent.tools.context import ToolContext, ToolError, require_workspace

READ_LIMIT = 100 * 1024
NUL_SCAN = 8 * 1024


async def run(args: dict[str, object], ctx: ToolContext) -> dict[str, object]:
    workspace = require_workspace(ctx)
    raw = args.get("path", "")
    if not isinstance(raw, str):
        raise ToolError("path must be a string")
    return await asyncio.to_thread(_read_sync, workspace, raw)


def _read_sync(workspace: Path, rel: str) -> dict[str, object]:
    target = resolve_in_workspace(workspace, rel, allow_root=False)
    shown = display_path(workspace, target)
    if not target.exists():
        raise ToolError("file not found")
    if not target.is_file():
        raise ToolError("not a file")
    with target.open("rb") as handle:
        data = handle.read(READ_LIMIT + 1)
    truncated = len(data) > READ_LIMIT
    data = data[:READ_LIMIT]
    if b"\x00" in data[:NUL_SCAN]:
        raise ToolError("file is not valid text")
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError("file is not valid UTF-8") from None
    return {"path": shown, "content": content, "truncated": truncated}
