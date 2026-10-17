from __future__ import annotations

import asyncio
from pathlib import Path

from atlas_agent.core.fsutil import ALLOWED_TEXT_EXTENSIONS, WRITE_LIMIT
from atlas_agent.core.pathjail import display_path, resolve_in_workspace
from atlas_agent.tools.context import ToolContext, ToolError, require_workspace


async def run(args: dict[str, object], ctx: ToolContext) -> dict[str, object]:
    workspace = require_workspace(ctx)
    raw = args.get("path", "")
    content = args.get("content", "")
    if not isinstance(raw, str):
        raise ToolError("path must be a string")
    if not isinstance(content, str):
        raise ToolError("content must be a string")
    return await asyncio.to_thread(_write_sync, workspace, raw, content)


def _write_sync(workspace: Path, rel: str, content: str) -> dict[str, object]:
    target = resolve_in_workspace(workspace, rel, allow_root=False, forbid_trash=True)
    shown = display_path(workspace, target)
    suffix = Path(shown).suffix.lower()
    if suffix not in ALLOWED_TEXT_EXTENSIONS:
        raise ToolError("file extension is not allowed")
    encoded = content.encode("utf-8")
    if len(encoded) > WRITE_LIMIT:
        raise ToolError("content is too large")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
    except FileExistsError:
        raise ToolError("file exists; overwriting is not available") from None
    return {"path": shown, "bytes": len(encoded)}
