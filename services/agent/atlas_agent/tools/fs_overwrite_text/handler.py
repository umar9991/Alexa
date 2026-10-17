from __future__ import annotations

import asyncio
import os
import shutil
import unicodedata
import uuid
from pathlib import Path

from atlas_agent.core.fsutil import ALLOWED_TEXT_EXTENSIONS, WRITE_LIMIT
from atlas_agent.core.pathjail import display_path, resolve_in_workspace
from atlas_agent.core.trash import new_trash_id, record_trash_entry, trash_file_path
from atlas_agent.tools.context import ToolContext, ToolError, require_workspace


async def describe(args: dict[str, object], ctx: ToolContext) -> str:
    workspace = require_workspace(ctx)
    raw = args.get("path", "")
    content = args.get("content", "")
    if not isinstance(raw, str):
        raise ToolError("path must be a string")
    if not isinstance(content, str):
        raise ToolError("content must be a string")
    return await asyncio.to_thread(_describe_sync, workspace, raw, content)


async def run(args: dict[str, object], ctx: ToolContext) -> dict[str, object]:
    workspace = require_workspace(ctx)
    raw = args.get("path", "")
    content = args.get("content", "")
    if not isinstance(raw, str):
        raise ToolError("path must be a string")
    if not isinstance(content, str):
        raise ToolError("content must be a string")
    return await asyncio.to_thread(_overwrite_sync, workspace, raw, content, ctx.run_id)


def _describe_sync(workspace: Path, rel: str, content: str) -> str:
    target = _checked_file(workspace, rel)
    shown = display_path(workspace, target)
    old_bytes = target.stat().st_size
    new_bytes = len(content.encode("utf-8"))
    preview = _preview(content)
    return (
        f"Overwrite '{shown}' ({_kb(old_bytes)} to {_kb(new_bytes)}), "
        f"old version saved to _trash, new starts: \"{preview}\""
    )


def _overwrite_sync(
    workspace: Path, rel: str, content: str, run_id: str
) -> dict[str, object]:
    target = _checked_file(workspace, rel)
    shown = display_path(workspace, target)
    encoded = content.encode("utf-8")
    if len(encoded) > WRITE_LIMIT:
        raise ToolError("content is too large")
    old_bytes = target.stat().st_size
    trash_id = new_trash_id()
    backup = trash_file_path(workspace, trash_id, shown)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, backup)
    backup_display = f"_trash/{trash_id}/{shown}"
    record_trash_entry(
        workspace,
        trash_id=trash_id,
        kind="overwrite",
        original_path=shown,
        trash_path=backup_display,
        run_id=run_id,
    )
    _atomic_write(target, content)
    return {
        "path": shown,
        "old_bytes": old_bytes,
        "new_bytes": len(encoded),
        "backup": backup_display,
    }


def _checked_file(workspace: Path, rel: str) -> Path:
    target = resolve_in_workspace(workspace, rel, allow_root=False, forbid_trash=True)
    shown = display_path(workspace, target)
    suffix = Path(shown).suffix.lower()
    if suffix not in ALLOWED_TEXT_EXTENSIONS:
        raise ToolError("file extension is not allowed")
    if not target.exists():
        raise ToolError("file not found; use fs.write_text to create it")
    if not target.is_file():
        raise ToolError("not a file")
    return target


def _atomic_write(target: Path, content: str) -> None:
    tmp = target.with_name(f".{target.name}.atlas_tmp_{uuid.uuid4().hex[:8]}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(tmp, target)
    except OSError:
        if tmp.exists():
            tmp.unlink()
        raise


def _preview(content: str) -> str:
    chunk = content[:120]
    cleaned: list[str] = []
    for ch in chunk:
        if ch in {"\n", "\r"} or unicodedata.category(ch) in {"Cc", "Cf"}:
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def _kb(n: int) -> str:
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"
