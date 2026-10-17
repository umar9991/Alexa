from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ToolError(Exception):
    """Safe to show to the model and the user. Never include absolute paths."""


@dataclass(frozen=True)
class ToolContext:
    workspace: Path | None
    run_id: str


def require_workspace(ctx: ToolContext) -> Path:
    if ctx.workspace is None:
        raise ToolError("workspace is not configured")
    return ctx.workspace
