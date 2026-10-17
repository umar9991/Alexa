from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from atlas_agent.core.pathjail import display_path

TrashKind = Literal["delete", "overwrite"]


def new_trash_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def trash_file_path(workspace: Path, trash_id: str, original_display: str) -> Path:
    root = workspace.resolve(strict=False)
    parts = [part for part in original_display.split("/") if part and part != "."]
    return (root / "_trash" / trash_id).joinpath(*parts)


def record_trash_entry(
    workspace: Path,
    *,
    trash_id: str,
    kind: TrashKind,
    original_path: str,
    trash_path: str,
    run_id: str,
) -> None:
    trash_dir = workspace.resolve(strict=False) / "_trash"
    trash_dir.mkdir(parents=True, exist_ok=True)
    index = trash_dir / ".index.jsonl"
    line = {
        "id": trash_id,
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "kind": kind,
        "original_path": original_path,
        "trash_path": trash_path,
        "run_id": run_id,
    }
    with index.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=True) + "\n")


def display_trash_path(workspace: Path, target: Path) -> str:
    return display_path(workspace, target)
