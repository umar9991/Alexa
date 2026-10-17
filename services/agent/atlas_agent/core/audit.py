from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ARG_LIMIT = 2000


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_env(cls) -> AuditLog:
        raw = os.environ.get("ATLAS_AUDIT_LOG", "data/audit.jsonl")
        return cls(Path(raw))

    def write(
        self,
        *,
        run_id: str,
        call_id: str,
        tool: str,
        args: dict[str, object],
        tier: str,
        decision: str,
        approved: bool,
        ok: bool,
        error: str | None,
        duration_ms: int,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "run_id": run_id,
            "call_id": call_id,
            "tool": tool,
            "args": truncate_args(args),
            "tier": tier,
            "decision": decision,
            "approved": approved,
            "ok": ok,
            "error": error,
            "duration_ms": duration_ms,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def truncate_args(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) > _ARG_LIMIT:
            return value[:_ARG_LIMIT]
        return value
    if isinstance(value, dict):
        return {str(key): truncate_args(item) for key, item in value.items()}
    if isinstance(value, list):
        return [truncate_args(item) for item in value]
    return value
