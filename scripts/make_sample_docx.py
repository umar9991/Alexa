#!/usr/bin/env python
"""Build workspace/assignments/sample.docx for a visual Word check."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "agent"))

from atlas_agent.tools.context import ToolContext  # noqa: E402
from atlas_agent.tools.registry import ToolRegistry  # noqa: E402


def _workspace() -> Path:
    raw = os.environ.get("ATLAS_WORKSPACE", "").strip()
    if raw:
        return Path(raw)
    return ROOT / "workspace"


def _args(path: str) -> dict[str, object]:
    long_cell = (
        "This cell has enough wrapping text that an eight-row table is likely "
        "to continue onto the next page, which is the point of this sample. "
        "Keep the row height automatic so nothing is clipped."
    )
    return {
        "path": path,
        "meta": {
            "student_name": "Test Student",
            "student_id": "00000",
            "instructor": "Test Instructor",
            "subject": "Software Construction",
            "deadline": "2099-12-31",
            "total_marks": "20",
            "assignment_number": 1,
        },
        "questions": [
            {
                "number": "Q1",
                "title": "Explain the house format",
                "marks": 8,
                "blocks": [
                    {
                        "type": "paragraph",
                        "text": "Write a short answer using the labeled lines below.",
                    },
                    {
                        "type": "labeled",
                        "label": "Answer:",
                        "text": "The assignment uses Times New Roman body text and a fixed banner.",
                    },
                    {
                        "type": "labeled",
                        "label": "Reason:",
                        "text": "A single house format keeps submissions consistent.",
                    },
                    {
                        "type": "bullets",
                        "items": [
                            "Do not overwrite an existing .docx",
                            "Keep table rows from splitting across pages",
                            "Repeat the header row on each page",
                        ],
                    },
                ],
            },
            {
                "number": "Q2",
                "title": "Complete the comparison table",
                "marks": 12,
                "blocks": [
                    {
                        "type": "paragraph",
                        "text": "The table below is long enough to cross a page boundary.",
                    },
                    {
                        "type": "table",
                        "headers": ["Row", "Observation", "Notes"],
                        "rows": [
                            [str(i), long_cell, f"tone-check-{i}"] for i in range(1, 9)
                        ],
                        "tones": [
                            ["none", "good", "none"],
                            ["none", "warn", "none"],
                            ["none", "info", "none"],
                            ["none", "bad", "none"],
                            ["none", "none", "good"],
                            ["none", "none", "warn"],
                            ["none", "none", "info"],
                            ["none", "none", "bad"],
                        ],
                    },
                    {
                        "type": "code",
                        "text": "def sample():\n    return 42\n",
                    },
                ],
            },
        ],
    }


async def _main() -> None:
    workspace = _workspace()
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "assignments" / "sample.docx"
    if target.exists():
        target.unlink()
    registry = ToolRegistry.load()
    result = await registry.run(
        "docx.build",
        _args("assignments/sample.docx"),
        ToolContext(workspace=workspace, run_id="sample-docx"),
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(_main())
