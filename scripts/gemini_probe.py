#!/usr/bin/env python
"""Dev-only Gemini request probe. Never imported by app code."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "agent"))
os.chdir(ROOT)

from atlas_agent.config import load_settings  # noqa: E402
from atlas_agent.core.prompts import SYSTEM_PROMPT  # noqa: E402
from atlas_agent.llm.base import LLMError  # noqa: E402
from atlas_agent.llm.gemini import GeminiProvider  # noqa: E402
from atlas_agent.tools.registry import ToolRegistry  # noqa: E402

USER = "use the echo tool to say hi"


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Gemini with Atlas settings")
    parser.add_argument(
        "--one-tool",
        action="store_true",
        help="Send only demo.echo instead of the full registry",
    )
    parser.add_argument(
        "--roundtrip",
        action="store_true",
        help="Run a first step then replay the tool result with provider_state",
    )
    return parser.parse_args()


def _tools(*, one_tool: bool) -> list[dict[str, object]]:
    specs = ToolRegistry.load().specs()
    if not one_tool:
        return specs
    return [spec for spec in specs if spec.get("name") == "demo.echo"]


async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    settings = load_settings()
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY is missing")
        return 2
    print(f"model={settings.gemini_model}")
    provider = GeminiProvider(settings)
    tools = _tools(one_tool=args.one_tool)
    messages: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER},
    ]
    try:
        first = await provider.step(messages, tools)
    except LLMError as exc:
        print(f"step=first class={type(exc).__name__} llm_error={exc}")
        return 1
    print(
        "step=first ok "
        f"text={bool(first.text)} calls={len(first.tool_calls)} "
        f"parse_error={any(c.parse_error for c in first.tool_calls)}"
    )
    if not args.roundtrip:
        return 0
    if not first.tool_calls:
        print("step=replay skipped (no tool call)")
        return 1
    call = first.tool_calls[0]
    recorded: dict[str, object] = {
        "id": call.call_id,
        "name": call.name,
        "args": call.args,
    }
    if call.provider_state is not None:
        recorded["provider_state"] = call.provider_state
    text = call.args.get("text", "hi")
    echo_result = json.dumps(
        {"ok": True, "output": {"echo": text}, "error": None}, ensure_ascii=True
    )
    replay = [
        *messages,
        {"role": "assistant", "content": first.text, "tool_calls": [recorded]},
        {
            "role": "tool",
            "tool_call_id": call.call_id,
            "name": call.name,
            "content": echo_result,
        },
    ]
    try:
        second = await provider.step(replay, tools)
    except LLMError as exc:
        print(f"step=replay class={type(exc).__name__} llm_error={exc}")
        return 1
    print(f"step=replay ok text={bool(second.text)} calls={len(second.tool_calls)}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run(_parse())))


if __name__ == "__main__":
    main()
