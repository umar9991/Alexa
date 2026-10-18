from __future__ import annotations

import json

from atlas_agent.llm.base import AgentTurn, ToolCall


class ScriptedProvider:
    """Test-only provider. Must be enabled explicitly with ATLAS_LLM_PROVIDER=scripted."""

    async def step(self, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> AgentTurn:
        del tools
        tool_message = _last_tool_message(messages)
        if tool_message is not None:
            return AgentTurn(text=_final_text(tool_message), tool_calls=[])
        user_text = _last_user_text(messages)
        if "danger" in user_text:
            return AgentTurn(
                tool_calls=[ToolCall(name="demo.dangerous", args={"action": user_text})]
            )
        return AgentTurn(tool_calls=[ToolCall(name="demo.echo", args={"text": user_text})])


def _last_user_text(messages: list[dict[str, object]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _last_tool_message(messages: list[dict[str, object]]) -> dict[str, object] | None:
    for message in reversed(messages):
        if message.get("role") == "tool":
            return message
    return None


def _final_text(tool_message: dict[str, object]) -> str:
    """A short sentence about the outcome. Never the raw tool JSON."""
    name = tool_message.get("name", "the tool")
    label = name if isinstance(name, str) and name else "the tool"
    content = tool_message.get("content", "")
    ok = False
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        ok = isinstance(parsed, dict) and parsed.get("ok") is True
    if ok:
        return f"Done: {label} completed."
    return f"Done: {label} did not complete."
