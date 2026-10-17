from __future__ import annotations

import json

from atlas_agent.llm.base import AgentTurn, ToolCall


class ScriptedProvider:
    async def step(self, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> AgentTurn:
        del tools
        tool_result = _last_tool_message(messages)
        if tool_result is not None:
            summary = _short_summary(tool_result)
            return AgentTurn(text=f"Done: {summary}", tool_calls=[])
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


def _last_tool_message(messages: list[dict[str, object]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "tool":
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)
    return None


def _short_summary(content: str) -> str:
    try:
        parsed = json.loads(content)
        content = json.dumps(parsed, separators=(",", ":"))
    except json.JSONDecodeError:
        pass
    if len(content) <= 120:
        return content
    return content[:117] + "..."
