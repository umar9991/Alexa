from typing import Protocol

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    args: dict[str, object] = Field(default_factory=dict)
    call_id: str | None = None


class AgentTurn(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class LLMProvider(Protocol):
    async def step(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> AgentTurn: ...
