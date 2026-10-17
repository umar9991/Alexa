"""Internal LLM message format (provider-agnostic).

Each `step()` receives a list of dicts with one of these shapes:

- `{"role": "system", "content": str}`
- `{"role": "user", "content": str}`
- `{"role": "assistant", "content": str | None, "tool_calls": [{"id": str, "name": str, "args": dict, "provider_state": dict | None}]}`
- `{"role": "tool", "tool_call_id": str, "name": str, "content": str}`
  where `content` is the JSON-encoded tool result or error.

`provider_state` is optional opaque data owned by the provider that produced
the assistant turn (for example Gemini thought signatures). The loop copies it
verbatim into history and never inspects it. Other providers ignore it. It is
never sent to the UI and never written to the audit log.
"""

from typing import Protocol

from pydantic import BaseModel, Field


class LLMError(Exception):
    """Safe, user-facing LLM failure. Never include keys or raw bodies."""


class LLMAuthError(LLMError):
    pass


class LLMRateLimited(LLMError):
    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class LLMUnavailable(LLMError):
    pass


class LLMBadRequest(LLMError):
    pass


class ToolCall(BaseModel):
    name: str
    args: dict[str, object] = Field(default_factory=dict)
    call_id: str | None = None
    parse_error: str | None = None
    provider_state: dict[str, object] | None = None


class AgentTurn(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class LLMProvider(Protocol):
    async def step(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> AgentTurn: ...
