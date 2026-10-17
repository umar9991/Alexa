from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from atlas_agent.config import Settings, env_text
from atlas_agent.llm.base import (
    AgentTurn,
    LLMAuthError,
    LLMBadRequest,
    LLMError,
    LLMRateLimited,
    LLMUnavailable,
    ToolCall,
)
from atlas_agent.llm.diag import log_provider_api_error
from atlas_agent.llm.names import (
    decode_tool_name,
    encode_tool_name,
    truncate_tool_content,
)
from atlas_agent.llm.retry import with_retry


class GroqChatClient(Protocol):
    chat: Any


def specs_to_groq_tools(tools: list[dict[str, object]]) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for spec in tools:
        name = spec.get("name")
        if not isinstance(name, str):
            continue
        description = spec.get("description", "")
        parameters = spec.get("input_schema", {"type": "object"})
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": encode_tool_name(name),
                    "description": description if isinstance(description, str) else "",
                    "parameters": parameters,
                },
            }
        )
    return converted


def internal_to_groq_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for message in messages:
        role = message.get("role")
        if role in ("system", "user"):
            content = message.get("content", "")
            converted.append(
                {"role": role, "content": content if isinstance(content, str) else str(content)}
            )
            continue
        if role == "assistant":
            converted.append(_assistant_to_groq(message))
            continue
        if role == "tool":
            converted.append(_tool_to_groq(message))
    return converted


def parse_groq_message(message: Any) -> AgentTurn:
    text = getattr(message, "content", None)
    if text is not None and not isinstance(text, str):
        text = str(text)
    raw_calls = getattr(message, "tool_calls", None) or []
    tool_calls = [_parse_tool_call(item) for item in raw_calls]
    return AgentTurn(text=text, tool_calls=tool_calls)


class GroqProvider:
    def __init__(
        self,
        settings: Settings,
        *,
        client: GroqChatClient | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._sleep = sleep

    async def step(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> AgentTurn:
        async def once() -> AgentTurn:
            return await self._step_once(messages, tools)

        return await with_retry(once, sleep=self._sleep)

    async def _step_once(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> AgentTurn:
        client = self._ensure_client()
        groq_messages = internal_to_groq_messages(messages)
        groq_tools = specs_to_groq_tools(tools)
        try:
            response = await client.chat.completions.create(
                model=self._settings.groq_model,
                messages=groq_messages,
                tools=groq_tools or None,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise _translate_groq_error(exc) from None
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMUnavailable("Groq returned an empty response")
        return parse_groq_message(choices[0].message)

    def _ensure_client(self) -> GroqChatClient:
        if self._client is not None:
            return self._client
        if not self._settings.groq_api_key:
            raise LLMAuthError("GROQ_API_KEY is missing or invalid")
        from groq import AsyncGroq

        env_text("GROQ_BASE_URL")
        env_text("GROQ_API_KEY")
        self._client = AsyncGroq(
            api_key=self._settings.groq_api_key,
            base_url=self._settings.groq_base_url,
            max_retries=0,
            timeout=self._settings.llm_timeout_s,
        )
        return self._client


def _assistant_to_groq(message: dict[str, object]) -> dict[str, object]:
    content = message.get("content")
    payload: dict[str, object] = {
        "role": "assistant",
        "content": content if isinstance(content, str) else None,
    }
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        return payload
    groq_calls: list[dict[str, object]] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        call_id = str(call.get("id", ""))
        name = str(call.get("name", ""))
        args = call.get("args", {})
        if not isinstance(args, dict):
            args = {}
        groq_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": encode_tool_name(name),
                    "arguments": json.dumps(args, ensure_ascii=True),
                },
            }
        )
    payload["tool_calls"] = groq_calls
    return payload


def _tool_to_groq(message: dict[str, object]) -> dict[str, object]:
    content = message.get("content", "")
    if not isinstance(content, str):
        content = str(content)
    name = message.get("name", "")
    return {
        "role": "tool",
        "tool_call_id": str(message.get("tool_call_id", "")),
        "name": encode_tool_name(name if isinstance(name, str) else str(name)),
        "content": truncate_tool_content(content),
    }


def _parse_tool_call(item: Any) -> ToolCall:
    call_id = getattr(item, "id", None)
    function = getattr(item, "function", None)
    raw_name = getattr(function, "name", "") if function is not None else ""
    raw_args = getattr(function, "arguments", "") if function is not None else ""
    name = decode_tool_name(raw_name if isinstance(raw_name, str) else str(raw_name))
    if not isinstance(raw_args, str):
        raw_args = str(raw_args)
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError:
        return ToolCall(
            name=name,
            args={},
            call_id=call_id if isinstance(call_id, str) else None,
            parse_error="invalid JSON arguments",
        )
    if not isinstance(parsed, dict):
        return ToolCall(
            name=name,
            args={},
            call_id=call_id if isinstance(call_id, str) else None,
            parse_error="invalid JSON arguments",
        )
    return ToolCall(
        name=name,
        args=parsed,
        call_id=call_id if isinstance(call_id, str) else None,
    )


def _translate_groq_error(exc: Exception) -> Exception:
    if isinstance(exc, LLMError):
        return exc
    log_provider_api_error("groq", exc)
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if name in {"AuthenticationError", "PermissionDeniedError"} or status in {401, 403}:
        return LLMAuthError("GROQ_API_KEY is missing or invalid")
    if name == "RateLimitError" or status == 429:
        return LLMRateLimited("Groq rate limit reached", retry_after_s=_retry_after_s(exc))
    if name in {"APITimeoutError", "APIConnectionError"}:
        return LLMUnavailable("Groq is temporarily unavailable")
    if name == "InternalServerError" or (isinstance(status, int) and status >= 500):
        return LLMUnavailable("Groq is temporarily unavailable")
    if name in {"BadRequestError", "UnprocessableEntityError", "NotFoundError"} or (
        isinstance(status, int) and 400 <= status < 500
    ):
        return LLMBadRequest("Groq rejected the request")
    return LLMUnavailable("Groq is temporarily unavailable")


def _retry_after_s(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
