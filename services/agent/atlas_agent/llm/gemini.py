from __future__ import annotations

import base64
import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from atlas_agent.config import Settings
from atlas_agent.llm.base import (
    AgentTurn,
    LLMAuthError,
    LLMBadRequest,
    LLMError,
    LLMRateLimited,
    LLMUnavailable,
    ToolCall,
)
from atlas_agent.llm.names import (
    decode_tool_name,
    encode_tool_name,
    truncate_tool_content,
)
from atlas_agent.llm.retry import with_retry

# google.genai.types.Part.thought_signature is Optional[bytes]
# (google-genai 1.52.0, google/genai/types.py). REST docs use the dummy
# strings "skip_thought_signature_validator" or
# "context_engineering_is_the_way_to_go" for non-Gemini history
# (https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures).
# google.genai._common.encode_unserializable_types urlsafe-b64encodes every
# bytes value, so a bytes dummy would never reach the API as that skip
# string. Placeholder is therefore passed as str.
SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"
GEMINI_SCHEMA_KEYS = frozenset(
    {
        "type",
        "description",
        "properties",
        "items",
        "required",
        "enum",
        "format",
        "nullable",
        "minItems",
        "maxItems",
    }
)


class GeminiGenerateClient(Protocol):
    async def generate_content(self, **kwargs: object) -> Any: ...


def sanitize_gemini_schema(schema: object) -> object:
    if isinstance(schema, dict):
        cleaned: dict[str, object] = {}
        for key, value in schema.items():
            if key not in GEMINI_SCHEMA_KEYS:
                continue
            if key == "properties" and isinstance(value, dict):
                cleaned[key] = {
                    str(name): sanitize_gemini_schema(item) for name, item in value.items()
                }
            elif key == "items":
                cleaned[key] = sanitize_gemini_schema(value)
            else:
                cleaned[key] = value
        return cleaned
    if isinstance(schema, list):
        return [sanitize_gemini_schema(item) for item in schema]
    return schema


def specs_to_gemini_tools(tools: list[dict[str, object]]) -> list[dict[str, object]]:
    declarations: list[dict[str, object]] = []
    for spec in tools:
        name = spec.get("name")
        if not isinstance(name, str):
            continue
        description = spec.get("description", "")
        raw_schema = spec.get("input_schema", {"type": "object"})
        parameters = sanitize_gemini_schema(raw_schema)
        if not isinstance(parameters, dict):
            parameters = {"type": "object"}
        declarations.append(
            {
                "name": encode_tool_name(name),
                "description": description if isinstance(description, str) else "",
                "parameters": parameters,
            }
        )
    if not declarations:
        return []
    return [{"function_declarations": declarations}]


def internal_to_gemini(
    messages: list[dict[str, object]],
) -> tuple[str | None, list[dict[str, object]]]:
    system_chunks: list[str] = []
    contents: list[dict[str, object]] = []
    pending_tools: list[dict[str, object]] = []

    def flush_tools() -> None:
        if not pending_tools:
            return
        contents.append({"role": "tool", "parts": list(pending_tools)})
        pending_tools.clear()

    for message in messages:
        role = message.get("role")
        if role == "system":
            content = message.get("content", "")
            system_chunks.append(content if isinstance(content, str) else str(content))
            continue
        if role == "tool":
            pending_tools.append(_tool_part(message))
            continue
        flush_tools()
        if role == "user":
            content = message.get("content", "")
            text = content if isinstance(content, str) else str(content)
            contents.append({"role": "user", "parts": [{"text": text}]})
            continue
        if role == "assistant":
            contents.append(_assistant_content(message))
    flush_tools()
    system = "\n".join(chunk for chunk in system_chunks if chunk) or None
    return system, contents


def parse_gemini_response(response: Any) -> AgentTurn:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise LLMUnavailable("Gemini returned an empty response")
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    texts: list[str] = []
    calls: list[ToolCall] = []
    index = 0
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            texts.append(text)
        function_call = getattr(part, "function_call", None)
        if function_call is None:
            continue
        calls.append(_parse_function_call(function_call, part, index))
        index += 1
    return AgentTurn(text="".join(texts) or None, tool_calls=calls)


class GeminiProvider:
    def __init__(
        self,
        settings: Settings,
        *,
        client: GeminiGenerateClient | None = None,
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
        system, contents = internal_to_gemini(messages)
        gemini_tools = specs_to_gemini_tools(tools)
        config: dict[str, object] = {
            "temperature": self._settings.llm_temperature,
            "max_output_tokens": self._settings.llm_max_tokens,
            "automatic_function_calling": {"disable": True},
        }
        if system:
            config["system_instruction"] = system
        if gemini_tools:
            config["tools"] = gemini_tools
        try:
            response = await client.generate_content(
                model=self._settings.gemini_model,
                contents=contents,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            raise _translate_gemini_error(exc) from None
        return parse_gemini_response(response)

    def _ensure_client(self) -> GeminiGenerateClient:
        if self._client is not None:
            return self._client
        if not self._settings.gemini_api_key:
            raise LLMAuthError("GEMINI_API_KEY is missing or invalid")
        from google import genai
        from google.genai import types

        http_kwargs: dict[str, object] = {
            "timeout": int(self._settings.llm_timeout_s * 1000),
            "retry_options": types.HttpRetryOptions(attempts=1),
        }
        if self._settings.gemini_base_url:
            http_kwargs["base_url"] = self._settings.gemini_base_url
        real = genai.Client(
            api_key=self._settings.gemini_api_key,
            http_options=types.HttpOptions(**http_kwargs),
        )
        self._client = real.aio.models
        return self._client


def _assistant_content(message: dict[str, object]) -> dict[str, object]:
    parts: list[dict[str, object]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        parts.append({"text": content})
    raw_calls = message.get("tool_calls")
    if isinstance(raw_calls, list):
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            name = encode_tool_name(str(call.get("name", "")))
            args = call.get("args", {})
            if not isinstance(args, dict):
                args = {}
            part: dict[str, object] = {
                "function_call": {
                    "name": name,
                    "args": args,
                    "id": str(call.get("id", "")),
                }
            }
            state = call.get("provider_state")
            signature = None
            if isinstance(state, dict):
                signature = state.get("thought_signature")
            if isinstance(signature, str) and signature:
                part["thought_signature"] = signature
            elif isinstance(signature, (bytes, bytearray)) and signature:
                part["thought_signature"] = bytes(signature)
            else:
                part["thought_signature"] = SKIP_THOUGHT_SIGNATURE
            parts.append(part)
    if not parts:
        parts.append({"text": ""})
    return {"role": "model", "parts": parts}


def _tool_part(message: dict[str, object]) -> dict[str, object]:
    raw_name = message.get("name", "")
    name = encode_tool_name(raw_name if isinstance(raw_name, str) else str(raw_name))
    content = message.get("content", "")
    if not isinstance(content, str):
        content = str(content)
    truncated = truncate_tool_content(content)
    try:
        parsed = json.loads(truncated)
        response = parsed if isinstance(parsed, dict) else {"result": parsed}
    except json.JSONDecodeError:
        response = {"result": truncated}
    payload: dict[str, object] = {
        "function_response": {
            "name": name,
            "response": response,
        }
    }
    call_id = message.get("tool_call_id")
    if isinstance(call_id, str) and call_id:
        payload["function_response"] = {
            "name": name,
            "response": response,
            "id": call_id,
        }
    return payload


def _parse_function_call(function_call: Any, part: Any, index: int) -> ToolCall:
    raw_name = getattr(function_call, "name", "") or ""
    name = decode_tool_name(raw_name if isinstance(raw_name, str) else str(raw_name))
    raw_id = getattr(function_call, "id", None)
    call_id = raw_id if isinstance(raw_id, str) and raw_id else f"gem-{index}"
    signature = getattr(part, "thought_signature", None)
    state: dict[str, object] | None = None
    if isinstance(signature, (bytes, bytearray)):
        state = {"thought_signature": base64.b64encode(bytes(signature)).decode("ascii")}
    elif isinstance(signature, str) and signature:
        state = {"thought_signature": signature}
    args, parse_error = _parse_args(getattr(function_call, "args", None))
    return ToolCall(
        name=name,
        args=args,
        call_id=call_id,
        parse_error=parse_error,
        provider_state=state,
    )


def _parse_args(raw: object) -> tuple[dict[str, object], str | None]:
    if raw is None:
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}, "invalid JSON arguments"
        if not isinstance(parsed, dict):
            return {}, "invalid JSON arguments"
        return parsed, None
    return {}, "invalid JSON arguments"


def _translate_gemini_error(exc: Exception) -> Exception:
    if isinstance(exc, LLMError):
        return exc
    name = type(exc).__name__
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(exc, "status_code", None)
    if name in {"ClientError"} and code in {401, 403}:
        return LLMAuthError("GEMINI_API_KEY is missing or invalid")
    if code in {401, 403}:
        return LLMAuthError("GEMINI_API_KEY is missing or invalid")
    if name in {"ClientError"} and code == 429:
        return LLMRateLimited("Gemini rate limit reached", retry_after_s=_retry_after_s(exc))
    if code == 429:
        return LLMRateLimited("Gemini rate limit reached", retry_after_s=_retry_after_s(exc))
    if name in {
        "APITimeoutError",
        "TimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutException",
        "APIConnectionError",
        "ConnectionError",
    }:
        return LLMUnavailable("Gemini is temporarily unavailable")
    if name in {"ServerError"} or (isinstance(code, int) and code >= 500):
        return LLMUnavailable("Gemini is temporarily unavailable")
    if name in {"ClientError"} and isinstance(code, int) and 400 <= code < 500:
        return LLMBadRequest("Gemini rejected the request")
    if isinstance(code, int) and 400 <= code < 500:
        return LLMBadRequest("Gemini rejected the request")
    return LLMUnavailable("Gemini is temporarily unavailable")


def _retry_after_s(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
