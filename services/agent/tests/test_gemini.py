from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from types import SimpleNamespace

import pytest

from atlas_agent.config import DEFAULT_GEMINI_MODEL, Settings
from atlas_agent.core.prompts import SYSTEM_PROMPT
from atlas_agent.llm.base import (
    AgentTurn,
    LLMAuthError,
    LLMBadRequest,
    LLMRateLimited,
    LLMUnavailable,
)
from atlas_agent.llm.gemini import (
    SKIP_THOUGHT_SIGNATURE,
    GeminiProvider,
    _retry_after_s,
    _translate_gemini_error,
    internal_to_gemini,
    parse_gemini_response,
    parse_retry_duration,
    sanitize_gemini_schema,
    specs_to_gemini_tools,
)
from atlas_agent.llm.names import TOOL_CONTENT_LIMIT
from atlas_agent.llm.retry import with_retry


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_provider": "gemini",
        "groq_api_key": "",
        "groq_model": "x",
        "llm_timeout_s": 30.0,
        "llm_temperature": 0.2,
        "llm_max_tokens": 1024,
        "gemini_api_key": "test-key",
        "gemini_model": DEFAULT_GEMINI_MODEL,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def test_sanitize_schema_strips_unsupported_and_keeps_nested() -> None:
    original = {
        "type": "object",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Args",
        "additionalProperties": False,
        "properties": {
            "path": {
                "type": "string",
                "description": "rel",
                "default": ".",
                "minLength": 1,
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {"name": {"type": "string", "title": "n"}},
                    "required": ["name"],
                },
            },
        },
        "required": ["path"],
    }
    snapshot = json.loads(json.dumps(original))
    cleaned = sanitize_gemini_schema(original)
    assert original == snapshot
    assert isinstance(cleaned, dict)
    assert "$schema" not in cleaned
    assert "title" not in cleaned
    assert "additionalProperties" not in cleaned
    assert cleaned["type"] == "object"
    assert cleaned["required"] == ["path"]
    props = cleaned["properties"]
    assert isinstance(props, dict)
    path = props["path"]
    assert isinstance(path, dict)
    assert path["type"] == "string"
    assert "default" not in path
    assert "minLength" not in path
    items = props["items"]
    assert isinstance(items, dict)
    nested = items["items"]
    assert isinstance(nested, dict)
    assert "additionalProperties" not in nested
    inner = nested["properties"]
    assert isinstance(inner, dict)
    name = inner["name"]
    assert isinstance(name, dict)
    assert "title" not in name


def test_internal_to_gemini_system_tools_truncation_and_state() -> None:
    long = "x" * (TOOL_CONTENT_LIMIT + 50)
    system, contents = internal_to_gemini(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "demo.echo",
                        "args": {"text": "a"},
                        "provider_state": {"thought_signature": "sig-1"},
                    },
                    {"id": "c2", "name": "demo.echo", "args": {"text": "b"}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "demo.echo", "content": '{"ok": true}'},
            {"role": "tool", "tool_call_id": "c2", "name": "demo.echo", "content": long},
        ]
    )
    assert system == "sys"
    assert contents[0]["role"] == "user"
    model = contents[1]
    assert model["role"] == "model"
    parts = model["parts"]
    assert isinstance(parts, list)
    first = parts[0]
    assert isinstance(first, dict)
    assert first["thought_signature"] == "sig-1"
    fn = first["function_call"]
    assert isinstance(fn, dict)
    assert fn["name"] == "demo__echo"
    second = parts[1]
    assert isinstance(second, dict)
    assert second["thought_signature"] == SKIP_THOUGHT_SIGNATURE
    grouped = contents[2]
    assert grouped["role"] == "tool"
    tool_parts = grouped["parts"]
    assert isinstance(tool_parts, list)
    assert len(tool_parts) == 2
    second_resp = tool_parts[1]
    assert isinstance(second_resp, dict)
    payload = second_resp["function_response"]
    assert isinstance(payload, dict)
    result = payload["response"]
    assert isinstance(result, dict)
    assert str(result["result"]).endswith("[truncated]")


def test_parse_gemini_variants() -> None:
    text_only = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="hello", function_call=None, thought_signature=None)]))]
    )
    assert parse_gemini_response(text_only) == AgentTurn(text="hello", tool_calls=[])

    call = SimpleNamespace(
        name="demo__echo",
        args={"text": "hi"},
        id="abc",
    )
    with_call = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text=None, function_call=call, thought_signature="sig")]
                )
            )
        ]
    )
    turn = parse_gemini_response(with_call)
    assert turn.tool_calls[0].name == "demo.echo"
    assert turn.tool_calls[0].args == {"text": "hi"}
    assert turn.tool_calls[0].call_id == "abc"
    assert turn.tool_calls[0].provider_state == {"thought_signature": "sig"}

    missing_id = SimpleNamespace(name="demo__echo", args={"text": "z"}, id=None)
    generated = parse_gemini_response(
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(text="also", function_call=None, thought_signature=None),
                            SimpleNamespace(text=None, function_call=missing_id, thought_signature=None),
                        ]
                    )
                )
            ]
        )
    )
    assert generated.text == "also"
    assert generated.tool_calls[0].call_id == "gem-0"

    bad = SimpleNamespace(name="demo__echo", args="{nope", id="x")
    parsed = parse_gemini_response(
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(text=None, function_call=bad, thought_signature=None)]
                    )
                )
            ]
        )
    )
    assert parsed.tool_calls[0].parse_error == "invalid JSON arguments"
    assert parsed.tool_calls[0].args == {}


def test_replay_decodes_stored_signature_bytes() -> None:
    raw = b"\x00sig\xff"
    stored = base64.b64encode(raw).decode("ascii")
    _system, contents = internal_to_gemini(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "demo.echo",
                        "args": {"text": "hi"},
                        "provider_state": {"thought_signature": stored},
                    }
                ],
            }
        ]
    )
    parts = contents[0]["parts"]
    assert isinstance(parts, list)
    part = parts[0]
    assert isinstance(part, dict)
    assert part["thought_signature"] == raw


def test_provider_state_round_trip() -> None:
    captured: list[dict[str, object]] = []

    class Client:
        async def generate_content(self, **kwargs: object) -> object:
            captured.append(dict(kwargs))
            if len(captured) == 1:
                call = SimpleNamespace(name="demo__echo", args={"text": "hi"}, id="c1")
                return SimpleNamespace(
                    candidates=[
                        SimpleNamespace(
                            content=SimpleNamespace(
                                parts=[
                                    SimpleNamespace(
                                        text=None, function_call=call, thought_signature="keep-me"
                                    )
                                ]
                            )
                        )
                    ]
                )
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[SimpleNamespace(text="done", function_call=None, thought_signature=None)]
                        )
                    )
                ]
            )

    async def _run() -> None:
        provider = GeminiProvider(_settings(), client=Client())
        first = await provider.step([{"role": "user", "content": "hi"}], [])
        assert first.tool_calls[0].provider_state == {"thought_signature": "keep-me"}
        await provider.step(
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "demo.echo",
                            "args": {"text": "hi"},
                            "provider_state": first.tool_calls[0].provider_state,
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "name": "demo.echo", "content": "{}"},
            ],
            [],
        )
        contents = captured[1]["contents"]
        assert isinstance(contents, list)
        model = contents[1]
        assert isinstance(model, dict)
        parts = model["parts"]
        assert isinstance(parts, list)
        part = parts[0]
        assert isinstance(part, dict)
        assert part["thought_signature"] == "keep-me"

    asyncio.run(_run())


def test_gemini_error_log_redacts_key_and_keeps_raised_safe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="atlas_agent")
    fake_key = "AIzaSyFakeTestKeyValue0000000000000"
    exc = _err("ClientError", 400)
    exc.status = "INVALID_ARGUMENT"  # type: ignore[attr-defined]
    exc.message = f"Unknown field in request {fake_key}"  # type: ignore[attr-defined]
    raised = _translate_gemini_error(exc)
    assert isinstance(raised, LLMBadRequest)
    assert str(raised) == "Gemini rejected the request"
    assert "Unknown field" not in str(raised)
    assert fake_key not in caplog.text
    assert "AIza" not in caplog.text
    assert "[redacted]" in caplog.text
    assert "INVALID_ARGUMENT" in caplog.text
    assert "400" in caplog.text


def test_parse_retry_duration_tokens() -> None:
    assert parse_retry_duration("2h21m49.85s") == pytest.approx(8509.85)
    assert parse_retry_duration("37s") == pytest.approx(37.0)
    assert parse_retry_duration("1m5s") == pytest.approx(65.0)
    assert parse_retry_duration("nope") is None


def test_gemini_429_retry_after_from_message_and_retry_delay() -> None:
    from_message = _err("ClientError", 429)
    from_message.message = "Please retry in 2h21m49.85s"  # type: ignore[attr-defined]
    assert _retry_after_s(from_message) == pytest.approx(8509.85)
    mapped = _translate_gemini_error(from_message)
    assert isinstance(mapped, LLMRateLimited)
    assert mapped.retry_after_s == pytest.approx(8509.85)

    from_delay = _err("ClientError", 429)
    from_delay.details = {  # type: ignore[attr-defined]
        "error": {
            "status": "RESOURCE_EXHAUSTED",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}],
        }
    }
    assert _retry_after_s(from_delay) == pytest.approx(37.0)

    unknown = _err("ClientError", 429)
    unknown.message = "quota exceeded"  # type: ignore[attr-defined]
    assert _retry_after_s(unknown) is None


def test_gemini_error_mapping_and_retry() -> None:
    assert isinstance(_translate_gemini_error(_err("ClientError", 401)), LLMAuthError)
    assert isinstance(_translate_gemini_error(_err("ClientError", 429)), LLMRateLimited)
    assert isinstance(_translate_gemini_error(_err("ClientError", 400)), LLMBadRequest)
    assert isinstance(_translate_gemini_error(_err("ServerError", 503)), LLMUnavailable)
    assert isinstance(_translate_gemini_error(_err("APITimeoutError", None)), LLMUnavailable)

    async def _run() -> None:
        sleeps: list[float] = []

        async def sleep(delay: float) -> None:
            sleeps.append(delay)

        client = _FakeGemini([_err("ClientError", 429), _text("ok")])
        provider = GeminiProvider(_settings(), client=client, sleep=sleep)
        turn = await provider.step([{"role": "user", "content": "hi"}], [])
        assert turn.text == "ok"
        assert sleeps == [1.0]

        auth = GeminiProvider(_settings(), client=_FakeGemini([_err("ClientError", 401)]))
        with pytest.raises(LLMAuthError):
            await auth.step([{"role": "user", "content": "hi"}], [])

        bad = GeminiProvider(_settings(), client=_FakeGemini([_err("ClientError", 400)]))
        with pytest.raises(LLMBadRequest):
            await bad.step([{"role": "user", "content": "hi"}], [])

        server = _FakeGemini([_err("ServerError", 500), _text("ok")])
        sleeps.clear()
        provider = GeminiProvider(_settings(), client=server, sleep=sleep)
        assert (await provider.step([{"role": "user", "content": "hi"}], [])).text == "ok"

        timeout = _FakeGemini([_err("APITimeoutError", None), _text("ok")])
        provider = GeminiProvider(_settings(), client=timeout, sleep=sleep)
        assert (await provider.step([{"role": "user", "content": "hi"}], [])).text == "ok"

        calls = {"n": 0}

        async def fail() -> str:
            calls["n"] += 1
            raise LLMBadRequest("bad")

        with pytest.raises(LLMBadRequest):
            await with_retry(fail)
        assert calls["n"] == 1

    asyncio.run(_run())


def test_specs_encode_names() -> None:
    tools = specs_to_gemini_tools(
        [{"name": "demo.echo", "description": "e", "input_schema": {"type": "object"}}]
    )
    decl = tools[0]["function_declarations"]
    assert isinstance(decl, list)
    assert decl[0]["name"] == "demo__echo"


def _live_gemini() -> tuple[GeminiProvider, list[dict[str, object]]]:
    from atlas_agent.config import load_settings
    from atlas_agent.tools.registry import ToolRegistry

    settings = load_settings()
    if os.environ.get("ATLAS_LIVE_TESTS") != "1" or not settings.gemini_api_key:
        pytest.skip("live Gemini tests disabled")
    return GeminiProvider(settings), ToolRegistry.load().specs()


@pytest.mark.live
def test_live_gemini_echo_parses() -> None:
    provider, tools = _live_gemini()

    async def _run() -> None:
        turn = await provider.step(
            [
                {"role": "system", "content": "Use the echo tool."},
                {"role": "user", "content": "Call demo.echo with text hi"},
            ],
            tools,
        )
        assert turn.text or turn.tool_calls

    asyncio.run(_run())


@pytest.mark.live
def test_live_groq_origin_history_on_gemini() -> None:
    provider, tools = _live_gemini()
    echo_result = json.dumps(
        {"ok": True, "output": {"echo": "hi"}, "error": None}, ensure_ascii=True
    )

    async def _run() -> None:
        turn = await provider.step(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "use the echo tool to say hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "name": "demo.echo", "args": {"text": "hi"}},
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "name": "demo.echo",
                    "content": echo_result,
                },
            ],
            tools,
        )
        assert isinstance(turn, AgentTurn)
        assert turn.text
        assert all(call.parse_error is None for call in turn.tool_calls)

    asyncio.run(_run())


@pytest.mark.live
def test_live_gemini_origin_history_round_trip() -> None:
    provider, tools = _live_gemini()

    async def _run() -> None:
        first = await provider.step(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "use the echo tool to say hi"},
            ],
            tools,
        )
        assert first.tool_calls
        call = first.tool_calls[0]
        assert call.parse_error is None
        text = call.args.get("text", "hi")
        echo_result = json.dumps(
            {"ok": True, "output": {"echo": text}, "error": None}, ensure_ascii=True
        )
        recorded = {
            "id": call.call_id,
            "name": call.name,
            "args": call.args,
        }
        if call.provider_state is not None:
            recorded["provider_state"] = call.provider_state
        second = await provider.step(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "use the echo tool to say hi"},
                {"role": "assistant", "content": first.text, "tool_calls": [recorded]},
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "name": call.name,
                    "content": echo_result,
                },
            ],
            tools,
        )
        assert isinstance(second, AgentTurn)
        assert second.text

    asyncio.run(_run())


def _err(name: str, code: int | None) -> Exception:
    cls = type(name, (Exception,), {})
    exc = cls(name)
    if code is not None:
        exc.code = code  # type: ignore[attr-defined]
        exc.status_code = code  # type: ignore[attr-defined]
    return exc


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text=text, function_call=None, thought_signature=None)]
                )
            )
        ]
    )


class _FakeGemini:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)

    async def generate_content(self, **_kwargs: object) -> object:
        item = self._outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
