from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atlas_agent.config import DEFAULT_GROQ_MODEL, Settings
from atlas_agent.llm.base import (
    AgentTurn,
    LLMAuthError,
    LLMBadRequest,
    LLMRateLimited,
)
from atlas_agent.llm.chain import ChainProvider
from atlas_agent.llm.factory import ProviderConfigError, build_provider
from atlas_agent.llm.gemini import GeminiProvider
from atlas_agent.llm.groq import (
    GroqProvider,
    _translate_groq_error,
    internal_to_groq_messages,
    parse_groq_message,
    specs_to_groq_tools,
)
from atlas_agent.llm.names import TOOL_CONTENT_LIMIT, decode_tool_name, encode_tool_name
from atlas_agent.llm.retry import with_retry
from atlas_agent.llm.scripted import ScriptedProvider
from atlas_agent.tools.registry import ToolRegistryError


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_provider": "groq",
        "groq_api_key": "test-key",
        "groq_model": DEFAULT_GROQ_MODEL,
        "llm_timeout_s": 30.0,
        "llm_temperature": 0.2,
        "llm_max_tokens": 1024,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def test_name_encoding_round_trip() -> None:
    assert encode_tool_name("demo.echo") == "demo__echo"
    assert decode_tool_name("demo__echo") == "demo.echo"


def test_registry_rejects_double_underscore(tmp_path: Path) -> None:
    from atlas_agent.tools.registry import ToolRegistry

    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "spec.json").write_text(
        json.dumps(
            {
                "name": "demo__echo",
                "description": "bad",
                "tier": "T0",
                "input_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    (folder / "handler.py").write_text("async def run(args):\n    return args\n", encoding="utf-8")
    with pytest.raises(ToolRegistryError, match="__"):
        ToolRegistry.load(tmp_path)


def test_internal_to_groq_messages_and_truncation() -> None:
    long_content = "x" * (TOOL_CONTENT_LIMIT + 50)
    internal = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "name": "demo.echo", "args": {"text": "hi"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "demo.echo",
            "content": long_content,
        },
    ]
    groq_messages = internal_to_groq_messages(internal)
    assert groq_messages[0] == {"role": "system", "content": "sys"}
    assistant = groq_messages[2]
    assert assistant["role"] == "assistant"
    calls = assistant["tool_calls"]
    assert isinstance(calls, list)
    first = calls[0]
    assert isinstance(first, dict)
    function = first["function"]
    assert isinstance(function, dict)
    assert function["name"] == "demo__echo"
    assert first["id"] == "c1"
    tool = groq_messages[3]
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "c1"
    content = tool["content"]
    assert isinstance(content, str)
    assert content.endswith("[truncated]")
    assert len(content) == TOOL_CONTENT_LIMIT + len("\n[truncated]")
    tools = specs_to_groq_tools(
        [{"name": "demo.echo", "description": "echo", "input_schema": {"type": "object"}}]
    )
    assert tools[0]["function"]["name"] == "demo__echo"  # type: ignore[index]


def test_parse_groq_text_only() -> None:
    message = SimpleNamespace(content="hello", tool_calls=None)
    turn = parse_groq_message(message)
    assert turn == AgentTurn(text="hello", tool_calls=[])


def test_parse_groq_valid_tool_call() -> None:
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="c1",
                function=SimpleNamespace(name="demo__echo", arguments='{"text": "hi"}'),
            )
        ],
    )
    turn = parse_groq_message(message)
    assert turn.tool_calls[0].name == "demo.echo"
    assert turn.tool_calls[0].args == {"text": "hi"}
    assert turn.tool_calls[0].parse_error is None


def test_parse_groq_invalid_json_args() -> None:
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="c1",
                function=SimpleNamespace(name="demo__echo", arguments="not-json"),
            )
        ],
    )
    turn = parse_groq_message(message)
    assert turn.tool_calls[0].args == {}
    assert turn.tool_calls[0].parse_error == "invalid JSON arguments"


def test_parse_groq_text_and_tool_calls() -> None:
    message = SimpleNamespace(
        content="calling",
        tool_calls=[
            SimpleNamespace(
                id="c1",
                function=SimpleNamespace(name="demo__echo", arguments='{"text": "x"}'),
            )
        ],
    )
    turn = parse_groq_message(message)
    assert turn.text == "calling"
    assert len(turn.tool_calls) == 1


def test_retry_429_then_success() -> None:
    async def _run() -> None:
        calls = {"n": 0}
        sleeps: list[float] = []

        async def op() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise LLMRateLimited("rate")
            return "ok"

        async def sleep(delay: float) -> None:
            sleeps.append(delay)

        result = await with_retry(op, sleep=sleep)
        assert result == "ok"
        assert calls["n"] == 3
        assert sleeps == [1.0, 2.0]

    asyncio.run(_run())


def test_retry_after_honored_and_capped() -> None:
    async def _run() -> None:
        sleeps: list[float] = []

        async def op() -> str:
            raise LLMRateLimited("rate", retry_after_s=30)

        async def sleep(delay: float) -> None:
            sleeps.append(delay)

        with pytest.raises(LLMRateLimited):
            await with_retry(op, sleep=sleep)
        assert sleeps == [20.0, 20.0]

    asyncio.run(_run())


def test_retry_three_failures_raise() -> None:
    async def _run() -> None:
        async def op() -> str:
            raise LLMRateLimited("rate")

        async def sleep(_delay: float) -> None:
            return None

        with pytest.raises(LLMRateLimited):
            await with_retry(op, sleep=sleep)
        # 3 attempts: two sleeps then raise on last
        calls = {"n": 0}

        async def counting() -> str:
            calls["n"] += 1
            raise LLMRateLimited("rate")

        with pytest.raises(LLMRateLimited):
            await with_retry(counting, sleep=sleep)
        assert calls["n"] == 3

    asyncio.run(_run())


def test_auth_and_bad_request_are_not_retried() -> None:
    async def _count(error: Exception) -> int:
        calls = {"n": 0}

        async def op() -> str:
            calls["n"] += 1
            raise error

        with pytest.raises(type(error)):
            await with_retry(op)
        return calls["n"]

    async def _run() -> None:
        assert await _count(LLMAuthError("nope")) == 1
        assert await _count(LLMBadRequest("bad")) == 1

    asyncio.run(_run())


def test_timeout_and_5xx_are_retried_via_provider() -> None:
    async def _once(error: Exception) -> None:
        fake = _FakeClient([error, _text_response("ok")])
        sleeps: list[float] = []

        async def sleep(delay: float) -> None:
            sleeps.append(delay)

        provider = GroqProvider(_settings(), client=fake, sleep=sleep)
        turn = await provider.step([{"role": "user", "content": "hi"}], [])
        assert turn.text == "ok"
        assert sleeps == [1.0]

    async def _run() -> None:
        await _once(_status_error("APITimeoutError", None))
        await _once(_status_error("InternalServerError", 503))

    asyncio.run(_run())


def test_provider_maps_status_codes() -> None:
    assert isinstance(_translate_groq_error(_status_error("AuthenticationError", 401)), LLMAuthError)
    assert isinstance(_translate_groq_error(_status_error("RateLimitError", 429)), LLMRateLimited)
    assert isinstance(_translate_groq_error(_status_error("BadRequestError", 400)), LLMBadRequest)

    async def _run() -> None:
        provider = GroqProvider(
            _settings(), client=_FakeClient([_status_error("AuthenticationError", 401)])
        )
        with pytest.raises(LLMAuthError):
            await provider.step([{"role": "user", "content": "hi"}], [])
        provider = GroqProvider(
            _settings(), client=_FakeClient([_status_error("BadRequestError", 400)])
        )
        with pytest.raises(LLMBadRequest):
            await provider.step([{"role": "user", "content": "hi"}], [])

    asyncio.run(_run())


def test_factory_default_scripted_and_lazy_key() -> None:
    scripted = build_provider(_settings(llm_provider="scripted", groq_api_key=""))
    assert isinstance(scripted, ScriptedProvider)
    groq = build_provider(_settings(llm_provider="groq", groq_api_key=""))
    assert isinstance(groq, GroqProvider)

    async def _run() -> None:
        with pytest.raises(LLMAuthError, match="GROQ_API_KEY"):
            await groq.step([{"role": "user", "content": "hi"}], [])

    asyncio.run(_run())


def test_factory_chain_unknown_and_gemini_lazy() -> None:
    chain = build_provider(
        _settings(llm_provider="groq,gemini", groq_api_key="", gemini_api_key="")
    )
    assert isinstance(chain, ChainProvider)
    alone = build_provider(_settings(llm_provider="groq"))
    assert isinstance(alone, GroqProvider)
    gemini = build_provider(_settings(llm_provider="gemini", gemini_api_key=""))
    assert isinstance(gemini, GeminiProvider)
    with pytest.raises(ProviderConfigError, match="unknown"):
        build_provider(_settings(llm_provider="nope"))

    async def _run() -> None:
        with pytest.raises(LLMAuthError, match="GEMINI_API_KEY"):
            await gemini.step([{"role": "user", "content": "hi"}], [])

    asyncio.run(_run())


@pytest.mark.live
def test_live_groq_echo_parses() -> None:
    if os.environ.get("ATLAS_LIVE_TESTS") != "1" or not os.environ.get("GROQ_API_KEY"):
        pytest.skip("live Groq tests disabled")
    from atlas_agent.config import load_settings
    from atlas_agent.tools.registry import ToolRegistry

    settings = load_settings()
    provider = GroqProvider(settings)
    registry = ToolRegistry.load()

    async def _run() -> None:
        turn = await provider.step(
            [
                {"role": "system", "content": "Use the echo tool."},
                {"role": "user", "content": "Call demo.echo with text hi"},
            ],
            registry.specs(),
        )
        assert turn.text or turn.tool_calls

    asyncio.run(_run())


def _status_error(name: str, status_code: int | None, retry_after: str | None = None) -> Exception:
    cls = type(name, (Exception,), {})
    exc = cls(name)
    exc.status_code = status_code  # type: ignore[attr-defined]
    headers = {"retry-after": retry_after} if retry_after else {}
    exc.response = SimpleNamespace(headers=headers)  # type: ignore[attr-defined]
    return exc


class _FakeCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)

    async def create(self, **_kwargs: object) -> object:
        item = self._outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(outcomes))


def _text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))])
