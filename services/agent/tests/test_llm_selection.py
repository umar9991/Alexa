from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from atlas_agent.config import Settings, load_settings
from atlas_agent.llm.base import AgentTurn, ToolCall
from atlas_agent.llm.chain import ChainProvider
from atlas_agent.llm.factory import (
    DebugLoggingProvider,
    ProviderConfigError,
    build_provider,
    describe_provider_chain,
)
from atlas_agent.llm.groq import GroqProvider
from atlas_agent.llm.scripted import ScriptedProvider

KEY_SENTINEL = "gsk_sentinel-key-do-not-log"
CONTENT_SENTINEL = "TOPSECRET-MESSAGE-CONTENT"
ARG_SENTINEL = "TOPSECRET-TOOL-ARG"


def _settings(provider: str, *, debug: bool = False) -> Settings:
    return Settings(
        llm_provider=provider,
        groq_api_key=KEY_SENTINEL,
        groq_model="openai/gpt-oss-20b",
        llm_timeout_s=30,
        llm_temperature=0.2,
        llm_max_tokens=1024,
        gemini_api_key=KEY_SENTINEL,
        gemini_model="gemini-3.1-flash-lite",
        llm_debug=debug,
    )


def test_describe_chain_lists_name_model_in_order() -> None:
    assert describe_provider_chain(_settings("groq,gemini")) == [
        "groq:openai/gpt-oss-20b",
        "gemini:gemini-3.1-flash-lite",
    ]
    assert describe_provider_chain(_settings(" Gemini , GROQ ")) == [
        "gemini:gemini-3.1-flash-lite",
        "groq:openai/gpt-oss-20b",
    ]
    assert describe_provider_chain(_settings("scripted")) == ["scripted:test-only"]
    blob = " ".join(describe_provider_chain(_settings("groq,gemini,scripted")))
    assert KEY_SENTINEL not in blob and "http" not in blob


@pytest.mark.parametrize("value", ["", "   ", " , "])
def test_unset_provider_fails_with_valid_values(value: str) -> None:
    for call in (build_provider, describe_provider_chain):
        with pytest.raises(ProviderConfigError) as exc_info:
            call(_settings(value))
        message = str(exc_info.value)
        assert "ATLAS_LLM_PROVIDER is not set" in message
        for valid in ("groq", "gemini", "groq,gemini", "scripted"):
            assert valid in message
        assert KEY_SENTINEL not in message


def test_unknown_provider_lists_valid_values() -> None:
    with pytest.raises(ProviderConfigError, match="unknown LLM provider: nope") as exc_info:
        describe_provider_chain(_settings("nope"))
    assert "groq" in str(exc_info.value)


def test_scripted_is_opt_in_unset_vs_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The suite sets it explicitly in conftest.
    assert os.environ["ATLAS_LLM_PROVIDER"] == "scripted"
    assert load_settings().llm_provider == "scripted"
    assert isinstance(build_provider(load_settings()), ScriptedProvider)
    # Unset (and no .env reachable from the working directory): nothing falls back to scripted.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ATLAS_LLM_PROVIDER")
    assert load_settings().llm_provider == ""
    monkeypatch.setenv("ATLAS_LLM_PROVIDER", "  ")
    assert load_settings().llm_provider == ""
    with pytest.raises(ProviderConfigError):
        build_provider(load_settings())


def _start_backend(tmp_path: Path, provider: str | None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "ATLAS_LLM_PROVIDER"}
    if provider is not None:
        env["ATLAS_LLM_PROVIDER"] = provider
    return subprocess.run(
        [sys.executable, "-c", "import atlas_agent.main"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_startup_fails_once_with_clear_error_when_unset(tmp_path: Path) -> None:
    for provider in (None, ""):
        result = _start_backend(tmp_path, provider)
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        assert "ATLAS_LLM_PROVIDER is not set" in result.stderr
        for valid in ("groq", "gemini", "scripted"):
            assert valid in result.stderr
        assert result.stderr.count("cannot start") == 1


def test_startup_succeeds_and_logs_chain_when_explicit(tmp_path: Path) -> None:
    result = _start_backend(tmp_path, "scripted")
    assert result.returncode == 0, result.stderr
    assert "LLM chain: scripted:test-only" in result.stderr


class _FakeInner:
    def __init__(self, turn: AgentTurn) -> None:
        self.turn = turn

    async def step(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> AgentTurn:
        del messages, tools
        return self.turn


def _secret_turn() -> AgentTurn:
    return AgentTurn(
        text=CONTENT_SENTINEL,
        tool_calls=[ToolCall(name="demo.echo", args={"text": ARG_SENTINEL})],
    )


def test_debug_flag_off_by_default_and_wraps_only_when_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    plain = build_provider(_settings("scripted"))
    assert isinstance(plain, ScriptedProvider)
    groq = build_provider(_settings("groq"))
    assert isinstance(groq, GroqProvider)
    assert isinstance(build_provider(_settings("groq,gemini")), ChainProvider)
    asyncio.run(plain.step([{"role": "user", "content": "hi"}], []))
    assert "llm step" not in caplog.text
    wrapped = build_provider(_settings("scripted", debug=True))
    assert isinstance(wrapped, DebugLoggingProvider)


def test_debug_logs_metadata_only(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    provider = DebugLoggingProvider(
        _FakeInner(_secret_turn()), name="groq", model="openai/gpt-oss-20b"
    )
    messages: list[dict[str, object]] = [
        {"role": "system", "content": CONTENT_SENTINEL},
        {"role": "user", "content": CONTENT_SENTINEL},
    ]
    turn = asyncio.run(provider.step(messages, []))
    assert turn.text == CONTENT_SENTINEL  # the turn itself is passed through untouched
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm step")]
    expected = (
        "llm step provider=groq model=openai/gpt-oss-20b finish_reason=unreported "
        "text_present=True tool_calls=['demo.echo']"
    )
    assert lines == [expected]
    assert CONTENT_SENTINEL not in caplog.text
    assert ARG_SENTINEL not in caplog.text
    assert KEY_SENTINEL not in caplog.text


def test_debug_wrapper_inside_chain_logs_each_provider_without_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    chain = build_provider(_settings("scripted,scripted", debug=True))
    assert isinstance(chain, ChainProvider)
    asyncio.run(chain.step([{"role": "user", "content": CONTENT_SENTINEL}], []))
    assert "llm step provider=scripted model=test-only" in caplog.text
    assert "text_present=False tool_calls=['demo.echo']" in caplog.text
    assert CONTENT_SENTINEL not in caplog.text
    assert KEY_SENTINEL not in caplog.text


def test_empty_text_turn_is_reported_as_not_present(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    provider = DebugLoggingProvider(_FakeInner(AgentTurn(text="  ")), name="gemini", model="m")
    asyncio.run(provider.step([], []))
    assert "text_present=False tool_calls=[]" in caplog.text


def test_debug_env_flag_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATLAS_LLM_DEBUG", raising=False)
    assert load_settings().llm_debug is False
    monkeypatch.setenv("ATLAS_LLM_DEBUG", "")
    assert load_settings().llm_debug is False
    monkeypatch.setenv("ATLAS_LLM_DEBUG", "1")
    assert load_settings().llm_debug is True
    monkeypatch.setenv("ATLAS_LLM_DEBUG", "0")
    assert load_settings().llm_debug is False
