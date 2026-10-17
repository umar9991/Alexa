from __future__ import annotations

import asyncio
import logging

import pytest

from atlas_agent.llm.base import (
    AgentTurn,
    LLMAuthError,
    LLMBadRequest,
    LLMRateLimited,
    LLMUnavailable,
)
from atlas_agent.llm.chain import ChainProvider


class FakeProvider:
    def __init__(
        self,
        *,
        name: str,
        outcomes: list[object],
        calls: list[str],
    ) -> None:
        self._name = name
        self._outcomes = list(outcomes)
        self._calls = calls

    async def step(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> AgentTurn:
        del messages, tools
        self._calls.append(self._name)
        item = self._outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, AgentTurn)
        return item


def test_chain_fallback_and_no_fallback() -> None:
    async def _run() -> None:
        calls: list[str] = []
        ok = AgentTurn(text="from-b")
        chain = ChainProvider(
            [
                ("groq", FakeProvider(name="groq", outcomes=[LLMRateLimited("r")], calls=calls)),
                ("gemini", FakeProvider(name="gemini", outcomes=[ok], calls=calls)),
            ],
            clock=lambda: 0.0,
        )
        turn = await chain.step([{"role": "user", "content": "hi"}], [])
        assert turn.text == "from-b"
        assert calls == ["groq", "gemini"]

        calls.clear()
        chain = ChainProvider(
            [
                ("groq", FakeProvider(name="groq", outcomes=[LLMUnavailable("down")], calls=calls)),
                ("gemini", FakeProvider(name="gemini", outcomes=[ok], calls=calls)),
            ],
            clock=lambda: 0.0,
        )
        assert (await chain.step([], [])).text == "from-b"

        calls.clear()
        chain = ChainProvider(
            [
                ("groq", FakeProvider(name="groq", outcomes=[LLMAuthError("bad key")], calls=calls)),
                ("gemini", FakeProvider(name="gemini", outcomes=[ok], calls=calls)),
            ],
            clock=lambda: 0.0,
        )
        with pytest.raises(LLMAuthError):
            await chain.step([], [])
        assert calls == ["groq"]

        calls.clear()
        chain = ChainProvider(
            [
                ("groq", FakeProvider(name="groq", outcomes=[LLMBadRequest("bad")], calls=calls)),
                ("gemini", FakeProvider(name="gemini", outcomes=[ok], calls=calls)),
            ],
            clock=lambda: 0.0,
        )
        with pytest.raises(LLMBadRequest):
            await chain.step([], [])
        assert calls == ["groq"]

    asyncio.run(_run())


def test_chain_cooldown_retry_after_and_all_failed(caplog: pytest.LogCaptureFixture) -> None:
    now = {"t": 0.0}

    def clock() -> float:
        return now["t"]

    async def _run() -> None:
        calls: list[str] = []
        caplog.set_level(logging.WARNING, logger="atlas_agent")
        groq = FakeProvider(
            name="groq",
            outcomes=[
                LLMRateLimited("r", retry_after_s=200),
                AgentTurn(text="should-not"),
            ],
            calls=calls,
        )
        gemini = FakeProvider(
            name="gemini",
            outcomes=[AgentTurn(text="g1"), AgentTurn(text="g2"), AgentTurn(text="g2")],
            calls=calls,
        )
        chain = ChainProvider([("groq", groq), ("gemini", gemini)], cooldown_s=60, clock=clock)
        assert (await chain.step([], [])).text == "g1"
        assert calls == ["groq", "gemini"]
        assert "LLMRateLimited" in caplog.text
        assert "groq" in caplog.text
        assert "sk-" not in caplog.text

        calls.clear()
        now["t"] = 10.0
        assert (await chain.step([], [])).text == "g2"
        assert calls == ["gemini"]

        calls.clear()
        now["t"] = 199.0
        assert (await chain.step([], [])).text == "g2"
        assert calls == ["gemini"]

        calls.clear()
        now["t"] = 200.0
        groq._outcomes.insert(0, AgentTurn(text="g-ok"))
        assert (await chain.step([], [])).text == "g-ok"
        assert calls == ["groq"]

        now["t"] = 0.0
        fail_calls: list[str] = []
        failing = ChainProvider(
            [
                (
                    "groq",
                    FakeProvider(name="groq", outcomes=[LLMRateLimited("r")], calls=fail_calls),
                ),
                (
                    "gemini",
                    FakeProvider(
                        name="gemini", outcomes=[LLMUnavailable("down")], calls=fail_calls
                    ),
                ),
            ],
            clock=clock,
        )
        with pytest.raises(LLMUnavailable, match="groq: rate limited; gemini: unavailable"):
            await failing.step([], [])

        primary: list[str] = []
        secondary: list[str] = []
        success = ChainProvider(
            [
                ("groq", FakeProvider(name="groq", outcomes=[AgentTurn(text="p")], calls=primary)),
                ("gemini", FakeProvider(name="gemini", outcomes=[AgentTurn(text="s")], calls=secondary)),
            ],
            clock=clock,
        )
        assert (await success.step([], [])).text == "p"
        assert primary == ["groq"]
        assert secondary == []

    asyncio.run(_run())


def test_chain_retry_after_capped() -> None:
    now = {"t": 0.0}

    async def _run() -> None:
        calls: list[str] = []
        chain = ChainProvider(
            [
                (
                    "groq",
                    FakeProvider(
                        name="groq",
                        outcomes=[LLMRateLimited("r", retry_after_s=5000), AgentTurn(text="late")],
                        calls=calls,
                    ),
                ),
                (
                    "gemini",
                    FakeProvider(
                        name="gemini",
                        outcomes=[AgentTurn(text="now"), AgentTurn(text="now")],
                        calls=calls,
                    ),
                ),
            ],
            clock=lambda: now["t"],
        )
        assert (await chain.step([], [])).text == "now"
        now["t"] = 3599.0
        calls.clear()
        assert (await chain.step([], [])).text == "now"
        assert calls == ["gemini"]
        now["t"] = 3600.0
        calls.clear()
        assert (await chain.step([], [])).text == "late"
        assert calls == ["groq"]

    asyncio.run(_run())
