from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from atlas_agent.core.events import AssistantMessage, RunFinished
from atlas_agent.core.prompts import FINAL_ANSWER_REQUEST
from atlas_agent.llm.base import AgentTurn, ToolCall
from tests.conftest import RecordingEmitter, run_scripted, wait_true


class RecordingProvider:
    def __init__(self, turns: list[AgentTurn]) -> None:
        self.turns = list(turns)
        self.calls: list[list[dict[str, object]]] = []

    async def step(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> AgentTurn:
        del tools
        self.calls.append([dict(message) for message in messages])
        return self.turns.pop(0) if self.turns else AgentTurn()


ECHO = AgentTurn(tool_calls=[ToolCall(name="demo.echo", args={"text": "hi"})])


def _assistant_texts(emitter: RecordingEmitter) -> list[str]:
    return [
        event.text for event in emitter.events if isinstance(event, AssistantMessage)
    ]


def _finished(emitter: RecordingEmitter) -> RunFinished:
    finished = emitter.of_type("run_finished")[0]
    assert isinstance(finished, RunFinished)
    return finished


def test_empty_final_text_gets_one_more_step(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    provider = RecordingProvider([ECHO, AgentTurn(), AgentTurn(text="I said hi for you.")])
    emitter = asyncio.run(run_scripted("say hi", tmp_path, provider=provider))
    assert len(provider.calls) == 3  # tool call, empty answer, exactly one extra step
    last = provider.calls[2][-1]
    assert last == {"role": "user", "content": FINAL_ANSWER_REQUEST}
    assert _assistant_texts(emitter) == ["I said hi for you."]
    finished = _finished(emitter)
    assert (finished.status, finished.summary) == ("done", "I said hi for you.")
    assert "no text after tool results" in caplog.text


@pytest.mark.parametrize("empty", [AgentTurn(), AgentTurn(text=""), AgentTurn(text="  \n ")])
def test_still_empty_finishes_with_neutral_message(tmp_path: Path, empty: AgentTurn) -> None:
    provider = RecordingProvider([ECHO, empty, empty, AgentTurn(text="never reached")])
    emitter = asyncio.run(run_scripted("say hi", tmp_path, provider=provider))
    assert len(provider.calls) == 3  # one extra step only, no loop
    assert _assistant_texts(emitter) == ["Done. Ran demo.echo."]
    finished = _finished(emitter)
    assert (finished.status, finished.summary) == ("done", "Done. Ran demo.echo.")


def test_neutral_message_lists_each_tool_once_and_never_tool_output(tmp_path: Path) -> None:
    two = AgentTurn(
        tool_calls=[
            ToolCall(name="demo.echo", args={"text": "TOPSECRET-OUTPUT"}),
            ToolCall(name="demo.echo", args={"text": "again"}),
        ]
    )
    provider = RecordingProvider([two, AgentTurn(), AgentTurn()])
    emitter = asyncio.run(run_scripted("go", tmp_path, provider=provider))
    texts = _assistant_texts(emitter)
    assert texts == ["Done. Ran demo.echo."]
    assert "TOPSECRET" not in texts[0] and "{" not in texts[0]


def test_empty_reply_without_any_tool_is_unchanged(tmp_path: Path) -> None:
    provider = RecordingProvider([AgentTurn()])
    emitter = asyncio.run(run_scripted("hello", tmp_path, provider=provider))
    assert len(provider.calls) == 1
    assert _assistant_texts(emitter) == []
    assert _finished(emitter).summary == "done"


def test_normal_text_after_tool_needs_no_extra_step(tmp_path: Path) -> None:
    provider = RecordingProvider([ECHO, AgentTurn(text="All set.")])
    emitter = asyncio.run(run_scripted("say hi", tmp_path, provider=provider))
    assert len(provider.calls) == 2
    assert _assistant_texts(emitter) == ["All set."]


def test_scripted_provider_never_emits_raw_tool_json(tmp_path: Path) -> None:
    for task, approve in (("Hy", True), ("hello there", True), ("danger zone", True), ("danger", False)):
        async def decide(call_id: str, approve: bool = approve) -> bool:
            del call_id
            return approve

        emitter = asyncio.run(run_scripted(task, tmp_path, wait_approval=decide))
        texts = _assistant_texts(emitter)
        assert texts, task
        for text in texts:
            assert "Done: {" not in text
            assert '"ok"' not in text and '"output"' not in text and "{" not in text
        assert _finished(emitter).summary == texts[-1]


def test_scripted_greeting_text_is_a_sentence(tmp_path: Path) -> None:
    emitter = asyncio.run(run_scripted("Hy", tmp_path, wait_approval=wait_true))
    assert _assistant_texts(emitter) == ["Done: demo.echo completed."]
