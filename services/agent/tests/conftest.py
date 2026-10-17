from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from atlas_agent.core.audit import AuditLog
from atlas_agent.core.events import AgentEvent
from atlas_agent.core.loop import run_agent
from atlas_agent.core.policy import decide
from atlas_agent.llm.base import AgentTurn, LLMProvider, ToolCall
from atlas_agent.llm.scripted import ScriptedProvider
from atlas_agent.tools.registry import ToolRegistry


class RecordingEmitter:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [event.type for event in self.events]

    def of_type(self, type_name: str) -> list[AgentEvent]:
        return [event for event in self.events if event.type == type_name]


class AlwaysToolProvider:
    def __init__(self, name: str, args: dict[str, object]) -> None:
        self.name = name
        self.args = args

    async def step(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> AgentTurn:
        del messages, tools
        return AgentTurn(tool_calls=[ToolCall(name=self.name, args=self.args)])


class SequenceProvider:
    def __init__(self, turns: list[AgentTurn]) -> None:
        self.turns = list(turns)

    async def step(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> AgentTurn:
        del messages, tools
        if not self.turns:
            return AgentTurn(text="empty")
        return self.turns.pop(0)


async def wait_true(_call_id: str) -> bool:
    return True


async def wait_false(_call_id: str) -> bool:
    return False


async def wait_forever(_call_id: str) -> bool:
    await asyncio.sleep(3600)
    return True


def make_audit(tmp_path: Path) -> AuditLog:
    return AuditLog(tmp_path / "audit.jsonl")


def default_registry() -> ToolRegistry:
    return ToolRegistry.load()


async def run_scripted(
    task: str,
    tmp_path: Path,
    *,
    wait_approval: Callable[[str], Awaitable[bool]] = wait_true,
    provider: LLMProvider | None = None,
    approval_timeout_s: float = 60,
    max_steps: int = 6,
    cancel_event: asyncio.Event | None = None,
    run_id: str = "run-1",
) -> RecordingEmitter:
    emitter = RecordingEmitter()
    await run_agent(
        task,
        provider=provider or ScriptedProvider(),
        registry=default_registry(),
        emit=emitter.emit,
        wait_approval=wait_approval,
        audit=make_audit(tmp_path),
        run_id=run_id,
        policy_decide=decide,
        approval_timeout_s=approval_timeout_s,
        max_steps=max_steps,
        cancel_event=cancel_event,
    )
    return emitter
