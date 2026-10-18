from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from atlas_agent.core.audit import AuditLog
from atlas_agent.core.events import AgentEvent
from atlas_agent.core.loop import run_agent
from atlas_agent.core.policy import decide
from atlas_agent.llm.base import AgentTurn, LLMProvider, ToolCall
from atlas_agent.llm.scripted import ScriptedProvider
from atlas_agent.tools.registry import ToolRegistry

# The scripted provider is opt-in, never a default. The test suite enables it explicitly,
# before any test module imports atlas_agent.main (which builds the provider at import).
# A real shell variable or a developer's .env can then never switch tests to a live LLM.
os.environ["ATLAS_LLM_PROVIDER"] = "scripted"


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
    workspace: Path | None = None,
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
        workspace=workspace,
    )
    return emitter


def make_junction(link: Path, target: Path) -> None:
    if sys.platform != "win32":
        pytest.skip("directory junctions require win32")
    link.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        completed.stderr.strip() or completed.stdout.strip() or "mklink /J failed"
    )


def make_symlink(link: Path, target: Path, *, target_is_directory: bool = True) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        pytest.skip(f"cannot create symlink: {exc}")


def remove_reparse(path: Path) -> None:
    if not path.exists():
        return
    if os.path.isjunction(path):
        os.rmdir(path)
        return
    if os.path.islink(path):
        path.unlink()


def outside_layout(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "ws"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret-bytes", encoding="utf-8")
    nested = outside / "sub"
    nested.mkdir()
    (nested / "nested.txt").write_text("nested-bytes", encoding="utf-8")
    return workspace, outside

