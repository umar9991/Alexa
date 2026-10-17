from __future__ import annotations

import asyncio
import json
from pathlib import Path

from atlas_agent.core.events import ToolResult
from atlas_agent.core.loop import run_agent
from atlas_agent.core.policy import decide
from atlas_agent.llm.base import AgentTurn, ToolCall
from atlas_agent.llm.scripted import ScriptedProvider
from tests.conftest import (
    AlwaysToolProvider,
    RecordingEmitter,
    SequenceProvider,
    default_registry,
    make_audit,
    run_scripted,
    wait_false,
    wait_forever,
    wait_true,
)


def test_loop_hello_event_order(tmp_path: Path) -> None:
    async def _run() -> None:
        emitter = await run_scripted("hello", tmp_path)
        assert emitter.types() == [
            "run_started",
            "tool_call",
            "tool_result",
            "assistant_message",
            "run_finished",
        ]
        result = emitter.of_type("tool_result")[0]
        assert isinstance(result, ToolResult)
        assert result.ok is True
        finished = emitter.of_type("run_finished")[0]
        assert finished.status == "done"  # type: ignore[union-attr]

    asyncio.run(_run())


def test_loop_danger_waits_then_approve(tmp_path: Path) -> None:
    async def _run() -> None:
        gate = asyncio.Event()
        decision: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        async def waiter(_call_id: str) -> bool:
            gate.set()
            return await decision

        emitter = RecordingEmitter()
        task = asyncio.create_task(
            run_agent(
                "danger zone",
                provider=ScriptedProvider(),
                registry=default_registry(),
                emit=emitter.emit,
                wait_approval=waiter,
                audit=make_audit(tmp_path),
                run_id="run-danger",
                policy_decide=decide,
            )
        )
        await asyncio.wait_for(gate.wait(), timeout=2)
        assert "approval_required" in emitter.types()
        assert "tool_result" not in emitter.types()
        decision.set_result(True)
        await task
        result = emitter.of_type("tool_result")[0]
        assert isinstance(result, ToolResult)
        assert result.ok is True

    asyncio.run(_run())


def test_loop_danger_reject(tmp_path: Path) -> None:
    async def _run() -> None:
        emitter = await run_scripted("danger zone", tmp_path, wait_approval=wait_false)
        result = emitter.of_type("tool_result")[0]
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert result.error == "rejected by user"

    asyncio.run(_run())


def test_loop_approval_timeout(tmp_path: Path) -> None:
    async def _run() -> None:
        emitter = await run_scripted(
            "danger zone",
            tmp_path,
            wait_approval=wait_forever,
            approval_timeout_s=0.05,
        )
        result = emitter.of_type("tool_result")[0]
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert result.error == "approval timed out"
        finished = emitter.of_type("run_finished")[0]
        assert finished.status == "done"  # type: ignore[union-attr]

    asyncio.run(_run())


def test_loop_invalid_args(tmp_path: Path) -> None:
    async def _run() -> None:
        provider = SequenceProvider(
            [
                AgentTurn(tool_calls=[ToolCall(name="demo.echo", args={})]),
                AgentTurn(text="Done: invalid"),
            ]
        )
        emitter = await run_scripted("hello", tmp_path, provider=provider)
        result = emitter.of_type("tool_result")[0]
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert result.error is not None
        assert "invalid args" in result.error
        assert "Traceback" not in result.error

    asyncio.run(_run())


def test_loop_handler_exception_has_no_traceback(tmp_path: Path) -> None:
    async def boom(_args: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("broken")

    async def _run() -> None:
        registry = default_registry()
        echo = registry.get("demo.echo")
        assert echo is not None
        echo.run = boom
        provider = SequenceProvider(
            [
                AgentTurn(tool_calls=[ToolCall(name="demo.echo", args={"text": "x"})]),
                AgentTurn(text="Done: failed"),
            ]
        )
        emitter = RecordingEmitter()
        await run_agent(
            "hello",
            provider=provider,
            registry=registry,
            emit=emitter.emit,
            wait_approval=wait_true,
            audit=make_audit(tmp_path),
            run_id="run-boom",
            policy_decide=decide,
        )
        result = emitter.of_type("tool_result")[0]
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert result.error is not None
        assert "broken" in result.error
        assert "Traceback" not in result.error

    asyncio.run(_run())


def test_loop_max_steps_failed(tmp_path: Path) -> None:
    async def _run() -> None:
        emitter = await run_scripted(
            "hello",
            tmp_path,
            provider=AlwaysToolProvider("demo.echo", {"text": "x"}),
            max_steps=2,
        )
        finished = emitter.of_type("run_finished")[0]
        assert finished.status == "failed"  # type: ignore[union-attr]
        assert finished.summary == "max steps reached"  # type: ignore[union-attr]

    asyncio.run(_run())


def test_loop_cancel(tmp_path: Path) -> None:
    async def _run() -> None:
        cancel = asyncio.Event()
        gate = asyncio.Event()

        async def waiter(_call_id: str) -> bool:
            gate.set()
            await cancel.wait()
            return False

        emitter = RecordingEmitter()
        task = asyncio.create_task(
            run_agent(
                "danger zone",
                provider=ScriptedProvider(),
                registry=default_registry(),
                emit=emitter.emit,
                wait_approval=waiter,
                audit=make_audit(tmp_path),
                run_id="run-cancel",
                policy_decide=decide,
                cancel_event=cancel,
            )
        )
        await asyncio.wait_for(gate.wait(), timeout=2)
        cancel.set()
        await task
        finished = emitter.of_type("run_finished")[0]
        assert finished.status == "cancelled"  # type: ignore[union-attr]

    asyncio.run(_run())


def test_audit_allowed_and_rejected(tmp_path: Path) -> None:
    async def _run() -> None:
        audit = make_audit(tmp_path)
        emitter = RecordingEmitter()
        await run_agent(
            "hello",
            provider=ScriptedProvider(),
            registry=default_registry(),
            emit=emitter.emit,
            wait_approval=wait_true,
            audit=audit,
            run_id="run-a",
            policy_decide=decide,
        )
        await run_agent(
            "danger zone",
            provider=ScriptedProvider(),
            registry=default_registry(),
            emit=emitter.emit,
            wait_approval=wait_false,
            audit=audit,
            run_id="run-b",
            policy_decide=decide,
        )
        lines = [line for line in audit.path.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 2
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert first["decision"] == "allow"
        assert first["approved"] is True
        assert second["decision"] == "needs_approval"
        assert second["approved"] is False

    asyncio.run(_run())
