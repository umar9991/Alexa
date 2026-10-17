from __future__ import annotations

import asyncio
import json
from pathlib import Path

from atlas_agent.core.events import ApprovalRequired, ToolResult
from atlas_agent.core.loop import run_agent
from atlas_agent.core.policy import Tier, decide
from atlas_agent.llm.base import (
    AgentTurn,
    LLMAuthError,
    LLMRateLimited,
    LLMUnavailable,
    ToolCall,
)
from atlas_agent.llm.chain import ChainProvider
from atlas_agent.llm.scripted import ScriptedProvider
from atlas_agent.tools.context import ToolContext, ToolError
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
    async def boom(_args: dict[str, object], _ctx: ToolContext) -> dict[str, object]:
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
        assert result.error == "internal tool error"
        assert "broken" not in result.error
        assert "Traceback" not in result.error
        assert str(tmp_path) not in result.error

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


def test_loop_llm_errors_finish_failed(tmp_path: Path) -> None:
    class BoomProvider:
        def __init__(self, error: Exception) -> None:
            self.error = error

        async def step(
            self, messages: list[dict[str, object]], tools: list[dict[str, object]]
        ) -> AgentTurn:
            del messages, tools
            raise self.error

    async def _run() -> None:
        for error, needle in (
            (LLMRateLimited("rate", retry_after_s=8), "rate limit"),
            (LLMAuthError("GROQ_API_KEY is missing or invalid"), "GROQ_API_KEY"),
        ):
            emitter = await run_scripted("hello", tmp_path, provider=BoomProvider(error))
            finished = emitter.of_type("run_finished")[0]
            assert finished.status == "failed"  # type: ignore[union-attr]
            summary = finished.summary  # type: ignore[union-attr]
            assert needle in summary
            assert "sk-" not in summary
            assert "Traceback" not in summary

    asyncio.run(_run())


def test_loop_parse_error_skips_handler_and_audits(tmp_path: Path) -> None:
    called = {"n": 0}

    async def guarded(_args: dict[str, object], _ctx: ToolContext) -> dict[str, object]:
        called["n"] += 1
        return {"echo": "nope"}

    async def _run() -> None:
        registry = default_registry()
        echo = registry.get("demo.echo")
        assert echo is not None
        echo.run = guarded
        provider = SequenceProvider(
            [
                AgentTurn(
                    tool_calls=[
                        ToolCall(
                            name="demo.echo",
                            args={},
                            parse_error="invalid JSON arguments",
                        )
                    ]
                ),
                AgentTurn(text="I will retry"),
            ]
        )
        audit = make_audit(tmp_path)
        emitter = RecordingEmitter()
        await run_agent(
            "hello",
            provider=provider,
            registry=registry,
            emit=emitter.emit,
            wait_approval=wait_true,
            audit=audit,
            run_id="run-parse",
            policy_decide=decide,
        )
        result = emitter.of_type("tool_result")[0]
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert result.error == "invalid JSON arguments"
        assert called["n"] == 0
        line = json.loads(audit.path.read_text(encoding="utf-8").splitlines()[0])
        assert line["decision"] == "invalid_args"

    asyncio.run(_run())


def test_loop_records_assistant_and_tool_ids(tmp_path: Path) -> None:
    class Capturing(SequenceProvider):
        def __init__(self) -> None:
            super().__init__(
                [
                    AgentTurn(
                        tool_calls=[ToolCall(name="demo.echo", args={"text": "hi"}, call_id="c1")]
                    ),
                    AgentTurn(text="Done"),
                ]
            )
            self.seen: list[list[dict[str, object]]] = []

        async def step(
            self, messages: list[dict[str, object]], tools: list[dict[str, object]]
        ) -> AgentTurn:
            self.seen.append([dict(item) for item in messages])
            return await super().step(messages, tools)

    async def _run() -> None:
        provider = Capturing()
        await run_scripted("hello", tmp_path, provider=provider)
        second = provider.seen[1]
        assistant = next(item for item in second if item.get("role") == "assistant")
        tool = next(item for item in second if item.get("role") == "tool")
        calls = assistant["tool_calls"]
        assert isinstance(calls, list)
        assert calls[0]["id"] == "c1"  # type: ignore[index]
        assert tool["tool_call_id"] == "c1"

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


def test_loop_tool_error_passes_through(tmp_path: Path) -> None:
    async def fail(_args: dict[str, object], _ctx: ToolContext) -> dict[str, object]:
        raise ToolError("safe tool message")

    async def _run() -> None:
        registry = default_registry()
        echo = registry.get("demo.echo")
        assert echo is not None
        echo.run = fail
        provider = SequenceProvider(
            [
                AgentTurn(tool_calls=[ToolCall(name="demo.echo", args={"text": "x"})]),
                AgentTurn(text="failed"),
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
            run_id="run-tool-error",
            policy_decide=decide,
        )
        result = emitter.of_type("tool_result")[0]
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert result.error == "safe tool message"

    asyncio.run(_run())


def test_loop_passes_tool_context(tmp_path: Path) -> None:
    seen: list[ToolContext] = []

    async def capture(_args: dict[str, object], ctx: ToolContext) -> dict[str, object]:
        seen.append(ctx)
        return {"echo": "ok"}

    async def _run() -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        registry = default_registry()
        echo = registry.get("demo.echo")
        assert echo is not None
        echo.run = capture
        provider = SequenceProvider(
            [
                AgentTurn(tool_calls=[ToolCall(name="demo.echo", args={"text": "x"})]),
                AgentTurn(text="ok"),
            ]
        )
        await run_agent(
            "hello",
            provider=provider,
            registry=registry,
            emit=RecordingEmitter().emit,
            wait_approval=wait_true,
            audit=make_audit(tmp_path),
            run_id="run-ctx",
            policy_decide=decide,
            workspace=workspace,
        )
        assert len(seen) == 1
        assert seen[0].run_id == "run-ctx"
        assert seen[0].workspace == workspace

    asyncio.run(_run())


def _no_abs(workspace: Path, blob: str) -> None:
    resolved = str(workspace.resolve())
    assert resolved not in blob
    assert resolved.replace("\\", "/") not in blob


def _events_blob(emitter: RecordingEmitter) -> str:
    return json.dumps([event.model_dump(mode="json") for event in emitter.events], default=str)


def test_loop_t2_fs_invalid_skips_approval(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    async def _run() -> None:
        for tool, bad_args in (
            ("fs.delete", {"path": ".."}),
            ("fs.overwrite_text", {"path": "..", "content": "x"}),
        ):
            provider = SequenceProvider(
                [
                    AgentTurn(tool_calls=[ToolCall(name=tool, args=bad_args)]),
                    AgentTurn(text="done"),
                ]
            )
            emitter = RecordingEmitter()
            await run_agent(
                "hello",
                provider=provider,
                registry=default_registry(),
                emit=emitter.emit,
                wait_approval=wait_true,
                audit=make_audit(tmp_path / tool),
                run_id=f"run-{tool}",
                policy_decide=decide,
                workspace=workspace,
            )
            assert "approval_required" not in emitter.types()
            result = emitter.of_type("tool_result")[0]
            assert isinstance(result, ToolResult)
            assert result.ok is False
            _no_abs(workspace, _events_blob(emitter))
            _no_abs(workspace, result.error or "")

    asyncio.run(_run())


def test_loop_t2_fs_approval_reject_timeout_approve(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "target.txt").write_text("keep", encoding="utf-8")
    (workspace / "notes.md").write_text("old", encoding="utf-8")

    async def _check(
        tool: str,
        args: dict[str, object],
        waiter: object,
        timeout: float,
        should_change: bool,
        expect_ok: bool,
    ) -> RecordingEmitter:
        provider = SequenceProvider(
            [
                AgentTurn(tool_calls=[ToolCall(name=tool, args=args)]),
                AgentTurn(text="done"),
            ]
        )
        emitter = RecordingEmitter()
        registry = default_registry()
        described = await registry.describe(
            tool, args, ToolContext(workspace=workspace, run_id="x")
        )
        await run_agent(
            "hello",
            provider=provider,
            registry=default_registry(),
            emit=emitter.emit,
            wait_approval=waiter,  # type: ignore[arg-type]
            audit=make_audit(tmp_path / f"{tool}-{timeout}"),
            run_id="run-t2",
            policy_decide=decide,
            workspace=workspace,
            approval_timeout_s=timeout,
        )
        required = emitter.of_type("approval_required")
        assert required
        card = required[0]
        assert isinstance(card, ApprovalRequired)
        assert described is not None
        assert card.summary == described
        assert "\n" not in card.summary
        _no_abs(workspace, card.summary)
        result = emitter.of_type("tool_result")[0]
        assert isinstance(result, ToolResult)
        assert result.ok is expect_ok
        _no_abs(workspace, _events_blob(emitter))
        del should_change
        return emitter

    async def _run() -> None:
        await _check("fs.delete", {"path": "target.txt"}, wait_false, 60, False, False)
        assert (workspace / "target.txt").read_text(encoding="utf-8") == "keep"
        await _check("fs.delete", {"path": "target.txt"}, wait_forever, 0.05, False, False)
        assert (workspace / "target.txt").read_text(encoding="utf-8") == "keep"
        await _check("fs.delete", {"path": "target.txt"}, wait_true, 60, True, True)
        assert not (workspace / "target.txt").exists()
        await _check(
            "fs.overwrite_text",
            {"path": "notes.md", "content": "new"},
            wait_false,
            60,
            False,
            False,
        )
        assert (workspace / "notes.md").read_text(encoding="utf-8") == "old"
        await _check(
            "fs.overwrite_text",
            {"path": "notes.md", "content": "new"},
            wait_forever,
            0.05,
            False,
            False,
        )
        assert (workspace / "notes.md").read_text(encoding="utf-8") == "old"
        await _check(
            "fs.overwrite_text",
            {"path": "notes.md", "content": "new"},
            wait_true,
            60,
            True,
            True,
        )
        assert (workspace / "notes.md").read_text(encoding="utf-8") == "new"

    asyncio.run(_run())


def test_loop_t2_toctou_file_vanishes(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = workspace / "gone.txt"
    target.write_text("x", encoding="utf-8")

    async def waiter(_call_id: str) -> bool:
        target.unlink()
        return True

    async def _run() -> None:
        provider = SequenceProvider(
            [
                AgentTurn(tool_calls=[ToolCall(name="fs.delete", args={"path": "gone.txt"})]),
                AgentTurn(text="done"),
            ]
        )
        emitter = RecordingEmitter()
        await run_agent(
            "hello",
            provider=provider,
            registry=default_registry(),
            emit=emitter.emit,
            wait_approval=waiter,
            audit=make_audit(tmp_path),
            run_id="run-toctou",
            policy_decide=decide,
            workspace=workspace,
        )
        assert "approval_required" in emitter.types()
        result = emitter.of_type("tool_result")[0]
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert result.error == "no longer exists"
        _no_abs(workspace, result.error or "")
        _no_abs(workspace, _events_blob(emitter))

    asyncio.run(_run())


def test_loop_describe_internal_error_skips_approval(tmp_path: Path) -> None:
    async def boom(_args: dict[str, object], _ctx: ToolContext) -> str:
        raise RuntimeError("describe crashed")

    async def _run() -> None:
        registry = default_registry()
        tool = registry.get("demo.dangerous")
        assert tool is not None
        tool.describe = boom
        provider = SequenceProvider(
            [
                AgentTurn(
                    tool_calls=[ToolCall(name="demo.dangerous", args={"action": "wipe"})]
                ),
                AgentTurn(text="done"),
            ]
        )
        emitter = RecordingEmitter()
        await run_agent(
            "danger",
            provider=provider,
            registry=registry,
            emit=emitter.emit,
            wait_approval=wait_true,
            audit=make_audit(tmp_path),
            run_id="run-desc",
            policy_decide=decide,
        )
        assert "approval_required" not in emitter.types()
        result = emitter.of_type("tool_result")[0]
        assert isinstance(result, ToolResult)
        assert result.error == "internal tool error"
        assert "Traceback" not in (result.error or "")

    asyncio.run(_run())


def test_loop_t2_audit_has_summary(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "x.md").write_text("a", encoding="utf-8")
    audit = make_audit(tmp_path)

    async def _run() -> None:
        provider = SequenceProvider(
            [
                AgentTurn(
                    tool_calls=[
                        ToolCall(name="fs.overwrite_text", args={"path": "x.md", "content": "b"})
                    ]
                ),
                AgentTurn(text="done"),
            ]
        )
        emitter = RecordingEmitter()
        await run_agent(
            "hello",
            provider=provider,
            registry=default_registry(),
            emit=emitter.emit,
            wait_approval=wait_true,
            audit=audit,
            run_id="run-aud",
            policy_decide=decide,
            workspace=workspace,
        )
        line = json.loads(audit.path.read_text(encoding="utf-8").splitlines()[0])
        assert line["approved"] is True
        assert line["decision"] == "needs_approval"
        assert isinstance(line["summary"], str)
        assert "x.md" in line["summary"]
        blob = json.dumps(line)
        _no_abs(workspace, blob)
        _no_abs(workspace, _events_blob(emitter))

    asyncio.run(_run())


def test_t2_specs_still_require_approval() -> None:
    registry = default_registry()
    delete = registry.get("fs.delete")
    overwrite = registry.get("fs.overwrite_text")
    move = registry.get("fs.move")
    assert delete is not None and overwrite is not None and move is not None
    assert delete.spec.tier is Tier.T2
    assert overwrite.spec.tier is Tier.T2
    assert move.spec.tier is Tier.T1
    assert decide(delete.spec, {"path": "a"}).action == "needs_approval"
    assert decide(overwrite.spec, {"path": "a", "content": "b"}).action == "needs_approval"
    assert decide(move.spec, {"source": "a", "destination": "b"}).action == "allow"


def test_loop_history_across_providers_hides_provider_state(tmp_path: Path) -> None:
    class First:
        def __init__(self) -> None:
            self.n = 0

        async def step(
            self, messages: list[dict[str, object]], tools: list[dict[str, object]]
        ) -> AgentTurn:
            del messages, tools
            self.n += 1
            if self.n == 1:
                return AgentTurn(
                    tool_calls=[
                        ToolCall(
                            name="demo.echo",
                            args={"text": "hi"},
                            call_id="c1",
                            provider_state={"thought_signature": "secret-sig"},
                        )
                    ]
                )
            raise LLMUnavailable("primary down")

    class Second:
        def __init__(self) -> None:
            self.seen: list[list[dict[str, object]]] = []

        async def step(
            self, messages: list[dict[str, object]], tools: list[dict[str, object]]
        ) -> AgentTurn:
            del tools
            self.seen.append([dict(item) for item in messages])
            return AgentTurn(text="from-secondary")

    async def _run() -> None:
        first = First()
        second = Second()
        audit = make_audit(tmp_path)
        emitter = RecordingEmitter()
        await run_agent(
            "hello",
            provider=ChainProvider([("groq", first), ("gemini", second)], clock=lambda: 0.0),
            registry=default_registry(),
            emit=emitter.emit,
            wait_approval=wait_true,
            audit=audit,
            run_id="run-chain",
            policy_decide=decide,
        )
        assert second.seen
        assistant = next(item for item in second.seen[0] if item.get("role") == "assistant")
        calls = assistant["tool_calls"]
        assert isinstance(calls, list)
        assert calls[0]["provider_state"] == {"thought_signature": "secret-sig"}  # type: ignore[index]
        blob = _events_blob(emitter) + audit.path.read_text(encoding="utf-8")
        assert "provider_state" not in blob
        assert "secret-sig" not in blob
        finished = emitter.of_type("run_finished")[0]
        assert finished.status == "done"  # type: ignore[union-attr]

    asyncio.run(_run())


