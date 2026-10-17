from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from jsonschema import ValidationError

from atlas_agent.core.audit import AuditLog
from atlas_agent.core.events import (
    AgentEvent,
    ApprovalRequired,
    AssistantMessage,
    RunFinished,
    RunStarted,
    ToolCallEvent,
    ToolResult,
)
from atlas_agent.core.policy import Decision, Tier, decide
from atlas_agent.core.prompts import SYSTEM_PROMPT
from atlas_agent.llm.base import (
    LLMAuthError,
    LLMBadRequest,
    LLMError,
    LLMProvider,
    LLMRateLimited,
    LLMUnavailable,
    ToolCall,
)
from atlas_agent.llm.groq import truncate_tool_content
from atlas_agent.tools.context import ToolContext, ToolError
from atlas_agent.tools.registry import ToolRegistry, ToolSpec, format_invalid_args

logger = logging.getLogger("atlas_agent")

EmitFn = Callable[[AgentEvent], Awaitable[None]]
ApprovalWaiter = Callable[[str], Awaitable[bool]]
DecideFn = Callable[[ToolSpec | None, dict[str, object]], Decision]


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


async def run_agent(
    task: str,
    *,
    provider: LLMProvider,
    registry: ToolRegistry,
    emit: EmitFn,
    wait_approval: ApprovalWaiter,
    audit: AuditLog,
    run_id: str,
    policy_decide: DecideFn = decide,
    approval_timeout_s: float = 60,
    max_steps: int = 6,
    cancel_event: asyncio.Event | None = None,
    workspace: Path | None = None,
) -> None:
    await emit(RunStarted(run_id=run_id, ts=utc_now(), task=task))
    messages: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    tools = registry.specs()

    for _step in range(max_steps):
        if _cancelled(cancel_event):
            await emit(_finished(run_id, "cancelled", "cancelled"))
            return
        try:
            turn = await provider.step(messages, tools)
        except LLMError as exc:
            await emit(_finished(run_id, "failed", _llm_failure_summary(exc)))
            return
        if _cancelled(cancel_event):
            await emit(_finished(run_id, "cancelled", "cancelled"))
            return
        if turn.text:
            await emit(AssistantMessage(run_id=run_id, ts=utc_now(), text=turn.text))
        if not turn.tool_calls:
            summary = turn.text if turn.text else "done"
            await emit(_finished(run_id, "done", summary))
            return
        prepared = _with_call_ids(turn.tool_calls)
        messages.append(_assistant_record(turn.text, prepared))
        for call in prepared:
            if _cancelled(cancel_event):
                await emit(_finished(run_id, "cancelled", "cancelled"))
                return
            await _handle_tool_call(
                call=call,
                run_id=run_id,
                registry=registry,
                emit=emit,
                wait_approval=wait_approval,
                audit=audit,
                policy_decide=policy_decide,
                approval_timeout_s=approval_timeout_s,
                cancel_event=cancel_event,
                messages=messages,
                workspace=workspace,
            )
            if _cancelled(cancel_event):
                await emit(_finished(run_id, "cancelled", "cancelled"))
                return

    await emit(_finished(run_id, "failed", "max steps reached"))


def _cancelled(cancel_event: asyncio.Event | None) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _finished(
    run_id: str, status: Literal["done", "failed", "cancelled"], summary: str
) -> RunFinished:
    return RunFinished(run_id=run_id, ts=utc_now(), status=status, summary=summary)


async def _handle_tool_call(
    *,
    call: ToolCall,
    run_id: str,
    registry: ToolRegistry,
    emit: EmitFn,
    wait_approval: ApprovalWaiter,
    audit: AuditLog,
    policy_decide: DecideFn,
    approval_timeout_s: float,
    cancel_event: asyncio.Event | None,
    messages: list[dict[str, object]],
    workspace: Path | None,
) -> None:
    call_id = call.call_id or str(uuid.uuid4())
    loaded = registry.get(call.name)
    spec = loaded.spec if loaded is not None else None
    tier = spec.tier if spec is not None else Tier.T2
    args = call.args
    await emit(
        ToolCallEvent(
            run_id=run_id,
            ts=utc_now(),
            call_id=call_id,
            tool=call.name,
            args=args,
            tier=tier,
        )
    )
    if call.parse_error:
        await _complete_call(
            run_id=run_id,
            call_id=call_id,
            tool=call.name,
            args=args,
            tier=tier,
            decision="invalid_args",
            approved=False,
            ok=False,
            output=None,
            error=call.parse_error,
            duration_ms=0,
            emit=emit,
            audit=audit,
            messages=messages,
        )
        return
    decision = policy_decide(spec, args)
    if decision.action == "deny":
        await _complete_call(
            run_id=run_id,
            call_id=call_id,
            tool=call.name,
            args=args,
            tier=tier,
            decision=decision.action,
            approved=False,
            ok=False,
            output=None,
            error=decision.reason,
            duration_ms=0,
            emit=emit,
            audit=audit,
            messages=messages,
        )
        return

    approved = True
    timed_out = False
    if decision.action == "needs_approval":
        await emit(
            ApprovalRequired(
                run_id=run_id,
                ts=utc_now(),
                call_id=call_id,
                tool=call.name,
                args=args,
                tier=tier,
                summary=_approval_summary(call.name, args),
            )
        )
        approved, timed_out = await _await_approval(
            wait_approval, call_id, approval_timeout_s
        )
        if _cancelled(cancel_event):
            await _complete_call(
                run_id=run_id,
                call_id=call_id,
                tool=call.name,
                args=args,
                tier=tier,
                decision=decision.action,
                approved=False,
                ok=False,
                output=None,
                error="rejected by user",
                duration_ms=0,
                emit=emit,
                audit=audit,
                messages=messages,
            )
            return
        if not approved:
            error = "approval timed out" if timed_out else "rejected by user"
            await _complete_call(
                run_id=run_id,
                call_id=call_id,
                tool=call.name,
                args=args,
                tier=tier,
                decision=decision.action,
                approved=False,
                ok=False,
                output=None,
                error=error,
                duration_ms=0,
                emit=emit,
                audit=audit,
                messages=messages,
            )
            return

    started = time.perf_counter()
    ctx = ToolContext(workspace=workspace, run_id=run_id)
    try:
        output = await registry.run(call.name, args, ctx)
        error = None
        ok = True
    except ValidationError as exc:
        output = None
        error = format_invalid_args(exc)
        ok = False
    except ToolError as exc:
        output = None
        error = str(exc)
        ok = False
    except Exception:
        logger.exception("internal tool error in %s", call.name)
        output = None
        error = "internal tool error"
        ok = False
    duration_ms = int((time.perf_counter() - started) * 1000)
    await _complete_call(
        run_id=run_id,
        call_id=call_id,
        tool=call.name,
        args=args,
        tier=tier,
        decision=decision.action,
        approved=approved,
        ok=ok,
        output=output,
        error=error,
        duration_ms=duration_ms,
        emit=emit,
        audit=audit,
        messages=messages,
    )


async def _await_approval(
    wait_approval: ApprovalWaiter, call_id: str, timeout_s: float
) -> tuple[bool, bool]:
    try:
        approved = await asyncio.wait_for(wait_approval(call_id), timeout=timeout_s)
        return approved, False
    except TimeoutError:
        return False, True


def _approval_summary(tool: str, args: dict[str, object]) -> str:
    return f"{tool} {json.dumps(args, ensure_ascii=True)}"


async def _complete_call(
    *,
    run_id: str,
    call_id: str,
    tool: str,
    args: dict[str, object],
    tier: Tier,
    decision: str,
    approved: bool,
    ok: bool,
    output: dict[str, object] | None,
    error: str | None,
    duration_ms: int,
    emit: EmitFn,
    audit: AuditLog,
    messages: list[dict[str, object]],
) -> None:
    await emit(
        ToolResult(
            run_id=run_id,
            ts=utc_now(),
            call_id=call_id,
            ok=ok,
            output=output,
            error=error,
            duration_ms=duration_ms,
        )
    )
    audit.write(
        run_id=run_id,
        call_id=call_id,
        tool=tool,
        args=args,
        tier=tier.value,
        decision=decision,
        approved=approved,
        ok=ok,
        error=error,
        duration_ms=duration_ms,
    )
    payload = {"ok": ok, "output": output, "error": error}
    content = json.dumps(payload, ensure_ascii=True)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool,
            "content": truncate_tool_content(content),
        }
    )


def _with_call_ids(calls: list[ToolCall]) -> list[ToolCall]:
    prepared: list[ToolCall] = []
    for call in calls:
        call_id = call.call_id or str(uuid.uuid4())
        prepared.append(call.model_copy(update={"call_id": call_id}))
    return prepared


def _assistant_record(text: str | None, calls: list[ToolCall]) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {"id": call.call_id, "name": call.name, "args": call.args} for call in calls
        ],
    }


def _llm_failure_summary(exc: LLMError) -> str:
    if isinstance(exc, LLMRateLimited):
        if exc.retry_after_s is not None:
            seconds = max(1, int(exc.retry_after_s))
            return f"Groq rate limit reached, try again in about {seconds} seconds"
        return "Groq rate limit reached, try again later"
    if isinstance(exc, LLMAuthError):
        return "GROQ_API_KEY is missing or invalid"
    if isinstance(exc, LLMBadRequest):
        return "Groq rejected the request"
    if isinstance(exc, LLMUnavailable):
        return "LLM is temporarily unavailable"
    return "LLM request failed"
