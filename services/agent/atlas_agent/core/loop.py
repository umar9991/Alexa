from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from jsonschema import ValidationError, validate

from atlas_agent.config import always_on_toolsets_from_env, llm_debug_enabled
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
from atlas_agent.core.prompts import FINAL_ANSWER_REQUEST, build_system_prompt
from atlas_agent.llm.base import (
    LLMError,
    LLMProvider,
    LLMRateLimited,
    ToolCall,
)
from atlas_agent.llm.names import truncate_tool_content
from atlas_agent.tools.context import ToolContext, ToolError
from atlas_agent.tools.registry import ToolRegistry, ToolSpec, format_invalid_args
from atlas_agent.tools.toolsets import (
    ENABLE_TOOL_NAME,
    MAX_ENABLE_CALLS,
    MAX_ENABLE_TOOLSETS,
    first_sentence,
    toolset_of,
)

logger = logging.getLogger("atlas_agent")

EmitFn = Callable[[AgentEvent], Awaitable[None]]
ApprovalWaiter = Callable[[str], Awaitable[bool]]
DecideFn = Callable[[ToolSpec | None, dict[str, object]], Decision]

TOOLSET_NOT_ENABLED = "toolset not enabled; call tools.enable first"

# Built-in meta tool. It has no tool folder: the loop handles it, but it still goes through
# schema validation, policy, audit log and trace events like every other call.
ENABLE_SPEC = ToolSpec(
    name=ENABLE_TOOL_NAME,
    description=(
        "Enable toolsets so their tools become available. "
        "Pass 1 to 4 toolset names from the toolset list."
    ),
    tier=Tier.T0,
    input_schema={
        "type": "object",
        "properties": {
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": MAX_ENABLE_TOOLSETS,
            }
        },
        "required": ["toolsets"],
        "additionalProperties": False,
    },
)


@dataclass
class _RunState:
    enabled: set[str]
    enable_calls: int = 0


def approx_tokens(tools: list[dict[str, object]]) -> int:
    """Rough token estimate (characters / 4) of a tools payload."""
    return len(json.dumps(tools, ensure_ascii=True)) // 4


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
    always_on_toolsets: Collection[str] | None = None,
) -> None:
    await emit(RunStarted(run_id=run_id, ts=utc_now(), task=task))
    available = registry.toolsets()
    messages: list[dict[str, object]] = [
        {"role": "system", "content": build_system_prompt(available)},
        {"role": "user", "content": task},
    ]
    initial = always_on_toolsets_from_env() if always_on_toolsets is None else always_on_toolsets
    state = _RunState(enabled={name for name in initial if name in available})
    tools_run: list[str] = []
    asked_for_final = False
    debug = llm_debug_enabled()

    steps_used = 0
    while steps_used < max_steps:
        steps_used += 1
        if _cancelled(cancel_event):
            await emit(_finished(run_id, "cancelled", "cancelled"))
            return
        tools = [ENABLE_SPEC.model_dump(mode="json"), *registry.specs_for(state.enabled)]
        if debug:
            logger.info("llm step tools_sent=%d approx_tokens=%d", len(tools), approx_tokens(tools))
        try:
            turn = await provider.step(messages, tools)
        except LLMError as exc:
            await emit(_finished(run_id, "failed", _llm_failure_summary(exc)))
            return
        if _cancelled(cancel_event):
            await emit(_finished(run_id, "cancelled", "cancelled"))
            return
        has_text = bool(turn.text and turn.text.strip())
        if has_text:
            await emit(AssistantMessage(run_id=run_id, ts=utc_now(), text=str(turn.text)))
        if not turn.tool_calls:
            if has_text or not tools_run:
                await emit(_finished(run_id, "done", str(turn.text) if has_text else "done"))
                return
            if not asked_for_final:
                asked_for_final = True
                logger.warning("model returned no text after tool results; asking once more")
                messages.append({"role": "user", "content": FINAL_ANSWER_REQUEST})
                continue
            neutral = _neutral_final(tools_run)
            await emit(AssistantMessage(run_id=run_id, ts=utc_now(), text=neutral))
            await emit(_finished(run_id, "done", neutral))
            return
        prepared = _with_call_ids(turn.tool_calls)
        tools_run.extend(call.name for call in prepared)
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
                state=state,
            )
            if _cancelled(cancel_event):
                await emit(_finished(run_id, "cancelled", "cancelled"))
                return
        if _only_enable_calls(prepared) and state.enable_calls <= MAX_ENABLE_CALLS:
            steps_used -= 1  # tools.enable does not consume a step (its calls are capped)

    await emit(_finished(run_id, "failed", "max steps reached"))


def _only_enable_calls(calls: list[ToolCall]) -> bool:
    return all(call.name == ENABLE_TOOL_NAME for call in calls)


def _enable_toolsets(
    registry: ToolRegistry, state: _RunState, args: dict[str, object]
) -> dict[str, object]:
    if state.enable_calls > MAX_ENABLE_CALLS:
        raise ToolError(f"{ENABLE_TOOL_NAME} limit reached ({MAX_ENABLE_CALLS} per run)")
    raw = args.get("toolsets")
    requested = [str(name) for name in raw] if isinstance(raw, list) else []
    names = list(dict.fromkeys(requested))
    available = registry.toolsets()
    unknown = [name[:40] for name in names if name not in available]
    if unknown:
        raise ToolError(
            f"unknown toolset: {', '.join(unknown)}. Known toolsets: {', '.join(available)}"
        )
    state.enabled.update(names)
    return {
        "enabled": names,
        "tools": [
            {
                "name": spec.name,
                "tier": spec.tier.value,
                "description": first_sentence(spec.description),
            }
            for spec in registry.tools_in(names)
        ],
    }


def _neutral_final(tools_run: list[str]) -> str:
    """Short fallback message. Names only; never tool output."""
    unique = list(dict.fromkeys(tools_run))
    return f"Done. Ran {', '.join(unique)}."


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
    state: _RunState,
) -> None:
    call_id = call.call_id or str(uuid.uuid4())
    is_meta = call.name == ENABLE_TOOL_NAME
    loaded = None if is_meta else registry.get(call.name)
    spec = ENABLE_SPEC if is_meta else (loaded.spec if loaded is not None else None)
    tier = spec.tier if spec is not None else Tier.T2
    if is_meta:
        state.enable_calls += 1
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
    ctx = ToolContext(workspace=workspace, run_id=run_id)
    if loaded is not None and toolset_of(call.name) not in state.enabled:
        await _complete_call(
            run_id=run_id,
            call_id=call_id,
            tool=call.name,
            args=args,
            tier=tier,
            decision="deny",
            approved=False,
            ok=False,
            output=None,
            error=TOOLSET_NOT_ENABLED,
            duration_ms=0,
            emit=emit,
            audit=audit,
            messages=messages,
        )
        return
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
    if loaded is not None or is_meta:
        try:
            if is_meta:
                validate(instance=args, schema=ENABLE_SPEC.input_schema)
            else:
                registry.validate_args(call.name, args)
        except ValidationError as exc:
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
                error=format_invalid_args(exc),
                duration_ms=0,
                emit=emit,
                audit=audit,
                messages=messages,
            )
            return
    summary = _approval_summary(call.name, args)
    if loaded is not None and loaded.describe is not None:
        try:
            described = await registry.describe(call.name, args, ctx)
        except ToolError as exc:
            await _complete_call(
                run_id=run_id,
                call_id=call_id,
                tool=call.name,
                args=args,
                tier=tier,
                decision="deny",
                approved=False,
                ok=False,
                output=None,
                error=str(exc),
                duration_ms=0,
                emit=emit,
                audit=audit,
                messages=messages,
            )
            return
        if described:
            summary = described
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
            summary=summary,
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
                summary=summary,
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
                summary=summary,
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
                summary=summary,
            )
            return

    started = time.perf_counter()
    try:
        if is_meta:
            output = _enable_toolsets(registry, state, args)
        else:
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
        summary=summary,
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
    summary: str | None = None,
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
        summary=summary,
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
    recorded: list[dict[str, object]] = []
    for call in calls:
        item: dict[str, object] = {
            "id": call.call_id,
            "name": call.name,
            "args": call.args,
        }
        if call.provider_state is not None:
            item["provider_state"] = call.provider_state
        recorded.append(item)
    return {"role": "assistant", "content": text, "tool_calls": recorded}


def _llm_failure_summary(exc: LLMError) -> str:
    if isinstance(exc, LLMRateLimited):
        if exc.retry_after_s is not None:
            seconds = max(1, int(exc.retry_after_s))
            return f"Groq rate limit reached, try again in about {seconds} seconds"
        return str(exc).strip() or "Groq rate limit reached, try again later"
    text = str(exc).strip()
    if text:
        return text
    return "LLM request failed"
