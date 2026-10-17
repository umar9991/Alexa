from typing import Annotated, Literal

from pydantic import BaseModel, Field

from atlas_agent.core.policy import Tier


class UserMessage(BaseModel):
    type: Literal["user_message"]
    text: str


class ApprovalMessage(BaseModel):
    type: Literal["approval"]
    call_id: str
    approved: bool


class CancelMessage(BaseModel):
    type: Literal["cancel"]
    run_id: str


ClientMessage = UserMessage | ApprovalMessage | CancelMessage


class RunStarted(BaseModel):
    type: Literal["run_started"] = "run_started"
    run_id: str
    ts: str
    task: str


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    run_id: str
    ts: str
    call_id: str
    tool: str
    args: dict[str, object]
    tier: Tier


class ApprovalRequired(BaseModel):
    type: Literal["approval_required"] = "approval_required"
    run_id: str
    ts: str
    call_id: str
    tool: str
    args: dict[str, object]
    tier: Tier
    summary: str


class ToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    run_id: str
    ts: str
    call_id: str
    ok: bool
    output: dict[str, object] | None = None
    error: str | None = None
    duration_ms: int


class AssistantMessage(BaseModel):
    type: Literal["assistant_message"] = "assistant_message"
    run_id: str
    ts: str
    text: str


class RunFinished(BaseModel):
    type: Literal["run_finished"] = "run_finished"
    run_id: str
    ts: str
    status: Literal["done", "failed", "cancelled"]
    summary: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


AgentEvent = Annotated[
    RunStarted
    | ToolCallEvent
    | ApprovalRequired
    | ToolResult
    | AssistantMessage
    | RunFinished,
    Field(discriminator="type"),
]
