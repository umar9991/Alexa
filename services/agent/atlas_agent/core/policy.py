from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from atlas_agent.tools.registry import ToolSpec


class Tier(StrEnum):
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"


class Decision(BaseModel):
    action: Literal["allow", "needs_approval", "deny"]
    reason: str


def decide(spec: ToolSpec | None, args: dict[str, object]) -> Decision:
    del args
    if spec is None:
        return Decision(action="deny", reason="unknown tool")
    if spec.tier is Tier.T2:
        return Decision(action="needs_approval", reason="T2 requires user approval")
    if spec.tier in (Tier.T0, Tier.T1):
        return Decision(action="allow", reason=f"{spec.tier.value} is allowed")
    return Decision(action="deny", reason="unknown tool")
