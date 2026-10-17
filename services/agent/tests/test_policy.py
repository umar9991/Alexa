from atlas_agent.core.policy import Decision, Tier, decide
from atlas_agent.tools.registry import ToolSpec


def _spec(tier: Tier) -> ToolSpec:
    return ToolSpec(
        name="demo.tool",
        description="test",
        tier=tier,
        input_schema={"type": "object"},
    )


def test_t0_is_allowed() -> None:
    result = decide(_spec(Tier.T0), {"text": "hi"})
    assert result == Decision(action="allow", reason="T0 is allowed")


def test_t2_needs_approval() -> None:
    result = decide(_spec(Tier.T2), {"action": "wipe"})
    assert result.action == "needs_approval"


def test_unknown_tool_is_denied() -> None:
    result = decide(None, {})
    assert result.action == "deny"
    assert "unknown" in result.reason
