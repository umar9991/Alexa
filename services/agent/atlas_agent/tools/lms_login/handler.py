from __future__ import annotations

from atlas_agent.browser import lms
from atlas_agent.config import ConfigError, load_lms_settings
from atlas_agent.tools.context import ToolContext, ToolError

DEFAULT_TIMEOUT_S = 180
MAX_TIMEOUT_S = 300


async def run(args: dict[str, object], ctx: ToolContext) -> dict[str, object]:
    del ctx
    raw = args.get("timeout_s", DEFAULT_TIMEOUT_S)
    if not isinstance(raw, int) or isinstance(raw, bool) or not 1 <= raw <= MAX_TIMEOUT_S:
        raise ToolError(f"timeout_s must be an integer from 1 to {MAX_TIMEOUT_S}")
    try:
        settings = load_lms_settings()
    except ConfigError as exc:
        raise ToolError(str(exc)) from None
    return await lms.run_login(settings, raw)
