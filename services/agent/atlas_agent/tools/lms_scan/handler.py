from __future__ import annotations

from atlas_agent.browser import lms
from atlas_agent.config import ConfigError, load_lms_settings
from atlas_agent.tools.context import ToolContext, ToolError


async def run(args: dict[str, object], ctx: ToolContext) -> dict[str, object]:
    del args, ctx
    try:
        settings = load_lms_settings()
    except ConfigError as exc:
        raise ToolError(str(exc)) from None
    return await lms.run_scan(settings)
