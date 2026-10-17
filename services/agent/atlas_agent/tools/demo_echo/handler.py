from atlas_agent.tools.context import ToolContext


async def run(args: dict[str, object], ctx: ToolContext) -> dict[str, object]:
    del ctx
    return {"echo": args["text"]}
