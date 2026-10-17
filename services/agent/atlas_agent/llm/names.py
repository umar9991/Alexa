from __future__ import annotations

TOOL_CONTENT_LIMIT = 8000


def encode_tool_name(name: str) -> str:
    return name.replace(".", "__")


def decode_tool_name(name: str) -> str:
    return name.replace("__", ".")


def truncate_tool_content(content: str) -> str:
    if len(content) <= TOOL_CONTENT_LIMIT:
        return content
    return content[:TOOL_CONTENT_LIMIT] + "\n[truncated]"
