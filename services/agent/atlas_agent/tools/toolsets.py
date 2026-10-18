from __future__ import annotations

import re

# A toolset is the part of a tool name before the first dot (fs.read_text -> fs).
ENABLE_TOOL_NAME = "tools.enable"
MAX_ENABLE_CALLS = 6
MAX_ENABLE_TOOLSETS = 4
DESCRIPTION_LIMIT = 120

TOOLSET_DESCRIPTIONS: dict[str, str] = {
    "fs": "files inside the workspace",
    "docx": "build Word assignment files",
    "lms": "read the university LMS",
    "demo": "test tools",
}

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def toolset_of(tool_name: str) -> str:
    return tool_name.split(".", 1)[0]


def toolset_description(name: str) -> str:
    """One-line description of a toolset. Unknown toolsets fall back to their name."""
    return TOOLSET_DESCRIPTIONS.get(name, name)


def first_sentence(text: str, limit: int = DESCRIPTION_LIMIT) -> str:
    """First sentence of a tool description, at most `limit` characters."""
    flat = " ".join(text.split())
    match = _SENTENCE_END.search(flat)
    sentence = flat[: match.start() + 1] if match else flat
    if len(sentence) > limit:
        return sentence[: limit - 3].rstrip() + "..."
    return sentence
