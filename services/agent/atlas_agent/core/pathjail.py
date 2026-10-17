from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

from atlas_agent.tools.context import ToolError

MAX_REL_LEN = 200
MAX_ABS_LEN = 240
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_FORBIDDEN_CHARS = set('<>:"|?*')
_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


def resolve_in_workspace(
    workspace: Path,
    rel: str,
    *,
    allow_root: bool = False,
    forbid_trash: bool = False,
) -> Path:
    normalized = _normalize_relative(rel, allow_root=allow_root)
    root = workspace.resolve(strict=False)
    segments = _segments(normalized)
    _reject_reparse_under_root(root, segments)
    candidate = (root / Path(*segments)).resolve(strict=False)
    if not _is_inside(root, candidate):
        raise ToolError("path is outside the workspace")
    if len(str(candidate)) > MAX_ABS_LEN:
        raise ToolError("path is too long")
    display = _relative_display(root, candidate)
    if forbid_trash and is_trash_display(display):
        raise ToolError("writes into _trash are not allowed")
    return candidate


def display_path(workspace: Path, resolved: Path) -> str:
    root = workspace.resolve(strict=False)
    final = resolved.resolve(strict=False)
    if not _is_inside(root, final):
        raise ToolError("path is outside the workspace")
    return _relative_display(root, final)


def _normalize_relative(rel: str, *, allow_root: bool) -> str:
    if not isinstance(rel, str):
        raise ToolError("path must be a string")
    if _has_hidden_or_control(rel):
        raise ToolError("path contains invalid characters")
    normalized = rel.replace("\\", "/")
    if len(normalized) > MAX_REL_LEN:
        raise ToolError("path is too long")
    if _is_absolute_form(normalized):
        raise ToolError("path must be relative")
    stripped = normalized.rstrip("/")
    if stripped in ("", "."):
        if allow_root:
            return "."
        if stripped == "":
            raise ToolError("path is empty")
        raise ToolError("path must not be '.'")
    segments = _segments(stripped)
    _validate_segments(segments)
    return "/".join(segments)


def _is_absolute_form(normalized: str) -> bool:
    if normalized.startswith(("~", "/")):
        return True
    return bool(_DRIVE_RE.match(normalized))


def _segments(path: str) -> list[str]:
    if path == ".":
        return []
    return [part for part in path.split("/") if part != "."]


def _validate_segments(segments: list[str]) -> None:
    if not segments:
        return
    for segment in segments:
        if segment == "":
            raise ToolError("path contains an empty segment")
        if segment == "..":
            raise ToolError("path must not contain '..'")
        if any(ch in _FORBIDDEN_CHARS for ch in segment):
            raise ToolError("path contains invalid characters")
        if segment.endswith((" ", ".")):
            raise ToolError("path has an invalid segment")
        if _is_reserved_segment(segment):
            raise ToolError("path uses a reserved name")


def _is_reserved_segment(segment: str) -> bool:
    stem = segment.split(".", 1)[0]
    return stem.upper() in _RESERVED


def _is_inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _relative_display(root: Path, candidate: Path) -> str:
    rel = candidate.relative_to(root).as_posix()
    if rel in ("", "."):
        return "."
    return rel


def is_reparse_point(path: Path) -> bool:
    return os.path.islink(path) or os.path.isjunction(path)


def _reject_reparse_under_root(root: Path, segments: list[str]) -> None:
    current = root
    for segment in segments:
        current = current / segment
        try:
            present = current.exists()
        except OSError:
            present = False
        if not present:
            continue
        if is_reparse_point(current):
            raise ToolError("links are not supported inside the workspace")


def is_trash_display(display: str) -> bool:
    first = display.split("/", 1)[0]
    return first.casefold() == "_trash"


def _has_hidden_or_control(rel: str) -> bool:
    for ch in rel:
        if ord(ch) < 32 or ord(ch) == 127:
            return True
        if unicodedata.category(ch) in {"Cc", "Cf"}:
            return True
    return False
