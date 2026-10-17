from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("atlas_agent")
_MESSAGE_LIMIT = 500
_KEY_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"gsk_[0-9A-Za-z]+"),
)


def redact_secret_text(text: str) -> str:
    redacted = text
    for pattern in _KEY_PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    return redacted[:_MESSAGE_LIMIT]


def log_provider_api_error(provider: str, exc: Exception) -> None:
    code, api_status, message = extract_api_error_fields(exc)
    if not message:
        message = str(exc)
    safe = redact_secret_text(message)
    logger.warning(
        "llm %s error: class=%s http_status=%s api_status=%s message=%s",
        provider,
        type(exc).__name__,
        code if code is not None else "none",
        api_status or "none",
        safe,
    )


def extract_api_error_fields(exc: Exception) -> tuple[int | None, str, str]:
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(exc, "status_code", None)
    if code is not None and not isinstance(code, int):
        try:
            code = int(code)
        except (TypeError, ValueError):
            code = None
    api_status = _as_text(getattr(exc, "status", None))
    message = _as_text(getattr(exc, "message", None))
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        nested = body.get("error")
        if isinstance(nested, dict):
            if not api_status:
                api_status = _as_text(nested.get("status") or nested.get("type") or nested.get("code"))
            if not message:
                message = _as_text(nested.get("message"))
    if not message:
        details = getattr(exc, "details", None)
        if isinstance(details, dict):
            nested = details.get("error")
            if isinstance(nested, dict):
                if not api_status:
                    api_status = _as_text(nested.get("status") or nested.get("type"))
                message = _as_text(nested.get("message"))
            elif not api_status:
                api_status = _as_text(details.get("status"))
                message = _as_text(details.get("message"))
    return code, api_status, message


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
