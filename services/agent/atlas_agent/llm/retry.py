from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from atlas_agent.llm.base import LLMRateLimited, LLMUnavailable

MAX_ATTEMPTS = 3
BACKOFF_S = (1.0, 2.0, 4.0)
MAX_RETRY_AFTER_S = 20.0
LONG_RETRY_AFTER_S = 60.0


async def with_retry[T](
    operation: Callable[[], Awaitable[T]],
    *,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> T:
    sleeper = sleep if sleep is not None else asyncio.sleep
    last_error: LLMRateLimited | LLMUnavailable | None = None
    for attempt in range(max_attempts):
        try:
            return await operation()
        except (LLMRateLimited, LLMUnavailable) as exc:
            last_error = exc
            if attempt >= max_attempts - 1:
                raise
            if (
                isinstance(exc, LLMRateLimited)
                and exc.retry_after_s is not None
                and exc.retry_after_s > LONG_RETRY_AFTER_S
            ):
                raise
            await sleeper(_delay_s(exc, attempt))
    assert last_error is not None
    raise last_error


def _delay_s(exc: LLMRateLimited | LLMUnavailable, attempt: int) -> float:
    if isinstance(exc, LLMRateLimited) and exc.retry_after_s is not None:
        return min(exc.retry_after_s, MAX_RETRY_AFTER_S)
    return BACKOFF_S[attempt]
