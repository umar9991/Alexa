from __future__ import annotations

import logging
import time
from collections.abc import Callable

from atlas_agent.llm.base import (
    AgentTurn,
    LLMAuthError,
    LLMBadRequest,
    LLMError,
    LLMProvider,
    LLMRateLimited,
    LLMUnavailable,
)

logger = logging.getLogger("atlas_agent")
MAX_COOLDOWN_S = 120.0


class ChainProvider:
    def __init__(
        self,
        providers: list[tuple[str, LLMProvider]],
        *,
        cooldown_s: float = 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._providers = providers
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._until: dict[str, float] = {}
        self._last: dict[str, LLMError] = {}

    async def step(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> AgentTurn:
        now = self._clock()
        failures: list[str] = []
        for name, provider in self._providers:
            until = self._until.get(name, 0.0)
            if now < until:
                failures.append(f"{name}: {_label(self._last.get(name))}")
                continue
            try:
                return await provider.step(messages, tools)
            except (LLMAuthError, LLMBadRequest):
                raise
            except (LLMRateLimited, LLMUnavailable) as exc:
                names = [item[0] for item in self._providers]
                idx = names.index(name)
                nxt = names[idx + 1] if idx + 1 < len(names) else "none"
                logger.warning(
                    "llm fallback: %s failed with %s; trying %s",
                    name,
                    type(exc).__name__,
                    nxt,
                )
                self._last[name] = exc
                delay = self._cooldown_s
                if isinstance(exc, LLMRateLimited) and exc.retry_after_s is not None:
                    delay = min(exc.retry_after_s, MAX_COOLDOWN_S)
                self._until[name] = now + delay
                failures.append(f"{name}: {_label(exc)}")
        message = "; ".join(failures) if failures else "all LLM providers unavailable"
        raise LLMUnavailable(message)


def _label(exc: LLMError | None) -> str:
    if isinstance(exc, LLMRateLimited):
        return "rate limited"
    if isinstance(exc, LLMUnavailable):
        return "unavailable"
    if isinstance(exc, LLMAuthError):
        return "auth"
    if isinstance(exc, LLMBadRequest):
        return "bad request"
    if exc is None:
        return "cooldown"
    return type(exc).__name__
