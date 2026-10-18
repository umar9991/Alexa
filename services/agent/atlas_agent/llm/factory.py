from __future__ import annotations

import logging

from atlas_agent.config import Settings
from atlas_agent.llm.base import AgentTurn, LLMProvider
from atlas_agent.llm.chain import ChainProvider
from atlas_agent.llm.gemini import GeminiProvider
from atlas_agent.llm.groq import GroqProvider
from atlas_agent.llm.scripted import ScriptedProvider

logger = logging.getLogger("atlas_agent")

SCRIPTED_MODEL_LABEL = "test-only"
VALID_PROVIDERS_HELP = (
    "Valid values: groq, gemini, or a comma-separated chain such as groq,gemini "
    "(fallback order). scripted is for tests only and must be set explicitly."
)


class ProviderConfigError(ValueError):
    pass


def _provider_names(settings: Settings) -> list[str]:
    names = [part.strip().lower() for part in settings.llm_provider.split(",") if part.strip()]
    if not names:
        raise ProviderConfigError(f"ATLAS_LLM_PROVIDER is not set. {VALID_PROVIDERS_HELP}")
    return names


def _model_label(name: str, settings: Settings) -> str:
    if name == "groq":
        return settings.groq_model
    if name == "gemini":
        return settings.gemini_model
    if name == "scripted":
        return SCRIPTED_MODEL_LABEL
    raise ProviderConfigError(f"unknown LLM provider: {name}. {VALID_PROVIDERS_HELP}")


def describe_provider_chain(settings: Settings) -> list[str]:
    """`name:model` entries in fallback order. Never keys or URLs."""
    return [f"{name}:{_model_label(name, settings)}" for name in _provider_names(settings)]


class DebugLoggingProvider:
    """Opt-in (ATLAS_LLM_DEBUG=1): one INFO line per LLM step, metadata only.

    Logs provider name, model, finish reason, whether text was present and the
    tool-call names. Never arguments, message content or keys.
    """

    def __init__(self, inner: LLMProvider, *, name: str, model: str) -> None:
        self._inner = inner
        self._name = name
        self._model = model

    async def step(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> AgentTurn:
        turn = await self._inner.step(messages, tools)
        finish = getattr(turn, "finish_reason", None)
        logger.info(
            "llm step provider=%s model=%s finish_reason=%s text_present=%s tool_calls=%s",
            self._name,
            self._model,
            finish if isinstance(finish, str) else "unreported",
            bool(turn.text and turn.text.strip()),
            [call.name for call in turn.tool_calls],
        )
        return turn


def build_provider(settings: Settings) -> LLMProvider:
    names = _provider_names(settings)
    built: list[tuple[str, LLMProvider]] = []
    for name in names:
        provider = _one_provider(name, settings)
        if settings.llm_debug:
            provider = DebugLoggingProvider(
                provider, name=name, model=_model_label(name, settings)
            )
        built.append((name, provider))
    if len(built) == 1:
        return built[0][1]
    return ChainProvider(built)


def _one_provider(name: str, settings: Settings) -> LLMProvider:
    if name == "scripted":
        return ScriptedProvider()
    if name == "groq":
        return GroqProvider(settings)
    if name == "gemini":
        return GeminiProvider(settings)
    raise ProviderConfigError(f"unknown LLM provider: {name}. {VALID_PROVIDERS_HELP}")
