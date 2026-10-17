from __future__ import annotations

from atlas_agent.config import Settings
from atlas_agent.llm.base import LLMProvider
from atlas_agent.llm.chain import ChainProvider
from atlas_agent.llm.gemini import GeminiProvider
from atlas_agent.llm.groq import GroqProvider
from atlas_agent.llm.scripted import ScriptedProvider


class ProviderConfigError(ValueError):
    pass


def build_provider(settings: Settings) -> LLMProvider:
    names = [part.strip().lower() for part in settings.llm_provider.split(",") if part.strip()]
    if not names:
        return ScriptedProvider()
    built: list[tuple[str, LLMProvider]] = []
    for name in names:
        built.append((name, _one_provider(name, settings)))
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
    raise ProviderConfigError(f"unknown LLM provider: {name}")
