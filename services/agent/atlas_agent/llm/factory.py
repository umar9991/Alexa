from __future__ import annotations

from atlas_agent.config import Settings
from atlas_agent.llm.base import LLMProvider
from atlas_agent.llm.groq import GroqProvider
from atlas_agent.llm.scripted import ScriptedProvider


def build_provider(settings: Settings) -> LLMProvider:
    name = settings.llm_provider.strip().lower()
    if name == "groq":
        return GroqProvider(settings)
    return ScriptedProvider()
