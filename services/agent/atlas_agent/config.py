from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger("atlas_agent")

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    groq_api_key: str
    groq_model: str
    llm_timeout_s: float
    llm_temperature: float
    llm_max_tokens: int
    workspace: Path | None = None


def load_settings() -> Settings:
    load_dotenv(find_dotenv(usecwd=True))
    model = os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL
    return Settings(
        llm_provider=os.environ.get("ATLAS_LLM_PROVIDER", "scripted").strip() or "scripted",
        groq_api_key=os.environ.get("GROQ_API_KEY", "").strip(),
        groq_model=model,
        llm_timeout_s=_float_env("ATLAS_LLM_TIMEOUT_S", DEFAULT_TIMEOUT_S),
        llm_temperature=_float_env("ATLAS_LLM_TEMPERATURE", DEFAULT_TEMPERATURE),
        llm_max_tokens=int(_float_env("ATLAS_LLM_MAX_TOKENS", float(DEFAULT_MAX_TOKENS))),
        workspace=_workspace_from_env(),
    )


def ensure_workspace(workspace: Path | None) -> Path | None:
    if workspace is None:
        return None
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "_trash").mkdir(parents=True, exist_ok=True)
    resolved = workspace.resolve(strict=False)
    logger.info("workspace ready at %s", resolved)
    return resolved


def _workspace_from_env() -> Path | None:
    raw = os.environ.get("ATLAS_WORKSPACE", "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        return None
    return path


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
