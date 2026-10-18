from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger("atlas_agent")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_REL = Path("data") / "browser-profile"

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
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
    gemini_api_key: str = ""
    gemini_model: str = ""
    groq_base_url: str | None = None
    gemini_base_url: str | None = None


def env_text(name: str) -> str | None:
    """Read an env var; empty or whitespace-only is unset (and removed)."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        os.environ.pop(name, None)
        return None
    if stripped != raw:
        os.environ[name] = stripped
    return stripped


def load_settings() -> Settings:
    load_dotenv(find_dotenv(usecwd=True))
    groq_model = env_text("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    gemini_model = env_text("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    return Settings(
        llm_provider=env_text("ATLAS_LLM_PROVIDER") or "scripted",
        groq_api_key=env_text("GROQ_API_KEY") or "",
        groq_model=groq_model,
        llm_timeout_s=_float_env("ATLAS_LLM_TIMEOUT_S", DEFAULT_TIMEOUT_S),
        llm_temperature=_float_env("ATLAS_LLM_TEMPERATURE", DEFAULT_TEMPERATURE),
        llm_max_tokens=int(_float_env("ATLAS_LLM_MAX_TOKENS", float(DEFAULT_MAX_TOKENS))),
        workspace=_workspace_from_env(),
        gemini_api_key=env_text("GEMINI_API_KEY") or "",
        gemini_model=gemini_model,
        groq_base_url=env_text("GROQ_BASE_URL"),
        gemini_base_url=env_text("GEMINI_BASE_URL"),
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
    raw = env_text("ATLAS_WORKSPACE")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        return None
    return path


def _float_env(name: str, default: float) -> float:
    raw = env_text(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class ConfigError(ValueError):
    """Safe-to-show configuration problem. Never includes secrets or paths."""


@dataclass(frozen=True)
class LmsSettings:
    base_url: str
    host: str
    base_path: str
    profile_dir: Path
    headless: bool


def load_lms_settings(repo_root: Path | None = None) -> LmsSettings:
    load_dotenv(find_dotenv(usecwd=True))
    root = (repo_root if repo_root is not None else REPO_ROOT).resolve(strict=False)
    base_url, host, base_path = _lms_base_url(env_text("LMS_BASE_URL"))
    return LmsSettings(
        base_url=base_url,
        host=host,
        base_path=base_path,
        profile_dir=_profile_dir(env_text("ATLAS_BROWSER_PROFILE"), root),
        headless=_bool_env("LMS_HEADLESS", default=True),
    )


def _lms_base_url(raw: str | None) -> tuple[str, str, str]:
    if raw is None:
        raise ConfigError("LMS_BASE_URL is not configured")
    try:
        parts = urlsplit(raw)
        port = parts.port
        host = parts.hostname
    except ValueError:
        raise ConfigError("LMS_BASE_URL must be a valid https URL") from None
    if parts.scheme != "https" or not host:
        raise ConfigError("LMS_BASE_URL must be a valid https URL")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ConfigError("LMS_BASE_URL must not contain credentials")
    if parts.query or parts.fragment or "\\" in raw or port not in (None, 443):
        raise ConfigError("LMS_BASE_URL must be a plain https URL without port or query")
    base_path = parts.path.rstrip("/")
    return f"https://{host}{base_path}", host, base_path


def _profile_dir(raw: str | None, root: Path) -> Path:
    data_root = (root / "data").resolve(strict=False)
    candidate = Path(raw) if raw else root / DEFAULT_PROFILE_REL
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    if resolved == data_root or data_root not in resolved.parents:
        raise ConfigError("ATLAS_BROWSER_PROFILE must be inside the data folder")
    return resolved


def _bool_env(name: str, *, default: bool) -> bool:
    raw = env_text(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in {"0", "false", "no", "off"}:
        return False
    if lowered in {"1", "true", "yes", "on"}:
        return True
    return default
