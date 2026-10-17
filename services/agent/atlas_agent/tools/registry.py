from __future__ import annotations

import importlib.util
import json
import logging
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from atlas_agent.core.policy import Tier
from atlas_agent.tools.context import ToolContext, ToolError

logger = logging.getLogger("atlas_agent")

DescribeFn = Callable[[dict[str, object], ToolContext], Awaitable[str]]
RunFn = Callable[[dict[str, object], ToolContext], Awaitable[dict[str, object]]]
SUMMARY_LIMIT = 300


class ToolSpec(BaseModel):
    name: str
    description: str
    tier: Tier
    input_schema: dict[str, Any]


class ToolRegistryError(ValueError):
    pass


@dataclass
class LoadedTool:
    spec: ToolSpec
    run: RunFn
    describe: DescribeFn | None = None


class ToolRegistry:
    def __init__(self, tools: dict[str, LoadedTool]) -> None:
        self._tools = tools

    @classmethod
    def load(cls, root: Path | None = None) -> ToolRegistry:
        tools_root = root if root is not None else Path(__file__).resolve().parent
        loaded: dict[str, LoadedTool] = {}
        for child in sorted(tools_root.iterdir()):
            if not child.is_dir():
                continue
            spec_path = child / "spec.json"
            if not spec_path.exists():
                continue
            spec = _read_spec(spec_path)
            if "__" in spec.name:
                raise ToolRegistryError(
                    f"invalid tool name {spec.name}: names must not contain '__'"
                )
            if spec.name in loaded:
                raise ToolRegistryError(f"duplicate tool name: {spec.name}")
            handler_path = child / "handler.py"
            if not handler_path.exists():
                raise ToolRegistryError(f"missing handler for tool {spec.name} in {child}")
            run, describe = _load_handler(handler_path, spec.name)
            loaded[spec.name] = LoadedTool(spec=spec, run=run, describe=describe)
        return cls(loaded)

    def get(self, name: str) -> LoadedTool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict[str, object]]:
        return [tool.spec.model_dump(mode="json") for tool in self._tools.values()]

    def validate_args(self, name: str, args: dict[str, object]) -> None:
        tool = self.get(name)
        if tool is None:
            raise ToolRegistryError(f"unknown tool: {name}")
        validate(instance=args, schema=tool.spec.input_schema)

    async def describe(
        self, name: str, args: dict[str, object], ctx: ToolContext
    ) -> str | None:
        tool = self.get(name)
        if tool is None or tool.describe is None:
            return None
        try:
            raw = await tool.describe(args, ctx)
        except ToolError:
            raise
        except Exception:
            logger.exception("internal tool error in %s describe", name)
            raise ToolError("internal tool error") from None
        if not isinstance(raw, str):
            raw = str(raw)
        return sanitize_summary(raw)

    async def run(
        self, name: str, args: dict[str, object], ctx: ToolContext
    ) -> dict[str, object]:
        tool = self.get(name)
        if tool is None:
            raise ToolRegistryError(f"unknown tool: {name}")
        self.validate_args(name, args)
        try:
            return await tool.run(args, ctx)
        except ToolError:
            raise
        except Exception:
            logger.exception("internal tool error in %s", name)
            raise ToolError("internal tool error") from None


def _read_spec(spec_path: Path) -> ToolSpec:
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolRegistryError(f"invalid spec in {spec_path}: {exc}") from exc
    try:
        return ToolSpec.model_validate(raw)
    except PydanticValidationError as exc:
        raise ToolRegistryError(f"invalid spec in {spec_path}: {exc}") from exc


def _load_handler(
    handler_path: Path, tool_name: str
) -> tuple[RunFn, DescribeFn | None]:
    module_name = f"atlas_agent.tools._loaded_{handler_path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, handler_path)
    if spec is None or spec.loader is None:
        raise ToolRegistryError(f"missing handler for tool {tool_name} in {handler_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if run is None or not callable(run):
        raise ToolRegistryError(f"missing handler for tool {tool_name} in {handler_path}")
    describe_attr = getattr(module, "describe", None)
    describe: DescribeFn | None = describe_attr if callable(describe_attr) else None
    return run, describe


def sanitize_summary(text: str) -> str:
    cleaned: list[str] = []
    for ch in text.replace("\r", " ").replace("\n", " "):
        if unicodedata.category(ch) in {"Cc", "Cf"}:
            continue
        cleaned.append(ch)
    out = "".join(cleaned).strip()
    if len(out) > SUMMARY_LIMIT:
        return out[:SUMMARY_LIMIT]
    return out


def format_invalid_args(error: ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    if path:
        return f"invalid args: {path}: {error.message}"
    return f"invalid args: {error.message}"
