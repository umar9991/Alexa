from __future__ import annotations

import importlib.util
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from atlas_agent.core.policy import Tier


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
    run: Callable[[dict[str, object]], Awaitable[dict[str, object]]]


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
            if spec.name in loaded:
                raise ToolRegistryError(f"duplicate tool name: {spec.name}")
            handler_path = child / "handler.py"
            if not handler_path.exists():
                raise ToolRegistryError(f"missing handler for tool {spec.name} in {child}")
            run = _load_run(handler_path, spec.name)
            loaded[spec.name] = LoadedTool(spec=spec, run=run)
        return cls(loaded)

    def get(self, name: str) -> LoadedTool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict[str, object]]:
        return [tool.spec.model_dump(mode="json") for tool in self._tools.values()]

    async def run(self, name: str, args: dict[str, object]) -> dict[str, object]:
        tool = self.get(name)
        if tool is None:
            raise ToolRegistryError(f"unknown tool: {name}")
        validate(instance=args, schema=tool.spec.input_schema)
        result = await tool.run(args)
        return result


def _read_spec(spec_path: Path) -> ToolSpec:
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolRegistryError(f"invalid spec in {spec_path}: {exc}") from exc
    try:
        return ToolSpec.model_validate(raw)
    except PydanticValidationError as exc:
        raise ToolRegistryError(f"invalid spec in {spec_path}: {exc}") from exc


def _load_run(handler_path: Path, tool_name: str) -> Callable[[dict[str, object]], Awaitable[dict[str, object]]]:
    module_name = f"atlas_agent.tools._loaded_{handler_path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, handler_path)
    if spec is None or spec.loader is None:
        raise ToolRegistryError(f"missing handler for tool {tool_name} in {handler_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if run is None or not callable(run):
        raise ToolRegistryError(f"missing handler for tool {tool_name} in {handler_path}")
    return run


def format_invalid_args(error: ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    if path:
        return f"invalid args: {path}: {error.message}"
    return f"invalid args: {error.message}"
