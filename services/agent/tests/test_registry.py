from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas_agent.core.policy import Tier
from atlas_agent.tools.registry import ToolRegistry, ToolRegistryError


def test_registry_loads_demo_tools() -> None:
    registry = ToolRegistry.load()
    echo = registry.get("demo.echo")
    dangerous = registry.get("demo.dangerous")
    assert echo is not None
    assert dangerous is not None
    assert echo.spec.tier is Tier.T0
    assert dangerous.spec.tier is Tier.T2
    names = {item["name"] for item in registry.specs()}
    assert names == {
        "demo.echo",
        "demo.dangerous",
        "fs.list",
        "fs.read_text",
        "fs.make_dir",
        "fs.write_text",
        "fs.move",
        "fs.delete",
        "fs.overwrite_text",
    }
    assert registry.get("fs.move") is not None
    assert registry.get("fs.move").spec.tier is Tier.T1  # type: ignore[union-attr]
    assert registry.get("fs.delete").spec.tier is Tier.T2  # type: ignore[union-attr]
    assert registry.get("fs.overwrite_text").spec.tier is Tier.T2  # type: ignore[union-attr]
    delete = registry.get("fs.delete")
    overwrite = registry.get("fs.overwrite_text")
    assert delete is not None and delete.describe is not None
    assert overwrite is not None and overwrite.describe is not None


def test_invalid_spec_is_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "spec.json").write_text(json.dumps({"name": "x"}), encoding="utf-8")
    (folder / "handler.py").write_text("async def run(args):\n    return args\n", encoding="utf-8")
    with pytest.raises(ToolRegistryError, match="invalid spec"):
        ToolRegistry.load(tmp_path)


def test_duplicate_name_is_rejected(tmp_path: Path) -> None:
    spec = {
        "name": "dup.tool",
        "description": "dup",
        "tier": "T0",
        "input_schema": {"type": "object"},
    }
    for name in ("one", "two"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        (folder / "handler.py").write_text(
            "async def run(args):\n    return args\n", encoding="utf-8"
        )
    with pytest.raises(ToolRegistryError, match="duplicate tool name"):
        ToolRegistry.load(tmp_path)


def test_missing_handler_is_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "no_handler"
    folder.mkdir()
    (folder / "spec.json").write_text(
        json.dumps(
            {
                "name": "missing.tool",
                "description": "no handler",
                "tier": "T0",
                "input_schema": {"type": "object"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ToolRegistryError, match="missing handler"):
        ToolRegistry.load(tmp_path)
