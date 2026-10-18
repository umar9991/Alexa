from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from atlas_agent.config import (
    ConfigError,
    always_on_toolsets_from_env,
    load_settings,
    parse_toolset_list,
)
from atlas_agent.core.events import (
    ApprovalRequired,
    AssistantMessage,
    RunFinished,
    ToolResult,
)
from atlas_agent.core.loop import ENABLE_SPEC, TOOLSET_NOT_ENABLED, approx_tokens
from atlas_agent.core.prompts import SYSTEM_PROMPT, build_system_prompt
from atlas_agent.llm.base import AgentTurn, ToolCall
from atlas_agent.tools.registry import ToolRegistry, discover_toolset_names
from atlas_agent.tools.toolsets import (
    ENABLE_TOOL_NAME,
    MAX_ENABLE_CALLS,
    first_sentence,
    toolset_description,
    toolset_of,
)
from tests.conftest import RecordingEmitter, run_scripted, wait_false, wait_true

REAL_TOOLSETS = ["demo", "docx", "fs", "lms"]


class TracingProvider:
    """Fake provider: records the tool names and a copy of the history for every step."""

    def __init__(self, turns: list[AgentTurn], *, repeat_last: bool = False) -> None:
        self.turns = list(turns)
        self.repeat_last = repeat_last
        self.tool_names: list[list[str]] = []
        self.tool_payloads: list[list[dict[str, object]]] = []
        self.histories: list[list[dict[str, object]]] = []

    async def step(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> AgentTurn:
        self.tool_payloads.append(copy.deepcopy(tools))
        self.tool_names.append([str(tool["name"]) for tool in tools])
        self.histories.append(copy.deepcopy(messages))
        if not self.turns:
            return AgentTurn(text="finished")
        if self.repeat_last and len(self.turns) == 1:
            return self.turns[0]
        return self.turns.pop(0)


def enable(*names: str) -> AgentTurn:
    return AgentTurn(tool_calls=[ToolCall(name=ENABLE_TOOL_NAME, args={"toolsets": list(names)})])


def call(name: str, **args: object) -> AgentTurn:
    return AgentTurn(tool_calls=[ToolCall(name=name, args=dict(args))])


def say(text: str) -> AgentTurn:
    return AgentTurn(text=text)


def lazy_run(
    task: str,
    tmp_path: Path,
    provider: TracingProvider,
    **kwargs: object,
) -> RecordingEmitter:
    workspace = tmp_path / "ws"
    (workspace / "_trash").mkdir(parents=True, exist_ok=True)
    return asyncio.run(
        run_scripted(
            task,
            tmp_path,
            provider=provider,
            workspace=workspace,
            always_on_toolsets=kwargs.pop("always_on_toolsets", ()),  # type: ignore[arg-type]
            **kwargs,  # type: ignore[arg-type]
        )
    )


def audit_lines(tmp_path: Path) -> list[dict[str, object]]:
    path = tmp_path / "audit.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def results(emitter: RecordingEmitter) -> list[ToolResult]:
    return [event for event in emitter.events if isinstance(event, ToolResult)]


def finished(emitter: RecordingEmitter) -> RunFinished:
    event = emitter.of_type("run_finished")[0]
    assert isinstance(event, RunFinished)
    return event


# --- helpers and registry ------------------------------------------------------------------


def test_toolset_name_is_part_before_first_dot() -> None:
    assert toolset_of("fs.read_text") == "fs"
    assert toolset_of("lms.assignment_detail") == "lms"
    assert toolset_of("plain") == "plain"


def test_unknown_toolset_description_falls_back_to_name() -> None:
    assert toolset_description("fs") == "files inside the workspace"
    assert toolset_description("zzz") == "zzz"


def test_first_sentence_is_short_and_single() -> None:
    assert first_sentence("Move a file. Never destroys data.") == "Move a file."
    assert first_sentence("No full stop here") == "No full stop here"
    assert first_sentence("Version 1.5 is fine. Second.") == "Version 1.5 is fine."
    long = first_sentence("x" * 300)
    assert len(long) <= 120 and long.endswith("...")


def test_registry_toolsets_only_for_registered_tools() -> None:
    registry = ToolRegistry.load()
    toolsets = registry.toolsets()
    assert list(toolsets) == REAL_TOOLSETS
    assert toolsets["lms"] == "read the university LMS"
    assert ToolRegistry({}).toolsets() == {}
    assert discover_toolset_names() == REAL_TOOLSETS


def test_registry_specs_for_returns_only_requested_toolsets() -> None:
    registry = ToolRegistry.load()
    fs_names = {str(spec["name"]) for spec in registry.specs_for({"fs"})}
    assert fs_names and all(name.startswith("fs.") for name in fs_names)
    assert registry.specs_for(set()) == []
    assert len(registry.specs_for(set(REAL_TOOLSETS))) == len(registry.specs())


def test_system_prompt_lists_exactly_the_registered_toolsets() -> None:
    registry = ToolRegistry.load()
    prompt = build_system_prompt(registry.toolsets())
    assert prompt.startswith(SYSTEM_PROMPT)
    marker = "Toolsets (call tools.enable before using their tools): "
    tail = prompt.split(marker, 1)[1]
    listing, closing = tail.split(". If no tool is needed, answer in plain text.", 1)
    assert closing == ""
    parsed = dict(item.split(": ", 1) for item in listing.split(", "))
    assert parsed == registry.toolsets()
    assert len(prompt) - len(SYSTEM_PROMPT) < 300  # stays short
    assert build_system_prompt({}) == SYSTEM_PROMPT


def test_initial_tools_payload_is_far_smaller_than_all_tools() -> None:
    registry = ToolRegistry.load()
    initial = [ENABLE_SPEC.model_dump(mode="json")]
    everything = registry.specs()
    assert len(json.dumps(initial)) < 0.25 * len(json.dumps(everything))
    assert approx_tokens(initial) < 0.25 * approx_tokens(everything)


# --- loop behavior -------------------------------------------------------------------------


def test_greeting_sends_only_the_meta_tool(tmp_path: Path) -> None:
    provider = TracingProvider([say("Hello!")])
    emitter = lazy_run("Hy", tmp_path, provider)
    assert provider.tool_names == [[ENABLE_TOOL_NAME]]
    assert emitter.of_type("tool_call") == []
    assert finished(emitter).status == "done"
    system = str(provider.histories[0][0]["content"])
    assert "fs: files inside the workspace" in system


def test_enabling_fs_exposes_fs_tools_on_the_next_step(tmp_path: Path) -> None:
    provider = TracingProvider(
        [
            enable("fs"),
            call("fs.write_text", path="notes.md", content="hello"),
            say("Created notes.md."),
        ]
    )
    emitter = lazy_run("create notes.md", tmp_path, provider)
    assert provider.tool_names[0] == [ENABLE_TOOL_NAME]
    second = set(provider.tool_names[1])
    assert ENABLE_TOOL_NAME in second and "fs.write_text" in second and "fs.delete" in second
    assert not any(name.startswith(("docx.", "lms.", "demo.")) for name in second)
    assert (tmp_path / "ws" / "notes.md").read_text(encoding="utf-8") == "hello"
    assert finished(emitter).summary == "Created notes.md."
    traced = [event.tool for event in emitter.events if event.type == "tool_call"]  # type: ignore[attr-defined]
    assert traced == [ENABLE_TOOL_NAME, "fs.write_text"]


def test_enable_result_shape_and_trace(tmp_path: Path) -> None:
    provider = TracingProvider([enable("fs", "fs"), say("ok")])
    emitter = lazy_run("go", tmp_path, provider)
    result = results(emitter)[0]
    assert result.ok and result.error is None and result.output is not None
    assert result.output["enabled"] == ["fs"]
    tools = result.output["tools"]
    assert isinstance(tools, list) and len(tools) == 7
    for item in tools:
        assert set(item) == {"name", "tier", "description"}
        assert len(item["description"]) <= 120
        assert item["tier"] in {"T0", "T1", "T2"}
    delete = next(item for item in tools if item["name"] == "fs.delete")
    assert delete["tier"] == "T2" and delete["description"] == "Move a file or folder to _trash."
    tool_message = next(m for m in provider.histories[1] if m["role"] == "tool")
    assert tool_message["name"] == ENABLE_TOOL_NAME


def test_disabled_tool_is_rejected_and_audited(tmp_path: Path) -> None:
    provider = TracingProvider(
        [call("fs.write_text", path="x.md", content="nope"), say("could not")]
    )
    emitter = lazy_run("write x.md", tmp_path, provider)
    result = results(emitter)[0]
    assert (result.ok, result.error) == (False, TOOLSET_NOT_ENABLED)
    assert result.error == "toolset not enabled; call tools.enable first"
    assert not (tmp_path / "ws" / "x.md").exists()
    lines = audit_lines(tmp_path)
    assert len(lines) == 1
    assert lines[0]["tool"] == "fs.write_text"
    assert lines[0]["decision"] == "deny"
    assert lines[0]["ok"] is False and lines[0]["approved"] is False
    assert lines[0]["error"] == TOOLSET_NOT_ENABLED


def test_unknown_tool_stays_denied_as_unknown(tmp_path: Path) -> None:
    provider = TracingProvider([call("fs.nope"), say("no")])
    emitter = lazy_run("x", tmp_path, provider)
    assert results(emitter)[0].error == "unknown tool"
    assert audit_lines(tmp_path)[0]["decision"] == "deny"


def test_unknown_toolset_returns_safe_error_and_enables_nothing(tmp_path: Path) -> None:
    provider = TracingProvider([enable("fs", "nope"), say("sorry")])
    emitter = lazy_run("x", tmp_path, provider)
    result = results(emitter)[0]
    assert result.ok is False
    assert result.error == "unknown toolset: nope. Known toolsets: demo, docx, fs, lms"
    assert provider.tool_names[1] == [ENABLE_TOOL_NAME]  # fs was not enabled either
    assert finished(emitter).status == "done"


@pytest.mark.parametrize(
    "args",
    [
        {"toolsets": []},
        {"toolsets": ["fs", "docx", "lms", "demo", "fs2"]},
        {"toolsets": "fs"},
        {"toolsets": [1]},
        {"toolsets": ["fs"], "extra": 1},
        {},
    ],
)
def test_enable_args_are_schema_validated(tmp_path: Path, args: dict[str, object]) -> None:
    provider = TracingProvider(
        [AgentTurn(tool_calls=[ToolCall(name=ENABLE_TOOL_NAME, args=args)]), say("x")]
    )
    emitter = lazy_run("x", tmp_path, provider)
    assert results(emitter)[0].ok is False
    assert str(results(emitter)[0].error).startswith("invalid args")
    assert audit_lines(tmp_path)[0]["decision"] == "invalid_args"
    assert provider.tool_names[1] == [ENABLE_TOOL_NAME]


def test_enable_call_limit_is_six_per_run(tmp_path: Path) -> None:
    turns = [enable("fs") for _ in range(MAX_ENABLE_CALLS + 1)] + [say("done")]
    provider = TracingProvider(turns)
    emitter = lazy_run("x", tmp_path, provider, max_steps=3)
    outcomes = results(emitter)
    assert [r.ok for r in outcomes] == [True] * MAX_ENABLE_CALLS + [False]
    assert outcomes[-1].error == "tools.enable limit reached (6 per run)"
    assert finished(emitter).status == "done"


def test_endless_enable_calls_still_terminate(tmp_path: Path) -> None:
    provider = TracingProvider([enable("fs")], repeat_last=True)
    emitter = lazy_run("x", tmp_path, provider, max_steps=3)
    assert finished(emitter).summary == "max steps reached"
    assert len(provider.tool_names) == MAX_ENABLE_CALLS + 3  # 6 free enables, then 3 real steps


def test_enable_does_not_consume_max_steps(tmp_path: Path) -> None:
    provider = TracingProvider(
        [
            enable("docx"),
            enable("fs"),
            call("fs.write_text", path="a.md", content="1"),
            say("All done."),
        ]
    )
    emitter = lazy_run("x", tmp_path, provider, max_steps=2)
    assert finished(emitter).status == "done"
    assert finished(emitter).summary == "All done."
    assert (tmp_path / "ws" / "a.md").exists()


def test_mixed_turn_with_enable_and_other_tool_costs_a_step(tmp_path: Path) -> None:
    mixed = AgentTurn(
        tool_calls=[
            ToolCall(name=ENABLE_TOOL_NAME, args={"toolsets": ["demo"]}),
            ToolCall(name="demo.echo", args={"text": "hi"}),
        ]
    )
    provider = TracingProvider([mixed, say("ok")])
    emitter = lazy_run("x", tmp_path, provider, max_steps=2)
    assert [r.ok for r in results(emitter)] == [True, True]  # enabled first, then echo allowed
    assert finished(emitter).status == "done"


def test_delete_still_requires_approval_after_enable(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "notes.md").write_text("hello", encoding="utf-8")
    provider = TracingProvider([enable("fs"), call("fs.delete", path="notes.md"), say("ok")])
    emitter = lazy_run("delete notes.md", tmp_path, provider, wait_approval=wait_false)
    approvals = [e for e in emitter.events if isinstance(e, ApprovalRequired)]
    assert [a.tool for a in approvals] == ["fs.delete"]
    assert results(emitter)[-1].error == "rejected by user"
    assert (tmp_path / "ws" / "notes.md").exists()
    delete_audit = audit_lines(tmp_path)[-1]
    assert delete_audit["tool"] == "fs.delete" and delete_audit["approved"] is False


def test_delete_runs_only_after_explicit_approval(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "notes.md").write_text("hello", encoding="utf-8")
    provider = TracingProvider([enable("fs"), call("fs.delete", path="notes.md"), say("ok")])
    asked: list[str] = []

    async def approve(call_id: str) -> bool:
        asked.append(call_id)
        return True

    emitter = lazy_run("delete notes.md", tmp_path, provider, wait_approval=approve)
    assert len(asked) == 1
    assert results(emitter)[-1].ok is True
    assert not (tmp_path / "ws" / "notes.md").exists()
    assert (tmp_path / "ws" / "_trash").exists()


def test_t2_tool_is_not_approved_by_enabling(tmp_path: Path) -> None:
    provider = TracingProvider([enable("demo"), call("demo.dangerous", action="x"), say("ok")])
    emitter = lazy_run("x", tmp_path, provider, wait_approval=wait_false)
    approvals = [e for e in emitter.events if isinstance(e, ApprovalRequired)]
    assert [a.tool for a in approvals] == ["demo.dangerous"]
    assert results(emitter)[-1].error == "rejected by user"


def test_enable_is_written_to_the_audit_log(tmp_path: Path) -> None:
    provider = TracingProvider([enable("fs"), say("ok")])
    lazy_run("x", tmp_path, provider)
    lines = audit_lines(tmp_path)
    assert len(lines) == 1
    entry = lines[0]
    assert entry["tool"] == ENABLE_TOOL_NAME
    assert entry["args"] == {"toolsets": ["fs"]}
    assert entry["tier"] == "T0" and entry["decision"] == "allow"
    assert entry["ok"] is True and entry["approved"] is True


def test_history_and_provider_state_round_trip_unchanged(tmp_path: Path) -> None:
    state = {"thought_signature": "opaque-123"}
    first = AgentTurn(
        text="enabling",
        tool_calls=[
            ToolCall(
                name=ENABLE_TOOL_NAME,
                args={"toolsets": ["fs"]},
                call_id="c1",
                provider_state=state,
            )
        ],
    )
    provider = TracingProvider(
        [first, call("fs.write_text", path="s.md", content="1"), say("done")]
    )
    lazy_run("x", tmp_path, provider)
    history = provider.histories[2]
    roles = [str(message["role"]) for message in history]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assistant = history[2]
    assert assistant["content"] == "enabling"
    assert assistant["tool_calls"] == [
        {"id": "c1", "name": ENABLE_TOOL_NAME, "args": {"toolsets": ["fs"]}, "provider_state": state}
    ]
    assert history[3]["tool_call_id"] == "c1" and history[3]["name"] == ENABLE_TOOL_NAME
    # Tools only ever grow within a run, so earlier calls always stay declared.
    assert set(provider.tool_names[0]) <= set(provider.tool_names[1]) <= set(provider.tool_names[2])


# --- always-on config ----------------------------------------------------------------------


def test_parse_toolset_list() -> None:
    assert parse_toolset_list(None) == ()
    assert parse_toolset_list("") == ()
    assert parse_toolset_list("  , ,") == ()
    assert parse_toolset_list("fs, DOCX,,fs") == ("fs", "docx")


def test_always_on_from_env_and_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ATLAS_ALWAYS_ON_TOOLSETS", raising=False)
    assert load_settings().always_on_toolsets == ()
    monkeypatch.setenv("ATLAS_ALWAYS_ON_TOOLSETS", "   ")
    assert load_settings().always_on_toolsets == ()
    monkeypatch.setenv("ATLAS_ALWAYS_ON_TOOLSETS", "fs, demo")
    assert always_on_toolsets_from_env() == ("fs", "demo")
    assert load_settings().always_on_toolsets == ("fs", "demo")


def test_unknown_always_on_name_fails_in_load_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_ALWAYS_ON_TOOLSETS", "fs,nope")
    with pytest.raises(ConfigError) as exc_info:
        load_settings()
    assert str(exc_info.value) == (
        "ATLAS_ALWAYS_ON_TOOLSETS has unknown toolset(s): nope. "
        "Known toolsets: demo, docx, fs, lms"
    )


def test_unknown_always_on_name_fails_backend_startup(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "ATLAS_ALWAYS_ON_TOOLSETS"}
    env["ATLAS_ALWAYS_ON_TOOLSETS"] = "nope"
    result = subprocess.run(
        [sys.executable, "-c", "import atlas_agent.main"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "unknown toolset(s): nope" in result.stderr


def test_always_on_config_enables_toolsets_from_the_first_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ATLAS_ALWAYS_ON_TOOLSETS", "fs")
    provider = TracingProvider([call("fs.list", path="."), say("listed")])
    workspace = tmp_path / "ws"
    workspace.mkdir()
    emitter = asyncio.run(
        run_scripted("list", tmp_path, provider=provider, workspace=workspace)
    )
    first = set(provider.tool_names[0])
    assert ENABLE_TOOL_NAME in first and "fs.list" in first
    assert not any(name.startswith(("docx.", "lms.", "demo.")) for name in first)
    assert results(emitter)[0].ok is True


def test_explicit_empty_always_on_beats_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ATLAS_ALWAYS_ON_TOOLSETS", "fs")
    provider = TracingProvider([say("hi")])
    asyncio.run(run_scripted("x", tmp_path, provider=provider, always_on_toolsets=()))
    assert provider.tool_names == [[ENABLE_TOOL_NAME]]


# --- measurement ---------------------------------------------------------------------------


def _debug_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("llm step tools_sent")]


def test_debug_line_reports_counts_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("ATLAS_LLM_DEBUG", "1")
    provider = TracingProvider([enable("fs"), say("ok")])
    lazy_run("TOPSECRET-TASK-TEXT", tmp_path, provider)
    lines = _debug_lines(caplog)
    expected_first = approx_tokens([ENABLE_SPEC.model_dump(mode="json")])
    assert lines[0] == f"llm step tools_sent=1 approx_tokens={expected_first}"
    assert lines[1].startswith("llm step tools_sent=8 approx_tokens=")
    second_tokens = int(lines[1].rsplit("=", 1)[1])
    assert second_tokens > expected_first
    assert "input_schema" not in caplog.text
    assert "TOPSECRET-TASK-TEXT" not in caplog.text


def test_debug_line_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.delenv("ATLAS_LLM_DEBUG", raising=False)
    lazy_run("x", tmp_path, TracingProvider([say("ok")]))
    assert _debug_lines(caplog) == []


def test_assistant_text_is_still_emitted(tmp_path: Path) -> None:
    provider = TracingProvider([say("Hello there")])
    emitter = lazy_run("Hy", tmp_path, provider)
    assert [e.text for e in emitter.events if isinstance(e, AssistantMessage)] == ["Hello there"]


_: Callable[[str], Awaitable[bool]] = wait_true
