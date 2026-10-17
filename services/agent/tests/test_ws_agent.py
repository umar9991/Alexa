from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from atlas_agent.main import app


def _drain_until(websocket: object, type_name: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    while True:
        event = websocket.receive_json()  # type: ignore[attr-defined]
        events.append(event)
        if event["type"] == type_name:
            return events


def test_ws_user_message_echo_and_busy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        hello = websocket.receive_json()
        assert hello == {"type": "hello"}
        websocket.send_json({"type": "user_message", "text": "hi"})
        events = _drain_until(websocket, "run_finished")
        types = [event["type"] for event in events]
        assert types[0] == "run_started"
        assert "run_finished" in types
        websocket.send_json({"text": "x"})
        echo = websocket.receive_json()
        assert echo == {"type": "echo", "data": {"text": "x"}}

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "user_message", "text": "danger hold"})
        _drain_until(websocket, "approval_required")
        websocket.send_json({"type": "user_message", "text": "hi"})
        error = websocket.receive_json()
        assert error == {"type": "error", "message": "run already in progress"}
