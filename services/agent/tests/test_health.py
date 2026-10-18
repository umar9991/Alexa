import pytest
from fastapi.testclient import TestClient

from atlas_agent import main
from atlas_agent.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "name": "atlas",
        "version": "0.0.1",
        "llm": ["scripted:test-only"],
    }


def test_health_reports_resolved_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main, "_llm_chain", ["groq:openai/gpt-oss-20b", "gemini:gemini-3.1-flash-lite"]
    )
    body = client.get("/health").json()
    assert body["llm"] == ["groq:openai/gpt-oss-20b", "gemini:gemini-3.1-flash-lite"]
    assert set(body) == {"status", "name", "version", "llm"}


def test_ws_hello_then_echo() -> None:
    with client.websocket_connect("/ws") as websocket:
        hello = websocket.receive_json()
        assert hello == {"type": "hello"}
        payload = {"ping": "pong"}
        websocket.send_json(payload)
        echo = websocket.receive_json()
        assert echo == {"type": "echo", "data": payload}
