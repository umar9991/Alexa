from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from atlas_agent import __version__
from atlas_agent.config import Settings, ensure_workspace, load_settings
from atlas_agent.core.audit import AuditLog
from atlas_agent.core.events import (
    AgentEvent,
    ApprovalMessage,
    CancelMessage,
    ErrorEvent,
    UserMessage,
)
from atlas_agent.core.loop import run_agent
from atlas_agent.llm.base import LLMProvider
from atlas_agent.llm.factory import (
    ProviderConfigError,
    build_provider,
    describe_provider_chain,
)
from atlas_agent.tools.registry import ToolRegistry

app = FastAPI(title="atlas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _configure_logging() -> None:
    """Make atlas_agent INFO lines visible under uvicorn (it only wires its own loggers)."""
    logger = logging.getLogger("atlas_agent")
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _start_llm(settings: Settings) -> tuple[LLMProvider, list[str]]:
    try:
        chain = describe_provider_chain(settings)
        provider = build_provider(settings)
    except ProviderConfigError as exc:
        raise SystemExit(f"atlas: cannot start: {exc}") from None
    return provider, chain


_configure_logging()
_settings = load_settings()
_workspace = ensure_workspace(_settings.workspace)
_registry = ToolRegistry.load()
_provider, _llm_chain = _start_llm(_settings)
logging.getLogger("atlas_agent").info("LLM chain: %s", ", ".join(_llm_chain))
if _settings.llm_debug:
    logging.getLogger("atlas_agent").info("LLM step debug logging is on (metadata only)")


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ok", "name": "atlas", "version": __version__, "llm": _llm_chain}


class ConnectionSession:
    def __init__(self) -> None:
        self.run_id: str | None = None
        self.run_task: asyncio.Task[None] | None = None
        self.cancel_event = asyncio.Event()
        self.pending: dict[str, asyncio.Future[bool]] = {}
        self.send_lock = asyncio.Lock()

    def run_active(self) -> bool:
        return self.run_task is not None and not self.run_task.done()

    def abort(self) -> None:
        self.cancel_event.set()
        for future in list(self.pending.values()):
            if not future.done():
                future.set_result(False)

    async def wait_approval(self, call_id: str) -> bool:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self.pending[call_id] = future
        try:
            return await future
        finally:
            self.pending.pop(call_id, None)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "hello"})
    session = ConnectionSession()
    try:
        while True:
            data = await websocket.receive_json()
            await _handle_client_message(websocket, session, data)
    except WebSocketDisconnect:
        session.abort()
        if session.run_task is not None:
            await session.run_task
        return


async def _handle_client_message(
    websocket: WebSocket, session: ConnectionSession, data: Any
) -> None:
    if not isinstance(data, dict):
        await websocket.send_json({"type": "echo", "data": data})
        return
    message_type = data.get("type")
    if message_type == "user_message":
        try:
            message = UserMessage.model_validate(data)
        except ValidationError:
            await websocket.send_json({"type": "echo", "data": data})
            return
        if session.run_active():
            await websocket.send_json(
                ErrorEvent(message="run already in progress").model_dump()
            )
            return
        await _start_run(websocket, session, message.text)
        return
    if message_type == "approval":
        try:
            message = ApprovalMessage.model_validate(data)
        except ValidationError:
            await websocket.send_json({"type": "echo", "data": data})
            return
        future = session.pending.get(message.call_id)
        if future is not None and not future.done():
            future.set_result(message.approved)
        return
    if message_type == "cancel":
        try:
            message = CancelMessage.model_validate(data)
        except ValidationError:
            await websocket.send_json({"type": "echo", "data": data})
            return
        if session.run_id == message.run_id:
            session.abort()
        return
    await websocket.send_json({"type": "echo", "data": data})


async def _start_run(websocket: WebSocket, session: ConnectionSession, text: str) -> None:
    run_id = str(uuid.uuid4())
    session.run_id = run_id
    session.cancel_event = asyncio.Event()
    audit = AuditLog.from_env()

    async def emit(event: AgentEvent) -> None:
        async with session.send_lock:
            try:
                await websocket.send_json(event.model_dump(mode="json"))
            except Exception:  # noqa: BLE001
                return

    session.run_task = asyncio.create_task(
        run_agent(
            text,
            provider=_provider,
            registry=_registry,
            emit=emit,
            wait_approval=session.wait_approval,
            audit=audit,
            run_id=run_id,
            cancel_event=session.cancel_event,
            workspace=_workspace,
        )
    )
