from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from atlas_agent import __version__

app = FastAPI(title="atlas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "name": "atlas", "version": __version__}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "hello"})
    try:
        while True:
            data = await websocket.receive_json()
            await websocket.send_json({"type": "echo", "data": data})
    except WebSocketDisconnect:
        return
