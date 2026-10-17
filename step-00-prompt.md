# Step 0: Repo scaffold + backend health check

Paste everything below the line into Cursor (Agent mode). Before that, put `AGENTS.md` in the repo root and `atlas.mdc` in `.cursor/rules/`.

---

Read `AGENTS.md` first and follow it.

## Task

Create the Atlas repo skeleton and a minimal FastAPI backend that proves the backend runs. Nothing else.

## Do this

1. Create the folder structure from `AGENTS.md` (empty folders get a `.gitkeep`). Do not create files inside `apps/desktop`, `tools`, `llm`, `memory`, or `core` yet other than `.gitkeep`.
2. In `services/agent`, create a Python 3.12 project:
   - `pyproject.toml` with dependencies `fastapi`, `uvicorn[standard]`, `pydantic`, `python-dotenv`, and optional `dev` dependencies `pytest`, `httpx`, `ruff`. Check the current stable versions and pin them.
   - `atlas_agent/__init__.py` with `__version__ = "0.0.1"`.
   - `atlas_agent/main.py` with:
     - `GET /health` returning `{"status": "ok", "name": "atlas", "version": "0.0.1"}`.
     - `WebSocket /ws` that sends `{"type": "hello"}` right after connecting, then for each JSON message received replies `{"type": "echo", "data": <the received JSON>}`.
     - CORS allowing `http://localhost:5173` (the future UI dev server).
   - `tests/test_health.py` using FastAPI `TestClient`: one test for `/health`, one test for `/ws` (hello message, then echo).
3. At the repo root add:
   - `.gitignore` (Python, Node, `.env`, `.venv`, `__pycache__`, `*.db`, `workspace/`).
   - `.env.example` with `GROQ_API_KEY=`, `GEMINI_API_KEY=`, `ATLAS_WORKSPACE=` (empty placeholders only, no real values).
   - `README.md` with the Windows PowerShell commands to set up, test, and run the backend.

## Do not

- Do not add any LLM code, tools, database, Electron, or React.
- Do not install packages other than the ones listed.
- Do not edit `AGENTS.md` or `.cursor/rules/atlas.mdc`.

## Acceptance test (I will run this)

```powershell
cd services/agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q
```
Expected: `2 passed`.

```powershell
uvicorn atlas_agent.main:app --reload --port 8000
```
Expected: opening `http://localhost:8000/health` shows `{"status":"ok","name":"atlas","version":"0.0.1"}`.

When done, tell me: what changed, how to run it, how to test it, and anything you did not do.
