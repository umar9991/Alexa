# Atlas

Personal AI desktop agent. Version 1 is local-first: Electron UI talks to a Python FastAPI backend.

## Backend (Windows PowerShell)

```powershell
cd services/agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q
python -m uvicorn atlas_agent.main:app --reload --port 8000
```

Health check: `http://localhost:8000/health` should return `{"status":"ok","name":"atlas","version":"0.0.1"}`.

Copy `.env.example` to `.env` and fill in keys when you add LLM providers. Do not commit `.env`.

## Desktop app

With the backend running on port 8000:

```powershell
cd apps/desktop
npm install
npm run typecheck
npm run build
npm run dev
```

Optional: copy `apps/desktop/.env.example` to `apps/desktop/.env` to override `VITE_ATLAS_BACKEND_WS`.

## Agent core

The backend WebSocket (`ws://localhost:8000/ws`) still sends `hello` on connect and echoes unknown JSON. Agent runs use:

- Client: `user_message` `{text}`, `approval` `{call_id, approved}`, `cancel` `{run_id}`
- Server: `run_started`, `tool_call`, `approval_required`, `tool_result`, `assistant_message`, `run_finished`
- T2 tools always wait for `approval`. A second `user_message` while a run is active returns `{"type":"error","message":"run already in progress"}`.
- Audit lines are appended to `data/audit.jsonl` (or `ATLAS_AUDIT_LOG`).

```powershell
python scripts/ws_demo.py "hello"
python scripts/ws_demo.py "danger delete everything"
```

## LLM provider

Copy `.env.example` to `.env`. Default `ATLAS_LLM_PROVIDER=scripted` needs no key. For Groq, set:

```
ATLAS_LLM_PROVIDER=groq
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant
```

Restart the backend after changing `.env`. Live API tests (skipped by default):

```powershell
$env:ATLAS_LIVE_TESTS="1"
pytest -q -m live
```

## Workspace tools

Set `ATLAS_WORKSPACE` in `.env` to an absolute folder (for example `D:\Atlas\workspace`). On startup the backend creates that folder and `_trash/` if needed. `fs.list`, `fs.read_text`, `fs.make_dir`, and `fs.write_text` only work inside it, using relative paths. Overwrite, move, and delete are not available yet. If `ATLAS_WORKSPACE` is empty, those tools fail with `workspace is not configured`.
