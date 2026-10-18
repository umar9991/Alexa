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

Copy `.env.example` to `.env`. `ATLAS_LLM_PROVIDER` is required: set it to one name or a comma-separated fallback chain (`groq`, `gemini`, or `groq,gemini`). If it is unset or empty the backend refuses to start and lists the valid values. `scripted` is a test-only provider (it answers every message by calling `demo.echo`) and is never a default; set `ATLAS_LLM_PROVIDER=scripted` explicitly if you want it, for example with `scripts/ws_demo.py`. Model names come from `GROQ_MODEL` / `GEMINI_MODEL`. Missing keys fail only when a run starts.

At startup the backend logs the resolved chain (for example `LLM chain: groq:openai/gpt-oss-20b, gemini:gemini-3.1-flash-lite`), and `GET /health` returns the same list in its `llm` field (`scripted:test-only` for the scripted provider). Restart the backend after editing `.env`; a real environment variable in the shell overrides `.env`. Set `ATLAS_LLM_DEBUG=1` to log one metadata line per LLM step (provider, model, finish reason, whether text was present, tool-call names; never content, arguments or keys).

```
ATLAS_LLM_PROVIDER=groq,gemini
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
```

Groq rate-limit or unavailability falls back to Gemini automatically. Auth and bad-request errors do not fall back. Optional `GROQ_BASE_URL` and `GEMINI_BASE_URL` override endpoints.

Restart the backend after changing `.env`. Dev probe (prints HTTP status and a redacted API message only): `python scripts/gemini_probe.py` (`--one-tool`, `--roundtrip`). Live API tests (skipped by default):

```powershell
$env:ATLAS_LIVE_TESTS="1"
pytest -q -m live
```

## Workspace tools

Set `ATLAS_WORKSPACE` in `.env` to an absolute folder (for example `D:\Atlas\workspace`). On startup the backend creates that folder and `_trash/` if needed. File tools only work inside it, using relative paths. `fs.list` / `fs.read_text` are T0; `fs.make_dir` / `fs.write_text` (create-only) / `fs.move` / `docx.build` (create-only `.docx` assignment) are T1; `fs.delete` and `fs.overwrite_text` are T2 and always need approval. Delete and overwrite move the previous data to `_trash/<id>/<original path>` and append `_trash/.index.jsonl`; `_trash` is never purged automatically. Restore by moving out of `_trash`. Optional `describe` hooks are read-only and must not change disk. Symlinks and directory junctions inside the workspace are rejected; the workspace root itself may be a link and is resolved once at startup. If `ATLAS_WORKSPACE` is empty, those tools fail with `workspace is not configured`.

## LMS

Read-only scan of your Moodle LMS through a saved browser session. Once, run `playwright install chromium`, then set `LMS_BASE_URL` (an https site root) in `.env` and restart the backend. Optional: `ATLAS_BROWSER_PROFILE` (must be inside `data/`) and `LMS_HEADLESS` (default `true`, used by `lms.scan` only).

1. Ask Atlas "log in to my LMS" once. `lms.login` (T1) opens a visible window; you sign in yourself, including 2FA. Atlas types nothing and stores no credentials.
2. Then ask "scan my LMS for assignments and quizzes". `lms.scan` (T0) lists each assignment and quiz with course, name, URL and due text. If the session expired it says so; run `lms.login` again.

3. Ask "show me the details of <assignment name>". `lms.assignment_detail` (T0) opens that one assignment or quiz page, using the URL from the scan result, and returns title, due and opened text, submission and grading status, description, attachment links and, for quizzes, time limit and attempts. It only accepts `/mod/assign/view.php?id=N` and `/mod/quiz/view.php?id=N` on your LMS host, never clicks or downloads, and its selectors target standard Moodle 4.x markup and were only tested against a fixture.

Navigation is limited to the LMS host and Microsoft sign-in hosts. `data/browser-profile` holds a live login session: never commit or share it.
