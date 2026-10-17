# Step 1: Desktop shell that connects to the backend

Paste everything below the line into Cursor (Agent mode). Step 0 must be done first (backend `/health` and `/ws` working). Node.js LTS must be installed (`node -v`).

---

Read `AGENTS.md` first and follow it.

## Task

Create the Atlas desktop app shell in `apps/desktop`: an Electron window with a React UI that connects to the backend WebSocket and shows the connection status. Nothing else.

## Do this

1. Scaffold `apps/desktop` with Electron + Vite + React + TypeScript (strict). Use `electron-vite` with its React + TypeScript template, but check its current docs first and follow what they say now. Add Tailwind CSS (current version, follow its current setup docs) and `zustand`.
2. Security defaults in the Electron main process and preload: `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`. The renderer must not get Node access. The preload exposes nothing yet.
3. Backend URL comes from an env variable `VITE_ATLAS_BACKEND_WS` with default `ws://localhost:8000/ws`. Add an `.env.example` in `apps/desktop` with that variable.
4. `src/renderer/lib/ws.ts`: a small WebSocket client that
   - connects to the backend URL,
   - reconnects automatically with backoff (1s, 2s, 4s, then 5s max),
   - exposes `status` (`connecting`, `connected`, `disconnected`), the last `hello` message, and a `send(json)` method,
   - closes cleanly on unmount.
5. `src/renderer/store/connection.ts`: a Zustand store holding `status`, `lastHello`, and a list of echo messages (keep the last 20).
6. UI (dark theme, minimal but clean):
   - Header with the name "Atlas" and a status pill: green "Connected", amber "Connecting...", red "Disconnected, retrying".
   - A small "Dev echo" panel: a text input and a Send button. Sending a message sends `{"text": "<input>"}` to the backend and shows the reply (`echo`) in a list below.
   - Show the received `hello` payload in a small muted line.
7. Add npm scripts: `dev`, `build`, `typecheck`. Add ESLint + Prettier only if the template already includes them; do not add extra tooling.
8. Add a short "Desktop app" section to the root `README.md` with the Windows PowerShell commands to install and run it (do not change other sections).

## Files you may touch

- Everything inside `apps/desktop/`
- Root `README.md` (only add the Desktop app section)

## Do not

- Do not edit anything in `services/agent`, `AGENTS.md`, or `.cursor/rules`.
- Do not add chat UI, approvals, panels, routing, shadcn/ui, or any other screen yet.
- Do not call any LLM or access the file system from the UI.
- Do not install packages other than the ones named here and what the template needs.

## Acceptance test (I will run this)

Terminal 1:
```powershell
cd services/agent
.venv\Scripts\Activate.ps1
uvicorn atlas_agent.main:app --reload --port 8000
```

Terminal 2:
```powershell
cd apps/desktop
npm install
npm run typecheck
npm run build
npm run dev
```

Expected:
1. `typecheck` and `build` finish without errors.
2. `npm run dev` opens an Atlas window with a green "Connected" pill and the hello message shown.
3. Typing "test" in the echo panel and pressing Send shows the echoed reply in the list.
4. Stop the backend (Ctrl+C): the pill turns red "Disconnected, retrying". Start it again: the pill returns to green by itself, without restarting the app.

When done, tell me: what changed, how to run it, how to test it, and anything you did not do.
