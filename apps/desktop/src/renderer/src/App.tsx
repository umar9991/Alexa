import { FormEvent, type JSX, useEffect, useState } from 'react'

import { connectAtlasWs, type WsStatus } from './lib/ws'
import { useConnectionStore } from './store/connection'

const BACKEND_WS =
  import.meta.env.VITE_ATLAS_BACKEND_WS ?? 'ws://localhost:8000/ws'

function statusLabel(status: WsStatus): string {
  if (status === 'connected') {
    return 'Connected'
  }
  if (status === 'connecting') {
    return 'Connecting...'
  }
  return 'Disconnected, retrying'
}

function statusClass(status: WsStatus): string {
  if (status === 'connected') {
    return 'bg-emerald-500/15 text-emerald-400 ring-emerald-500/40'
  }
  if (status === 'connecting') {
    return 'bg-amber-500/15 text-amber-400 ring-amber-500/40'
  }
  return 'bg-red-500/15 text-red-400 ring-red-500/40'
}

function formatPayload(payload: unknown): string {
  try {
    return JSON.stringify(payload)
  } catch {
    return String(payload)
  }
}

export function App(): JSX.Element {
  const status = useConnectionStore((s) => s.status)
  const lastHello = useConnectionStore((s) => s.lastHello)
  const echoes = useConnectionStore((s) => s.echoes)
  const [text, setText] = useState('')
  const [sendJson, setSendJson] = useState<((json: unknown) => void) | null>(null)

  useEffect(() => {
    const store = useConnectionStore.getState()
    const client = connectAtlasWs(BACKEND_WS, {
      onStatus: store.setStatus,
      onHello: store.setLastHello,
      onEcho: store.addEcho
    })
    setSendJson(() => client.send)
    return () => {
      client.close()
      setSendJson(null)
    }
  }, [])

  const onSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault()
    const value = text.trim()
    if (value.length === 0 || sendJson === null) {
      return
    }
    sendJson({ text: value })
    setText('')
  }

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="flex items-center justify-between border-b border-zinc-800 px-6 py-4">
        <h1 className="text-lg font-semibold tracking-tight">Atlas</h1>
        <span
          className={`rounded-full px-3 py-1 text-sm font-medium ring-1 ${statusClass(status)}`}
        >
          {statusLabel(status)}
        </span>
      </header>

      <main className="mx-auto flex w-full max-w-xl flex-1 flex-col gap-6 px-6 py-6">
        <p className="text-sm text-zinc-500">
          {lastHello === null ? 'Waiting for hello…' : formatPayload(lastHello)}
        </p>

        <section className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
          <h2 className="mb-3 text-sm font-medium text-zinc-300">Dev echo</h2>
          <form className="flex gap-2" onSubmit={onSubmit}>
            <input
              className="min-w-0 flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm outline-none ring-emerald-500/40 focus:ring-2"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Message"
              aria-label="Echo message"
            />
            <button
              type="submit"
              className="rounded-lg bg-zinc-100 px-4 py-2 text-sm font-medium text-zinc-900 disabled:opacity-40"
              disabled={status !== 'connected'}
            >
              Send
            </button>
          </form>
          <ul className="mt-4 flex max-h-64 flex-col gap-2 overflow-y-auto text-sm">
            {echoes.length === 0 ? (
              <li className="text-zinc-500">No echoes yet.</li>
            ) : (
              echoes.map((item) => (
                <li
                  key={item.id}
                  className="rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 font-mono text-xs text-zinc-300"
                >
                  {formatPayload(item.payload)}
                </li>
              ))
            )}
          </ul>
        </section>
      </main>
    </div>
  )
}
