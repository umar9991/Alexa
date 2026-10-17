import { type FormEvent, type JSX, useState } from 'react'

import { formatJson } from '../lib/format'
import type { EchoItem } from '../store/connection'

export function DevTools({
  lastHello,
  echoes,
  connected,
  onSend
}: {
  lastHello: unknown
  echoes: EchoItem[]
  connected: boolean
  onSend: (payload: unknown) => void
}): JSX.Element {
  const [text, setText] = useState('')

  const onSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault()
    const value = text.trim()
    if (value.length === 0) {
      return
    }
    onSend({ text: value })
    setText('')
  }

  return (
    <details className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <summary className="cursor-pointer text-sm text-zinc-300">Dev tools</summary>
      <p className="mt-2 text-xs text-zinc-500">
        {lastHello === null ? 'Waiting for hello…' : formatJson(lastHello)}
      </p>
      <form className="mt-3 flex gap-2" onSubmit={onSubmit}>
        <input
          className="min-w-0 flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm outline-none"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Echo"
          aria-label="Echo message"
        />
        <button
          type="submit"
          className="rounded-lg bg-zinc-100 px-3 py-1 text-sm text-zinc-900 disabled:opacity-40"
          disabled={!connected}
        >
          Send
        </button>
      </form>
      <ul className="mt-3 flex max-h-40 flex-col gap-1 overflow-y-auto">
        {echoes.length === 0 ? (
          <li className="text-xs text-zinc-500">No echoes yet.</li>
        ) : (
          echoes.map((item) => (
            <li
              key={item.id}
              className="rounded border border-zinc-800 bg-zinc-950 px-2 py-1 font-mono text-[11px] text-zinc-400"
            >
              {formatJson(item.payload)}
            </li>
          ))
        )}
      </ul>
    </details>
  )
}
