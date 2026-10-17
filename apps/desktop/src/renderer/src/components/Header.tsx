import type { JSX } from 'react'

import type { WsStatus } from '../lib/ws'

export function Header({ status }: { status: WsStatus }): JSX.Element {
  return (
    <header className="flex items-center justify-between border-b border-zinc-800 px-6 py-3">
      <h1 className="text-lg font-semibold tracking-tight">Atlas</h1>
      <span
        className={`rounded-full px-3 py-1 text-sm font-medium ring-1 ${statusClass(status)}`}
      >
        {statusLabel(status)}
      </span>
    </header>
  )
}

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
