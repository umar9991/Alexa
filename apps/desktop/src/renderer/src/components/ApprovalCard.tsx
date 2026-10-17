import type { JSX } from 'react'

import { formatJson } from '../lib/format'
import type { ToolCallCard } from '../store/run'

export function ApprovalCard({
  card,
  onDecide
}: {
  card: ToolCallCard
  onDecide: (approved: boolean) => void
}): JSX.Element {
  const locked = card.decision !== null || card.expired
  return (
    <div className="mt-3 rounded-md border border-amber-500/30 bg-amber-950/30 p-3">
      <p className="text-sm font-medium text-amber-200">Approval required</p>
      <p className="mt-1 text-xs text-zinc-400">
        {card.tool} · {card.tier}
      </p>
      {card.summary !== null ? (
        <p className="mt-2 text-sm text-zinc-200">{card.summary}</p>
      ) : null}
      <pre className="mt-2 max-h-32 overflow-auto whitespace-pre-wrap break-all rounded bg-zinc-950 p-2 font-mono text-[11px] text-zinc-400">
        {formatJson(card.args)}
      </pre>
      {card.expired ? (
        <p className="mt-2 text-xs text-red-300">Connection lost</p>
      ) : card.decision !== null ? (
        <p className="mt-2 text-xs text-zinc-400">
          {card.decision === 'approved' ? 'Approved' : 'Rejected'}
        </p>
      ) : (
        <div className="mt-3 flex gap-2">
          <button
            type="button"
            className="rounded bg-emerald-500/20 px-3 py-1 text-sm text-emerald-300 ring-1 ring-emerald-500/40 disabled:opacity-40"
            disabled={locked}
            onClick={() => onDecide(true)}
          >
            Approve
          </button>
          <button
            type="button"
            className="rounded bg-red-500/20 px-3 py-1 text-sm text-red-300 ring-1 ring-red-500/40 disabled:opacity-40"
            disabled={locked}
            onClick={() => onDecide(false)}
          >
            Reject
          </button>
        </div>
      )}
    </div>
  )
}
