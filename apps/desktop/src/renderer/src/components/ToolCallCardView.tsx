import type { JSX } from 'react'

import { formatJson } from '../lib/format'
import type { Tier } from '../lib/protocol'
import type { ToolCallCard } from '../store/run'
import { ApprovalCard } from './ApprovalCard'

export function ToolCallCardView({
  card,
  onDecide
}: {
  card: ToolCallCard
  onDecide: (callId: string, approved: boolean) => void
}): JSX.Element {
  const showApproval = card.status === 'awaiting_approval'
  return (
    <article className="rounded-lg border border-zinc-800 bg-zinc-900/70 p-3">
      <div className="flex items-center justify-between gap-2">
        <h3 className="truncate text-sm font-medium">{card.tool}</h3>
        <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${tierClass(card.tier)}`}>
          {card.tier}
        </span>
      </div>
      <p className="mt-1 text-xs text-zinc-400">{statusLabel(card)}</p>
      {card.durationMs !== null ? (
        <p className="text-xs text-zinc-500">{card.durationMs} ms</p>
      ) : null}
      <details className="mt-2">
        <summary className="cursor-pointer text-xs text-zinc-400">Args</summary>
        <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all rounded bg-zinc-950 p-2 font-mono text-[11px] text-zinc-400">
          {formatJson(card.args)}
        </pre>
      </details>
      {card.output !== null ? (
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-zinc-400">Output</summary>
          <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all rounded bg-zinc-950 p-2 font-mono text-[11px] text-zinc-400">
            {formatJson(card.output)}
          </pre>
        </details>
      ) : null}
      {card.status === 'failed' && card.error !== null ? (
        <p className="mt-2 whitespace-pre-wrap break-all text-xs text-red-300">{card.error}</p>
      ) : null}
      {showApproval && card.status !== 'ok' ? (
        <ApprovalCard card={card} onDecide={(approved) => onDecide(card.callId, approved)} />
      ) : null}
    </article>
  )
}

function tierClass(tier: Tier): string {
  if (tier === 'T2') {
    return 'bg-red-500/20 text-red-300'
  }
  if (tier === 'T1') {
    return 'bg-sky-500/20 text-sky-300'
  }
  return 'bg-zinc-700 text-zinc-300'
}

function statusLabel(card: ToolCallCard): string {
  if (card.expired && card.status === 'awaiting_approval') {
    return 'connection lost'
  }
  if (card.status === 'awaiting_approval') {
    return 'awaiting approval'
  }
  if (card.status === 'ok') {
    return 'ok'
  }
  if (card.status === 'failed') {
    return 'failed'
  }
  return 'running'
}
