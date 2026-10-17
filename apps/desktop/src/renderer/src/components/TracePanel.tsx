import { useEffect, useRef, type JSX } from 'react'

import type { EchoItem } from '../store/connection'
import type { ToolCallCard } from '../store/run'
import { DevTools } from './DevTools'
import { ToolCallCardView } from './ToolCallCardView'

export function TracePanel({
  cards,
  lastHello,
  echoes,
  connected,
  onEcho,
  onDecide
}: {
  cards: ToolCallCard[]
  lastHello: unknown
  echoes: EchoItem[]
  connected: boolean
  onEcho: (payload: unknown) => void
  onDecide: (callId: string, approved: boolean) => void
}): JSX.Element {
  const scroller = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const node = scroller.current
    if (node !== null) {
      node.scrollTop = node.scrollHeight
    }
  }, [cards])

  return (
    <aside className="flex w-[380px] shrink-0 flex-col border-l border-zinc-800 bg-zinc-950">
      <h2 className="border-b border-zinc-800 px-4 py-3 text-sm font-medium text-zinc-300">
        Live trace
      </h2>
      <div ref={scroller} className="flex-1 overflow-y-auto px-3 py-3">
        {cards.length === 0 ? (
          <p className="text-xs text-zinc-500">Tool calls will appear here.</p>
        ) : (
          <div className="flex flex-col gap-3">
            {cards.map((card) => (
              <ToolCallCardView key={card.callId} card={card} onDecide={onDecide} />
            ))}
          </div>
        )}
      </div>
      <div className="border-t border-zinc-800 p-3">
        <DevTools lastHello={lastHello} echoes={echoes} connected={connected} onSend={onEcho} />
      </div>
    </aside>
  )
}
