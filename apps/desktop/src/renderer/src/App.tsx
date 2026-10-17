import { type JSX, useEffect, useMemo, useState } from 'react'

import { ChatTranscript } from './components/ChatTranscript'
import { CommandBar } from './components/CommandBar'
import { ErrorBanner } from './components/ErrorBanner'
import { Header } from './components/Header'
import { TracePanel } from './components/TracePanel'
import type { ClientMessage, ServerMessage } from './lib/protocol'
import { connectAtlasWs } from './lib/ws'
import { useConnectionStore } from './store/connection'
import { isCommandBarLocked, useRunStore, type RunRecord } from './store/run'

const BACKEND_WS = import.meta.env.VITE_ATLAS_BACKEND_WS ?? 'ws://localhost:8000/ws'

export function App(): JSX.Element {
  const status = useConnectionStore((s) => s.status)
  const lastHello = useConnectionStore((s) => s.lastHello)
  const echoes = useConnectionStore((s) => s.echoes)
  const runState = useRunStore()
  const [sendJson, setSendJson] = useState<((json: unknown) => void) | null>(null)

  useEffect(() => {
    const connection = useConnectionStore.getState()
    const run = useRunStore.getState()
    const client = connectAtlasWs(BACKEND_WS, {
      onStatus: (next) => {
        connection.setStatus(next)
        if (next === 'disconnected') {
          run.dispatch({ kind: 'disconnect' })
        }
      },
      onMessage: (message: ServerMessage) => {
        if (message.type === 'hello') {
          connection.setLastHello(message)
          return
        }
        if (message.type === 'echo') {
          connection.addEcho(message.data)
          return
        }
        run.dispatch({ kind: 'server', message })
      }
    })
    setSendJson(() => client.send)
    return () => {
      client.close()
      setSendJson(null)
    }
  }, [])

  const sendClient = (message: ClientMessage): void => {
    if (sendJson === null) {
      return
    }
    sendJson(message)
  }

  const locked = isCommandBarLocked(runState)
  const runs = useMemo(
    () =>
      runState.runOrder
        .map((id) => runState.runs[id])
        .filter((run): run is RunRecord => run !== undefined),
    [runState.runOrder, runState.runs]
  )
  const activeRun = runState.activeRunId === null ? undefined : runState.runs[runState.activeRunId]
  const cards = runs.flatMap((run) => run.toolCalls)

  const onSend = (text: string): void => {
    sendClient({ type: 'user_message', text })
  }

  const onStop = (): void => {
    if (activeRun === undefined) {
      return
    }
    sendClient({ type: 'cancel', run_id: activeRun.runId })
  }

  const onDecide = (callId: string, approved: boolean): void => {
    const before = useRunStore.getState()
    const already = before.runOrder.some((id) =>
      before.runs[id]?.toolCalls.some((card) => card.callId === callId && card.decision !== null)
    )
    if (already) {
      return
    }
    useRunStore.getState().dispatch({ kind: 'decision', callId, approved })
    sendClient({ type: 'approval', call_id: callId, approved })
  }

  return (
    <div className="flex h-screen flex-col bg-zinc-950 text-zinc-100">
      <Header status={status} />
      {runState.banner !== null ? (
        <ErrorBanner
          message={runState.banner}
          onDismiss={() => useRunStore.getState().dispatch({ kind: 'dismiss_banner' })}
        />
      ) : null}
      <div className="flex min-h-0 flex-1">
        <section className="flex min-w-0 flex-1 flex-col">
          <ChatTranscript runs={runs} />
          <CommandBar
            disabled={locked}
            canSend={status === 'connected'}
            showStop={locked}
            onSend={onSend}
            onStop={onStop}
          />
        </section>
        <TracePanel
          cards={cards}
          lastHello={lastHello}
          echoes={echoes}
          connected={status === 'connected'}
          onEcho={(payload) => sendJson?.(payload)}
          onDecide={onDecide}
        />
      </div>
    </div>
  )
}
