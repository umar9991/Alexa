import { useEffect, useRef, type JSX } from 'react'

import type { RunRecord } from '../store/run'

export function ChatTranscript({ runs }: { runs: RunRecord[] }): JSX.Element {
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' })
  }, [runs])

  return (
    <div className="flex-1 overflow-y-auto px-6 py-4">
      {runs.length === 0 ? (
        <p className="text-sm text-zinc-500">Send a command to start a run.</p>
      ) : (
        <ol className="flex flex-col gap-6">
          {runs.map((run) => (
            <li key={run.runId} className="flex flex-col gap-2">
              <div className="self-end max-w-[85%] rounded-lg bg-zinc-800 px-3 py-2 text-sm">
                {run.task}
              </div>
              {run.assistantMessages.map((text, index) => (
                <div
                  key={`${run.runId}-a-${index}`}
                  className="self-start max-w-[85%] rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200"
                >
                  {text}
                </div>
              ))}
              {isTerminal(run.status) ? (
                <p className="text-xs text-zinc-500">{finishLabel(run.status)}</p>
              ) : null}
            </li>
          ))}
        </ol>
      )}
      <div ref={bottom} />
    </div>
  )
}

function isTerminal(status: RunRecord['status']): boolean {
  return (
    status === 'done' ||
    status === 'failed' ||
    status === 'cancelled' ||
    status === 'connection_lost'
  )
}

function finishLabel(status: RunRecord['status']): string {
  if (status === 'done') {
    return 'done'
  }
  if (status === 'failed') {
    return 'failed'
  }
  if (status === 'cancelled') {
    return 'cancelled'
  }
  if (status === 'connection_lost') {
    return 'connection lost'
  }
  return status
}
