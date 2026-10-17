import { create } from 'zustand'

import type { ServerMessage, Tier } from '../lib/protocol'

export type RunStatus =
  | 'running'
  | 'awaiting_approval'
  | 'done'
  | 'failed'
  | 'cancelled'
  | 'connection_lost'

export type ToolCallStatus = 'running' | 'awaiting_approval' | 'ok' | 'failed'

export type LocalDecision = 'approved' | 'rejected' | null

export interface ToolCallCard {
  callId: string
  tool: string
  tier: Tier
  args: Record<string, unknown>
  status: ToolCallStatus
  output: Record<string, unknown> | null
  error: string | null
  durationMs: number | null
  decision: LocalDecision
  summary: string | null
  expired: boolean
}

export interface RunRecord {
  runId: string
  task: string
  status: RunStatus
  assistantMessages: string[]
  toolCalls: ToolCallCard[]
  finishedSummary: string | null
}

export interface RunState {
  runs: Record<string, RunRecord>
  runOrder: string[]
  activeRunId: string | null
  banner: string | null
}

export type RunAction =
  | { kind: 'server'; message: ServerMessage }
  | { kind: 'decision'; callId: string; approved: boolean }
  | { kind: 'disconnect' }
  | { kind: 'dismiss_banner' }

export const initialRunState: RunState = {
  runs: {},
  runOrder: [],
  activeRunId: null,
  banner: null
}

export function isCommandBarLocked(state: RunState): boolean {
  if (state.activeRunId === null) {
    return false
  }
  const run = state.runs[state.activeRunId]
  return run !== undefined && (run.status === 'running' || run.status === 'awaiting_approval')
}

export function reduceRunState(state: RunState, action: RunAction): RunState {
  if (action.kind === 'dismiss_banner') {
    return { ...state, banner: null }
  }
  if (action.kind === 'disconnect') {
    return applyDisconnect(state)
  }
  if (action.kind === 'decision') {
    return applyDecision(state, action.callId, action.approved)
  }
  return applyServerMessage(state, action.message)
}

export const useRunStore = create<RunState & { dispatch: (action: RunAction) => void }>(
  (set) => ({
    ...initialRunState,
    dispatch: (action) => set((state) => reduceRunState(state, action))
  })
)

function applyServerMessage(state: RunState, message: ServerMessage): RunState {
  switch (message.type) {
    case 'hello':
    case 'echo':
      return state
    case 'error':
      return { ...state, banner: message.message }
    case 'run_started':
      return applyRunStarted(state, message)
    case 'tool_call':
      return applyToolCall(state, message)
    case 'approval_required':
      return applyApprovalRequired(state, message)
    case 'tool_result':
      return applyToolResult(state, message)
    case 'assistant_message':
      return applyAssistant(state, message)
    case 'run_finished':
      return applyRunFinished(state, message)
  }
}

function applyRunStarted(
  state: RunState,
  message: Extract<ServerMessage, { type: 'run_started' }>
): RunState {
  const run: RunRecord = {
    runId: message.run_id,
    task: message.task,
    status: 'running',
    assistantMessages: [],
    toolCalls: [],
    finishedSummary: null
  }
  return {
    ...state,
    runs: { ...state.runs, [message.run_id]: run },
    runOrder: state.runOrder.includes(message.run_id)
      ? state.runOrder
      : [...state.runOrder, message.run_id],
    activeRunId: message.run_id
  }
}

function applyToolCall(
  state: RunState,
  message: Extract<ServerMessage, { type: 'tool_call' }>
): RunState {
  return updateRun(state, message.run_id, (run) => {
    if (run.toolCalls.some((card) => card.callId === message.call_id)) {
      return run
    }
    const card: ToolCallCard = {
      callId: message.call_id,
      tool: message.tool,
      tier: message.tier,
      args: message.args,
      status: 'running',
      output: null,
      error: null,
      durationMs: null,
      decision: null,
      summary: null,
      expired: false
    }
    return { ...run, toolCalls: [...run.toolCalls, card] }
  })
}

function applyApprovalRequired(
  state: RunState,
  message: Extract<ServerMessage, { type: 'approval_required' }>
): RunState {
  return updateRun(state, message.run_id, (run) => {
    const existing = run.toolCalls.find((card) => card.callId === message.call_id)
    const card: ToolCallCard = existing
      ? {
          ...existing,
          status: 'awaiting_approval',
          summary: message.summary,
          tool: message.tool,
          tier: message.tier,
          args: message.args
        }
      : {
          callId: message.call_id,
          tool: message.tool,
          tier: message.tier,
          args: message.args,
          status: 'awaiting_approval',
          output: null,
          error: null,
          durationMs: null,
          decision: null,
          summary: message.summary,
          expired: false
        }
    const toolCalls = existing
      ? run.toolCalls.map((item) => (item.callId === message.call_id ? card : item))
      : [...run.toolCalls, card]
    return { ...run, status: 'awaiting_approval', toolCalls }
  })
}

function applyToolResult(
  state: RunState,
  message: Extract<ServerMessage, { type: 'tool_result' }>
): RunState {
  return updateRun(state, message.run_id, (run) => {
    const toolCalls = run.toolCalls.map((card) => {
      if (card.callId !== message.call_id) {
        return card
      }
      return {
        ...card,
        status: message.ok ? 'ok' : 'failed',
        output: message.output,
        error: message.error,
        durationMs: message.duration_ms,
        expired: false
      } satisfies ToolCallCard
    })
    const stillWaiting = toolCalls.some((card) => card.status === 'awaiting_approval')
    const nextStatus: RunStatus = stillWaiting
      ? 'awaiting_approval'
      : run.status === 'awaiting_approval'
        ? 'running'
        : run.status
    return { ...run, toolCalls, status: nextStatus }
  })
}

function applyAssistant(
  state: RunState,
  message: Extract<ServerMessage, { type: 'assistant_message' }>
): RunState {
  return updateRun(state, message.run_id, (run) => ({
    ...run,
    assistantMessages: [...run.assistantMessages, message.text]
  }))
}

function applyRunFinished(
  state: RunState,
  message: Extract<ServerMessage, { type: 'run_finished' }>
): RunState {
  return updateRun(state, message.run_id, (run) => ({
    ...run,
    status: message.status,
    finishedSummary: message.summary
  }))
}

function applyDecision(state: RunState, callId: string, approved: boolean): RunState {
  const runId = findRunIdByCall(state, callId)
  if (runId === null) {
    return state
  }
  return updateRun(state, runId, (run) => {
    const card = run.toolCalls.find((item) => item.callId === callId)
    if (card === undefined || card.decision !== null || card.expired) {
      return run
    }
    return {
      ...run,
      toolCalls: run.toolCalls.map((item) =>
        item.callId === callId
          ? { ...item, decision: approved ? 'approved' : 'rejected' }
          : item
      )
    }
  })
}

function applyDisconnect(state: RunState): RunState {
  if (state.activeRunId === null) {
    return state
  }
  const run = state.runs[state.activeRunId]
  if (run === undefined) {
    return state
  }
  if (run.status !== 'running' && run.status !== 'awaiting_approval') {
    return state
  }
  const toolCalls = run.toolCalls.map((card) => {
    if (card.status !== 'awaiting_approval') {
      return card
    }
    return { ...card, expired: true }
  })
  return {
    ...state,
    runs: {
      ...state.runs,
      [run.runId]: { ...run, status: 'connection_lost', toolCalls }
    }
  }
}

function updateRun(
  state: RunState,
  runId: string,
  updater: (run: RunRecord) => RunRecord
): RunState {
  const run = state.runs[runId]
  if (run === undefined) {
    return state
  }
  return { ...state, runs: { ...state.runs, [runId]: updater(run) } }
}

function findRunIdByCall(state: RunState, callId: string): string | null {
  for (const runId of state.runOrder) {
    const run = state.runs[runId]
    if (run !== undefined && run.toolCalls.some((card) => card.callId === callId)) {
      return runId
    }
  }
  return null
}
