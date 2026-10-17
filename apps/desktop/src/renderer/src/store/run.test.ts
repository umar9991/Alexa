import { expect, test } from 'vitest'

import {
  initialRunState,
  isCommandBarLocked,
  reduceRunState,
  type RunState
} from './run'
import type { ServerMessage } from '../lib/protocol'

function applyAll(messages: ServerMessage[], start: RunState = initialRunState): RunState {
  return messages.reduce(
    (state, message) => reduceRunState(state, { kind: 'server', message }),
    start
  )
}

const started: ServerMessage = {
  type: 'run_started',
  run_id: 'r1',
  ts: 't0',
  task: 'hello'
}

const toolCall: ServerMessage = {
  type: 'tool_call',
  run_id: 'r1',
  ts: 't1',
  call_id: 'c1',
  tool: 'demo.echo',
  args: { text: 'hello' },
  tier: 'T0'
}

const toolOk: ServerMessage = {
  type: 'tool_result',
  run_id: 'r1',
  ts: 't2',
  call_id: 'c1',
  ok: true,
  output: { echo: 'hello' },
  error: null,
  duration_ms: 4
}

const assistant: ServerMessage = {
  type: 'assistant_message',
  run_id: 'r1',
  ts: 't3',
  text: 'Done: ok'
}

const finished: ServerMessage = {
  type: 'run_finished',
  run_id: 'r1',
  ts: 't4',
  status: 'done',
  summary: 'Done: ok'
}

test('happy-path sequence produces expected state', () => {
  const state = applyAll([started, toolCall, toolOk, assistant, finished])
  const run = state.runs.r1
  expect(run.status).toBe('done')
  expect(run.task).toBe('hello')
  expect(run.assistantMessages).toEqual(['Done: ok'])
  expect(run.toolCalls).toHaveLength(1)
  expect(run.toolCalls[0].status).toBe('ok')
  expect(run.toolCalls[0].durationMs).toBe(4)
  expect(isCommandBarLocked(state)).toBe(false)
})

test('approval_required then failed tool_result', () => {
  const required: ServerMessage = {
    type: 'approval_required',
    run_id: 'r1',
    ts: 't1',
    call_id: 'c2',
    tool: 'demo.dangerous',
    args: { action: 'wipe' },
    tier: 'T2',
    summary: 'demo.dangerous wipe'
  }
  const failed: ServerMessage = {
    type: 'tool_result',
    run_id: 'r1',
    ts: 't2',
    call_id: 'c2',
    ok: false,
    output: null,
    error: 'rejected by user',
    duration_ms: 0
  }
  const waiting = applyAll([started, required])
  expect(waiting.runs.r1.status).toBe('awaiting_approval')
  expect(waiting.runs.r1.toolCalls[0].status).toBe('awaiting_approval')
  expect(waiting.runs.r1.toolCalls[0].summary).toBe('demo.dangerous wipe')
  const after = applyAll([failed], waiting)
  expect(after.runs.r1.toolCalls[0].status).toBe('failed')
  expect(after.runs.r1.toolCalls[0].error).toBe('rejected by user')
})

test('disconnect during active run marks connection_lost', () => {
  const required: ServerMessage = {
    type: 'approval_required',
    run_id: 'r1',
    ts: 't1',
    call_id: 'c2',
    tool: 'demo.dangerous',
    args: { action: 'wipe' },
    tier: 'T2',
    summary: 'need approval'
  }
  const waiting = applyAll([started, required])
  expect(isCommandBarLocked(waiting)).toBe(true)
  const lost = reduceRunState(waiting, { kind: 'disconnect' })
  expect(lost.runs.r1.status).toBe('connection_lost')
  expect(lost.runs.r1.toolCalls[0].expired).toBe(true)
  expect(isCommandBarLocked(lost)).toBe(false)
})

test('second decision for the same call_id is ignored', () => {
  const required: ServerMessage = {
    type: 'approval_required',
    run_id: 'r1',
    ts: 't1',
    call_id: 'c2',
    tool: 'demo.dangerous',
    args: { action: 'wipe' },
    tier: 'T2',
    summary: 'need approval'
  }
  const waiting = applyAll([started, required])
  const first = reduceRunState(waiting, { kind: 'decision', callId: 'c2', approved: true })
  expect(first.runs.r1.toolCalls[0].decision).toBe('approved')
  const second = reduceRunState(first, { kind: 'decision', callId: 'c2', approved: false })
  expect(second.runs.r1.toolCalls[0].decision).toBe('approved')
})
