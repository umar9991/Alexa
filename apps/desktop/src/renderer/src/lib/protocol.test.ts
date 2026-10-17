import { expect, test } from 'vitest'

import { parseServerMessage } from './protocol'

test('parseServerMessage accepts every valid message type', () => {
  expect(parseServerMessage({ type: 'hello' })).toEqual({ type: 'hello' })
  expect(parseServerMessage({ type: 'echo', data: { text: 'x' } })).toEqual({
    type: 'echo',
    data: { text: 'x' }
  })
  expect(parseServerMessage({ type: 'error', message: 'run already in progress' })).toEqual({
    type: 'error',
    message: 'run already in progress'
  })
  expect(
    parseServerMessage({
      type: 'run_started',
      run_id: 'r1',
      ts: 't',
      task: 'hello'
    })?.type
  ).toBe('run_started')
  expect(
    parseServerMessage({
      type: 'tool_call',
      run_id: 'r1',
      ts: 't',
      call_id: 'c1',
      tool: 'demo.echo',
      args: { text: 'hello' },
      tier: 'T0'
    })?.type
  ).toBe('tool_call')
  expect(
    parseServerMessage({
      type: 'approval_required',
      run_id: 'r1',
      ts: 't',
      call_id: 'c1',
      tool: 'demo.dangerous',
      args: { action: 'wipe' },
      tier: 'T2',
      summary: 'demo.dangerous'
    })?.type
  ).toBe('approval_required')
  expect(
    parseServerMessage({
      type: 'tool_result',
      run_id: 'r1',
      ts: 't',
      call_id: 'c1',
      ok: true,
      output: { echo: 'hello' },
      error: null,
      duration_ms: 3
    })?.type
  ).toBe('tool_result')
  expect(
    parseServerMessage({
      type: 'assistant_message',
      run_id: 'r1',
      ts: 't',
      text: 'Done: ok'
    })?.type
  ).toBe('assistant_message')
  expect(
    parseServerMessage({
      type: 'run_finished',
      run_id: 'r1',
      ts: 't',
      status: 'done',
      summary: 'Done: ok'
    })?.type
  ).toBe('run_finished')
})

test('parseServerMessage rejects unknown types and missing fields', () => {
  expect(parseServerMessage({ type: 'nope' })).toBeNull()
  expect(parseServerMessage({})).toBeNull()
  expect(parseServerMessage(null)).toBeNull()
  expect(parseServerMessage({ type: 'run_started', run_id: 'r1' })).toBeNull()
  expect(parseServerMessage({ type: 'error' })).toBeNull()
  expect(
    parseServerMessage({
      type: 'tool_call',
      run_id: 'r1',
      ts: 't',
      call_id: 'c1',
      tool: 'demo.echo',
      args: { text: 'hello' },
      tier: 'T9'
    })
  ).toBeNull()
  expect(
    parseServerMessage({
      type: 'run_finished',
      run_id: 'r1',
      ts: 't',
      status: 'connection_lost',
      summary: 'x'
    })
  ).toBeNull()
})
