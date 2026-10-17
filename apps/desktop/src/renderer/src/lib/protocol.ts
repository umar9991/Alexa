export type JsonObject = Record<string, unknown>

export type Tier = 'T0' | 'T1' | 'T2'

export type HelloMessage = {
  type: 'hello'
}

export type EchoMessage = {
  type: 'echo'
  data: unknown
}

export type ErrorMessage = {
  type: 'error'
  message: string
}

export type RunStartedMessage = {
  type: 'run_started'
  run_id: string
  ts: string
  task: string
}

export type ToolCallMessage = {
  type: 'tool_call'
  run_id: string
  ts: string
  call_id: string
  tool: string
  args: JsonObject
  tier: Tier
}

export type ApprovalRequiredMessage = {
  type: 'approval_required'
  run_id: string
  ts: string
  call_id: string
  tool: string
  args: JsonObject
  tier: Tier
  summary: string
}

export type ToolResultMessage = {
  type: 'tool_result'
  run_id: string
  ts: string
  call_id: string
  ok: boolean
  output: JsonObject | null
  error: string | null
  duration_ms: number
}

export type AssistantMessage = {
  type: 'assistant_message'
  run_id: string
  ts: string
  text: string
}

export type RunFinishedMessage = {
  type: 'run_finished'
  run_id: string
  ts: string
  status: 'done' | 'failed' | 'cancelled'
  summary: string
}

export type ServerMessage =
  | HelloMessage
  | EchoMessage
  | ErrorMessage
  | RunStartedMessage
  | ToolCallMessage
  | ApprovalRequiredMessage
  | ToolResultMessage
  | AssistantMessage
  | RunFinishedMessage

export type UserMessage = {
  type: 'user_message'
  text: string
}

export type ApprovalClientMessage = {
  type: 'approval'
  call_id: string
  approved: boolean
}

export type CancelClientMessage = {
  type: 'cancel'
  run_id: string
}

export type ClientMessage = UserMessage | ApprovalClientMessage | CancelClientMessage

export function parseServerMessage(raw: unknown): ServerMessage | null {
  if (!isRecord(raw) || typeof raw.type !== 'string') {
    return null
  }
  switch (raw.type) {
    case 'hello':
      return { type: 'hello' }
    case 'echo':
      if (!('data' in raw)) {
        return null
      }
      return { type: 'echo', data: raw.data }
    case 'error':
      if (typeof raw.message !== 'string') {
        return null
      }
      return { type: 'error', message: raw.message }
    case 'run_started':
      if (!isRunBase(raw) || typeof raw.task !== 'string') {
        return null
      }
      return {
        type: 'run_started',
        run_id: raw.run_id,
        ts: raw.ts,
        task: raw.task
      }
    case 'tool_call':
      return parseToolCall(raw)
    case 'approval_required':
      return parseApprovalRequired(raw)
    case 'tool_result':
      return parseToolResult(raw)
    case 'assistant_message':
      if (!isRunBase(raw) || typeof raw.text !== 'string') {
        return null
      }
      return {
        type: 'assistant_message',
        run_id: raw.run_id,
        ts: raw.ts,
        text: raw.text
      }
    case 'run_finished':
      if (
        !isRunBase(raw) ||
        (raw.status !== 'done' && raw.status !== 'failed' && raw.status !== 'cancelled') ||
        typeof raw.summary !== 'string'
      ) {
        return null
      }
      return {
        type: 'run_finished',
        run_id: raw.run_id,
        ts: raw.ts,
        status: raw.status,
        summary: raw.summary
      }
    default:
      return null
  }
}

function parseToolCall(raw: Record<string, unknown>): ToolCallMessage | null {
  if (!isRunBase(raw) || !isToolFields(raw)) {
    return null
  }
  return {
    type: 'tool_call',
    run_id: raw.run_id,
    ts: raw.ts,
    call_id: raw.call_id,
    tool: raw.tool,
    args: raw.args,
    tier: raw.tier
  }
}

function parseApprovalRequired(raw: Record<string, unknown>): ApprovalRequiredMessage | null {
  if (!isRunBase(raw) || !isToolFields(raw) || typeof raw.summary !== 'string') {
    return null
  }
  return {
    type: 'approval_required',
    run_id: raw.run_id,
    ts: raw.ts,
    call_id: raw.call_id,
    tool: raw.tool,
    args: raw.args,
    tier: raw.tier,
    summary: raw.summary
  }
}

function parseToolResult(raw: Record<string, unknown>): ToolResultMessage | null {
  if (
    !isRunBase(raw) ||
    typeof raw.call_id !== 'string' ||
    typeof raw.ok !== 'boolean' ||
    typeof raw.duration_ms !== 'number' ||
    !isOptionalObject(raw.output) ||
    !isOptionalString(raw.error)
  ) {
    return null
  }
  return {
    type: 'tool_result',
    run_id: raw.run_id,
    ts: raw.ts,
    call_id: raw.call_id,
    ok: raw.ok,
    output: isRecord(raw.output) ? raw.output : null,
    error: typeof raw.error === 'string' ? raw.error : null,
    duration_ms: raw.duration_ms
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isRunBase(
  raw: Record<string, unknown>
): raw is Record<string, unknown> & { run_id: string; ts: string } {
  return typeof raw.run_id === 'string' && typeof raw.ts === 'string'
}

function isTier(value: unknown): value is Tier {
  return value === 'T0' || value === 'T1' || value === 'T2'
}

function isToolFields(
  raw: Record<string, unknown>
): raw is Record<string, unknown> & {
  call_id: string
  tool: string
  args: JsonObject
  tier: Tier
} {
  return (
    typeof raw.call_id === 'string' &&
    typeof raw.tool === 'string' &&
    isRecord(raw.args) &&
    isTier(raw.tier)
  )
}

function isOptionalObject(value: unknown): boolean {
  return value === undefined || value === null || isRecord(value)
}

function isOptionalString(value: unknown): boolean {
  return value === undefined || value === null || typeof value === 'string'
}
