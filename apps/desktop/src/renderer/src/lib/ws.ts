import { parseServerMessage, type ServerMessage } from './protocol'

export type WsStatus = 'connecting' | 'connected' | 'disconnected'

export interface AtlasWsClient {
  send: (json: unknown) => void
  close: () => void
}

export interface AtlasWsHandlers {
  onStatus: (status: WsStatus) => void
  onMessage: (message: ServerMessage) => void
}

const BACKOFF_MS = [1000, 2000, 4000] as const
const MAX_BACKOFF_MS = 5000

function backoffMs(failCount: number): number {
  if (failCount < BACKOFF_MS.length) {
    return BACKOFF_MS[failCount]
  }
  return MAX_BACKOFF_MS
}

export function connectAtlasWs(url: string, handlers: AtlasWsHandlers): AtlasWsClient {
  let socket: WebSocket | null = null
  let closed = false
  let failCount = 0
  let retryTimer: ReturnType<typeof setTimeout> | undefined

  const clearRetry = (): void => {
    if (retryTimer !== undefined) {
      clearTimeout(retryTimer)
      retryTimer = undefined
    }
  }

  const open = (): void => {
    if (closed) {
      return
    }
    handlers.onStatus('connecting')
    const ws = new WebSocket(url)
    socket = ws

    ws.addEventListener('open', () => {
      if (closed || socket !== ws) {
        return
      }
      failCount = 0
      handlers.onStatus('connected')
    })

    ws.addEventListener('message', (event: MessageEvent<string>) => {
      if (closed || socket !== ws) {
        return
      }
      let payload: unknown
      try {
        payload = JSON.parse(event.data) as unknown
      } catch {
        console.warn('ignored malformed websocket payload', event.data)
        return
      }
      const message = parseServerMessage(payload)
      if (message === null) {
        console.warn('ignored unknown websocket message', payload)
        return
      }
      handlers.onMessage(message)
    })

    ws.addEventListener('close', () => {
      if (socket !== ws) {
        return
      }
      socket = null
      if (closed) {
        return
      }
      handlers.onStatus('disconnected')
      const delay = backoffMs(failCount)
      failCount += 1
      clearRetry()
      retryTimer = setTimeout(open, delay)
    })
  }

  open()

  return {
    send: (json: unknown) => {
      if (socket === null || socket.readyState !== WebSocket.OPEN) {
        return
      }
      socket.send(JSON.stringify(json))
    },
    close: () => {
      closed = true
      clearRetry()
      if (socket !== null) {
        socket.close()
        socket = null
      }
    }
  }
}
