import { create } from 'zustand'

import type { WsStatus } from '../lib/ws'

export interface EchoItem {
  id: number
  payload: unknown
}

interface ConnectionState {
  status: WsStatus
  lastHello: unknown | null
  echoes: EchoItem[]
  setStatus: (status: WsStatus) => void
  setLastHello: (payload: unknown) => void
  addEcho: (payload: unknown) => void
}

let echoSeq = 0

export const useConnectionStore = create<ConnectionState>((set) => ({
  status: 'connecting',
  lastHello: null,
  echoes: [],
  setStatus: (status) => set({ status }),
  setLastHello: (lastHello) => set({ lastHello }),
  addEcho: (payload) =>
    set((state) => ({
      echoes: [...state.echoes, { id: ++echoSeq, payload }].slice(-20)
    }))
}))
