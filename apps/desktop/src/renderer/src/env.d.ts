/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ATLAS_BACKEND_WS?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
