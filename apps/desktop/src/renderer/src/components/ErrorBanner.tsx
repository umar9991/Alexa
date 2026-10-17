import type { JSX } from 'react'

export function ErrorBanner({
  message,
  onDismiss
}: {
  message: string
  onDismiss: () => void
}): JSX.Element {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-red-900/60 bg-red-950/70 px-6 py-2 text-sm text-red-200">
      <p>{message}</p>
      <button
        type="button"
        className="rounded px-2 py-0.5 text-xs text-red-100 ring-1 ring-red-400/40"
        onClick={onDismiss}
      >
        Dismiss
      </button>
    </div>
  )
}
