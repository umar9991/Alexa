import { type JSX, type KeyboardEvent, useState } from 'react'

export function CommandBar({
  disabled,
  canSend,
  showStop,
  onSend,
  onStop
}: {
  disabled: boolean
  canSend: boolean
  showStop: boolean
  onSend: (text: string) => void
  onStop: () => void
}): JSX.Element {
  const [text, setText] = useState('')

  const submit = (): void => {
    const value = text.trim()
    if (value.length === 0 || disabled || !canSend) {
      return
    }
    onSend(value)
    setText('')
  }

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <div className="border-t border-zinc-800 p-4">
      <div className="flex items-end gap-2">
        <textarea
          className="min-h-[44px] max-h-32 flex-1 resize-y rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm outline-none ring-emerald-500/40 focus:ring-2 disabled:opacity-40"
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask Atlas…"
          aria-label="Command"
          disabled={disabled}
          rows={2}
        />
        {showStop ? (
          <button
            type="button"
            className="rounded-lg bg-red-500/20 px-4 py-2 text-sm font-medium text-red-300 ring-1 ring-red-500/40"
            onClick={onStop}
          >
            Stop
          </button>
        ) : (
          <button
            type="button"
            className="rounded-lg bg-zinc-100 px-4 py-2 text-sm font-medium text-zinc-900 disabled:opacity-40"
            onClick={submit}
            disabled={disabled || !canSend}
          >
            Send
          </button>
        )}
      </div>
    </div>
  )
}
