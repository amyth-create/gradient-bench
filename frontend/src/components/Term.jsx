import React, { useCallback, useEffect, useId, useRef, useState } from 'react'
import { TERMS } from '../glossary'

/* Once one tooltip has opened, its neighbours open with no animation: reading
   a dense page of definitions should not cost 125ms a term. */
let instantUntil = 0
let decay = null
function markOpened() {
  instantUntil = Date.now() + 2500
  document.body.setAttribute('data-tip-instant', '')
  clearTimeout(decay)
  decay = setTimeout(() => {
    if (Date.now() >= instantUntil) document.body.removeAttribute('data-tip-instant')
  }, 2600)
}

/**
 * A glossary term with its definition attached where it appears.
 *
 * Accessibility fixes from the design critique:
 *   · the panel is referenced by aria-describedby from the trigger
 *   · Escape closes it
 *   · terms are NOT in the tab order by default (sixty focus stops before the
 *     Record button is a tax on keyboard users); the "Definitions" switch in
 *     the chrome puts them in, for a reader who wants to walk them.
 * The panel is mounted on open and unmounted on close, so forty hidden 330px
 * boxes never widen the page (a real bug in the first build).
 */
export default function Term({ k, children }) {
  const def = TERMS[k]
  const host = useRef(null)
  const id = useId()
  const [open, setOpen] = useState(false)
  const [flip, setFlip] = useState(false)

  const show = useCallback(() => {
    markOpened()
    const el = host.current
    if (el) setFlip(el.getBoundingClientRect().left + 330 > window.innerWidth - 16)
    setOpen(true)
  }, [])
  const hide = useCallback(() => setOpen(false), [])
  useEffect(() => {
    if (!open) return
    const onKey = (e) => { if (e.key === 'Escape') hide() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, hide])

  if (!def) return <>{children || k}</>
  const focusable = document.body.getAttribute('data-defs') !== 'off'
  return (
    <span className="term" tabIndex={focusable ? 0 : -1} ref={host}
          aria-describedby={open ? id : undefined}
          onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}>
      {children || k}
      {open && (
        <span className={`tip${flip ? ' flip' : ''}`} role="tooltip" id={id}>
          <b>{k}</b>{def}
        </span>
      )}
    </span>
  )
}
