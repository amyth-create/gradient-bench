import { useCallback, useEffect, useState } from 'react'

/* One async runner per screen. `busy` is the label shown beside a spinner,
   `err` the inline error. The live region in App.jsx mirrors both, so a
   screen reader hears "recording…" and "recorded" without a focus change. */
export function useAsync() {
  const [busy, setBusy] = useState(null)
  const [err, setErr] = useState(null)
  const run = useCallback(async (label, fn) => {
    setErr(null); setBusy(label)
    announce(label)
    try { const r = await fn((p) => { setBusy(p || label); announce(p || label) }); announce(`${label}: done`); return r }
    catch (e) { setErr(e.message); announce(`Failed: ${e.message}`); throw e }
    finally { setBusy(null) }
  }, [])
  return { busy, err, setErr, run }
}

const listeners = new Set()
export function announce(text) { listeners.forEach((l) => l(text)) }
export function useAnnouncements() {
  const [msg, setMsg] = useState('')
  useEffect(() => { listeners.add(setMsg); return () => listeners.delete(setMsg) }, [])
  return msg
}

/* Per-browser conveniences. Never for anything that must persist. */
export function useLocal(key, initial) {
  const [v, setV] = useState(() => {
    try { const s = localStorage.getItem(key); return s == null ? initial : JSON.parse(s) }
    catch { return initial }
  })
  const set = useCallback((nv) => {
    setV(nv)
    try { localStorage.setItem(key, JSON.stringify(nv)) } catch { /* private window */ }
  }, [key])
  return [v, set]
}

/* system | light | dark, stamped on <html data-theme>. "system" stamps
   nothing so prefers-color-scheme decides, which is the product default. */
export function useTheme() {
  const [theme, setTheme] = useLocal('gb2.theme', 'system')
  useEffect(() => {
    const el = document.documentElement
    if (theme === 'light' || theme === 'dark') el.setAttribute('data-theme', theme)
    else el.removeAttribute('data-theme')
  }, [theme])
  return [theme, setTheme]
}

export const f = (v, d = 2) => (v == null || Number.isNaN(Number(v)) ? '—' : Number(v).toFixed(d))
export const pct = (v, d = 1) => (v == null ? '—' : (Number(v) * 100).toFixed(d))
export const plural = (n, one, many) => `${n} ${n === 1 ? one : (many || one + 's')}`
