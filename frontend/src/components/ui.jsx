import React, { useEffect, useId, useState } from 'react'
import Icon from './Icon.jsx'
import Term from './Term.jsx'
import { f } from '../hooks'

/* ── banner: eyebrow category, sentence-case headline, body, actions ───── */
export function Banner({ kind = 'INFO', eyebrow, title, children, actions, form, role, className = '' }) {
  return (
    <div className={`banner ${kind} ${className}`} role={role}>
      {eyebrow && <div className="label">{eyebrow}</div>}
      {title && <h4>{title}</h4>}
      {children}
      {actions && actions.length > 0 && <ul>{actions.map((a, i) => <li key={i}>{a}</li>)}</ul>}
      {form}
    </div>
  )
}

export function Err({ children }) {
  if (!children) return null
  return <div className="err" role="alert"><Icon name="warn" />{children}</div>
}

export function Busy({ label }) {
  if (!label) return null
  return <span className="busy"><span className="spin" />{label}</span>
}

/* ── instrument lamp ──────────────────────────────────────────────────── */
export function Light({ v, children, title }) {
  return (
    <span className={`light ${v || ''}`} title={title}>
      <i />{children || v}
    </span>
  )
}

export function Tag({ kind = '', children, title }) {
  return <span className={`tag ${kind}`} title={title}>{children}</span>
}

/* ── the run budget meter ─────────────────────────────────────────────── */
export function Meter({ b, big = false }) {
  const [filled, setFilled] = useState(false)
  useEffect(() => { const r = requestAnimationFrame(() => setFilled(true))
                    return () => cancelAnimationFrame(r) }, [])
  const w = Math.max(2, Math.min(100, (b.fraction || 0) * 100))
  return (
    <span className={`meter ${b.state} ${big ? 'big' : ''}${filled ? ' in' : ''}`} data-fill=""
          role="progressbar" aria-valuemin={0} aria-valuemax={b.limit} aria-valuenow={b.used}
          aria-label={`${b.used} of ${b.limit} methods used`}>
      <i style={{ width: `${w}%` }} />
    </span>
  )
}

/* ── segmented control: a selection, never a primary action ───────────── */
export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => {
        const v = typeof o === 'string' ? o : o.value
        const l = typeof o === 'string' ? o : o.label
        return (
          <button key={v} type="button" aria-pressed={value === v} onClick={() => onChange(v)}>{l}</button>
        )
      })}
    </div>
  )
}

/* ── a labelled field ─────────────────────────────────────────────────── */
export function Field({ label, term, help, error, children, unit, className = '' }) {
  const id = useId()
  const child = React.isValidElement(children)
    ? React.cloneElement(children, { id, 'aria-describedby': help ? `${id}-h` : undefined,
                                     'aria-invalid': error ? true : undefined })
    : children
  return (
    <div className={`field ${className}`}>
      <label htmlFor={id}>{term ? <Term k={term}>{label}</Term> : label}</label>
      {unit ? <div className="inputrow">{child}<span className="unit">{unit}</span></div> : child}
      {help && <div className="help" id={`${id}-h`}>{help}</div>}
      <Err>{error}</Err>
    </div>
  )
}

/* ── number with a slider beside it ───────────────────────────────────── */
export function NumberSlider({ value, onChange, min, max, step, unit, id }) {
  const v = value ?? ''
  return (
    <div className="inputrow">
      <input type="range" min={min} max={max} step={step} value={v === '' ? min : v}
             onChange={(e) => onChange(parseFloat(e.target.value))} aria-hidden="true" tabIndex={-1} />
      <input id={id} type="number" min={min} max={max} step={step} value={v}
             onChange={(e) => onChange(e.target.value === '' ? '' : parseFloat(e.target.value))} />
      {unit && <span className="unit">{unit}</span>}
    </div>
  )
}

/* ── stat tiles ───────────────────────────────────────────────────────── */
export function Tile({ label, term, value, unit, note }) {
  return (
    <div className="tile">
      <div className="label">{term ? <Term k={term}>{label}</Term> : label}</div>
      <div className="v">{value}{unit && <small> {unit}</small>}</div>
      {note && <div className="note">{note}</div>}
    </div>
  )
}

/* ── option card (radio semantics) ────────────────────────────────────── */
export function Choice({ checked, onSelect, name, desc, meta, badge, children }) {
  return (
    <button type="button" role="radio" aria-checked={checked} className="choice" onClick={onSelect}>
      <div className="head">
        <span className="name">{name}</span>
        <span className="row" style={{ gap: 8 }}>
          {badge}
          <span className="radio" aria-hidden="true" />
        </span>
      </div>
      {desc && <div className="desc">{desc}</div>}
      {meta && <div className="meta">{meta}</div>}
      {children}
    </button>
  )
}

/* ── the gradient table an analyst keys into the instrument ───────────── */
export function GradientTable({ rows, T, compact = false }) {
  return (
    <div className={`tablewrap ${compact ? 'tight' : ''}`}>
      <table>
        <thead><tr><th className="l">time (min)</th><th>%B</th><th className="l">phase</th></tr></thead>
        <tbody>
          {(rows || []).map((g, i) => (
            <tr key={i}><td className="l num">{f(g.time_min)}</td>
              <td className="num">{f(g.pct_b, 1)}</td><td className="l">{g.phase}</td></tr>
          ))}
          {T != null && (
            <tr><td className="l" colSpan={2}>column temperature</td><td className="l num">{f(T, 1)} °C</td></tr>
          )}
        </tbody>
      </table>
    </div>
  )
}

/* ── skeleton while a screen reads its data ───────────────────────────── */
export function Skeleton({ rows = 3 }) {
  return (
    <div className="card" aria-busy="true" aria-live="polite">
      <div className="stack">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="skel" style={{ width: `${70 - i * 12}%`, height: i === 0 ? 22 : 14 }} />
        ))}
      </div>
    </div>
  )
}

export function Disclose({ summary, children, open = false }) {
  return (
    <details className="disclose" open={open}>
      <summary>{summary}</summary>
      {children}
    </details>
  )
}
