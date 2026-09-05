import React, { useCallback, useEffect, useRef, useState } from 'react'
import Term from './Term.jsx'
import Icon from './Icon.jsx'
import { Segmented } from './ui.jsx'

const PAD = { l: 58, r: 16, t: 18, b: 40 }
export const CAT = { clean: 'var(--clean)', shoulder: 'var(--shoulder)', hump: 'var(--onhump)' }
const GRAB = 8          // px either side of an edge that counts as grabbing it
const MIN_SPAN = 0.05   // min, below which a drawn region is a stray click

export function ticks(lo, hi, target) {
  const span = hi - lo
  if (!(span > 0)) return [lo]
  const raw = span / target
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const n = raw / mag
  const step = (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag
  const out = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step)
    out.push(Math.round(v / step) * step)
  return out
}

/** Peak category carries a SHAPE as well as a colour: circle = clean,
 *  triangle = shoulder, square = on-hump. A correctness requirement. */
export function Marker({ cat, x, y, r = 3.6 }) {
  const stroke = CAT[cat] || 'var(--ink-soft)'
  const common = { fill: 'var(--surface)', stroke, strokeWidth: 1.5 }
  if (cat === 'shoulder') {
    const p = `${x},${y - r * 1.15} ${x - r},${y + r * 0.8} ${x + r},${y + r * 0.8}`
    return <polygon points={p} {...common} />
  }
  if (cat === 'hump') {
    return <rect x={x - r * 0.9} y={y - r * 0.9} width={r * 1.8} height={r * 1.8} {...common} />
  }
  return <circle cx={x} cy={y} r={r} {...common} />
}

export function LegendKey({ cat, label, term }) {
  return (
    <span>
      <svg width="13" height="13" style={{ verticalAlign: '-2px', marginRight: 5 }}>
        <Marker cat={cat} x={6.5} y={6.5} r={4} />
      </svg><Term k={term || label}>{label}</Term>
    </span>
  )
}

/**
 * The chromatogram, and the interaction the review screen turns on: drag
 * across the trace to draw an unresolved region, drag either edge of a drawn
 * region to adjust it, remove one with its handle. Every change is committed
 * on RELEASE, because each re-runs the picker on the server; while the pick
 * is in flight the drawn regions stay where the analyst put them and the
 * counts show as provisional.
 */
export default function Chromatogram({ data, manualHumps, onDrawHump, onEditHump,
                                       onDeleteHump, onAdoptHumps, onClearHumps,
                                       height = 340, busy = false, showRaw = false }) {
  const wrapRef = useRef(null)
  const svgRef = useRef(null)
  const [w, setW] = useState(880)
  const [scale, setScale] = useState('linear')
  const [hover, setHover] = useState(null)
  const [drag, setDrag] = useState(null)
  const [edit, setEdit] = useState(null)

  useEffect(() => {
    if (!wrapRef.current || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => setW(Math.max(wrapRef.current.clientWidth - 16, 320)))
    ro.observe(wrapRef.current)
    setW(Math.max(wrapRef.current.clientWidth - 16, 320))
    return () => ro.disconnect()
  }, [])

  /* The trace draws itself in once, on first view, and never again. Nothing is
     hidden until JS adds .arm, and the dash is stripped once the draw is done. */
  useEffect(() => {
    const svg = svgRef.current
    const line = svg && svg.querySelector('polyline[data-draw]')
    if (!svg || !line || typeof line.getTotalLength !== 'function') return
    let len
    try { len = Math.ceil(line.getTotalLength()) } catch { return }
    if (!len || !isFinite(len)) return
    line.style.setProperty('--len', String(len))
    svg.classList.add('arm')
    requestAnimationFrame(() => requestAnimationFrame(() => svg.classList.add('drawn')))
    const done = setTimeout(() => { svg.classList.remove('arm'); line.style.removeProperty('--len') }, 1200)
    return () => clearTimeout(done)
  }, [])

  if (!data) return null
  const sqrt = scale === 'sqrt'
  const iw = w - PAD.l - PAD.r
  const ih = height - PAD.t - PAD.b
  const x0 = data.t[0], x1 = data.t[data.t.length - 1]
  const ty = (v) => (sqrt ? Math.sqrt(Math.max(v, 0)) : v)
  const series = showRaw && data.raw ? data.raw : data.y
  const ymaxRaw = Math.max(...series)
  const ymax = ty(ymaxRaw) * 1.06 || 1
  const X = (t) => PAD.l + ((t - x0) / (x1 - x0)) * iw
  const Y = (v) => PAD.t + ih - (ty(v) / ymax) * ih
  const invX = (px) => x0 + ((px - PAD.l) / iw) * (x1 - x0)
  const perPx = (x1 - x0) / Math.max(iw, 1)

  const toT = useCallback((ev) => {
    const svg = svgRef.current
    const r = svg.getBoundingClientRect()
    const sx = ((ev.clientX - r.left) / r.width) * w
    return Math.min(Math.max(invX(sx), x0), x1)
  }, [w, x0, x1])

  const nearestY = (t) => {
    let lo = 0, hi = data.t.length - 1
    while (hi - lo > 1) { const m = (lo + hi) >> 1; data.t[m] < t ? (lo = m) : (hi = m) }
    return series[Math.abs(data.t[lo] - t) < Math.abs(data.t[hi] - t) ? lo : hi]
  }

  const manual = !!(manualHumps && manualHumps.length)
  const humps = manual ? manualHumps : (data.humps || [])
  const editable = manual && !!onEditHump
  const interactive = !!onDrawHump

  const edgeAt = (t) => {
    if (!editable) return null
    const tol = GRAB * perPx
    for (let i = 0; i < manualHumps.length; i++) {
      const [a, b] = manualHumps[i]
      if (Math.abs(t - a) <= tol) return { i, edge: 0 }
      if (Math.abs(t - b) <= tol) return { i, edge: 1 }
    }
    return null
  }

  const onDown = (ev) => {
    if (!interactive) return
    const t = toT(ev)
    try { svgRef.current.setPointerCapture(ev.pointerId) } catch { /* synthetic or already captured */ }
    const hit = edgeAt(t)
    if (hit) { setEdit({ ...hit, span: [...manualHumps[hit.i]] }); return }
    setDrag([t, t])
  }
  const onMove = (ev) => {
    const t = toT(ev)
    setHover(t)
    if (edit) { const span = [...edit.span]; span[edit.edge] = t; setEdit({ ...edit, span }); return }
    if (drag) setDrag([drag[0], t])
  }
  const onUp = () => {
    if (edit) {
      const [a, b] = [Math.min(...edit.span), Math.max(...edit.span)]
      if (b - a > MIN_SPAN) onEditHump(edit.i, [+a.toFixed(3), +b.toFixed(3)])
      setEdit(null); return
    }
    if (drag) {
      const [a, b] = [Math.min(...drag), Math.max(...drag)]
      if (b - a > MIN_SPAN) onDrawHump && onDrawHump([+a.toFixed(3), +b.toFixed(3)])
    }
    setDrag(null)
  }

  const shown = edit
    ? humps.map((h, i) => (i === edit.i ? [Math.min(...edit.span), Math.max(...edit.span)] : h))
    : humps
  const fmt = (v, d = 2) => Number(v).toFixed(d)
  const hoverEdge = hover != null && !drag && !edit ? edgeAt(hover) : null
  const yTicks = ticks(0, sqrt ? ymaxRaw * 1.06 : ymax, 5)

  return (
    <div>
      <div className="row between" style={{ marginBottom: 8 }}>
        <div className="row" style={{ gap: 10 }}>
          <Segmented label="y-axis scale" value={scale} onChange={setScale}
                     options={[{ value: 'linear', label: 'linear' }, { value: 'sqrt', label: '√ scale' }]} />
          {interactive && manual && (
            <button className="sm" onClick={onClearHumps}><Icon name="undo" />clear drawn regions</button>
          )}
          {interactive && !manual && (data.humps || []).length > 0 && onAdoptHumps && (
            <button className="sm" onClick={() => onAdoptHumps(data.humps)}
                    title="Copy the detected regions so their edges can be dragged. The score does not change until you move one.">
              <Icon name="drag" />adjust detected regions
            </button>
          )}
          <div className="legend">
            <LegendKey cat="clean" label="clean" term="clean peak" />
            <LegendKey cat="shoulder" label="shoulder" />
            <LegendKey cat="hump" label="on-hump" />
            <span><i style={{ background: 'var(--humpfill)', borderRadius: 2, opacity: .5 }} /><Term k="hump">unresolved region</Term></span>
          </div>
        </div>
        <div className="readout" aria-live="off">
          {busy
            ? <><span className="spin" />re-measuring…</>
            : edit
              ? <>adjusting <b>{fmt(Math.min(...edit.span))} – {fmt(Math.max(...edit.span))}</b> min</>
              : drag
                ? <>drawing <b>{fmt(Math.min(...drag))} – {fmt(Math.max(...drag))}</b> min · width <b>{fmt(Math.abs(drag[1] - drag[0]))}</b> min</>
                : hoverEdge
                  ? <>drag to move this edge</>
                  : hover != null
                    ? <><b>{fmt(hover, 3)}</b> min · signal <b>{fmt(nearestY(hover), 1)}</b></>
                    : interactive
                      ? <>drag across the trace to mark a stretch as unresolved</>
                      : <>as measured, under the settings stamped on this run</>}
        </div>
      </div>

      <div className="chart" ref={wrapRef}>
        <svg ref={svgRef} className="chart-svg" viewBox={`0 0 ${w} ${height}`} height={height}
             style={{ cursor: !interactive ? 'default' : hoverEdge ? 'ew-resize' : 'crosshair' }}
             onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}
             onPointerLeave={() => setHover(null)} role="img"
             aria-label={`chromatogram ${data.name || ''}: ${data.n_clean} clean peaks, ${data.n_shoulder} shoulders, ${data.n_on_hump} on a hump`}>
          {ticks(x0, x1, 9).map((t) => (
            <g key={`x${t}`}>
              <line x1={X(t)} x2={X(t)} y1={PAD.t} y2={PAD.t + ih} stroke="var(--grid)" />
              <text x={X(t)} y={PAD.t + ih + 15} textAnchor="middle" className="svgtext">{+t.toFixed(2)}</text>
            </g>
          ))}
          {yTicks.map((v) => {
            const py = Y(v)
            if (py < PAD.t - 1 || py > PAD.t + ih + 1) return null
            return (
              <g key={`y${v}`}>
                <line x1={PAD.l} x2={PAD.l + iw} y1={py} y2={py} stroke="var(--grid)" />
                <text x={PAD.l - 7} y={py + 3.5} textAnchor="end" className="svgtext">{v >= 1000 ? `${(v / 1000).toFixed(v % 1000 ? 1 : 0)}k` : v.toFixed(0)}</text>
              </g>
            )
          })}
          <line x1={PAD.l} x2={PAD.l + iw} y1={PAD.t + ih} y2={PAD.t + ih} stroke="var(--axis)" />
          <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={PAD.t + ih} stroke="var(--axis)" />
          <text x={PAD.l + iw / 2} y={height - 4} textAnchor="middle" className="svgtext">retention time (min)</text>
          <text transform={`rotate(-90 12 ${PAD.t + ih / 2})`} x="12" y={PAD.t + ih / 2} textAnchor="middle" className="svgtext">
            detector signal{sqrt ? ' (√)' : ''}{showRaw ? ' — raw' : ' — baseline-corrected'}
          </text>

          <rect x="0" y="0" width={w} height={height} fill="none" pointerEvents="all" />

          {shown.map(([a, b], i) => {
            const live = edit && edit.i === i
            return (
              <g key={`h${i}`}>
                <rect x={X(a)} y={PAD.t} width={Math.max(X(b) - X(a), 1)} height={ih}
                      fill="var(--humpfill)" fillOpacity={manual ? (live ? 0.3 : 0.2) : 0.14}
                      stroke={manual ? 'var(--humpfill)' : 'none'} strokeDasharray="3 3" />
                <text x={(X(a) + X(b)) / 2} y={PAD.t + 12} textAnchor="middle" className="svgtext"
                      style={{ fill: 'var(--shoulder)', fontWeight: 700 }}>
                  {manual ? 'drawn ' : 'unresolved '}{a.toFixed(2)}–{b.toFixed(2)}
                </text>
                {editable && (
                  <>
                    {[a, b].map((e, k) => (
                      <rect key={k} x={X(e) - GRAB / 2} y={PAD.t} width={GRAB} height={ih}
                            fill="var(--humpfill)"
                            fillOpacity={hoverEdge && hoverEdge.i === i && hoverEdge.edge === k ? 0.6 : 0.3}
                            style={{ cursor: 'ew-resize' }} />
                    ))}
                    <g style={{ cursor: 'pointer' }} role="button" aria-label="remove this region"
                       onPointerDown={(ev) => { ev.stopPropagation(); onDeleteHump && onDeleteHump(i) }}>
                      <circle cx={X(b) - 11} cy={PAD.t + 28} r="8" fill="var(--surface)" stroke="var(--humpfill)" strokeWidth="1.2" />
                      <path d={`M${X(b) - 14} ${PAD.t + 25} l6 6 M${X(b) - 8} ${PAD.t + 25} l-6 6`} stroke="var(--humpfill)" strokeWidth="1.4" fill="none" />
                    </g>
                  </>
                )}
              </g>
            )
          })}
          {drag && (
            <rect x={X(Math.min(...drag))} y={PAD.t} width={Math.max(X(Math.max(...drag)) - X(Math.min(...drag)), 1)} height={ih}
                  fill="var(--accent)" fillOpacity=".14" stroke="var(--accent)" strokeDasharray="3 3" />
          )}
          {showRaw && data.baseline && (
            <polyline fill="none" stroke="var(--ink-faint)" strokeWidth="1" strokeDasharray="4 3"
                      points={data.t.map((t, i) => `${X(t).toFixed(1)},${Y(data.baseline[i]).toFixed(1)}`).join(' ')} />
          )}
          <polyline data-draw fill="none" stroke="var(--ink)" strokeWidth="1.15" strokeOpacity=".78"
                    points={data.t.map((t, i) => `${X(t).toFixed(1)},${Y(series[i]).toFixed(1)}`).join(' ')} />
          {(data.peaks || []).map((p, i) => (
            <g key={`p${i}`} data-pt opacity={busy ? 0.45 : 1}>
              <line x1={X(p.t)} x2={X(p.t)} y1={Y(p.h) - 10} y2={Y(p.h) - 4} stroke={CAT[p.cat]} strokeWidth="1.4" />
              <Marker cat={p.cat} x={X(p.t)} y={Y(p.h)} />
              <title>{p.cat} · {p.t.toFixed(2)} min · height {p.h.toFixed(0)} · FWHM {p.w.toFixed(3)} min</title>
            </g>
          ))}
          {hover != null && (
            <line x1={X(hover)} x2={X(hover)} y1={PAD.t} y2={PAD.t + ih} stroke="var(--accent)" strokeDasharray="2 3" strokeOpacity=".8" pointerEvents="none" />
          )}
        </svg>
      </div>
    </div>
  )
}
