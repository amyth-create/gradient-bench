import React, { useState } from 'react'
import Term from './Term.jsx'
import { ticks } from './Chromatogram.jsx'
import { f } from '../hooks'

/* Every chart: Tier-3 well, grid token, mono axis text, axis titles, nice
   ticks. Designed at n = 40, because a 40-method product must read at 40. */

const T = ({ x, y, anchor = 'middle', children, style, dy }) => (
  <text x={x} y={y} dy={dy} textAnchor={anchor} className="svgtext" style={style}>{children}</text>
)

/** The learning curve: mean posterior SD over the fixed grid after each fit. */
export function LearningCurve({ points, height = 260 }) {
  const [hover, setHover] = useState(null)
  const pts = points || []
  const W = 860, L = 62, R = 24, TT = 16, B = 44
  const iw = W - L - R, ih = height - TT - B
  if (!pts.length) return null
  const ys = pts.map((p) => p.mean_sd).filter((v) => v != null)
  const ymax = Math.max(...ys, 0.01) * 1.1
  const yt = ticks(0, ymax, 5)
  const X = (i) => (pts.length < 2 ? L + iw / 2 : L + (i / (pts.length - 1)) * iw)
  const Y = (v) => TT + ih - (v / ymax) * ih
  const every = pts.length > 24 ? 5 : pts.length > 12 ? 2 : 1
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label="learning curve: mean posterior SD after each recorded method"
           onMouseLeave={() => setHover(null)}>
        {yt.map((v) => (
          <g key={v}>
            <line x1={L} x2={L + iw} y1={Y(v)} y2={Y(v)} stroke="var(--grid)" />
            <T x={L - 8} y={Y(v) + 3.5} anchor="end">{v.toFixed(yt[1] < 1 ? 2 : 1)}</T>
          </g>
        ))}
        <line x1={L} x2={L + iw} y1={TT + ih} y2={TT + ih} stroke="var(--axis)" />
        <line x1={L} x2={L} y1={TT} y2={TT + ih} stroke="var(--axis)" />
        <polyline fill="none" stroke="var(--series)" strokeWidth="2.2" strokeLinejoin="round"
                  points={pts.map((p, i) => `${X(i)},${Y(p.mean_sd)}`).join(' ')} />
        {pts.map((p, i) => (
          <g key={i} onMouseEnter={() => setHover(i)}>
            <rect x={X(i) - iw / (2 * Math.max(pts.length - 1, 1))} y={TT} width={iw / Math.max(pts.length - 1, 1)} height={ih} fill="transparent" />
            <circle cx={X(i)} cy={Y(p.mean_sd)} r={hover === i ? 5 : 3.5} fill="var(--surface)" stroke="var(--series)" strokeWidth="1.8" />
            {((i + 1) % every === 0 || i === 0) && (
              <T x={X(i)} y={TT + ih + 16}>{p.method}</T>
            )}
          </g>
        ))}
        {hover != null && (
          <g>
            <line x1={X(hover)} x2={X(hover)} y1={TT} y2={TT + ih} stroke="var(--accent)" strokeDasharray="2 3" />
            <rect x={Math.min(X(hover) + 8, W - 200)} y={TT + 4} width={192} height={38} rx="6" fill="var(--surface)" stroke="var(--rule-soft)" />
            <T x={Math.min(X(hover) + 16, W - 192)} y={TT + 19} anchor="start" style={{ fill: 'var(--ink)' }}>
              {pts[hover].method} · {pts[hover].n_runs} runs</T>
            <T x={Math.min(X(hover) + 16, W - 192)} y={TT + 34} anchor="start">
              SD {f(pts[hover].mean_sd, 3)} (was {f(pts[hover].mean_sd_before, 3)})</T>
          </g>
        )}
        <T x={L + iw / 2} y={height - 6}>method, in the order recorded{every > 1 ? ` (every ${every}th labelled)` : ''}</T>
        <text transform={`rotate(-90 14 ${TT + ih / 2})`} x="14" y={TT + ih / 2} textAnchor="middle" className="svgtext">mean posterior SD (CRF)</text>
      </svg>
    </div>
  )
}

/** The control chart: reference CRF against the instrument position it ran at. */
export function ControlChart({ series, verdict, limits, height = 280 }) {
  const W = 860, L = 58, R = 104, TT = 18, B = 42
  if (!series?.length) return null
  const iw = W - L - R, ih = height - TT - B
  const anchor = verdict.anchor
  const xs = series.map((s) => s.run_order)
  const x0 = 0, x1 = Math.max(...xs, 5)
  const lo = Math.min(anchor - limits.halt - 1.5, ...series.map((s) => s.crf - 0.5))
  const hi = Math.max(anchor + 2, ...series.map((s) => s.crf + 1))
  const X = (v) => L + ((v - x0) / Math.max(x1 - x0, 1)) * iw
  const Y = (v) => TT + ih - ((v - lo) / Math.max(hi - lo, 1e-9)) * ih
  const band = (a, b, fill, key) => (
    <rect key={key} x={L} y={Y(Math.max(a, b))} width={iw} height={Math.max(Math.abs(Y(a) - Y(b)), 0)} fill={fill} fillOpacity="0.45" />
  )
  const trend = Number.isFinite(verdict.slope) && series.length > 1
  const yt = ticks(lo, hi, 5)
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`control chart, verdict ${verdict.verdict}`}>
        {band(anchor - limits.halt, anchor - limits.watch, 'var(--watch-soft)', 'w')}
        {band(lo, anchor - limits.halt, 'var(--halt-soft)', 'h')}
        {yt.map((v) => (
          <g key={v}>
            <line x1={L} x2={L + iw} y1={Y(v)} y2={Y(v)} stroke="var(--grid)" />
            {Math.abs(Y(v) - Y(anchor)) > 9 && <T x={L - 8} y={Y(v) + 3.5} anchor="end">{v.toFixed(0)}</T>}
          </g>
        ))}
        {[['anchor', anchor, 'var(--ink-soft)', ''], ['watch', anchor - limits.watch, 'var(--watch)', '4 3'],
          ['halt', anchor - limits.halt, 'var(--halt)', '4 3']].map(([lab, v, col, dash]) => (
          <g key={lab}>
            <line x1={L} x2={L + iw} y1={Y(v)} y2={Y(v)} stroke={col} strokeDasharray={dash} strokeWidth="1.1" />
            <T x={L + iw + 8} y={Y(v) + 3.5} anchor="start" style={{ fill: col, fontWeight: 700 }}>{lab} {v.toFixed(2)}</T>
          </g>
        ))}
        <line x1={L} x2={L + iw} y1={TT + ih} y2={TT + ih} stroke="var(--axis)" />
        <line x1={L} x2={L} y1={TT} y2={TT + ih} stroke="var(--axis)" />
        {trend && (
          <line x1={X(xs[0])} x2={X(xs[xs.length - 1])}
                y1={Y(verdict.intercept + verdict.slope * xs[0])}
                y2={Y(verdict.intercept + verdict.slope * xs[xs.length - 1])}
                stroke="var(--series)" strokeWidth="1.4" strokeDasharray="6 4" />
        )}
        <polyline fill="none" stroke="var(--ink)" strokeWidth="1.6" points={series.map((s) => `${X(s.run_order)},${Y(s.crf)}`).join(' ')} />
        {series.map((s, i) => (
          <g key={i}>
            <circle cx={X(s.run_order)} cy={Y(s.crf)} r="4.5" fill="var(--surface)" stroke="var(--ink)" strokeWidth="1.6" />
            <T x={X(s.run_order)} y={Y(s.crf) - 10} style={{ fill: 'var(--ink-soft)' }}>{s.crf.toFixed(2)}</T>
            <T x={X(s.run_order)} y={TT + ih + 15}>{s.name}</T>
            <T x={X(s.run_order)} y={TT + ih + 27} style={{ fontSize: 9.5 }}>run {s.run_order}</T>
          </g>
        ))}
        {trend && <T x={L + iw + 8} y={TT + ih - 4} anchor="start" style={{ fill: 'var(--series)' }}>trend {verdict.slope.toFixed(3)}/run</T>}
        <text transform={`rotate(-90 14 ${TT + ih / 2})`} x="14" y={TT + ih / 2} textAnchor="middle" className="svgtext">reference CRF</text>
        <T x={L + iw / 2} y={height - 4}>instrument run position</T>
      </svg>
    </div>
  )
}

/** The improvement record: a step line for the running best, runs behind it. */
export function BestSoFar({ design, anchor, height = 280 }) {
  const W = 860, L = 52, R = 96, TT = 22, B = 44
  const iw = W - L - R, ih = height - TT - B
  const order = [], byMethod = {}
  design.forEach((r) => {
    const m = r.method.replace(/r\d+$/, '')
    if (!byMethod[m]) { byMethod[m] = []; order.push(m) }
    byMethod[m].push(r)
  })
  const n = order.length
  const ymax = Math.max(...design.map((r) => r.CRF ?? 0), anchor ?? 0, 1) * 1.12
  const X = (i) => L + ((i + 0.5) / Math.max(n, 1)) * iw
  const Y = (v) => TT + ih - (v / ymax) * ih
  const yt = ticks(0, ymax, 5)
  let best = -Infinity
  const steps = [], records = []
  order.forEach((m, i) => {
    const mx = Math.max(...byMethod[m].map((r) => r.CRF ?? -Infinity))
    if (mx > best) { best = mx; records.push({ i, v: mx, m }) }
    steps.push(best)
  })
  let path = `M${X(0).toFixed(1)} ${Y(steps[0]).toFixed(1)}`
  for (let i = 1; i < n; i++) {
    path += ` H${X(i).toFixed(1)}`
    if (steps[i] !== steps[i - 1]) path += ` V${Y(steps[i]).toFixed(1)}`
  }
  path += ` H${(L + iw).toFixed(1)}`
  const labelled = []
  records.forEach((r, k) => {
    if (k !== records.length - 1 && (anchor == null || r.v < anchor)) return
    const x = X(r.i)
    let y = Y(r.v) - 9
    labelled.forEach((p) => { if (Math.abs(p.x - x) < 80 && Math.abs(p.y - y) < 16) y = p.y - 15 })
    labelled.push({ ...r, x, y })
  })
  const every = n > 24 ? 5 : n > 12 ? 2 : 1
  return (
    <div>
      <div className="chart">
        <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label="best score so far, by method">
          {yt.map((v) => (
            <g key={v}>
              <line x1={L} x2={L + iw} y1={Y(v)} y2={Y(v)} stroke="var(--grid)" />
              <T x={L - 8} y={Y(v) + 3.5} anchor="end">{v}</T>
            </g>
          ))}
          <line x1={L} x2={L + iw} y1={TT + ih} y2={TT + ih} stroke="var(--axis)" />
          <line x1={L} x2={L} y1={TT} y2={TT + ih} stroke="var(--axis)" />
          {anchor != null && (
            <g>
              <line x1={L} x2={L + iw} y1={Y(anchor)} y2={Y(anchor)} stroke="var(--ink-soft)" strokeDasharray="5 4" strokeWidth="1.1" />
              <T x={L + iw + 6} y={Y(anchor) + 3.5} anchor="start" style={{ fill: 'var(--ink-soft)' }}>anchor {f(anchor, 2)}</T>
            </g>
          )}
          {order.map((m, i) => {
            const rs = byMethod[m], vs = rs.map((r) => r.CRF ?? 0)
            return (
              <g key={m}>
                {vs.length > 1 && <line x1={X(i)} x2={X(i)} y1={Y(Math.min(...vs))} y2={Y(Math.max(...vs))} stroke="var(--rule)" strokeWidth="1.4" />}
                {rs.map((r, k) => (
                  <circle key={k} cx={X(i)} cy={Y(r.CRF ?? 0)} r="2.8" fill="var(--surface)" stroke="var(--ink-faint)" strokeWidth="1.2">
                    <title>{r.method} — CRF {f(r.CRF, 2)}</title>
                  </circle>
                ))}
              </g>
            )
          })}
          {n > 0 && <path d={path} fill="none" stroke="var(--series)" strokeWidth="2.2" strokeLinejoin="round" />}
          {records.map((r) => (
            <circle key={r.m} cx={X(r.i)} cy={Y(r.v)} r="4.5" fill="var(--series)" stroke="var(--surface)" strokeWidth="2">
              <title>new best — {r.m} at CRF {f(r.v, 2)}</title>
            </circle>
          ))}
          {labelled.map((r) => (
            <T key={r.m} x={r.x} y={r.y} style={{ fill: 'var(--ink)' }}>{r.m} · {f(r.v, 1)}</T>
          ))}
          {order.map((m, i) => ((i + 1) % every === 0 || i === 0 ? (
            <T key={m} x={X(i)} y={TT + ih + 15}>{m}</T>
          ) : null))}
          <T x={L + iw / 2} y={height - 5}>methods, in the order they were run</T>
          <text transform={`rotate(-90 12 ${TT + ih / 2})`} x="12" y={TT + ih / 2} textAnchor="middle" className="svgtext">CRF</text>
        </svg>
      </div>
      <div className="legend mt2">
        <span><svg width="16" height="8"><line x1="0" x2="16" y1="4" y2="4" stroke="var(--series)" strokeWidth="2.2" /></svg>best so far — steps up only when a run breaks the record</span>
        <span><svg width="10" height="10"><circle cx="5" cy="5" r="2.8" fill="var(--surface)" stroke="var(--ink-faint)" strokeWidth="1.2" /></svg>individual runs, replicate pairs joined</span>
        {anchor != null && <span><svg width="16" height="8"><line x1="0" x2="16" y1="4" y2="4" stroke="var(--ink-soft)" strokeWidth="1.2" strokeDasharray="4 3" /></svg>anchor method</span>}
      </div>
    </div>
  )
}

/** Two chromatograms on shared axes. */
export function Overlay({ a, b, height = 300 }) {
  const PAD = { l: 58, r: 16, t: 16, b: 38 }
  const W = 880
  if (!a) return null
  const series = [a, b].filter(Boolean)
  const iw = W - PAD.l - PAD.r, ih = height - PAD.t - PAD.b
  const x1 = Math.max(...series.map((s) => s.t[s.t.length - 1]))
  const ymax = Math.max(...series.map((s) => Math.max(...s.y))) * 1.06 || 1
  const X = (t) => PAD.l + (t / x1) * iw
  const Y = (v) => PAD.t + ih - (v / ymax) * ih
  const COL = ['var(--series)', 'var(--series-2)']
  return (
    <div>
      <div className="chart">
        <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label="two chromatograms on shared axes">
          {ticks(0, ymax, 5).map((v) => (
            <g key={v}>
              <line x1={PAD.l} x2={PAD.l + iw} y1={Y(v)} y2={Y(v)} stroke="var(--grid)" />
              <T x={PAD.l - 7} y={Y(v) + 3.5} anchor="end">{v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0)}</T>
            </g>
          ))}
          {ticks(0, x1, 9).map((t) => <T key={t} x={X(t)} y={PAD.t + ih + 15}>{+t.toFixed(1)}</T>)}
          <line x1={PAD.l} x2={PAD.l + iw} y1={PAD.t + ih} y2={PAD.t + ih} stroke="var(--axis)" />
          {series.map((s, i) => (
            <polyline key={i} fill="none" stroke={COL[i]} strokeWidth="1.3" strokeOpacity={i === 0 ? 0.95 : 0.85}
                      points={s.t.map((t, k) => `${X(t).toFixed(1)},${Y(s.y[k]).toFixed(1)}`).join(' ')} />
          ))}
          <T x={PAD.l + iw / 2} y={height - 4}>retention time (min)</T>
          <text transform={`rotate(-90 12 ${PAD.t + ih / 2})`} x="12" y={PAD.t + ih / 2} textAnchor="middle" className="svgtext">detector signal</text>
        </svg>
      </div>
      <div className="legend mt2">
        {series.map((s, i) => (
          <span key={i}><i style={{ background: COL[i] }} />{s.method} — CRF {f(s.recorded_crf, 3)}, {s.n_clean} clean</span>
        ))}
      </div>
    </div>
  )
}

/** A descriptor's series across the reference runs. */
export function Spark({ d, width = 132, height = 30 }) {
  if (!d.values?.length || d.n < 2) return <span className="muted">—</span>
  const v = d.values, lo = Math.min(...v), hi = Math.max(...v)
  const X = (i) => (i / (v.length - 1)) * (width - 4) + 2
  const Y = (x) => height - 3 - ((x - lo) / Math.max(hi - lo, 1e-12)) * (height - 6)
  return (
    <svg width={width} height={height} style={{ verticalAlign: 'middle' }} role="img" aria-label={`${d.n} readings`}>
      {d.step_at != null && (
        <line x1={X(d.run_order.indexOf(d.step_at))} x2={X(d.run_order.indexOf(d.step_at))} y1="0" y2={height} stroke="var(--watch)" strokeDasharray="2 2" />
      )}
      <polyline fill="none" stroke="var(--series)" strokeWidth="1.5" points={v.map((x, i) => `${X(i)},${Y(x)}`).join(' ')} />
      {v.map((x, i) => <circle key={i} cx={X(i)} cy={Y(x)} r="1.8" fill="var(--series)" />)}
    </svg>
  )
}

/** ARD lengthscale bar, capped at 1.0 with an overflow mark. */
export function LsBar({ value, reading }) {
  const cap = 1.0
  const col = reading.startsWith('at the floor') ? 'var(--halt)' : reading.startsWith('inert') ? 'var(--ink-faint)' : 'var(--series)'
  const w = Math.max(3, Math.min(100, (value / cap) * 100))
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, width: '100%' }}>
      <span style={{ flex: 1, height: 8, borderRadius: 2, background: 'var(--container-highest)', position: 'relative' }}>
        <span style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${w}%`, borderRadius: 2, background: col }} />
      </span>
      {value > cap && <span className="svgtext" style={{ fontFamily: 'var(--f-mono)', fontSize: 10.5, color: 'var(--ink-faint)' }}>&gt;1</span>}
    </span>
  )
}

export { Term }
