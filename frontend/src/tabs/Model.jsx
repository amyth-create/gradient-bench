import React, { useEffect, useState } from 'react'
import { get } from '../api'
import { f } from '../hooks'
import Term from '../components/Term.jsx'
import { Banner, Tag, Tile, Skeleton } from '../components/ui.jsx'
import { LearningCurve, LsBar } from '../components/charts.jsx'

/* The recent trend, not endpoint-to-endpoint: the first build's caption said
   "falling" while the last ten points plainly rose. */
function recentTrend(pts) {
  if (pts.length < 4) return null
  const k = Math.min(8, Math.floor(pts.length / 2))
  const tail = pts.slice(-k).map((p) => p.mean_sd)
  const head = pts.slice(-2 * k, -k).map((p) => p.mean_sd)
  if (!head.length) return null
  const a = head.reduce((s, v) => s + v, 0) / head.length
  const b = tail.reduce((s, v) => s + v, 0) / tail.length
  const d = (b - a) / Math.max(Math.abs(a), 1e-9)
  if (d < -0.05) return `Over the last ${k} methods the mean posterior SD fell ${(-d * 100).toFixed(0)}% — recent runs are still teaching the model.`
  if (d > 0.05) return `Over the last ${k} methods the mean posterior SD ROSE ${(d * 100).toFixed(0)}%. Recent runs disagreed with their neighbours and the noise term absorbed them; read this beside the Instrument tab.`
  return `Over the last ${k} methods the mean posterior SD is flat. The model may have learned what this budget can teach it.`
}

export default function Model({ st }) {
  const [d, setD] = useState(null)
  const [err, setErr] = useState(null)
  useEffect(() => { get('/api/model').then(setD).catch((e) => setErr(e.message)) }, [])
  if (err) return (
    <main><h1>Model</h1>
      <Banner kind="WATCH" eyebrow="diagnostics" title="Could not read the model diagnostics"><p>{err}</p>
        <p className="muted mt2">These are read from the campaign’s uncertainty history file, a derived record — the runs themselves are unaffected.</p></Banner></main>)
  if (!d) return <main><h1>Model</h1><Skeleton /></main>
  const t = d.trend, pts = t.points || []
  const o = d.optimiser || st?.optimiser || {}
  const trend = recentTrend(pts)

  return (
    <main>
      <h1>Model</h1>
      <p className="sub">Whether the campaign is learning, and which parameters it has learned something about. Every number here was computed when a run was recorded and read back from the campaign’s history — opening this page does not refit anything.</p>

      <div className="card">
        <div className="cardhead"><h3>What this campaign is fitted with</h3><Tag><Term k="locked">locked at creation</Term></Tag></div>
        <div className="tiles">
          <Tile label="surrogate" term="surrogate" value={o.kernel_label || d.kernel} note={`BoTorch SingleTaskGP · ${o.kernel || d.kernel}`} />
          <Tile label="acquisition" term="acquisition function" value={o.acquisition_short || 'qLogNEI'} note={o.description || ''} />
          <Tile label="fits recorded" value={d.n_design_runs} unit="design runs" note={`history rows: ${pts.length}`} />
          <Tile label="cold start" term="cold start" value={o.n_seed ?? '—'} unit="methods" note={`${o.n_replicates ?? 2} repeats each`} />
        </div>
      </div>

      <div className="card">
        <h3>Is it learning?</h3>
        <p className="lead"><b>{t.reading}</b></p>
        {pts.length === 0 ? (
          <p className="muted mb0">The history starts once three design runs exist — the fewest a before/after comparison can be made from.</p>
        ) : (
          <>
            <LearningCurve points={pts} />
            {trend && <p className="mt3">{trend}</p>}
            <p className="muted mb0">Mean <Term k="posterior SD">posterior SD</Term> over the campaign’s <Term k="evaluation grid">fixed evaluation grid</Term>, after each recorded method. The grid is fixed and cached on purpose: if the set of points moved between runs, this line would be an artefact of resampling rather than a measure of learning.</p>
          </>
        )}
      </div>

      <div className="card">
        <h3>What the model has learned about each parameter</h3>
        {d.ard.length === 0 ? (
          <p className="muted mb0">Empty until a model has been fitted, which happens on the third recorded design run. It will then show one <Term k="lengthscale">lengthscale</Term> per parameter — how far you have to move along each before the score changes. A short one is a parameter that matters.</p>
        ) : (
          <>
            <p className="muted" style={{ maxWidth: '76ch' }}><Term k="ARD">ARD</Term> <Term k="lengthscale">lengthscales</Term> from the most recent fit, in normalised units — the design space is mapped to 0–1 before fitting, so a lengthscale near 1 means the score changes over about the whole range of that parameter, and one near 0.1 means it changes ten times faster. The bar is capped at 1.</p>
            <div className="tablewrap">
              <table>
                <thead><tr><th className="l">parameter</th><th>lengthscale</th><th className="l" style={{ width: '38%' }}>scale (0–1)</th><th className="l">reading</th></tr></thead>
                <tbody>
                  {d.ard.map((a) => (
                    <tr key={a.name}>
                      <td className="l mono">{a.name}</td>
                      <td>{f(a.value, 3)}</td>
                      <td className="l"><LsBar value={a.value} reading={a.reading} /></td>
                      <td className="l" style={{ whiteSpace: 'normal', fontFamily: 'var(--f-ui)', fontSize: 13 }}>{a.reading}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>

      <div className="card">
        <h3>Noise</h3>
        <div className="tiles">
          <Tile label="measured σ" term="sigma" value={d.noise.sigma_measured == null ? '—' : f(d.noise.sigma_measured, 3)} unit={d.noise.sigma_measured == null ? '' : 'CRF'}
                note={d.noise.sigma_measured == null ? 'no method has two recorded runs yet' : `pooled over ${d.noise.sigma_groups} method(s), ${d.noise.sigma_dof} dof`} />
          <Tile label="fitted noise" value={f(d.noise.noise_sd_crf, 3)} unit="CRF" note="what the last fit settled on" />
          <Tile label="floor_z" term="floor_z" value={f(d.noise.noise_floor_z, 4)} note={d.noise.floor_reading || ''} />
          <Tile label="fallback σ" value={f(d.noise.fallback, 2)} unit="CRF" note="used until a replicate pair lands" />
        </div>
        {d.noise.floor_dominated && (
          <Banner kind="WATCH" eyebrow="noise floor" title="The noise floor is at its ceiling" className="mt4">
            <p>Replicate noise is as large as the entire spread of scores in this campaign, so the posterior is nearly flat and the next method is close to a guess. <b>Do not lower the floor</b> — it is reporting the replicates faithfully.</p>
            <ul>
              <li><b>More replicates.</b> σ is pooled over {d.noise.sigma_groups} method(s) at {d.noise.sigma_dof} dof; more pairs sharpen it.</li>
              <li><b>Explore further apart.</b> Methods further apart in the design space grow the spread the noise is compared against.</li>
            </ul>
          </Banner>
        )}
        <p className="mt3 mb0"><b>Read together with the instrument.</b> {d.campaign_reading}</p>
      </div>
    </main>
  )
}
