import React, { useCallback, useEffect, useState } from 'react'
import { get, post } from '../api'
import { useAsync, f } from '../hooks'
import Term from '../components/Term.jsx'
import { Banner, Busy, Err, Light, Tag, Tile, Skeleton, Segmented } from '../components/ui.jsx'
import { ControlChart, Spark } from '../components/charts.jsx'

export default function Instrument({ reload }) {
  const { busy, err, run } = useAsync()
  const [d, setD] = useState(null)
  const load = useCallback(() => get('/api/drift').then(setD), [])
  useEffect(() => { load() }, [load])
  if (!d) return <main><h1>Instrument</h1><Skeleton /></main>
  const v = d.verdict, inc = d.incumbent
  const fmt = (x, n = 3) => (x == null ? '—' : Number(x).toFixed(n))
  const lampFor = (verdict) => verdict === 'steady' ? 'PASS' : verdict === 'changed' ? 'HALT' : verdict === 'unmeasured' ? '' : 'WATCH'

  return (
    <main>
      <div className="pagehead">
        <div><h1>Instrument</h1>
          <p className="sub mb0">One method held fixed and re-run on a schedule. Its movement can only be the instrument — the only thing in this campaign that separates <Term k="drift">drift</Term> from chemistry.</p></div>
        <Light v={v.verdict}><Term k="verdict">instrument {v.verdict}</Term></Light>
      </div>

      <Banner kind={v.verdict} eyebrow="instrument check" title={v.verdict === 'PASS' ? 'The reference is holding' : v.verdict === 'HALT' ? 'Halt: proposing is blocked until the instrument is serviced and re-anchored' : 'Watch: the reference has moved, or is not yet measured'} actions={v.actions} role={v.verdict === 'HALT' ? 'alert' : undefined} className="mt3">
        <p>{v.reasons.join(' ')}</p>
      </Banner>

      {inc.phantom && (
        <Banner kind="HALT" eyebrow="incumbent" title="Phantom incumbent — the raw and drift-adjusted winners are different methods">
          <p>{inc.warning}</p>
          <div className="tiles mt3">
            <Tile label="raw winner" value={inc.raw_method} note={`${fmt(inc.raw_best)} CRF at run ${inc.raw_run_order}`} />
            <Tile label="adjusted winner" value={inc.adj_method} note={`${fmt(inc.adj_best)} CRF at run ${inc.adj_run_order}`} />
            <Tile label="raw winner, adjusted" value={fmt(inc.adj_of_raw_best)} unit="CRF" note="once the day it ran on is accounted for" />
          </div>
        </Banner>
      )}

      <div className="card">
        <div className="cardhead"><h3><Term k="control chart">Control chart</Term></h3>
          <span className="muted">one reference every {d.limits.every} methods · {d.schedule.reason}</span></div>
        {d.series.length === 0 ? (
          <p className="muted mb0">No reference run yet. The <Term k="anchor">anchor</Term> is the zero every later reference is a difference from, and it cannot be added afterwards.</p>
        ) : (
          <>
            <ControlChart series={d.series} verdict={v} limits={d.limits} />
            <div className="tiles mt3">
              <Tile label="anchor" term="anchor" value={fmt(v.anchor, 2)} unit="CRF" note="the first reference" />
              <Tile label="latest" value={fmt(v.last, 2)} unit="CRF" note={`${v.delta > 0 ? '+' : ''}${fmt(v.delta, 2)} from the anchor`} />
              <Tile label="trend" value={fmt(v.slope, 3)} unit="CRF/run" note={Number.isFinite(v.implied_fall) ? `implying ${fmt(v.implied_fall, 2)} over ${fmt(v.span, 0)} runs` : 'least squares through the references'} />
              <Tile label="limits" value={`−${d.limits.watch} / −${d.limits.halt}`} note={<>WATCH / HALT below the anchor, or {d.limits.monotone_k} <Term k="monotone">consecutive falls</Term></>} />
            </div>
          </>
        )}
      </div>

      <div className="card">
        <h3>What moved, and what that implicates</h3>
        <p className="muted" style={{ maxWidth: '76ch' }}>The control chart says <i>that</i> something moved. These channels are read from descriptors already recorded on every run — on the <b>reference runs only</b>, because a design run’s peaks change when the method changes. Each reading is a hypothesis, not a diagnosis.</p>
        <p className="lead"><b>{d.channel_summary.line}</b></p>
        <div className="stack">
          {d.channels.map((ch) => (
            <details key={ch.key} className="detail" open={ch.verdict !== 'steady' && ch.verdict !== 'unmeasured'}>
              <summary style={{ cursor: 'pointer', listStyle: 'none' }}>
                <div className="row between">
                  <div><b>{ch.label}</b> <span className="muted">— implicates {ch.implicates}</span></div>
                  <Light v={lampFor(ch.verdict)}>{ch.verdict}</Light>
                </div>
                <p className="muted mb0 mt1">{ch.headline}</p>
              </summary>
              {ch.verdict === 'unmeasured' && <p className="muted mt2">The dashes are not zeros — a channel needs at least two reference runs before a change can be seen at all.</p>}
              <div className="tablewrap mt2">
                <table>
                  <thead><tr><th className="l">descriptor</th><th>anchor</th><th>latest</th><th>change</th><th>per run</th><th className="l">shape</th></tr></thead>
                  <tbody>
                    {ch.descriptors.map((x) => (
                      <tr key={x.key}>
                        <td className="l mono">{x.label}{x.unit ? ` (${x.unit})` : ''}</td>
                        <td>{x.anchor == null ? '—' : Number(x.anchor).toPrecision(4)}</td>
                        <td>{x.last == null ? '—' : Number(x.last).toPrecision(4)}</td>
                        <td>{x.rel_change == null ? '—' : `${x.rel_change > 0 ? '+' : ''}${(x.rel_change * 100).toFixed(1)}%`}</td>
                        <td>{x.slope == null ? '—' : Number(x.slope).toPrecision(3)}</td>
                        <td className="l"><Spark d={x} />{' '}
                          {x.step_at != null && <Tag kind="watch"><Term k="step">step at run {x.step_at}</Term></Tag>}
                          {x.monotone && x.step_at == null && <Tag><Term k="monotone">monotone</Term></Tag>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <details className="disclose mt2"><summary>why this channel means what it means</summary>
                <p style={{ maxWidth: '74ch' }}>{ch.why}</p>
                <b className="muted">If it is flagged</b>
                <ul>{ch.then.map((t, i) => <li key={i}>{t}</li>)}</ul>
              </details>
            </details>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="cardhead"><h3><Term k="drift-adjusted">Drift-adjusted reporting</Term></h3>
          <Segmented label="drift-adjusted reporting" value={d.use_drift_adjusted ? 'on' : 'off'}
                     onChange={(vv) => run('switching', () => post('/api/toggle', { key: 'use_drift_adjusted', value: vv === 'on' }).then(() => Promise.all([load(), reload()])))}
                     options={['off', 'on']} /></div>
        <p className="muted" style={{ maxWidth: '74ch' }}><b>Adjusting assumes the drift is reversible, and nothing here can check that.</b> Correcting scores for a column that has irreversibly degraded produces numbers the instrument can no longer produce. That is why it is off by default.</p>
        <p className="muted mb0" style={{ maxWidth: '74ch' }}>It changes what is <i>reported</i> as best. It never changes what the model is fitted to: the <Term k="GP">GP</Term> always sees raw scores.</p>
        <Busy label={busy} /><Err>{err}</Err>
      </div>
    </main>
  )
}
