import React, { useEffect, useState } from 'react'
import { get, runJob } from '../api'
import { useAsync, f, pct, plural } from '../hooks'
import Term from '../components/Term.jsx'
import Icon from '../components/Icon.jsx'
import Chromatogram from '../components/Chromatogram.jsx'
import { Banner, Busy, Err, Tag, Meter, Segmented, GradientTable, Tile, Skeleton } from '../components/ui.jsx'
import { BestSoFar, Overlay } from '../components/charts.jsx'

function SortTh({ k, label, sort, setSort, first = false }) {
  const active = sort.key === k
  return (
    <th className={first ? 'l' : ''} aria-sort={active ? (sort.dir < 0 ? 'descending' : 'ascending') : 'none'}>
      <button type="button" onClick={() => setSort({ key: k, dir: active ? -sort.dir : -1 })}>
        {label}{active && <Icon name={sort.dir < 0 ? 'chevd' : 'up'} size={11} />}
      </button>
    </th>
  )
}

export default function Results({ st }) {
  const { busy, err, run } = useAsync()
  const [d, setD] = useState(null)
  const [ref, setRef] = useState(null)
  const [meth, setMeth] = useState(null)
  const [view, setView] = useState('methods')
  const [sort, setSort] = useState({ key: 'crf_mean', dir: -1 })
  const [only, setOnly] = useState('all')
  const [best, setBest] = useState(null)
  const [cmp, setCmp] = useState({ a: '', b: '' })
  const [traces, setTraces] = useState({})
  const [exported, setExported] = useState(null)
  const b = st?.budget
  useEffect(() => {
    get('/api/runs').then(setD); get('/api/reference').then(setRef)
    get('/api/methods').then((r) => {
      setMeth(r)
      const bm = r.summary?.best_method
      if (bm) get(`/api/trace/${encodeURIComponent(bm)}`).then((t) => setBest(t.trace)).catch(() => {})
    })
  }, [])
  const loadTrace = (m) => {
    if (!m || traces[m]) return
    get(`/api/trace/${encodeURIComponent(m)}`).then((t) => setTraces((p) => ({ ...p, [m]: t.trace }))).catch(() => {})
  }
  useEffect(() => { loadTrace(cmp.a); loadTrace(cmp.b) }, [cmp.a, cmp.b])   // eslint-disable-line
  if (!d) return <main><h1>Results</h1><Skeleton /></main>
  const design = d.rows.filter((r) => r.source === 'design')
  const bestBase = meth?.summary?.best_method?.replace(/r\d+$/, '')
  const sorted = [...(meth?.methods || [])].sort((x, y) => {
    const a = x[sort.key], b2 = y[sort.key]
    if (a == null) return 1
    if (b2 == null) return -1
    return (a > b2 ? 1 : a < b2 ? -1 : 0) * sort.dir
  })

  return (
    <main>
      <h1>Results</h1>
      <p className="sub">Every run, in the order the instrument saw them. Reference runs are shown in place because they occupy real positions on the clock, but they are excluded from the model by construction.</p>

      {d.phantom && (
        <Banner kind="HALT" eyebrow="incumbent" title="Phantom incumbent — the raw and drift-adjusted winners differ" role="alert">
          <p>{d.phantom_warning}</p>
          <p className="muted mt2">The Instrument tab shows the control chart this comes from.</p>
        </Banner>
      )}

      {design.length === 0 && (
        <div className="card">
          <h3>No design runs yet</h3>
          <p>This page fills up as methods are recorded. Once there are runs it shows the best <Term k="CRF">separation score</Term> so far and the chromatogram behind it, every method with its <Term k="within-method SD">replicate spread</Term>, and any two runs drawn on shared axes.</p>
          <p className="muted mb0">The loop starts on the <b>Run</b> tab, with the <Term k="anchor">instrument-check anchor</Term>.</p>
        </div>
      )}

      {design.length > 0 && (
        <div className="grid g84">
          <div className="card" style={{ margin: 0 }}>
            <div className="cardhead"><h3>Best <Term k="CRF">CRF</Term> so far</h3>
              {d.use_drift_adjusted && <Tag kind="info"><Term k="drift-adjusted">adjusted reporting on</Term></Tag>}</div>
            <div className="row" style={{ gap: 24, alignItems: 'flex-end' }}>
              <div className="hero">{f(d.best, 3)}</div>
              <div className="muted">
                raw winner <b>{d.best_raw_method || '—'}</b> at {f(d.best_raw, 3)}
                {d.best_adjusted != null && <> · <Term k="drift-adjusted">drift-adjusted</Term> winner <b>{d.best_adjusted_method || '—'}</b> at {f(d.best_adjusted, 3)}</>}
              </div>
            </div>
            {design.length > 1 && <div className="mt3"><BestSoFar design={design} anchor={ref?.anchor} /></div>}
            {design.length === 1 && <p className="muted mb0 mt2">One method recorded. The improvement record appears once there are two to compare.</p>}
          </div>
          <div className="stack">
            {b && (
              <div className="card" style={{ margin: 0 }}>
                <div className="cardhead"><h3><Term k="run budget">Budget</Term></h3><Tag kind={b.state === 'WATCH' ? 'watch' : ''}>{b.state.toLowerCase()}</Tag></div>
                <div className="row" style={{ gap: 12 }}><span className="hero sm">{b.used}/{b.limit}</span><span className="muted">methods</span></div>
                <Meter b={b} big />
                <p className="muted mt2 mb0">{b.reason} {plural(b.injections, 'injection')} recorded; the whole budget is about {b.cost?.injections} injections, roughly {b.cost?.hours} hours.</p>
                {b.n_extensions > 0 && <p className="muted mt2 mb0">Extended {plural(b.n_extensions, 'time')} from an original {b.original} methods.</p>}
                {b.history?.length > 1 && (
                  <details className="disclose mt2"><summary>Budget history</summary>
                    {b.history.map((e, i) => (
                      <div className="histline" key={i}><b>{e.event}</b><span className="when">{(e.at || '').replace('T', ' ')}</span>
                        {e.n != null && <span>{e.n > 0 ? `+${e.n}` : e.n} → {e.to}</span>}<span>at method {e.used}</span>{e.reason && <span style={{ flex: 1 }}>{e.reason}</span>}</div>
                    ))}
                  </details>
                )}
              </div>
            )}
            <div className="card" style={{ margin: 0 }}>
              <div className="cardhead"><h3><Term k="reference run">Instrument check</Term></h3>{ref?.verdict && <Tag kind={ref.verdict === 'PASS' ? 'pass' : ref.verdict === 'HALT' ? 'halt' : 'watch'}>{ref.verdict}</Tag>}</div>
              {!ref?.series?.length
                ? <p className="muted mb0">No <Term k="anchor">anchor</Term> has been run yet. Until one is, instrument drift is unmeasured — which is not the same as absent.</p>
                : <>
                    <div className="data muted">{ref.series.map((s) => `${s.name}@${s.run_order} = ${f(s.crf, 2)}`).join('   ')}</div>
                    <p className="muted mt2 mb0">{ref.reasons?.[0]}</p>
                  </>}
            </div>
          </div>
        </div>
      )}

      {design.length > 0 && !best && (
        <div className="card mt4">
          <h3>Best separation</h3>
          <p className="muted mb0">The winning chromatogram could not be loaded — its trace file may have been moved out of the campaign’s traces folder. Every score is still recorded; only the picture is missing.</p>
        </div>
      )}
      {best && (
        <div className="card mt4">
          <div className="cardhead"><h3>Best separation — <span className="mono">{best.method}</span></h3>
            <span className="row" style={{ gap: 6 }}>{best.deviates && <Tag><Term k="tuned">tuned</Term></Tag>}{best.hump_drawn && <Tag>region drawn</Tag>}</span></div>
          {best.exact === false && <Banner kind="WATCH" eyebrow="provenance"><p>{best.exact_note}</p></Banner>}
          <div className="grid g84">
            <Chromatogram data={best} manualHumps={[]} height={300} />
            <div className="stack">
              <GradientTable rows={best.gradient} T={best.u?.[3]} />
              <div className="tiles">
                <Tile label="recorded CRF" value={f(best.recorded_crf, 3)} />
                <Tile label="clean peaks" value={best.n_clean} note={`${best.n_shoulder} shoulder · ${best.n_on_hump} on-hump`} />
                <Tile label="hump share" value={`${pct(best.hump_time_fraction)}%`} note={`run ${best.run_order}`} />
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="cardhead">
          <h3>{view === 'methods' ? 'Methods' : 'Every run'}</h3>
          <div className="row">
            <Segmented label="table view" value={view} onChange={setView}
                       options={[{ value: 'methods', label: 'group replicates' }, { value: 'runs', label: 'every run' }]} />
            {view === 'runs' && <Segmented label="source" value={only} onChange={setOnly} options={['all', 'design', 'reference']} />}
          </div>
        </div>
        <p className="muted" style={{ marginTop: 0 }}>
          {view === 'methods'
            ? 'One row per method — the unit the optimiser works in. A two-replicate method is one decision and one score. sd is the spread over its replicates, blank for a method run once.'
            : 'One row per instrument run, in the order the instrument saw them.'}
        </p>
        <div className="tablewrap">
          {view === 'methods' ? (
            <table>
              <thead><tr>
                {[['method', 'method'], ['crf_mean', 'CRF'], ['crf_sd', 'sd'], ['n', 'n'], ['crf_adjusted', 'adjusted'],
                  ['n_clean_peaks', 'clean'], ['hump_time_fraction', 'hump %'], ['start_phi', 'start %B'], ['end_phi', 'end %B'],
                  ['duration_min', 'min'], ['T', '°C'], ['run_order', 'first run']].map(([k, label], i) => (
                  <SortTh key={k} k={k} label={label} sort={sort} setSort={setSort} first={i === 0} />))}
                <th className="l">flags</th>
              </tr></thead>
              <tbody>
                {sorted.length === 0 && <tr><td className="l muted" colSpan={13}>No methods recorded yet.</td></tr>}
                {sorted.map((m) => (
                  <tr key={m.method} className={m.method === bestBase ? 'best' : ''}>
                    <td className="l mono">{m.method}</td>
                    <td><b>{f(m.crf_mean, 3)}</b></td>
                    <td>{m.crf_sd == null ? '—' : f(m.crf_sd, 3)}{m.tied ? <> <Tag><Term k="tied">tied</Term></Tag></> : ''}</td>
                    <td>{m.n}</td><td>{f(m.crf_adjusted, 3)}</td><td>{f(m.n_clean_peaks, 1)}</td>
                    <td>{pct(m.hump_time_fraction)}</td><td>{f(m.start_phi * 100, 1)}</td><td>{f(m.end_phi * 100, 1)}</td>
                    <td>{f(m.duration_min, 1)}</td><td>{f(m.T, 1)}</td><td>{m.run_order}</td>
                    <td className="l">{m.deviates && <Tag><Term k="tuned">tuned</Term></Tag>}{' '}{m.manual_humps && <Tag>region drawn</Tag>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <table>
              <thead><tr>
                <th className="l">method</th><th>rep</th><th>run</th><th>CRF</th><th>adjusted</th><th>clean</th><th>shoulder</th><th>on-hump</th>
                <th>hump %</th><th>start %B</th><th>end %B</th><th>min</th><th>°C</th><th className="l">flags</th><th className="l">notes</th>
              </tr></thead>
              <tbody>
                {d.rows.filter((r) => only === 'all' || r.source === only).length === 0 && (
                  <tr><td className="l muted" colSpan={15}>{d.rows.length === 0 ? 'No runs recorded yet.' : `No ${only} runs yet.`}</td></tr>)}
                {d.rows.filter((r) => only === 'all' || r.source === only).map((r, i) => (
                  <tr key={i} className={r.source === 'reference' ? 'ref' : ''}>
                    <td className="l mono">{r.method}</td><td>{r.replicate}</td><td>{r.run_order}</td><td><b>{f(r.CRF, 3)}</b></td>
                    <td>{f(r.CRF_adjusted, 3)}</td><td>{r.n_clean_peaks}</td><td>{r.n_shoulder_fronting}</td><td>{r.n_on_hump_peaks}</td>
                    <td>{pct(r.hump_time_fraction)}</td><td>{f((r.phi1 ?? 0) * 100, 1)}</td><td>{f((r.phi2 ?? 0) * 100, 1)}</td>
                    <td>{f(r.t1, 1)}</td><td>{f(r.T, 1)}</td>
                    <td className="l">{r.picker_deviates ? <Tag>tuned</Tag> : ''}{' '}{r.manual_humps ? <Tag>region drawn</Tag> : ''}</td>
                    <td className="l" style={{ whiteSpace: 'normal', fontFamily: 'var(--f-ui)' }}>{r.notes || ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="card">
        <h3>Compare two runs</h3>
        <div className="row mb2">
          {['a', 'b'].map((slot) => (
            <select key={slot} value={cmp[slot]} style={{ width: 'auto', minWidth: 180 }} aria-label={slot === 'a' ? 'first run' : 'second run'}
                    onChange={(e) => setCmp({ ...cmp, [slot]: e.target.value })}>
              <option value="">{slot === 'a' ? 'first run…' : 'second run…'}</option>
              {(d.rows || []).map((r) => <option key={r.method} value={r.method}>{r.method} — CRF {f(r.CRF, 2)}</option>)}
            </select>
          ))}
          {(cmp.a || cmp.b) && <button className="sm" onClick={() => setCmp({ a: '', b: '' })}><Icon name="x" />clear</button>}
        </div>
        {cmp.a && traces[cmp.a]
          ? <><Overlay a={traces[cmp.a]} b={traces[cmp.b]} />
              <p className="muted mt2 mb0">Same axes, same scale. Both traces are re-measured under the settings stamped on their own rows, so each picture is the one that produced its own score.</p></>
          : <p className="muted mb0">Pick a run to draw it; pick a second to overlay them on shared axes.</p>}
      </div>

      <div className="card">
        <div className="cardhead"><h3>Export</h3><Busy label={busy} /></div>
        <p className="muted" style={{ marginTop: 0, maxWidth: '74ch' }}>Both files are written into the campaign folder, because an export is part of the record. Each states its own provenance: what ran, under which configuration and which model, how the budget moved, and what the instrument was doing throughout.</p>
        <div className="row">
          <button className="primary" onClick={() => run('exporting', (prog) => runJob('/api/export', {}, prog).then(setExported))}><Icon name="download" />Export workbook and PDF report</button>
        </div>
        {exported && (
          <Banner kind="PASS" eyebrow="written" title={`${exported.files.length} files in ${exported.folder}`} className="mt3 mb0">
            {exported.files.map((x) => <div className="histline" key={x.name}><b className="mono">{x.name}</b><span>{(x.bytes / 1024).toFixed(0)} kB</span></div>)}
          </Banner>
        )}
        <Err>{err}</Err>
      </div>
    </main>
  )
}
