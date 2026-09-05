import React, { useCallback, useEffect, useState } from 'react'
import { get, post, runJob } from '../api'
import { useAsync, f, plural } from '../hooks'
import Term from '../components/Term.jsx'
import Icon from '../components/Icon.jsx'
import { Banner, Busy, Err, Tag, Field, Skeleton, Tile, NumberSlider } from '../components/ui.jsx'

function runsPhrase(d) {
  const runs = d.n_runs || 0
  const checks = Math.max(0, (d.n_injections || 0) - runs)
  const r = plural(runs, 'recorded run')
  if (!checks) return r
  const c = plural(checks, 'instrument check')
  return runs ? `${r} (plus ${c})` : `${c} and no runs yet`
}

function CampaignInfo({ reload }) {
  const { busy, err, run } = useAsync()
  const [d, setD] = useState(null)
  const [edit, setEdit] = useState({})
  const [why, setWhy] = useState('')
  const load = useCallback(() => get('/api/campaign/info').then((r) => { setD(r); setEdit({}); setWhy(''); return r }), [])
  useEffect(() => { load() }, [load])
  if (!d) return <Skeleton />
  const changed = Object.keys(edit).filter((k) => edit[k] !== (d.info[k] ?? ''))
  const material = changed.filter((k) => d.material.includes(k))
  const needsReason = material.length > 0 && (d.n_injections ?? d.n_runs) > 0
  const save = () => run('saving', () => post('/api/campaign/info', {
    updates: Object.fromEntries(changed.map((k) => [k, edit[k]])), reason: why }).then(() => Promise.all([load(), reload()])))
  return (
    <div className="card">
      <div className="cardhead"><h3>Campaign information</h3><Busy label={busy} /></div>
      <div className="grid g2">
        {d.fields.map(([k, label]) => (
          <Field key={k} label={label} help={d.material.includes(k) ? <Term k="material">material</Term> : ''}>
            <input value={edit[k] ?? (d.info[k] ?? '')} onChange={(e) => setEdit({ ...edit, [k]: e.target.value })} />
          </Field>
        ))}
      </div>
      {changed.length > 0 && (
        <>
          {needsReason && (
            <Banner kind="WATCH" eyebrow="material change" title="This changes what the runs are measurements of" className="mt4">
              <p>{material.join(', ')} on a campaign with {runsPhrase(d)}. Correcting how it is written down is routine; changing which column or sample was on the instrument means the earlier runs and the later ones are not the same experiment. Say which this is.</p>
            </Banner>
          )}
          <div className="row mt3">
            <input style={{ flex: 1, minWidth: 240 }} value={why} placeholder={needsReason ? 'why this is changing (required)' : 'why this is changing (optional)'} onChange={(e) => setWhy(e.target.value)} />
            <button className="primary" disabled={needsReason && !why.trim()} onClick={save}>Save {plural(changed.length, 'change')}</button>
            <button onClick={() => { setEdit({}); setWhy('') }}>Cancel</button>
          </div>
        </>
      )}
      <Err>{err}</Err>
      {d.history?.length > 0 && (
        <details className="disclose mt3"><summary>Changes since creation ({d.history.length})</summary>
          {d.history.map((h, i) => h.changes.map((ch, k) => (
            <div className="histline" key={`${i}-${k}`}><b>{ch.field}</b><span className="when">{(h.at || '').replace('T', ' ')}</span>
              <span>{ch.from || '(blank)'} → {ch.to}</span><span>at {plural(h.runs_recorded, 'run')}</span>
              {ch.material && <Tag kind="watch"><Term k="material">material</Term></Tag>}{h.reason && <span style={{ flex: 1 }}>{h.reason}</span>}</div>
          )))}
        </details>
      )}
    </div>
  )
}

const RANGES = {
  snr: [2, 12, 0.5], hump_min_floor_ratio: [0.5, 1.0, 0.01], hump_min_span_abs: [0.2, 8, 0.1],
  hump_level_frac: [0.001, 0.05, 0.001], hump_min_span_factor: [1, 10, 0.5], hump_min_real_peaks: [0, 5, 1],
  hump_merge_gap: [0, 10, 0.5], arpls_lam: [1e4, 1e6, 1e4], min_width_min: [0.01, 0.5, 0.01],
  merge_min_sep_min: [0.01, 0.5, 0.01], d2_snr: [1, 8, 0.5], hybrid_shoulder_snr: [1, 10, 0.5],
}
function Ctrl({ c, value, fallback, onChange }) {
  const base = fallback ?? c.default
  const v = value ?? base
  const changed = value != null && value !== base
  const [lo, hi, st] = RANGES[c.key] || [0, 10, c.step ?? 0.1]
  return (
    <div className="ctrl">
      <div className="top"><label htmlFor={`s-${c.key}`}>{c.term ? <Term k={c.term}>{c.label}</Term> : c.label}</label>
        <span className={`delta ${changed ? 'changed' : ''}`}>{changed ? `${base} → ${v}` : `now ${base}`}</span></div>
      {c.type === 'choice'
        ? <select id={`s-${c.key}`} value={v} onChange={(e) => onChange(c.key, e.target.value)}>{c.choices.map((o) => <option key={o} value={o}>{o}</option>)}</select>
        : <NumberSlider id={`s-${c.key}`} value={v} min={Math.min(lo, v)} max={Math.max(hi, v)} step={st} onChange={(nv) => onChange(c.key, nv)} />}
      {(c.raise_it || c.lower_it) && <div className="effects"><span><b>Raise it</b>{c.raise_it}</span><span><b>Lower it</b>{c.lower_it}</span></div>}
      <details><summary>what it is</summary><p className="mb0 mt1">{c.note}</p></details>
    </div>
  )
}

export default function Setup({ reload }) {
  const { busy, err, run } = useAsync()
  const [d, setD] = useState(null)
  const [next, setNext] = useState({})
  const [prev, setPrev] = useState(null)
  const [why, setWhy] = useState('')
  const [st, setSt] = useState(null)
  const load = useCallback(() => get('/api/picker/controls').then((r) => { setD(r); setNext({}); setPrev(null); return r }), [])
  useEffect(() => { load(); get('/api/state').then(setSt).catch(() => {}) }, [load])
  if (!d) return <main><h1>Setup</h1><Skeleton /></main>
  const merged = { ...d.campaign, ...next }
  const changed = Object.keys(merged).filter((k) => merged[k] !== d.campaign[k])
  const preview = () => run('working out what would change', (prog) => runJob('/api/picker/rescore', { config: merged }, prog).then(setPrev))
  const commit = () => run('re-scoring the campaign', (prog) => runJob('/api/picker/rescore', { config: merged, commit: true, reason: why }, prog)
    .then(() => { setWhy(''); return Promise.all([load(), reload()]) }))
  const o = st?.optimiser

  return (
    <main>
      <h1>Setup</h1>
      <p className="sub">What this campaign is, and the ruler its scores are read against. Changing the <Term k="picker">picker</Term> here is the sanctioned way to change it mid-campaign, because it <Term k="re-score">re-measures every stored trace</Term> so the <Term k="CRF">CRF</Term> column keeps meaning one thing.</p>

      <CampaignInfo reload={reload} />

      {o && (
        <div className="card">
          <div className="cardhead"><h3>The model</h3><Tag><Icon name="lock" size={11} /><Term k="locked">locked at creation</Term></Tag></div>
          <div className="tiles">
            <Tile label="surrogate" term="surrogate" value={o.kernel_label} note={o.kernel} />
            <Tile label="acquisition" term="acquisition function" value={o.acquisition_short} note={o.description} />
            <Tile label="cold start" term="cold start" value={o.n_seed} unit="methods" />
            <Tile label="repeats" term="replicate" value={o.n_replicates} unit="per method" />
          </div>
          <p className="muted mt3 mb0">A campaign fitted under one kernel and refitted under another is not the same campaign, so these cannot change here. To run the same sample under a different model, duplicate the settings into a new campaign and choose there.</p>
        </div>
      )}

      <div className="card">
        <div className="cardhead"><h3>Campaign picker settings</h3><Busy label={busy} /></div>
        <div className="grid g3">
          {d.controls.map((c) => <Ctrl key={c.key} c={c} value={next[c.key]} fallback={d.campaign[c.key]} onChange={(k, v) => setNext({ ...next, [k]: v })} />)}
        </div>
        <details className="adv mt3">
          <summary>Advanced settings{changed.filter((k) => !d.controls.some((c) => c.key === k)).length ? ` — ${changed.filter((k) => !d.controls.some((c) => c.key === k)).length} changed` : ''}</summary>
          {d.groups.map((g) => (
            <div key={g.key} className="mt3">
              <h4>{g.label}</h4>
              <p className="muted" style={{ marginTop: 0, maxWidth: '76ch' }}>{g.why}</p>
              <div className="grid g3">
                {g.controls.map((c) => <Ctrl key={c.key} c={c} value={next[c.key]} fallback={d.campaign[c.key] ?? d.defaults?.[c.key]} onChange={(k, v) => setNext({ ...next, [k]: v })} />)}
              </div>
            </div>
          ))}
        </details>
        <div className="row mt4">
          <button className="primary" disabled={!changed.length} onClick={preview}>
            {changed.length ? `Preview the effect of ${plural(changed.length, 'change')}` : 'Change a setting to preview'}</button>
          {changed.length > 0 && <button onClick={() => { setNext({}); setPrev(null) }}><Icon name="undo" />Reset</button>}
        </div>
        <Err>{err}</Err>
      </div>

      {prev && (
        <div className="card">
          <h3>What would change</h3>
          <p className="lead">{prev.summary}</p>
          {prev.n_missing > 0 && (
            <Banner kind="HALT" eyebrow="cannot re-score" title={`${prev.n_missing} trace file(s) are missing from the campaign’s traces folder`}>
              <p>Re-scoring the rest would leave the campaign measured two different ways with nothing in the sheet to say so.</p>
            </Banner>
          )}
          <div className="tablewrap">
            <table>
              <thead><tr><th className="l">method</th><th>run</th><th>CRF now</th><th>CRF after</th><th>Δ</th><th>peaks</th><th className="l">notes</th></tr></thead>
              <tbody>
                {prev.rows.map((r, i) => (
                  <tr key={i} className={r.source === 'reference' ? 'ref' : ''}>
                    <td className="l mono">{r.method}</td><td>{r.run_order}</td><td>{f(r.old_crf, 3)}</td><td><b>{f(r.new_crf, 3)}</b></td>
                    <td className={r.moved ? 'moved' : ''}>{r.delta == null ? '—' : (r.delta > 0 ? '+' : '') + f(r.delta, 3)}</td>
                    <td>{r.old_peaks} → {r.new_peaks < 0 ? '—' : r.new_peaks}</td>
                    <td className="l">
                      {!r.found && <Tag kind="halt">trace missing</Tag>}{' '}
                      {r.error && r.found ? <Tag kind="watch">{r.error}</Tag> : null}{' '}
                      {r.deviates && <Tag>tuned — kept</Tag>}{' '}{r.manual && <Tag>region drawn — kept</Tag>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted mt3">Runs that were tuned by hand keep their corrections: the deviation is re-applied on top of the new settings. The workbook is backed up first and the whole Data sheet is replaced in one write.</p>
          <div className="row">
            <input style={{ flex: 1, minWidth: 240 }} value={why} placeholder="why the campaign's settings are changing (goes on the record)" onChange={(e) => setWhy(e.target.value)} />
            <button className="primary" disabled={!prev.ok || !why.trim()} onClick={commit}>Re-score all {prev.n_rows} runs</button>
          </div>
        </div>
      )}

      {d.history?.length > 0 && (
        <div className="card">
          <h3>Configuration history</h3>
          {d.history.map((h, i) => (
            <div className="histline" key={i}><span className="when">{(h.at || '').replace('T', ' ')}</span>
              <span>{Object.entries(h.changed || {}).map(([k, [a, b]]) => `${k}: ${a} → ${b}`).join(', ')}</span>
              <span>{h.n_moved}/{h.n_rows} moved</span>{h.reason && <span style={{ flex: 1 }}>{h.reason}</span>}</div>
          ))}
        </div>
      )}
    </main>
  )
}
