import React, { useEffect, useMemo, useRef, useState } from 'react'
import { post, upload, runJob } from '../api'
import { useAsync, f, pct, plural } from '../hooks'
import Term from '../components/Term.jsx'
import Icon from '../components/Icon.jsx'
import Chromatogram from '../components/Chromatogram.jsx'
import { Banner, Busy, Err, Tag, Meter, GradientTable, Field, NumberSlider, Tile } from '../components/ui.jsx'

/* ── one picker threshold: slider + number, effect text, delta ────────── */
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
      <div className="top">
        <label htmlFor={`ctrl-${c.key}`}>{c.term ? <Term k={c.term}>{c.label}</Term> : c.label}</label>
        <span className={`delta ${changed ? 'changed' : ''}`}>{changed ? `${base} → ${v}` : `default ${base}`}</span>
      </div>
      {c.type === 'choice'
        ? <select id={`ctrl-${c.key}`} value={v} onChange={(e) => onChange(c.key, e.target.value)}>
            {c.choices.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        : <NumberSlider id={`ctrl-${c.key}`} value={v} min={Math.min(lo, v)} max={Math.max(hi, v)} step={st}
                        onChange={(nv) => onChange(c.key, nv)} />}
      {(c.raise_it || c.lower_it) && (
        <div className="effects">
          <span><b>Raise it</b>{c.raise_it}</span>
          <span><b>Lower it</b>{c.lower_it}</span>
        </div>
      )}
      <details><summary>what it is</summary><p className="mb0 mt1">{c.note}</p></details>
    </div>
  )
}

function PickerControls({ controls, groups, values, defaults, onChange }) {
  const n = Object.keys(values || {}).length
  return (
    <>
      <div className="grid g3">
        {(controls || []).map((c) => <Ctrl key={c.key} c={c} value={values[c.key]} fallback={defaults?.[c.key]} onChange={onChange} />)}
      </div>
      {(groups || []).length > 0 && (
        <details className="adv mt3">
          <summary>Advanced settings{n > 0 ? ` — ${n} changed` : ''}: how the estimator works, not just where its thresholds sit</summary>
          {groups.map((g) => (
            <div key={g.key} className="mt3">
              <h4>{g.label}</h4>
              <p className="muted" style={{ marginTop: 0, maxWidth: '76ch' }}>{g.why}</p>
              <div className="grid g3">
                {g.controls.map((c) => <Ctrl key={c.key} c={c} value={values[c.key]} fallback={defaults?.[c.key]} onChange={onChange} />)}
              </div>
            </div>
          ))}
        </details>
      )}
    </>
  )
}

/* ── extend / finish / reopen ────────────────────────────────────────── */
export function BudgetActions({ b, reload, show, compact = false }) {
  const { busy, err, run } = useAsync()
  const [n, setN] = useState(10)
  const [why, setWhy] = useState('')
  const [end, setEnd] = useState('')
  const call = (label, url, body) => run(label, () => post(url, body).then(() => { setWhy(''); setEnd(''); return reload() }))

  if (b.finished) {
    return (
      <div className="mt3">
        <Field label="Reopen this campaign — why?">
          <div className="row">
            <input style={{ flex: 1, minWidth: 220 }} value={why} onChange={(e) => setWhy(e.target.value)} placeholder="e.g. one more confirmation method at the winner" />
            <button disabled={!why.trim()} onClick={() => call('reopening', '/api/campaign/reopen', { reason: why })}>Reopen</button>
          </div>
        </Field>
        <Busy label={busy} /><Err>{err}</Err>
      </div>
    )
  }
  return (
    <div className={`grid ${show.extend && show.finish ? 'g2' : ''} mt3`}>
      {show.extend && (
        <div className="ctrl">
          <div className="top"><label>Extend the budget</label></div>
          <div className="inputrow">
            <input type="number" min="1" step="1" value={n} onChange={(e) => setN(+e.target.value)} />
            <span className="unit">more methods</span>
          </div>
          <input value={why} placeholder="why these methods are needed (goes on the record)" onChange={(e) => setWhy(e.target.value)} />
          <div className="row">
            <button className="primary" disabled={!why.trim() || !(n >= 1)} onClick={() => call('extending', '/api/budget/extend', { n, reason: why })}>Extend by {plural(n, 'method')}</button>
          </div>
          {!compact && <p className="muted mb0" style={{ fontSize: 12.5 }}>Before extending, check the Model tab: if the posterior has stopped narrowing and the instrument check still passes, more methods buy precision rather than a better one.</p>}
        </div>
      )}
      {show.finish && (
        <div className="ctrl">
          <div className="top"><label><Term k="finished campaign">Finish this campaign</Term></label></div>
          <input value={end} placeholder="why you are stopping (goes on the record)" onChange={(e) => setEnd(e.target.value)} />
          <div className="chips">
            {['separation is good enough', 'run budget spent', 'sample or column changed'].map((p) => (
              <button key={p} className="chip" aria-pressed={end === p} onClick={() => setEnd(p)}>{p}</button>
            ))}
          </div>
          <div className="row">
            <button disabled={!end.trim()} onClick={() => call('finishing', '/api/campaign/finish', { reason: end })}>Finish the campaign</button>
          </div>
          {!compact && <p className="muted mb0" style={{ fontSize: 12.5 }}>Stops new methods being served. Nothing is deleted or locked — the workbook, every trace and the report keep working, and it can be reopened.</p>}
        </div>
      )}
      <div><Busy label={busy} /><Err>{err}</Err></div>
    </div>
  )
}

/* Provisional CRF while a drawn region is in flight: the same rule the
   server applies (clean iff the prominence pass found it AND it sits outside
   every region), so the number the analyst sees mid-edit is not a guess. */
function provisional(rep, humps) {
  if (!rep || !humps) return null
  const inside = (t) => humps.some(([a, b]) => t >= a && t <= b)
  let clean = 0, sh = 0, on = 0
  rep.peaks.forEach((p) => { if (inside(p.t)) on++; else if (p.prom) clean++; else sh++ })
  const span = humps.reduce((s, [a, b]) => s + Math.max(0, b - a), 0)
  const frac = Math.min(1, span / Math.max(rep.tmax, 1e-9))
  return { n_clean: clean, n_shoulder: sh, n_on_hump: on, hump_time_fraction: frac, crf: clean * (1 - frac) ** 2 }
}

function Counts({ r, prov }) {
  const v = prov || r
  return (
    <span className="counts" aria-live="polite">
      <span><b>{v.n_clean}</b> <Term k="clean peak">clean</Term></span>
      <span><b>{v.n_shoulder}</b> <Term k="shoulder">shoulder</Term></span>
      <span><b>{v.n_on_hump}</b> <Term k="on-hump">on-hump</Term></span>
      <span><Term k="hump_time_fraction">hump</Term> <b>{pct(v.hump_time_fraction)}%</b></span>
      <span className="crf"><Term k="CRF">CRF</Term> <b>{f(v.crf, 3)}</b>{prov && <Tag kind="info">provisional</Tag>}</span>
    </span>
  )
}

/* ── the tab ─────────────────────────────────────────────────────────── */
export default function Run({ st, reload, setTab }) {
  const { busy, err, run, setErr } = useAsync()
  const [pick, setPick] = useState(null)
  const [manual, setManual] = useState([])
  const [pending, setPending] = useState(null)      // humps sent but not yet confirmed
  const [ov, setOv] = useState({})
  const [notes, setNotes] = useState('')
  const [over, setOver] = useState(false)
  const [recorded, setRecorded] = useState(null)
  const [showRaw, setShowRaw] = useState(false)
  const fileRef = useRef(null)
  const p = st.pending
  const step = p?.step || 'suggest'
  const b = st.budget
  const o = st.optimiser || {}

  /* A reload mid-run restores what the analyst had drawn or tuned from the
     persisted state, then re-picks so the numbers on screen come from the
     code that is running now rather than a stale cache. */
  useEffect(() => {
    setPick(null); setPending(null)
    setManual((p?.manual_humps || []).map(([a, b]) => [a, b]))
    setOv({ ...(p?.picker_overrides || {}) })
  }, [p?.u?.join()])
  useEffect(() => {
    if (p && p.uploads.length >= p.n_replicates && !pick && !busy)
      doPick(p.picker_overrides || {}, (p.manual_humps || []).map(([a, b]) => [a, b]))
  }, [p?.uploads?.length])   // eslint-disable-line

  const doPick = (overrides, humps) => {
    const hs = humps ?? manual
    setPending(hs)
    return run('picking peaks', (prog) => runJob('/api/pick', { overrides: overrides ?? ov, manual_humps: hs }, prog)
      .then((r) => { setPick(r); setPending(null); return r }))
      .catch(() => setPending(null))
  }
  const setHumps = (nx) => { setManual(nx); doPick(undefined, nx) }
  const onFiles = async (files) => {
    for (const file of files) await run('uploading', () => upload(file))
    await reload()
  }

  const blocked = st.next.blocked
  const halted = st.next.action === 'halted'
  const stepTitle = p
    ? (p.kind === 'reference' ? (p.phase === 'anchor' ? 'Instrument-check method — the anchor' : 'Instrument-check method')
       : p.phase === 'cold_start' ? `Exploring method${p.seed_index ? ` ${p.seed_index}` : ''}` : 'The model’s next method')
    : blocked ? 'Nothing to run right now' : st.next.title
  const blockedNext = b?.finished
    ? 'The campaign is closed. Reopen it above to serve another method, or read the record on the Results and Method tabs.'
    : b?.exhausted
      ? 'The budget is spent. Extend it above with a reason, or finish the campaign; everything recorded stays readable and exportable.'
      : 'Proposing is blocked until the instrument has been serviced and re-anchored. The Instrument tab shows what moved.'
  const instruction = p
    ? `Run this on the HPLC${p.n_replicates > 1 ? `, ${p.n_replicates} times back to back, changing nothing between them` : ''}, then upload the exported trace${p.n_replicates > 1 ? 's' : ''} below.`
    : null

  return (
    <main>
      {/* ONE authoritative banner for the campaign's state. The step card
          below shows what to do next rather than repeating it. */}
      {b?.finished && (
        <Banner kind="FINISHED" eyebrow="campaign" title="This campaign is closed" actions={b.actions}
                form={<BudgetActions b={b} reload={reload} show={{}} />}>
          <p>{b.reason}</p>
        </Banner>
      )}
      {!b?.finished && b?.exhausted && (
        <Banner kind="EXHAUSTED" eyebrow="budget" title="Out of methods — extend the budget or finish the campaign"
                form={<BudgetActions b={b} reload={reload} show={{ extend: true, finish: true }} />}>
          <p>{st.next.detail}</p>
        </Banner>
      )}
      {halted && (
        <Banner kind="HALT" eyebrow="instrument check" title="Stopped — the instrument needs attention" actions={st.drift.actions} role="alert">
          <p>{st.drift.reasons.join(' ')}</p>
        </Banner>
      )}
      {!halted && !blocked && st.drift.verdict === 'WATCH' && (
        <Banner kind="WATCH" eyebrow="instrument check" title={st.drift.n ? 'Watch: the reference has moved' : 'Watch: drift is unmeasured until the anchor is run'}>
          <p>{st.drift.reasons.join(' ')}</p>
        </Banner>
      )}
      {!blocked && b?.warn && !b?.exhausted && (
        <Banner kind="NEUTRAL" eyebrow="budget" title="Near the end of the run budget" actions={b.actions}
                form={<BudgetActions b={b} reload={reload} show={{ extend: true }} compact />}>
          <p>{b.reason}</p>
        </Banner>
      )}

      {/* step 1 */}
      <section className={`card step ${step === 'suggest' ? 'now' : 'done'}`} aria-labelledby="s1">
        <div className="stephead">
          <span className="n">Step 1</span><h2 id="s1">{stepTitle}</h2>
          {p && <span className="state"><Tag kind={p.kind === 'reference' ? 'info' : ''}>{p.kind === 'reference' ? 'instrument check' : p.phase === 'cold_start' ? 'cold start' : `${o.acquisition_short || 'model'} proposal`}</Tag></span>}
        </div>
        {!p && (
          <>
            <p className="lead">{blocked ? blockedNext : st.next.detail}</p>
            {!blocked && (
              <button className="primary lg" onClick={() => run('working out the next method', (prog) => runJob('/api/suggest', {}, prog).then(reload))}>
                <Icon name="flask" />{st.next.kind === 'reference' ? 'Get the instrument-check method' : st.next.action === 'seed' ? 'Get the next exploring method' : 'Ask the model for the next method'}
              </button>
            )}
            {blocked && <p className="muted mb0">Uploading, picking and recording a run already on the instrument still work.</p>}
          </>
        )}
        {p && (
          <>
            <p className="lead"><b>{instruction}</b></p>
            <div className="grid g48">
              <GradientTable rows={p.gradient} T={p.u[3]} />
              <div className="tiles">
                <Tile label="repeat runs" term="replicate" value={p.n_replicates} note={p.n_replicates > 1 ? 'back to back, nothing changed between them' : 'a single injection'} />
                {p.posterior_mean != null
                  ? <Tile label="predicted CRF" term="posterior mean" value={f(p.posterior_mean)} unit={`± ${f(p.posterior_sd)}`} note={`${o.acquisition_label || ''}${p.acq_value != null ? ` · acquisition score ${f(p.acq_value, 3)}` : ''}`} />
                  : <Tile label="phase" term={p.phase === 'cold_start' ? 'cold start' : p.phase === 'anchor' ? 'anchor' : 'reference run'} value={p.phase === 'cold_start' ? 'exploring' : p.phase} note={p.reason} />}
                <Tile label="model" term="surrogate" value={o.kernel_label || '—'} note={`${o.acquisition_label || ''}${o.acq_params?.ucb_beta != null ? `, β ${o.acq_params.ucb_beta}` : ''}`} />
              </div>
            </div>
            {p.posterior_mean != null && <p className="muted mt3 mb0">{p.reason}.</p>}
          </>
        )}
      </section>

      {/* step 2 */}
      {p && (
        <section className={`card step ${step === 'upload' ? 'now' : (p.uploads.length ? 'done' : '')}`} aria-labelledby="s2">
          <div className="stephead"><span className="n">Step 2</span><h2 id="s2">Upload the chromatogram{p.n_replicates > 1 ? 's' : ''}</h2>
            <span className="state"><Tag kind={p.uploads.length >= p.n_replicates ? 'pass' : ''}>{p.uploads.length} of {p.n_replicates} uploaded</Tag></span></div>
          <div className={`drop ${over ? 'over' : ''}`} role="button" tabIndex={0}
               onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') fileRef.current.click() }}
               onDragOver={(e) => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
               onDrop={(e) => { e.preventDefault(); setOver(false); onFiles([...e.dataTransfer.files]) }}
               onClick={() => fileRef.current.click()}>
            <input ref={fileRef} type="file" style={{ display: 'none' }} multiple accept=".txt,.TXT,.csv,.asc" onChange={(e) => onFiles([...e.target.files])} />
            <Icon name="upload" size={22} />
            <div className="big">{p.uploads.length >= p.n_replicates ? 'All traces are in' : `Drop the exported ASCII .txt here, or click to choose`}</div>
            <div className="muted">Two columns: time in minutes, then signal. The file is copied into the campaign folder and kept.</div>
            {p.uploads.length > 0 && (
              <ol aria-label="uploaded traces">{p.uploads.map((u, i) => <li key={i}><Tag kind="pass"><Icon name="check" size={11} />{u}</Tag></li>)}</ol>
            )}
          </div>
        </section>
      )}

      {/* step 3 */}
      {p && p.uploads.length > 0 && (
        <section className={`card step ${step === 'pick' ? 'now' : (pick ? 'done' : '')}`} aria-labelledby="s3">
          <div className="stephead"><span className="n">Step 3</span><h2 id="s3">Check the peaks</h2>
            {pick && <span className="state row" style={{ gap: 8 }}>
              {(pick.deviates || manual.length > 0) && <Tag kind="watch" title="This run's picker settings differ from the campaign's; the deviation is stamped on the row."><Term k="stamped">will be stamped</Term></Tag>}
              <button className="sm ghost" aria-pressed={showRaw} onClick={() => setShowRaw(!showRaw)}>{showRaw ? 'baseline-corrected' : 'show raw + baseline'}</button>
            </span>}
          </div>
          {!pick && p.uploads.length < p.n_replicates && (
            <p className="lead">Waiting for {p.n_replicates - p.uploads.length} more trace{p.n_replicates - p.uploads.length === 1 ? '' : 's'} before the peaks are picked — both runs of a method are checked and recorded together.</p>
          )}
          {!pick && p.uploads.length >= p.n_replicates && (
            <div className="row">{busy ? <Busy label={busy} /> : <button className="primary" onClick={() => doPick()}>Pick peaks</button>}</div>
          )}
          {pick && (
            <>
              <p className="lead">Circles are counted. Triangles are shoulders — recorded, not counted. Squares sit on an unresolved region and are excluded.{!pick.is_reference && <> If a stretch is unresolved and the picker did not call it, <b>drag across it</b>.</>}</p>
              {pick.replicates.map((r, i) => {
                const prov = pending ? provisional(r, pending) : null
                return (
                  <div key={i} style={{ marginBottom: 20 }}>
                    <div className="tracehead">
                      <b className="mono">{r.name}</b>
                      <Counts r={r} prov={prov} />
                    </div>
                    {pick.is_reference
                      ? <Chromatogram data={r} manualHumps={[]} showRaw={showRaw} />
                      : <Chromatogram data={r} manualHumps={pending || manual} busy={!!pending} showRaw={showRaw}
                          onDrawHump={(span) => setHumps([...manual, span])}
                          onEditHump={(k, span) => setHumps(manual.map((h, j) => (j === k ? span : h)))}
                          onDeleteHump={(k) => setHumps(manual.filter((_, j) => j !== k))}
                          onAdoptHumps={(hs) => setHumps(hs.map(([a, b]) => [a, b]))}
                          onClearHumps={() => setHumps([])} />}
                  </div>
                )
              })}
              {pick.is_reference ? (
                <Banner kind="INFO" eyebrow="reference run" title="The picker is locked on an instrument check">
                  <p>The reference series is a differential measurement of the instrument: the only thing allowed to change between ref01 and ref06 is the instrument. Retuning a threshold or drawing a region on one reference would make a picking artefact look like drift, so this run is measured exactly as the campaign's settings measure every reference. To change how the whole series is read, re-score it from the Setup tab.</p>
                </Banner>
              ) : (
                <details className="adv" open={Object.keys(ov).length > 0}>
                  <summary>If the automatic pick is wrong: move the thresholds the <Term k="picker">picker</Term> uses{Object.keys(ov).length ? ` — ${Object.keys(ov).length} changed` : ''}</summary>
                  <p className="muted mt2">Drawing a region annotates <i>this</i> chromatogram and is usually the faster, more honest fix. Moving a threshold changes where the estimator draws its line for this run; the change is stamped on the row and kept if the campaign is later re-scored.</p>
                  <PickerControls controls={pick.controls} groups={pick.groups} values={ov} defaults={pick.campaign_picker_config}
                                  onChange={(k, v) => setOv({ ...ov, [k]: v })} />
                  <div className="row mt3">
                    <button className="primary" disabled={!Object.keys(ov).length} onClick={() => doPick()}><Icon name="refresh" />Re-pick with these settings</button>
                    <button disabled={!Object.keys(ov).length && !pick.deviates} onClick={() => { setOv({}); doPick({}) }}><Icon name="undo" />Reset to campaign defaults</button>
                  </div>
                </details>
              )}
            </>
          )}
        </section>
      )}

      {/* step 4 */}
      {pick && (
        <section className="card step now" aria-labelledby="s4">
          <div className="stephead"><span className="n">Step 4</span><h2 id="s4">Record</h2></div>
          <div className="grid g48" style={{ alignItems: 'end' }}>
            <div>
              <div className="label"><Term k="CRF">separation score</Term></div>
              <div className="hero">{f(pick.crf_mean, 3)}</div>
              {pick.crf_values.length > 1 && (
                <div className="muted">repeats {pick.crf_values.map((v) => f(v, 3)).join(' · ')}{pick.tied && <> — <Term k="tied">identical</Term></>}</div>
              )}
            </div>
            <div className="stack">
              <Field label="Notes for this run (optional)">
                <input value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="anything the trace alone would not tell a reader" />
              </Field>
              <div className="row">
                <button className="primary lg" disabled={p.uploads.length < p.n_replicates || !!pending || !!busy}
                  onClick={() => run('recording', (prog) => runJob('/api/record', { notes }, prog).then((r) => {
                    setRecorded(r); setPick(null); setManual([]); setNotes(''); return reload()
                  }))}>
                  <Icon name="check" />{p.kind === 'reference' ? 'Record the instrument check' : 'Record and update the model'}
                </button>
                <Busy label={busy} />
              </div>
            </div>
          </div>
          {pick.tied && (
            <p className="muted mt3 mb0">Both repeats returned an identical score. With an integer peak count that is a tie, not evidence of zero noise: σ is smaller than one clean peak and this pair cannot resolve it.</p>
          )}
        </section>
      )}

      {recorded && !p && (
        <Banner kind="PASS" eyebrow="recorded" title={recorded.recorded.map((r) => `${r.method} at run ${r.run_order} scored ${f(r.crf, 3)}`).join('; ')}>
          {recorded.within_sd != null && <p><Term k="within-method SD">Within-method SD</Term> {f(recorded.within_sd, 3)} CRF.</p>}
          {recorded.uncertainty?.reading && <p className="mt1">{recorded.uncertainty.reading}</p>}
          {recorded.uncertainty?.skipped && <p className="muted mt1">Model diagnostic skipped: {recorded.uncertainty.skipped}.</p>}
          {recorded.uncertainty?.failed && <p className="muted mt1">The run is recorded; the model diagnostic could not be computed ({recorded.uncertainty.failed}).</p>}
          {recorded.post_write_failed && <p className="muted mt1">{recorded.note}</p>}
          <div className="row actions">
            <button className="ghost sm" onClick={() => setTab('Results')}>See the results<Icon name="chev" /></button>
            <button className="ghost sm" onClick={() => setRecorded(null)}>Dismiss</button>
          </div>
        </Banner>
      )}

      {/* the budget, and the human exit - collapsed until it is wanted */}
      {b && !b.finished && !b.exhausted && (
        <section className="card flat">
          <div className="row between">
            <div className="row" style={{ gap: 12, flex: 1 }}>
              <span className="label">budget</span>
              <Meter b={b} />
              <span className="muted">{b.reason}</span>
            </div>
            <span className="muted">{plural(b.injections, 'injection')} so far</span>
          </div>
          <details className="disclose mt2">
            <summary>Finish this campaign early, or extend it…</summary>
            <BudgetActions b={b} reload={reload} show={{ extend: true, finish: true }} />
          </details>
        </section>
      )}

      <div className="row">
        <Busy label={busy && !pick ? busy : null} />
        {p && <button className="ghost" onClick={() => { if (window.confirm('Discard this run? The suggested method and any uploaded traces for it are dropped; nothing recorded is affected.')) run('discarding', () => post('/api/discard').then(() => { setPick(null); setRecorded(null); setErr(null); return reload() })) }}>
          <Icon name="x" />Discard this run</button>}
      </div>
      <Err>{err}</Err>
    </main>
  )
}
