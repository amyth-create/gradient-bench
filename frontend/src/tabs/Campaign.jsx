import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { get, post } from '../api'
import { useAsync, f, plural } from '../hooks'
import Term from '../components/Term.jsx'
import Icon from '../components/Icon.jsx'
import { Banner, Busy, Err, Light, Tag, Field, Choice, GradientTable, Tile, NumberSlider } from '../components/ui.jsx'

/* Bench-time arithmetic, mirrored from core/budget.py so it can be shown
   BEFORE the campaign exists. The server's copy is what the record uses. */
function cost(nMethods, nReps, every = 5, minutesPerRun = 30) {
  const m = Math.max(0, Math.floor(nMethods) || 0)
  const r = Math.max(1, Math.floor(nReps) || 1)
  const design = m * r
  const checks = m ? 1 + Math.floor(m / Math.max(1, every)) : 0
  const injections = design + checks
  return { design, checks, injections, hours: Math.round(injections * minutesPerRun / 60) }
}

const INFO_FIELDS = [
  ['name', 'Campaign name', 'e.g. OLA-2 gradient optimisation'],
  ['analyst', 'Analyst', ''],
  ['sample', 'Sample', 'what is being separated, and its concentration'],
  ['column', 'Column', 'make, phase, dimensions, serial'],
  ['mobile_phase', 'Mobile phase', 'A and B, additives'],
  ['flow_rate', 'Flow rate (mL/min)', ''],
  ['detector', 'Detector / wavelength', ''],
  ['instrument', 'Instrument / serial', ''],
]

/* ── one recent campaign ─────────────────────────────────────────────── */
function CampaignCard({ c, onOpen, onDuplicate, onReveal, onArchive }) {
  if (c.error) {
    return (
      <div className="card">
        <div className="cardhead"><h3>{c.slug}</h3><Tag kind="halt">unreadable</Tag></div>
        <p className="muted mb0">{c.error}</p>
        <div className="path mt2"><Icon name="folder" />{c.root}</div>
      </div>
    )
  }
  const o = c.optimiser || {}
  return (
    <div className="card" style={{ margin: 0 }}>
      <div className="cardhead">
        <div>
          <h3 style={{ marginBottom: 2 }}>{c.name || c.slug}</h3>
          <div className="muted">{c.sample || 'no sample recorded'} · {c.analyst || 'no analyst'} · started {c.created}</div>
        </div>
        {c.drift && <Light v={c.drift}><Term k="verdict">{c.drift}</Term></Light>}
      </div>
      <div className="tiles" style={{ marginBottom: 12 }}>
        <Tile label="methods" term="run budget" value={c.n_methods} note={`${plural(c.n_runs, 'run')} recorded`} />
        <Tile label="budget" value={`${c.runs_used}/${c.run_budget}`}
              note={c.finished ? 'closed' : c.budget_state === 'EXHAUSTED' ? 'spent' : `${plural(c.methods_left, 'method')} left`} />
        <Tile label="best CRF" term="CRF" value={f(c.best_crf)} note={c.best_method ? `at ${c.best_method}` : '—'} />
        <Tile label="next" value={c.finished ? 'closed' : c.next_is === 'reference' ? 'check' : 'method'}
              note={c.finished ? '' : c.next_is === 'reference' ? 'an instrument check' : 'a proposed method'} />
      </div>
      <div className="row" style={{ gap: 6, marginBottom: 12 }}>
        <Tag title="surrogate kernel">{o.kernel_label || c.kernel || '—'}</Tag>
        <Tag title="acquisition function">{o.acquisition_short || 'qLogNEI'}</Tag>
        {c.drift_adjusted && <Tag kind="info">drift-adjusted reporting</Tag>}
        {c.phantom && <Tag kind="watch">phantom incumbent</Tag>}
        {c.archived && <Tag>archived</Tag>}
      </div>
      <div className="row">
        <button className="primary" onClick={onOpen}>Open</button>
        <button onClick={onDuplicate} title="A new campaign with this one's settings and none of its data. It runs its own anchor.">Duplicate settings</button>
        <button className="ghost" onClick={onReveal}><Icon name="external" />Reveal folder</button>
        <button className="ghost" onClick={onArchive} title="Hides it from this list. Nothing is moved or deleted.">{c.archived ? 'Restore' : 'Archive'}</button>
      </div>
    </div>
  )
}

/* ── the folder browser ──────────────────────────────────────────────── */
function Browser({ browse, go, onOpenCampaign, choosing, onChoose, busy }) {
  if (!browse) return null
  return (
    <div className="card">
      <div className="cardhead">
        <h3>{choosing ? 'Where should the campaign folder be created?' : 'Choose a folder'}</h3>
        <Busy label={busy} />
      </div>
      <div className="path"><Icon name="folder" /><b>{browse.path}</b></div>
      <div className="row mt3 mb2">
        <button onClick={() => go(browse.parent)}><Icon name="up" />Up one level</button>
        {browse.is_campaign && <button className="primary" onClick={() => onOpenCampaign(browse.path)}>Open this campaign</button>}
        {choosing && !browse.is_campaign && (
          <button className="primary" onClick={() => onChoose(browse.path)}><Icon name="check" />Create the campaign inside this folder</button>
        )}
        {choosing && browse.is_campaign && <span className="muted">This folder is already a campaign. Step up, or into another folder.</span>}
      </div>
      <div className="folders" role="list" aria-label="sub-folders">
        {browse.entries.map((e) => (
          <button key={e.path} role="listitem" className={e.campaign ? 'camp' : ''} onClick={() => go(e.path)}
                  aria-label={e.campaign ? `${e.name}, a campaign folder` : e.name}>
            <Icon name={e.campaign ? 'campaign' : 'folder'} /><span>{e.name}</span>
            {e.campaign && <Tag kind="info">campaign</Tag>}
          </button>
        ))}
        {!browse.entries.length && <span className="muted" style={{ padding: 8 }}>no sub-folders here</span>}
      </div>
      <p className="muted mt2 mb0">A campaign folder holds a workbook, its traces and its history, and is marked with the peak icon. Step into a folder to look further.</p>
    </div>
  )
}

/* ── the new-campaign wizard ─────────────────────────────────────────── */
const STEPS = ['Folder', 'Campaign', 'Instrument check', 'Plan', 'Model', 'Review']

function Wizard({ opts, folder, onPickFolder, onCreate, busy, err }) {
  const [step, setStep] = useState(folder ? 1 : 0)
  const [info, setInfo] = useState(Object.fromEntries(INFO_FIELDS.map(([k]) => [k, ''])))
  const preset = opts?.reference_presets?.shallow_survey?.u || [0.10, 0.30, 50.0, 45.0]
  const [ref, setRef] = useState(preset)
  const [nrep, setNrep] = useState(2)
  const [nseed, setNseed] = useState(5)
  const [every, setEvery] = useState(5)
  const [budget, setBudget] = useState(opts?.budget_default || 40)
  const [kernel, setKernel] = useState(opts?.default_kernel || 'matern52')
  const [acq, setAcq] = useState(opts?.default_acquisition || 'qlognei')
  const [acqParams, setAcqParams] = useState({ ...(opts?.acq_param_defaults || { ucb_beta: 2.0 }) })
  const [restarts, setRestarts] = useState(opts?.acq_num_restarts || 10)
  const [raw, setRaw] = useState(opts?.acq_raw_samples || 512)

  useEffect(() => { if (folder && step === 0) setStep(1) }, [folder])   // eslint-disable-line

  const lo = opts?.space?.lower || [0.02, 0.10, 10, 25], hi = opts?.space?.upper || [0.40, 1.00, 60, 60]
  const refErrors = useMemo(() => {
    const e = {}
    if (!(ref[0] >= lo[0] && ref[0] <= hi[0])) e[0] = `start %B must be between ${lo[0] * 100} and ${hi[0] * 100}`
    if (!(ref[1] >= lo[1] && ref[1] <= hi[1])) e[1] = `end %B must be between ${lo[1] * 100} and ${hi[1] * 100}`
    if (ref[1] < ref[0]) e[1] = 'end %B must not be below start %B'
    if (Math.abs(ref[1] - ref[0]) < 1e-6) e[1] = 'an isocratic reference has little structure to lose — choose a gradient'
    if (!(ref[2] >= lo[2] && ref[2] <= hi[2])) e[2] = `duration must be between ${lo[2]} and ${hi[2]} min`
    if (!(ref[3] >= lo[3] && ref[3] <= hi[3])) e[3] = `temperature must be between ${lo[3]} and ${hi[3]} °C`
    return e
  }, [ref])
  const okStep = [!!folder, !!info.name.trim(), Object.keys(refErrors).length === 0,
                  nrep >= 1 && nseed >= 2 && budget >= 1 && every >= 1, true, true]
  const canCreate = okStep.slice(0, 5).every(Boolean)
  const cst = cost(budget, nrep, every)
  const sur = (opts?.surrogates || []).find((s) => s.key === kernel)
  const ao = (opts?.acquisitions || []).find((a) => a.key === acq)
  const gradient = [
    { time_min: 0, pct_b: ref[0] * 100, phase: 'Start' },
    { time_min: ref[2], pct_b: ref[1] * 100, phase: 'End of gradient' },
  ]

  const create = () => onCreate({
    parent_dir: folder, info, reference_method: ref, n_replicates: nrep, n_seed: nseed,
    run_budget: budget, reference_every: every, kernel, acquisition: acq,
    acq_params: acqParams, acq_num_restarts: restarts, acq_raw_samples: raw,
  })

  return (
    <div className="card" id="new-campaign">
      <div className="cardhead"><h2 style={{ margin: 0 }}>New campaign</h2><Busy label={busy} /></div>
      <div className="wizard">
        <aside className="wrail">
          <ol className="wsteps" aria-label="steps">
            {STEPS.map((s, i) => {
              const done = i < step && okStep[i]
              const bad = i < step && !okStep[i]
              return (
                <li key={s}>
                  <button type="button" aria-current={step === i ? 'step' : undefined}
                          className={done ? 'ok' : bad ? 'bad' : ''} onClick={() => setStep(i)}>
                    <span className="dot">{done ? <Icon name="check" size={12} /> : bad ? '!' : i + 1}</span>{s}
                  </button>
                </li>
              )
            })}
          </ol>
          <div className="summary" aria-label="what will be created">
            <dl>
              <dt>folder</dt><dd style={{ wordBreak: 'break-all' }}>{folder || '—'}</dd>
              <dt>name</dt><dd>{info.name || '—'}</dd>
              <dt>check method</dt><dd>{(ref[0] * 100).toFixed(0)}→{(ref[1] * 100).toFixed(0)} %B, {ref[2]} min, {ref[3]} °C</dd>
              <dt>plan</dt><dd>{budget} methods × {nrep} · {nseed} seeds</dd>
              <dt>model</dt><dd>{sur?.label || kernel} · {ao?.short || acq}{acq === 'ucb' ? ` β ${acqParams.ucb_beta}` : ''}</dd>
            </dl>
          </div>
        </aside>

        <div>
          {step === 0 && (
            <div>
              <h3>Where the campaign lives</h3>
              <p className="muted">A campaign is a folder you choose the location of. The workbook, every uploaded trace and the model’s history live inside it, so it can be copied to another machine and opened intact. The campaign folder is created <b>inside</b> the folder you pick, named after the campaign.</p>
              {folder
                ? <div className="path"><Icon name="check" /><span>will be created inside <b>{folder}</b></span></div>
                : <div className="path"><Icon name="folder" /><span>no folder chosen yet</span></div>}
              <div className="row mt3">
                <button className={folder ? '' : 'primary'} onClick={onPickFolder}><Icon name="folder" />{folder ? 'Choose a different folder…' : 'Choose a folder…'}</button>
              </div>
            </div>
          )}

          {step === 1 && (
            <div>
              <h3>What this campaign is</h3>
              <p className="muted">The six fields marked <Term k="material">material</Term> say what the runs are measurements <i>of</i>. They can be corrected later, but changing one after runs are recorded needs a reason, because it decides whether the earlier and later runs are the same experiment.</p>
              <div className="grid g2">
                {INFO_FIELDS.map(([k, label, ph]) => (
                  <Field key={k} label={label} error={k === 'name' && !info.name.trim() && step > 1 ? 'required' : ''}
                         help={['sample', 'column', 'mobile_phase', 'instrument', 'flow_rate', 'detector'].includes(k) ? 'material' : (k === 'name' ? 'names the folder' : '')}>
                    <input value={info[k]} placeholder={ph} onChange={(e) => setInfo({ ...info, [k]: e.target.value })} />
                  </Field>
                ))}
              </div>
            </div>
          )}

          {step === 2 && (
            <div>
              <h3><Term k="reference run">Instrument-check method</Term></h3>
              <p className="muted">One fixed method, re-run on a schedule. Its movement can only be the instrument, which is what separates <Term k="drift">drift</Term> from chemistry. The first run of it is the <Term k="anchor">anchor</Term> and must be measured before anything else. Pick a method you know this instrument runs well — it only has to stay fixed.</p>
              <div className="grid g48">
                <div>
                  <div className="row mb2">
                    <button className="sm" onClick={() => setRef(preset)}><Icon name="refresh" />shallow survey preset</button>
                    <span className="muted small">10→30 %B over 50 min at 45 °C: 0.4 %B/min spreads most samples out, so retention and width shifts are obvious.</span>
                  </div>
                  <div className="grid g2">
                    {[['start %B', 0, 0.5, 'start_phi'], ['end %B', 1, 0.5, 'end_phi'], ['duration', 2, 1, 'min'], ['temperature', 3, 1, '°C']].map(([label, i, stp, unit]) => (
                      <Field key={label} label={label} error={refErrors[i]} unit={i < 2 ? '%B' : unit}>
                        <input type="number" step={stp} value={i < 2 ? +(ref[i] * 100).toFixed(2) : ref[i]}
                               onChange={(e) => { const v = parseFloat(e.target.value); const nx = [...ref]; nx[i] = i < 2 ? v / 100 : v; setRef(nx) }} />
                      </Field>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="label mb2">as the instrument will see it</div>
                  <GradientTable rows={gradient} T={ref[3]} />
                  <p className="muted mt2 mb0">Allowed box: {lo[0] * 100}–{hi[0] * 100} %B start, {lo[1] * 100}–{hi[1] * 100} %B end, {lo[2]}–{hi[2]} min, {lo[3]}–{hi[3]} °C. The box is the instrument’s envelope, the same for every sample.</p>
                </div>
              </div>
            </div>
          )}

          {step === 3 && (
            <div>
              <h3>The plan: how many runs, and how they are spent</h3>
              <div className="grid g2">
                <Field label="Run budget" term="run budget" unit="methods"
                       help="Unique methods, not injections. Repeats and instrument checks cost bench time and spend no budget. You can extend it later, on the record.">
                  <input type="number" min="1" step="1" value={budget} onChange={(e) => setBudget(+e.target.value)} />
                </Field>
                <Field label="Repeat runs per method" term="replicate" unit="injections"
                       help="Two runs at identical settings differ only by noise, which is the only way to measure σ directly. Set 1 and σ becomes unmeasurable.">
                  <input type="number" min="1" max="5" value={nrep} onChange={(e) => setNrep(+e.target.value)} />
                </Field>
                <Field label="Exploring methods before the model starts" term="cold start" unit="methods"
                       help="Spread-out methods, recorded as measured. Five is the minimum that gives a four-parameter model a first look; eight to twelve buys a better-conditioned first fit for a few methods of budget.">
                  <input type="number" min="2" max="20" value={nseed} onChange={(e) => setNseed(+e.target.value)} />
                </Field>
                <Field label="Instrument check every" term="cadence" unit="methods"
                       help="Enforced, not advised: when a check is due the loop will not serve a design method until it has been run.">
                  <input type="number" min="1" max="20" value={every} onChange={(e) => setEvery(+e.target.value)} />
                </Field>
              </div>
              <div className="detail mt4">
                <div className="label mb2">what this plan costs on the instrument</div>
                <div className="tiles">
                  <Tile label="injections" value={cst.injections} note={`${cst.design} design runs + ${cst.checks} checks`} />
                  <Tile label="bench time" value={`≈ ${cst.hours}`} unit="h" note="at a 30-minute gradient" />
                  <Tile label="first model fit" value={`after ${nseed}`} unit="methods" note="the cold start" />
                </div>
              </div>
            </div>
          )}

          {step === 4 && opts && (
            <div>
              <h3>The model that chooses the next method</h3>
              <p className="muted">Two choices, made once and <Term k="locked">locked</Term> for the life of the campaign. The defaults are what the science was validated with; the alternatives are offered for campaigns with a clear brief that wants a different balance. Every choice is written to the campaign’s record and printed on the Method page.</p>
              <h4 className="mt3"><Term k="surrogate">Surrogate</Term> — how smooth the response is assumed to be</h4>
              <div className="choices" role="radiogroup" aria-label="surrogate kernel">
                {opts.surrogates.map((s) => (
                  <Choice key={s.key} checked={kernel === s.key} onSelect={() => setKernel(s.key)} name={s.label}
                          desc={s.what} meta={s.smoothness} badge={s.recommended ? <Tag kind="rec">recommended</Tag> : null} />
                ))}
              </div>
              {sur && (
                <div className="detail mt3">
                  <dl>
                    <dt>choose it when</dt><dd>{sur.when}</dd>
                    <dt>trade-off</dt><dd>{sur.tradeoff}</dd>
                  </dl>
                </div>
              )}
              <h4 className="mt6"><Term k="acquisition function">Acquisition</Term> — how the next method is chosen from the model</h4>
              <div className="choices" role="radiogroup" aria-label="acquisition function">
                {opts.acquisitions.map((a) => (
                  <Choice key={a.key} checked={acq === a.key} onSelect={() => setAcq(a.key)} name={a.label}
                          desc={a.what} meta={a.class} badge={a.recommended ? <Tag kind="rec">recommended</Tag> : null} />
                ))}
              </div>
              {ao && (
                <div className="detail mt3">
                  <dl>
                    <dt>choose it when</dt><dd>{ao.when}</dd>
                    <dt>trade-off</dt><dd>{ao.tradeoff}</dd>
                  </dl>
                  {ao.params.map((p) => (
                    <Field key={p.key} label={p.label} term={p.key === 'ucb_beta' ? 'beta' : undefined} help={p.note} className="mt3">
                      <NumberSlider value={acqParams[p.key] ?? p.default} min={p.min} max={p.max} step={p.step}
                                    onChange={(v) => setAcqParams({ ...acqParams, [p.key]: v })} />
                    </Field>
                  ))}
                </div>
              )}
              <details className="adv mt3">
                <summary>Acquisition optimiser settings (rarely changed)</summary>
                <div className="grid g2 mt2">
                  <Field label="Restarts" term="restarts" help="starting points for the climb over the acquisition surface">
                    <input type="number" min="1" max="64" value={restarts} onChange={(e) => setRestarts(+e.target.value)} />
                  </Field>
                  <Field label="Raw samples" term="raw samples" help="candidates scored before the restarts are chosen">
                    <input type="number" min="16" max="4096" step="16" value={raw} onChange={(e) => setRaw(+e.target.value)} />
                  </Field>
                </div>
              </details>
              {!opts.torch && (
                <Banner kind="WATCH" eyebrow="environment" title="torch is not installed here" className="mt3">
                  <p>The campaign can still be created, opened, picked and recorded. Proposing a method — the step that fits the model — needs torch and BoTorch installed in this environment.</p>
                </Banner>
              )}
            </div>
          )}

          {step === 5 && (
            <div>
              <h3>Review</h3>
              <p className="muted">Everything below is written to the campaign’s Config and Campaign sheets on creation. The model choices are locked; the budget and the metadata can change later, on the record.</p>
              <dl className="kv wide">
                <dt>folder</dt><dd><b>{folder}</b></dd>
                <dt>name</dt><dd><b>{info.name || '—'}</b></dd>
                <dt>sample · column</dt><dd>{info.sample || '—'} · {info.column || '—'}</dd>
                <dt>instrument check</dt><dd>{(ref[0] * 100).toFixed(1)}→{(ref[1] * 100).toFixed(1)} %B over {ref[2]} min at {ref[3]} °C, every {every} methods</dd>
                <dt>budget</dt><dd>{budget} methods × {nrep} repeats ≈ {cst.injections} injections, ≈ {cst.hours} h</dd>
                <dt>cold start</dt><dd>{nseed} exploring methods</dd>
                <dt>surrogate</dt><dd><b>{sur?.label}</b> — {sur?.smoothness}</dd>
                <dt>acquisition</dt><dd><b>{ao?.label}</b>{acq === 'ucb' ? ` with β = ${acqParams.ucb_beta}` : ''} · {restarts} restarts, {raw} raw samples</dd>
              </dl>
              {!canCreate && (
                <Banner kind="WATCH" eyebrow="not ready" title="Something is missing" className="mt3">
                  <ul>
                    {!okStep[0] && <li>Choose a folder (step 1).</li>}
                    {!okStep[1] && <li>Name the campaign (step 2).</li>}
                    {!okStep[2] && <li>Fix the instrument-check method (step 3).</li>}
                    {!okStep[3] && <li>Check the plan numbers (step 4).</li>}
                  </ul>
                </Banner>
              )}
              <Err>{err}</Err>
            </div>
          )}

          <div className="wfoot">
            <button disabled={step === 0} onClick={() => setStep(step - 1)}>Back</button>
            {step < STEPS.length - 1
              ? <button className="primary" onClick={() => setStep(step + 1)}>Continue<Icon name="chev" /></button>
              : <button className="primary lg" disabled={!canCreate || !!busy} onClick={create}><Icon name="flask" />Create campaign</button>}
          </div>
        </div>
      </div>
    </div>
  )
}

/* ── the tab ─────────────────────────────────────────────────────────── */
export default function Campaign({ onOpen, st }) {
  const { busy, err, run, setErr } = useAsync()
  const [list, setList] = useState(null)
  const [nArchived, setNArchived] = useState(0)
  const [showArchived, setShowArchived] = useState(false)
  const [browse, setBrowse] = useState(null)
  const [mode, setMode] = useState(null)          // null | 'open' | 'choose'
  const [creating, setCreating] = useState(false)
  const [folder, setFolder] = useState('')
  const [opts, setOpts] = useState(null)
  const [migration, setMigration] = useState(null)
  const [version, setVersion] = useState(null)

  const refresh = useCallback(() => get(`/api/campaigns${showArchived ? '?archived=1' : ''}`)
    .then((r) => { setList(r.campaigns || []); setNArchived(r.n_archived || 0) }), [showArchived])
  useEffect(() => { refresh() }, [refresh])
  useEffect(() => { get('/api/optimiser/options').then(setOpts).catch(() => {}); get('/api/version').then(setVersion).catch(() => {}) }, [])

  const openCampaign = (path) => run('opening', () =>
    get(`/api/migrate?path=${encodeURIComponent(path)}`)
      .then((r) => (r.report.needed ? setMigration({ ...r.report, path })
                                    : post('/api/campaigns/open', { path }).then(onOpen))))
  const go = (p) => run('reading folder', () => get(`/api/browse?path=${encodeURIComponent(p || '')}`).then(setBrowse))
  const startBrowse = (m) => { setMode(m); if (!browse) go('') }

  const createCampaign = (body) => run('creating', () => post('/api/campaigns/create', body).then(onOpen))

  return (
    <main>
      <div className="pagehead">
        <div>
          <h1>Campaigns</h1>
          <p className="sub mb0">A campaign is a folder you choose the location of. Everything it records — the workbook, every uploaded trace, the model’s history — lives inside it, so it can be copied to another machine and opened intact.</p>
        </div>
        <div className="row">
          <button onClick={() => startBrowse('open')}><Icon name="folder" />Open a folder…</button>
          <button className="primary" onClick={() => { setCreating(!creating); setErr(null); if (!creating) setTimeout(() => document.getElementById('new-campaign')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50) }}>
            <Icon name={creating ? 'x' : 'plus'} />{creating ? 'Cancel' : 'New campaign'}
          </button>
        </div>
      </div>
      <div className="row mb4"><Busy label={busy} /><Err>{!creating && err}</Err></div>

      {migration && (
        <Banner kind="WATCH" eyebrow="older campaign" title="This campaign was written by an older build">
          <p>{migration.summary}</p>
          <p className="muted mt2">Bringing it forward only adds — a missing column, a missing default, the marker file. No measured value is rewritten and nothing is deleted, and the workbook is backed up first. A version-1 campaign keeps the model it was created with: Matérn-5/2 and log noisy expected improvement.</p>
          <div className="row actions">
            <button className="primary" onClick={() => run('migrating', () => post('/api/migrate', { path: migration.path })
              .then(() => post('/api/campaigns/open', { path: migration.path })).then(() => { setMigration(null); return onOpen() }))}>Bring it forward and open</button>
            <button onClick={() => run('opening', () => post('/api/campaigns/open', { path: migration.path }).then(() => { setMigration(null); return onOpen() }))}>Open without changing it</button>
            <button className="ghost" onClick={() => setMigration(null)}>Cancel</button>
          </div>
        </Banner>
      )}

      {(mode === 'open' || (creating && mode === 'choose')) && (
        <Browser browse={browse} go={go} busy={busy} onOpenCampaign={openCampaign}
                 choosing={mode === 'choose'} onChoose={(p) => { setFolder(p); setMode(null); setTimeout(() => document.getElementById('new-campaign')?.scrollIntoView({ behavior: 'smooth' }), 50) }} />
      )}

      {creating && (
        <Wizard opts={opts} folder={folder} busy={busy} err={err}
                onPickFolder={() => startBrowse('choose')} onCreate={createCampaign} />
      )}

      {list === null && <div className="card"><div className="skel" style={{ height: 60 }} /></div>}
      {list && list.length === 0 && !creating && (
        <div className="card">
          <h3>Nothing here yet</h3>
          <p>Press <b>New campaign</b>, choose where its folder should live, describe the sample and the column, set the instrument-check method and the plan, and pick the model. The first thing a new campaign asks for is the instrument-check run, and it serves nothing else until that has been recorded.</p>
          <p className="muted mb0">Already have a campaign folder from an earlier session or another machine? <b>Open a folder…</b> and step to it.</p>
        </div>
      )}
      {list && list.length > 0 && (
        <>
          <div className="row between mb2">
            <div className="label">Recent</div>
            {nArchived > 0 && <button className="ghost sm" onClick={() => setShowArchived(!showArchived)}>{showArchived ? 'Hide' : 'Show'} {nArchived} archived</button>}
          </div>
          <div className="grid g2">
            {list.map((c) => (
              <CampaignCard key={c.root} c={c}
                onOpen={() => openCampaign(c.root)}
                onDuplicate={() => run('copying settings', () => post('/api/campaigns/duplicate', { path: c.root }).then(() => refresh().then(onOpen)))}
                onReveal={() => run('revealing', () => post('/api/campaigns/reveal', { path: c.root }))}
                onArchive={() => run(c.archived ? 'restoring' : 'archiving', () => post('/api/campaigns/archive', { path: c.root, archived: !c.archived }).then(refresh))} />
            ))}
          </div>
        </>
      )}
      {version && (
        <p className="muted mt6 mb0" style={{ fontSize: 12 }}>Gradient Bench {version.tag} ({version.version}) · botorch {version.libraries?.botorch} · torch {version.libraries?.torch}{version.torch ? '' : ' — not importable here'}</p>
      )}
    </main>
  )
}
