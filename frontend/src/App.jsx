import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { get } from './api'
import { useAnnouncements, useLocal, useTheme, f } from './hooks'
import Term from './components/Term.jsx'
import Mark from './components/Mark.jsx'
import Icon from './components/Icon.jsx'
import { Light, Meter } from './components/ui.jsx'
import Campaign from './tabs/Campaign.jsx'
import Run from './tabs/Run.jsx'
import Results from './tabs/Results.jsx'
import Model from './tabs/Model.jsx'
import Instrument from './tabs/Instrument.jsx'
import Setup from './tabs/Setup.jsx'
import Method from './tabs/Method.jsx'

const TABS = [
  ['Campaign', 'Open or create a campaign'],
  ['Run', 'The four-step loop'],
  ['Results', 'Every run and the best so far'],
  ['Model', 'Is it learning, and what about'],
  ['Instrument', 'The drift monitor'],
  ['Setup', 'Metadata and picker settings'],
  ['Method', 'The printable methods statement'],
]

/* ── the chrome: brand, campaign, priority-collapsing stats, lamp, tools ── */
function Bar({ st, tab, setTab, theme, setTheme, defs, setDefs }) {
  const c = st?.campaign, b = st?.budget
  const nav = useRef(null), ind = useRef(null)

  useLayoutEffect(() => {
    const n = nav.current, i = ind.current
    if (!n || !i) return
    const move = () => {
      const el = n.querySelector('button[aria-selected="true"]')
      if (!el) { i.style.width = '0px'; return }
      i.style.width = `${el.offsetWidth}px`
      i.style.transform = `translateX(${el.offsetLeft}px)`
      if (!n.dataset.ready) requestAnimationFrame(() => { n.dataset.ready = '1' })
    }
    move()
    const ro = new ResizeObserver(move)
    ro.observe(n)
    Array.from(n.querySelectorAll('button')).forEach((el) => ro.observe(el))
    if (document.fonts?.ready) document.fonts.ready.then(move).catch(() => {})
    return () => ro.disconnect()
  }, [tab, !!c])

  /* WAI-ARIA tabs: roving tabindex, arrow keys, Home/End. */
  const onKey = (e) => {
    const enabled = TABS.map(([t]) => t).filter((t) => t === 'Campaign' || c)
    const i = enabled.indexOf(tab)
    let next = null
    if (e.key === 'ArrowRight') next = enabled[(i + 1) % enabled.length]
    if (e.key === 'ArrowLeft') next = enabled[(i - 1 + enabled.length) % enabled.length]
    if (e.key === 'Home') next = enabled[0]
    if (e.key === 'End') next = enabled[enabled.length - 1]
    if (next) { e.preventDefault(); setTab(next); requestAnimationFrame(() => nav.current?.querySelector('button[aria-selected="true"]')?.focus()) }
  }

  const themeNext = { system: 'light', light: 'dark', dark: 'system' }
  const themeIcon = { system: 'system', light: 'sun', dark: 'moon' }
  return (
    <header className="appchrome">
      <div className="bar">
        <span className="brand"><Mark decorative height={26} />Gradient Bench<span className="ver">v1</span></span>
        {c && <>
          <span className="campaign" title={c.name}>{c.name}</span>
          <span className="stats">
            <span className="stat p2"><Term k="run budget">methods</Term> <b>{c.n_methods}</b> · runs <b>{c.n_runs}</b></span>
            {b && <span className="stat" title={b.reason}>budget <b>{b.used}/{b.limit}</b> <Meter b={b} />{b.finished ? ' closed' : ''}</span>}
            <span className="stat p4">best <Term k="CRF">CRF</Term> <b>{f(c.best_crf)}</b></span>
            <span className="stat p3"><Term k="sigma">σ</Term> <b>{c.sigma_crf == null ? '—' : f(c.sigma_crf)}</b>{c.sigma_dof ? <> (<Term k="dof">{c.sigma_dof} dof</Term>)</> : ''}</span>
          </span>
        </>}
        <span className="spacer" />
        {c && <Light v={st.drift.verdict} title={st.drift.reasons?.join(' ')}><Term k="verdict">instrument {st.drift.verdict}</Term></Light>}
        <span className="tools">
          <button className="ghost sm" aria-pressed={defs} onClick={() => setDefs(!defs)}
                  title={defs ? 'Definitions are in the tab order and underlined. Click to hide them.' : 'Show dotted definitions and put them in the tab order.'}>
            <Icon name="book" /><span>{defs ? 'definitions on' : 'definitions'}</span>
          </button>
          <button className="ghost sm" onClick={() => setTheme(themeNext[theme])} title={`Theme: ${theme}. Click to change.`} aria-label={`theme ${theme}`}>
            <Icon name={themeIcon[theme]} /><span>{theme}</span>
          </button>
        </span>
      </div>
      <nav className="tabs" ref={nav} role="tablist" aria-label="Gradient Bench" onKeyDown={onKey}>
        {TABS.map(([t, desc]) => (
          <button key={t} role="tab" id={`tab-${t}`} aria-selected={tab === t} aria-controls={`panel-${t}`}
                  tabIndex={tab === t ? 0 : -1} onClick={() => setTab(t)} title={desc}
                  disabled={t !== 'Campaign' && !c}>{t}</button>
        ))}
        <span className="tab-ind" ref={ind} aria-hidden="true" />
      </nav>
    </header>
  )
}

/* Tab CONTENT swaps in 0ms; the stagger runs once per panel, on first visit. */
function TabPanel({ name, seen, children }) {
  const first = !seen.current.has(name)
  const [entered, setEntered] = useState(!first)
  useEffect(() => {
    if (!first) return
    seen.current.add(name)
    const r = requestAnimationFrame(() => setEntered(true))
    return () => cancelAnimationFrame(r)
  }, [name])          // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className={`tabpanel${entered ? ' entered' : ''}`} role="tabpanel" id={`panel-${name}`}
         aria-labelledby={`tab-${name}`} {...(first ? { 'data-enter': '' } : {})}>
      {children}
    </div>
  )
}

function Live() {
  const msg = useAnnouncements()
  return <div className="sr-only live" aria-live="polite" aria-atomic="true">{msg}</div>
}

export default function App() {
  const [st, setSt] = useState(null)
  const [tab, setTab] = useLocal('gb2.tab', 'Campaign')
  const [theme, setTheme] = useTheme()
  const [defs, setDefs] = useLocal('gb2.defs', false)
  const seen = useRef(new Set())
  useEffect(() => { document.body.setAttribute('data-defs', defs ? 'on' : 'off') }, [defs])

  const reload = useCallback(() => get('/api/state').then((r) => {
    setSt(r.no_campaign ? null : r); return r
  }), [])
  useEffect(() => { reload().then((r) => { if (r?.no_campaign) setTab('Campaign') }) }, [reload])
  const onOpen = () => reload().then(() => setTab('Run'))
  const active = (st || tab === 'Campaign') ? tab : 'Campaign'

  return (
    <>
      <Live />
      <Bar st={st} tab={active} setTab={setTab} theme={theme} setTheme={setTheme} defs={defs} setDefs={setDefs} />
      <TabPanel name={active} seen={seen} key={active}>
        {active === 'Campaign' && <Campaign onOpen={onOpen} st={st} />}
        {active === 'Run' && st && <Run st={st} reload={reload} setTab={setTab} />}
        {active === 'Results' && st && <Results st={st} reload={reload} />}
        {active === 'Model' && st && <Model st={st} />}
        {active === 'Instrument' && st && <Instrument reload={reload} />}
        {active === 'Setup' && st && <Setup reload={reload} />}
        {active === 'Method' && st && <Method />}
      </TabPanel>
    </>
  )
}
