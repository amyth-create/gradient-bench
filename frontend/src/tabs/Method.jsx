import React, { useEffect, useState } from 'react'
import { get } from '../api'
import Term from '../components/Term.jsx'
import Mark from '../components/Mark.jsx'
import Icon from '../components/Icon.jsx'
import { Banner, Skeleton } from '../components/ui.jsx'

/* The page you print when someone asks what you did. Assembled by the
   server from the running code and the campaign's own workbook. */
export default function Method() {
  const [d, setD] = useState(null)
  const [err, setErr] = useState(null)
  useEffect(() => { get('/api/setup').then(setD).catch((e) => setErr(e.message)) }, [])
  if (err) return (
    <main><h1>Method</h1>
      <Banner kind="WATCH" eyebrow="method page" title="Could not assemble this page"><p>{err}</p>
        <p className="muted mt2">Nothing is wrong with the campaign’s record — this page only reads it. Reopen the campaign, or read the workbook directly.</p></Banner></main>)
  if (!d) return <main><h1>Method</h1><Skeleton rows={5} /></main>
  return (
    <main className="doc">
      <div className="printmast"><Mark height={22} decorative />Gradient Bench · version 2 · {d.campaign?.name}</div>
      <div className="pagehead noprint">
        <div><h1>Method</h1>
          <p className="sub mb0">Everything this campaign is doing and why, read from the running code and this campaign’s own workbook — not from a page someone maintains by hand. Generated {(d.generated_at || '').replace('T', ' ')}.</p></div>
        <button onClick={() => window.print()}><Icon name="print" />Print</button>
      </div>
      <nav className="toc noprint" aria-label="sections">
        {d.sections.map((s) => <a key={s.key} href={`#sec-${s.key}`}>{s.title}</a>)}
      </nav>
      {d.sections.map((s) => (
        <section className="card" key={s.key} id={`sec-${s.key}`}>
          <h2>{s.title}</h2>
          {s.blurb && <p className="muted" style={{ marginTop: 0, maxWidth: '76ch' }}>{s.blurb}</p>}
          {s.math?.length > 0 && <pre className="math">{s.math.join('\n')}</pre>}
          {s.rows?.length > 0 && (
            <dl className="kv wide mt3">
              {s.rows.map((r, i) => (
                <React.Fragment key={i}>
                  <dt>{r.term ? <Term k={r.term}>{r.label}</Term> : r.label}</dt>
                  <dd><b>{r.value === null || r.value === '' ? '—' : String(r.value)}</b>
                    {r.note && <div className="muted" style={{ marginTop: 2 }}>{r.note}</div>}</dd>
                </React.Fragment>
              ))}
            </dl>
          )}
          {(s.tables || []).map((t, i) => (
            <div key={i} className="mt4">
              <h3>{t.title}</h3>
              {t.note && <p className="muted" style={{ marginTop: -2, maxWidth: '76ch' }}>{t.note}</p>}
              <div className="tablewrap">
                <table>
                  <thead><tr>{t.columns.map((cn, k) => <th key={k} className={k === 0 ? 'l' : ''}>{cn}</th>)}</tr></thead>
                  <tbody>{t.rows.map((row, k) => (
                    <tr key={k}>{row.map((cell, m) => <td key={m} className={m === 0 ? 'l mono' : ''} style={m > 0 && typeof cell === 'string' && cell.length > 24 ? { whiteSpace: 'normal', textAlign: 'left', fontFamily: 'var(--f-ui)' } : undefined}>
                      {cell === null || cell === undefined || cell === '' ? '—' : String(cell)}</td>)}</tr>))}</tbody>
                </table>
              </div>
            </div>
          ))}
          {(s.warnings || []).map((w, i) => <Banner key={i} kind="WATCH" className="mt3 mb0"><p>{w}</p></Banner>)}
        </section>
      ))}
    </main>
  )
}
