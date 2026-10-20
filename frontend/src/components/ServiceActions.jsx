import React, { useState } from 'react'
import { post } from '../api'
import { useAsync } from '../hooks'
import Term from './Term.jsx'
import { Busy, Err } from './ui.jsx'

/* ── the instrument was serviced: close the block, re-anchor ───────────
   A HALT has exactly one exit. Lowering the limit makes the chart pass
   without making the instrument right; running the reference again inside
   the same block compares it to an anchor the instrument may no longer be
   able to reach. So the analyst fixes the instrument, says what was done,
   and the next instrument check anchors a NEW block. */
const CHIPS = ['fresh mobile phase', 'column replaced', 'column re-equilibrated', 'oven settled, injector checked']

export default function ServiceActions({ reload, compact = false }) {
  const { busy, err, run } = useAsync()
  const [why, setWhy] = useState('')
  const go = () => run('recording the service', () => post('/api/campaign/service', { reason: why }).then(() => { setWhy(''); return reload() }))
  return (
    <div className="ctrl mt3">
      <div className="top"><label><Term k="block">The instrument has been serviced — re-anchor</Term></label></div>
      <input value={why} placeholder="what was done (goes on the record)" onChange={(e) => setWhy(e.target.value)} />
      <div className="chips">
        {CHIPS.map((p) => <button key={p} className="chip" aria-pressed={why === p} onClick={() => setWhy(p)}>{p}</button>)}
      </div>
      <div className="row">
        <button className="primary" disabled={!why.trim()} onClick={go}>Serviced — run the instrument check again</button>
      </div>
      {!compact && <p className="muted mb0" style={{ fontSize: 12.5 }}>Closes the current block. The next thing on the instrument is the instrument-check method, which anchors a new block: the chart, its limits and the monotone rule restart from it. Nothing before the service is corrected or discarded — it stays on the record as its own block.</p>}
      <Busy label={busy} /><Err>{err}</Err>
    </div>
  )
}
