// One place that talks to Flask. Every long call goes through a job the page
// polls, so a GP fit never leaves a button dead.
const J = async (r) => {
  const body = await r.json().catch(() => ({ ok: false, error: r.statusText }))
  if (!r.ok || body.ok === false) throw new Error(body.error || `HTTP ${r.status}`)
  return body
}
export const get = (p) => fetch(p).then(J)
export const post = (p, body) =>
  fetch(p, { method: 'POST', headers: { 'Content-Type': 'application/json' },
             body: JSON.stringify(body || {}) }).then(J)

export async function upload(file) {
  const fd = new FormData()
  fd.append('file', file)
  return fetch('/api/upload', { method: 'POST', body: fd }).then(J)
}

export async function runJob(path, body, onProgress) {
  const { job } = await post(path, body)
  for (;;) {
    await new Promise((r) => setTimeout(r, 250))
    const s = (await get(`/api/jobs/${job.id}`)).job
    if (s.progress && onProgress) onProgress(s.progress)
    if (s.state === 'done') return s.result
    if (s.state === 'error') throw new Error(s.error)
  }
}
