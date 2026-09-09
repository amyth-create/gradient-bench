# gradient_bench — the package

Developer notes on the package itself. For what the app is, how to install it and
how to use it, see the [README at the repository root](../README.md).

Two layers: the science (the 4-parameter optimiser and the peak picker, as a library
an API can call) and the loop on top of it (a Flask API, a React page, and the cycle
from proposal to record).

## Run the app

```bash
pip install -r gradient_bench/requirements.txt flask
python -m gradient_bench.api.app                 # then open http://127.0.0.1:5051
python -m gradient_bench.api.app --open /path/to/campaign   # skip the picker
```

The frontend is pre-built into `gradient_bench/api/static/`, so nothing needs
Node to *run* it. To change the interface:

```bash
cd frontend && npm install && npm run build      # builds straight into api/static
npm run dev                                      # or dev server, proxying /api to :5051
```

Everything is bundled - no CDN, no web fonts - because the lab PC may have no
internet.

## Layout

| module | what it owns |
|---|---|
| `core/crf.py` | the objective. One definition, versioned, imported everywhere and edited nowhere. |
| `core/space.py` | the 4-parameter box, the constraint, `check4` - the single feasibility gate. |
| `core/optimiser.py` | the GP, the measured noise floor, qLogNEI, the optional run-order covariate. |
| `core/drift.py` | reference schedule, PASS/WATCH/HALT, the drift baseline, the phantom incumbent. |
| `core/uncertainty.py` | the fixed Sobol grid, the four posterior panels, the per-run audit. |
| `core/picker.py` | the adapter over `hplc_picker`, campaign vs per-run configuration. |
| `store/sheet.py` | the workbook - Data, Campaign, Config. Atomic writes, timestamped backups. |
| `store/campaign.py` | a campaign is a folder; create, open, the recent registry. |
| `store/state.py` | the in-progress loop, mirrored to disk so closing the app loses nothing. |
| `loop.py` | the engine: every gate in one place - anchor first, cold start, cadence, halt. |
| `jobs.py` | background work, so a GP fit never freezes the page. |
| `api/app.py` | HTTP. Thin on purpose: if a decision is being made here, it is in the wrong place. |
| `frontend/` | React + Vite. The chromatogram, and drag-to-draw hump regions. |

## Run it

```bash
python -m pytest gradient_bench/tests -q                    # 130 tests
python -m gradient_bench.scripts.replay_corpus             # the whole corpus, headless
```

`hplc_picker.py` must be importable - it is found automatically if it sits
beside the package. The replay and the golden-trace tests need the trace
corpus; point at it with `--trace-dir` or `GB_TRACE_DIR`.

## The golden files

`golden_files.json` holds the expected peak count and CRF for all twenty real
traces, adjudicated by Amyth against the rendered chromatograms. Fifteen
reproduce at campaign defaults. Five were contested and settled:

| trace | outcome |
|---|---|
| `method4` | 13 peaks - the picker was right, the old record was wrong |
| `method7` | 1 peak - the picker was right, the old record was wrong |
| `method2` | 3 peaks with a hand-drawn hump at 6.74-13.85 min |
| `method12` | 7 peaks with a hand-drawn hump at 19.25-22.39 min |
| `method9` | 13 peaks, needs `snr=2.5` - neither the record (10) nor the default picker (8) |

Do not edit these without re-adjudicating the trace.

## Two decisions the data made

**`snr = 5.0` is the campaign default.** Tested against every adjudicated peak
count: 5.0 gets 19/20, 4.0 gets 11/20, 3.0 gets 5/20, 2.5 gets 5/20. Lowering
it to rescue `method9` would break fifteen traces. `method9` is a per-run tune.

**`arpls_lam` stays at 1e5.** On `method2` the baseline fitter demonstrably
absorbs the hump - it sits at 99.9-127.7 inside the span against 43-101 outside
- but stiffening it never recovers a hump at any lam from 1e6 to 1e9, and
1e7 breaks two traces that currently agree. The manual hump override is the
answer, which is why it is a first-class feature rather than an escape hatch.

## What is deliberately not here

Run-order modelling works (`model_run_order=True`) but is off. It makes the
surrogate 5-D while the candidate stays 4-D, and whether it helps should be
settled by leave-one-out comparison on a recorded campaign, not by running two
campaigns.


## The loop, and its gates

    anchor first   a new campaign cannot serve a design method until the
                   instrument-check run is measured - it is the zero every later
                   comparison is made against and cannot be added afterwards
    cold start     until the seeds are done, serve spread-out methods and record
                   what they score; no model is fitted
    cadence        every N proposals the next thing on the instrument is the
                   reference. ENFORCED - `Schedule.blocks_proposing`
    halt           if the reference has left its limits, proposing is blocked

All four live in `loop.py::next_step`. The API translates JSON and nothing else.

## Drawing a hump

Drag across the chromatogram. It replaces automatic detection for that run,
which is the right semantics: it annotates one trace rather than moving a
threshold, so it implies nothing about any other run. Verified against the
adjudicated corpus - drawing 19.25-22.39 min on `method12` through the UI gives
exactly 7 clean peaks and CRF 5.310, matching the golden file.

Per-run tuning is unrestricted on design runs: the picker settings ESTIMATE a
physical truth rather than defining it, so an analyst correcting a failed
estimate reduces error. Reference runs are the exception and
`picker.resolve_config` refuses an override on one - the reference series is a
differential measurement, and retuning one of its runs makes a picking artefact
look like instrument drift.
