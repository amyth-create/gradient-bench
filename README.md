# Gradient Bench — version 2

Closed-loop HPLC gradient method development. The app proposes a gradient method, you
run it on the instrument, upload the trace, check the peaks, and it records the result
and proposes the next one — with a Bayesian optimiser choosing the methods and a
control chart watching the instrument for drift.

**Version 2** keeps the science of version 1 unchanged — the same peak picker, the same
CRF objective, the same Matérn-5/2 surrogate with BoTorch's priors and the same
log noisy expected improvement acquisition by default — and rebuilds the interface
around it. It also lets a campaign choose its surrogate kernel and acquisition
function when it is created (and only then).

Tag: **version 2** (`gradient_bench.APP_VERSION == "2.0.0"`).

## What changed from version 1

**Science (additive only)**

- The surrogate kernel is chosen at campaign creation from Matérn-5/2 (default),
  Matérn-3/2, squared exponential (RBF) or the installed BoTorch default. Every choice
  is built through BoTorch's own constructor so the dimension-scaled LogNormal
  lengthscale priors are kept. Matérn-3/2 copies the prior and constraint off the
  module BoTorch builds and changes only `nu`.
- The acquisition function is chosen at campaign creation from log noisy expected
  improvement (default), log expected improvement, upper confidence bound (with a
  `beta` the analyst sets and defends) or log probability of improvement.
- Both choices are written to the campaign's Config sheet (`kernel`, `acquisition`,
  `acquisition_note`, `acq_params`, `app_version`), printed on the Method page, carried
  by "Duplicate settings", and **locked for the life of the campaign** — a campaign
  fitted under one kernel and refitted under another is not the same campaign.
- Version-1 campaigns open unchanged and read as Matérn-5/2 + qLogNEI, which is what
  they were proposed with.
- The review payload now says which picks the prominence pass found, so the Run tab can
  show a provisional score the instant a region is drawn (same rule as the server).

**Interface (rebuilt)**

- New-campaign wizard: folder → campaign → instrument check → plan → model → review,
  with a live summary rail, validation per step, and the bench-time cost of the plan
  stated before anything exists.
- Run tab: one authoritative state banner (never repeated in the step card), 26px step
  titles with the operative instruction as the lead line, an 8/4 layout for the method
  table and its facts, provisional counts while a drawn region is being confirmed,
  picker thresholds as slider + number with the change against the campaign default
  and raise-it / lower-it effects, advanced settings behind a disclosure, the
  finish/extend controls collapsed at the foot.
- Results: 8/4 dashboard (best-so-far record beside budget and instrument check), a
  real segmented control for the table view, tags that are never styled as chips.
- Model: what the campaign is fitted with, a learning curve that stays legible at 40
  methods (nice ticks, every-5th label, hover readout), a recent-trend caption,
  lengthscale bars capped at 1.
- Instrument: control chart with a y-axis, band fills that no longer flood the plot,
  channels as disclosures with lamps, drift-adjusted reporting as a segmented switch.
- Design system revisions from the critique: peak-category colours no longer share hex
  values with instrument-state colours (teal / amber / violet, each still with its
  shape); control strokes clear 3:1; chips (interactive) and tags (static) are
  different components; the label-caps style is reserved for eyebrows, table headers
  and `<dt>`; banner headlines are 15px sentence case; a 5-glyph inline SVG icon set
  replaces the Unicode glyphs; a system / light / dark theme switch persisted per
  browser; WAI-ARIA tabs with arrow keys; glossary terms out of the tab order by
  default with a "definitions" switch that puts them back; one polite live region for
  every async job; chromatograms carry an axis title; nothing scrolls sideways at 375px.

## Install and run

Version 2 reuses version 1's virtual environment (torch and BoTorch are 3 GB; there is no
reason to install them twice). From this folder:

```bash
../gradient-bench/.venv/bin/python -m gradient_bench.api.app --port 5051
```

then open http://127.0.0.1:5051. The frontend is pre-built into
`gradient_bench/api/static/`, so nothing needs Node to run it. Nothing loads from a CDN
and both fonts are vendored — the lab PC may have no internet.

To install from scratch instead:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r gradient_bench/requirements.txt flask
```

To change the interface:

```bash
cd frontend && npm install && npm run build   # builds into ../gradient_bench/api/static
npm run dev                                   # dev server, proxies /api to :5051
```

## Check it works

```bash
../gradient-bench/.venv/bin/python -m pytest gradient_bench/tests/test_core.py -q
../gradient-bench/.venv/bin/python -m gradient_bench.scripts.verify_install
```

`testdata/` holds the twenty real chromatograms and their sheet, so the golden-file
tests run straight out of this folder.

## What is in here

```
gradient_bench/     the app: core science, store, loop engine, re-scoring, Flask API
  core/optimiser.py   surrogate + acquisition options (SURROGATE_OPTIONS, ACQUISITION_OPTIONS)
  store/campaign.py   Campaign.acquisition(), acq_params(), optimiser_summary()
  api/app.py          GET /api/optimiser/options, GET /api/version; create takes the choices
frontend/           React source (Vite): tabs/, components/, styles/, glossary.js
testdata/           21 real traces + their sheet
hplc_picker.py      three-line shim to the vendored picker
```

The complete account of the version-1 build this is derived from — every decision and
why — is `../GRADIENT_BENCH_BUILD.md`.
