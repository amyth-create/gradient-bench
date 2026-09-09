# Changelog

## 1.0.0 — first release (2026-09-09)

Closed-loop HPLC gradient method development: the app proposes a gradient, you run it and
upload the trace, it records the result and proposes the next one, and a control chart
watches the instrument for drift.

### The science

- Objective frozen at CRF version 1.0.0: `n_clean_peaks × (1 − hump_time_fraction)²`,
  chosen by a 56-candidate study against 20 expert-ranked chromatograms. The 15 other
  trace-health descriptors are structurally barred from the score by an assertion that runs
  on every analysis.
- Four-parameter design space (start %B, end %B, duration, temperature) with a linear
  constraint handed natively to the acquisition optimiser. Minimum span is zero, so
  isocratic methods stay reachable.
- Gaussian-process surrogate built through BoTorch's own constructors, preserving its
  dimension-scaled LogNormal lengthscale priors. BoTorch pinned to `>=0.18,<0.19` because
  its default kernel and priors change between releases.
- Surrogate chosen at campaign creation: Matérn-5/2 (default), Matérn-3/2, RBF, or the
  installed BoTorch default. Acquisition chosen at creation: log noisy expected improvement
  (default), log expected improvement, upper confidence bound with a `beta` the analyst
  sets, or log probability of improvement. Both are recorded in the campaign's Config
  sheet, printed on the Method page, carried by "Duplicate settings", and locked for the
  life of the campaign.
- Replicate-measured σ, pooled and installed as the model's noise floor; `floor_z` reported
  rather than hidden.
- Peak picker `hplc_picker` 3.0.0, vendored and self-contained. Changing it campaign-wide
  re-scores every stored trace.
- Drift control chart against a t = 0 anchor: WATCH at 1.5 CRF below, HALT at 3.0 or three
  consecutive falls, with six diagnostic channels read from reference runs only — including
  detection that the acquisition itself changed. Drift-adjusted reporting available but off
  by default, and never fed to the model.

### The app

- A campaign is a folder: workbook, every uploaded trace, and the model's history live
  inside it and can be copied to another machine and opened intact.
- Six-step new-campaign wizard (folder, campaign, instrument check, plan, model, review)
  with a live summary rail, per-step validation, and the bench-time cost of the plan stated
  before anything is created.
- Run loop: one authoritative state banner, the operative instruction as the lead line,
  provisional counts while a region is drawn, picker thresholds with their campaign default
  and raise-it / lower-it effects, advanced settings behind a disclosure.
- Results ledger with a best-so-far record that steps only when a run breaks it; Model tab
  with a learning curve legible at 40 methods, lengthscale bars and the noise readout;
  Instrument tab with the control chart and its channels; Method tab assembling a full
  methods statement from the running code.
- Budget denominated in unique methods and always converted to injections and bench hours.
  Extensions require a reason and are written to the record.
- Workbook and PDF export into the campaign folder.
- Design system: peak-category colours distinct from instrument-state colours, each also
  carrying a shape; interactive chips and static tags as separate components; WAI-ARIA tabs
  with arrow keys; system / light / dark theme switch; one polite live region per async
  job; nothing scrolls sideways at 375px.
- Runs fully offline: the frontend is pre-built, fonts are vendored, nothing loads from a
  CDN.

### Compatibility

Campaign folders written by earlier pre-release builds open unchanged and read as
Matérn-5/2 with log noisy expected improvement, which is what they were proposed with.
Bringing a folder forward only ever adds — no measured value is rewritten, nothing is
deleted, and the workbook is backed up first.
