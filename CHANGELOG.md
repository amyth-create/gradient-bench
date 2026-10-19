# Changelog

## 1.2.0 — a way out of a HALT, and quieter metadata (2026-10-19)

### The science

- **A HALT now has an exit.** Before, a HALT blocked proposing and nothing could lift it:
  the cadence had just been met, so no reference was due, and any reference that did run
  was compared to the same anchor the instrument could no longer reach. Now the analyst
  records that the instrument was **serviced**, with a reason, and the next instrument check
  anchors a **new block**. The verdict, the schedule and the health channels read the current
  block only: the chart, its limits and the monotone rule restart from the new anchor, and
  the monotone rule never counts a fall across a service. Everything before the service
  stays on the record as its own block — charted against its own anchor, never corrected
  across the boundary. A service is refused while a run is on the instrument (it belongs to
  the block that is closing) and when the current block has no anchor yet. It is not gated
  on the verdict: a column replaced on a PASS is still a new instrument state.
- **Drift adjustment is per block.** Each run is corrected against its own block's anchor;
  a baseline is never interpolated through a service.

### The app

- On a HALT the Run and Instrument tabs carry the service form; the control chart draws a
  boundary at every service and the Instrument tab lists the service record.
- The new-campaign wizard and the Setup tab no longer caption each metadata box *material*;
  the sentence above the boxes names the six fields that are, and a material change still
  asks for a reason.
- The export's reference series carries a `block` column with each run's delta from its own
  block's anchor; the PDF chart marks services; the Method page prints the service record.

### Compatibility

Campaign folders written by 1.0.0 and 1.1.0 open unchanged: a folder without a
`service_history` field is one block. The Data sheet's schema does not change.

## 1.1.0 — version 3: a second design space and shape-aware surrogates (2026-09-15)

After Boelrijk, Detlefsen, Dürholt and Haas, *Shape-aware Gaussian process kernels for
closed-loop Bayesian optimization of gradients in automated LC method development*
(ChemRxiv, 2026).

### The science

- **The design space is a per-campaign choice, locked at creation** like the kernel.
  `ramp4` (the default, unchanged) is the single ramp. `steps8` is a four-node pump
  timetable written in deltas: a starting %B, three composition steps, three time steps
  (the first node is always at t = 0) and the column temperature. A composition step of
  zero is a hold, a time step of zero an instantaneous jump. Monotonicity lives in the
  box and nothing else is bounded per segment; the only constraints are on the sums (final
  %B at most 100, duration at most 30 min, and a 1-minute technical floor on the total),
  every one handed to the acquisition optimiser natively. Cold-start seeds come from the
  paper's window-relative sampler. The instrument check stays a single ramp.
- **Four shape-aware surrogates.** The distance between two methods is the area between
  their gradient profiles on a normalised 0–1 time axis with %B as a fraction (the
  Wasserstein-1 distance, Eq. 16), turned into a covariance as `exp(−d_W/ℓ_W)` (Eq. 17)
  and multiplied by a second kernel over duration and temperature (Eq. 18) — `wasserstein`
  with the paper's RBF, `wasserstein_matern` with a Matérn-5/2 — or over start %B, end %B,
  duration and temperature as well, the paper's Wasserstein+ (Eq. 19): `wasserstein_plus`
  and `wasserstein_plus_matern`. In `steps8` the endpoint features are derived from the
  timetable inside the kernel.
- The profile distance is implemented for piecewise-linear profiles in general, exactly
  (union of breakpoints, closed form per interval), in numpy and as a differentiable torch
  kernel; tested against quadrature, the metric axioms, the Gram matrix's eigenvalues, the
  identity `d_W = 0` between two timetables that draw the same profile, and vertical steps.
- Priors stay BoTorch's: `ℓ_W` carries the dimension-scaled LogNormal with D = 1 and the
  `GreaterThan(0.025)` constraint; the second factor is built by BoTorch's own constructor.
- **The longest method is 30 min, not 60.** `duration_min` is bounded 10–30 in `ramp4`, the
  same ceiling `steps8` has on its total duration, so the two spaces share one envelope. The
  shallow-survey reference preset becomes 10→30 %B over 30 min at 45 °C, so the reference is
  never longer than anything the loop proposes. A campaign created under the old box keeps
  its runs; a run longer than 30 min now sits outside the envelope the acquisition searches.
- Matérn-5/2 on the single ramp remains the recommended default. The shape-aware kernels
  earn their keep in `steps8`, where many timetables draw one profile; in `ramp4` they only
  change what *nearby* means. None is yet validated on this bench.

### The app

- The new-campaign wizard's model step now makes three locked choices: design space,
  surrogate, acquisition. Shape-aware kernels carry a *shape-aware* tag.
- The Run and Results tabs render a gradient table of any number of nodes, and the Results
  methods table gains a *segments* column. Campaign cards, the Setup tab and the Model tab
  name the space.
- The Model tab reports a `shape` lengthscale — in units of area between profiles — with a
  note on how to read it, and names derived features as such in `steps8`.
- The Method page prints the campaign's space (parameters, every constraint in absolute and
  normalised form, the sampler, the gate) and each factor of a product kernel with its
  features, prior and constraint, assembled from the running module as before.
- The uncertainty history widens with `ls_shape` and the `steps8` names only on campaigns
  that use them. `replay_corpus` takes `--kernel`.
- Glossary: *design space*, *timetable redundancy*, *shape-aware*, *Wasserstein*,
  *gradient profile*, *shape lengthscale*.

### Compatibility

Campaign folders written by 1.0.0 open unchanged: a folder without a `space` key reads as
the single ramp, and keeps the kernel it was created with. Nothing about the objective, the
noise model, the acquisition or the drift chart has changed.

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
