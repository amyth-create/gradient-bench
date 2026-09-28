# Roadmap

Noted 2026-09-16. Versions to date: v1 (`gradient-bench/`), v2 (`v2/`, released 1.0.0),
v3 (`v3/`, branch `v3-shape-aware`: a second design space and shape-aware kernels).

## v4 — a changeable objective, chosen from the analyst's own ranking

The objective is frozen at CRF 1.0.0 (`n_clean_peaks × (1 − hump_time_fraction)²`), chosen
once by a 56-candidate study against expert-ranked chromatograms. v4 makes that study a
per-campaign step, so the objective the optimiser climbs is the one that agrees with how
*this* analyst judges *this* sample.

### The planned pipeline

1. **Scouting.** The campaign opens with 10 scouting runs (spread-out methods, as the cold
   start does today), recorded as measured.
2. **Ranking.** The analyst ranks the 10 chromatograms by quality of separation, in the
   interface.
3. **Fitting an objective to the ranking.** Every candidate objective is scored on the 10
   runs and compared with the analyst's ranking (rank agreement, as the original study did
   with Kendall τ and Spearman). A plot shows how closely each candidate aligns with the
   ranking. The analyst chooses from the **top 5**.
4. **Custom objectives.** The analyst can also build their own: pick from the features the
   peak picker already extracts (clean peaks, shoulders, hump fraction, sum and mean
   resolution, fraction baseline-separated, first and last peak retention, …) and write a
   numerical expression over them. It is scored against the ranking like any other
   candidate.
5. **Lock and go.** Once an objective is chosen, it is written to the campaign's record
   (its expression, its version, and the ranking it was chosen against) and the Bayesian
   optimisation campaign begins.

### What already exists that this builds on

- The picker records every feature on every row (`gradient_bench/core/hplc_picker.py`,
  `RECORD_KEYS`), so candidate objectives can be evaluated on stored runs without
  re-running anything.
- Every row is stamped with `crf_version`, so a campaign whose objective was chosen in
  step 3 stays interpretable beside one that used 1.0.0.
- `rescore.py` re-measures every stored trace under new settings; the same machinery
  re-scores the scouting runs under a candidate objective.
- The structural guard in `crf.py` (trace-health descriptors can never enter the objective)
  must survive: the expression builder should expose separation features only.

### Open questions, to decide when v4 starts

- Whether the 10 scouting runs replace the cold start or precede it, and whether they
  count against the run budget.
- Ties in the analyst's ranking, and whether a partial ranking (best few, worst few) is
  enough.
- Whether the chosen objective may change mid-campaign (a new `crf_version` and a
  re-score) or is locked like the space and the kernel.
