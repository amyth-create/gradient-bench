# Changelog

## 2.0.0 — version 2 (2026-09-05)

Science unchanged by default: the same peak picker (hplc_picker 3.0.0), the same
CRF = n_clean_peaks × (1 − hump_time_fraction)², the same Matérn-5/2 ARD surrogate
with BoTorch's dimension-scaled priors and replicate-measured noise floor, the same
qLogNoisyExpectedImprovement acquisition.

Added
- Surrogate choice at campaign creation: matern52 (default), matern32, rbf, default.
- Acquisition choice at campaign creation: qlognei (default), qlogei, ucb (beta), logpi.
- Choices recorded in Config, printed on the Method page, carried by Duplicate settings,
  locked afterwards. Version-1 campaigns read as matern52 + qlognei.
- GET /api/optimiser/options, GET /api/version; /api/state carries `optimiser`.
- `prom` flag per pick in the review payload (provisional score while drawing).
- Tests for every option, parameter validation, recording, and legacy campaigns.

Rebuilt
- The whole interface: wizard, Run loop, Results, Model, Instrument, Setup, Method.
- Design tokens revised after the version-1 critique (see README).
