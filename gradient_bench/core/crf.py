"""crf.py - the objective.

ONE definition, versioned, imported by everything and edited by nothing.

    CRF = n_clean_peaks * (1 - hump_time_fraction) ** 2

A high count of well-separated peaks is good; time spent under an unresolved
complex mixture is bad, and the square makes the penalty bite only once the
hump occupies a real fraction of the run.

WHY THIS OBJECTIVE.  56 pre-declared candidates (34 hand-written, 22 from the
literature) were scored on 20 expert-ranked chromatograms against both of two
reviewers' orderings and one reviewer's 0-10 ratings.  This one finished 1st of
56 on the composite and 1st on tau vs the reviewers (0.857 against an
inter-rater ceiling of 0.937), Spearman vs the ratings (0.970), tie-aware tau,
out-of-sample isotonic calibration (RMSE 0.734 rating points, best of 56) and
NDCG@20 (0.997).

THE OBJECTIVE IS FIXED.  A moving objective makes every campaign incomparable
with every other, so it is settled here and never re-derived downstream.  If it
ever must change, bump CRF_VERSION: every recorded run is stamped with the
version that scored it, so a mixed sheet stays interpretable instead of
silently averaging two objectives.

TRACE HEALTH IS NOT AN INPUT.  The trace_* descriptors describe the INSTRUMENT,
not the separation.  Feeding them to the objective would let a drifting
detector move the score the optimiser is climbing.  `assert_objective_is_clean`
is the guard, and it runs on EVERY analysis.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

CRF_VERSION = "1.0.0"

#: The only measurements compute_crf reads.
CRF_COLUMNS: list[str] = ["n_clean_peaks", "hump_time_fraction"]

#: Recorded beside the objective and useful for diagnosis, but never inputs.
CRF_DIAGNOSTIC_COLUMNS: list[str] = [
    "sum_resolution", "mean_resolution", "frac_baseline_sep",
    "n_shoulder_fronting", "n_ucm_humps",
]

RS_BASELINE = 1.5   # textbook baseline resolution; descriptive only


def compute_crf(row: Mapping[str, Any]):
    """Score one method, or a whole table, from its raw measurements.

    `row` may be a dict, a pandas Series or a pandas DataFrame carrying
    CRF_COLUMNS; the return type follows the input, so this works elementwise
    on a DataFrame without a special case.
    """
    return row["n_clean_peaks"] * (1.0 - row["hump_time_fraction"]) ** 2


def assert_objective_is_clean(trace_health_keys: Sequence[str]) -> None:
    """Fail loudly if an instrument descriptor has leaked into the objective."""
    leaked = sorted(set(CRF_COLUMNS) & set(trace_health_keys))
    if leaked:
        raise AssertionError(
            f"trace-health descriptor(s) {leaked} reached CRF_COLUMNS. The "
            "objective must describe the separation, not the instrument.")
