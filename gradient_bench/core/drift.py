"""drift.py - the instrument control chart, and the phantom incumbent.

THE PROBLEM.  A GP models y = f(x) + iid noise. Reality is y = f(x) + g(t) +
noise, where g is whatever moved across the campaign - mobile phase, column
condition, injected mass, ambient temperature. The model has exactly one place
to put g: sigma^2. That inflation is not local. It flattens the posterior
EVERYWHERE, including regions the drift never touched, which makes the
acquisition surface nearly uniform and pushes proposals to the box boundary.
The campaign then looks converged when it has merely been outrun by the clock.

REPLICATION CANNOT SEE IT.  Two runs back to back share their moment in time.
They measure sigma WITHIN a moment, not the trend BETWEEN moments, and are
structurally blind to it. Tight replicates alongside a flat learning curve is
the SIGNATURE of drift, not evidence against it.

SO ONE METHOD IS HELD FIXED AND RE-RUN.  Tagged separately so it can never
reach the GP, stamped with its instrument position, and charted against a t = 0
anchor with an explicit PASS / WATCH / HALT rule. It is the only thing in the
loop whose movement can only be g.

WHAT THIS DOES NOT DO.  It MEASURES drift; it does not remove it. The physical
fix - fresh mobile phase, a re-equilibrated column, a settled oven - is still
the first-choice response. A HALT is an instruction to go and service
something, not a number to correct against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# ── the decision rule ───────────────────────────────────────────────────────
WATCH_DELTA = 1.5      # CRF below the anchor -> WATCH
HALT_DELTA = 3.0       # CRF below the anchor -> HALT
MONOTONE_K = 3         # this many consecutive falls -> HALT
REFERENCE_EVERY = 5    # proposals between reference runs

assert HALT_DELTA > WATCH_DELTA > 0

#: Suggested reference methods offered as presets at campaign creation. The
#: analyst may enter any feasible method instead; these are starting points.
REFERENCE_PRESETS = {
    "best_so_far": {
        "u": None,   # filled from the campaign's best warm-start run
        "label": "Best warm-start candidate",
        "why": "The most peak-rich method available has the most structure to "
               "lose, so it is the most sensitive detector of degradation.",
    },
    "shallow_survey": {
        "u": [0.10, 0.30, 50.0, 45.0],
        "label": "Shallow survey gradient",
        "why": "0.4 %B/min over 50 min spreads the sample out, so retention and "
               "width shifts are obvious. Independent of how a campaign went, "
               "which makes it comparable between campaigns.",
    },
}


@dataclass
class ReferenceRun:
    run_order: float
    crf: float
    name: str = ""


@dataclass
class DriftVerdict:
    verdict: str                     # PASS | WATCH | HALT
    n: int
    anchor: float = float("nan")
    last: float = float("nan")
    delta: float = float("nan")
    slope: float = float("nan")
    intercept: float = float("nan")
    rho: float = float("nan")
    span: float = float("nan")
    implied_fall: float = float("nan")
    monotone: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def blocks_proposing(self) -> bool:
        return self.verdict == "HALT"

    @property
    def falling(self) -> bool:
        """Is the reference below its anchor? A scientific judgement, so it
        lives on the verdict rather than being re-derived by each caller."""
        return bool(np.isfinite(self.delta) and self.delta < 0)

    def actions(self) -> list[str]:
        return {
            "PASS": ["Continue the loop."],
            "WATCH": [
                "Keep proposing, but do not trust the incumbent: read the raw and "
                "the drift-adjusted best-so-far side by side.",
                "Bring the next reference run forward rather than waiting the full "
                "cadence.",
            ],
            "HALT": [
                "STOP PROPOSING AND SERVICE THE INSTRUMENT.",
                "Fresh mobile phase; re-equilibrate or replace the column; check "
                "injection volume and vial age; let the column oven settle.",
                "Then re-anchor with a new reference run and treat everything "
                "before the service as a SEPARATE BLOCK - not as data to be "
                "corrected.",
                "Do not lower the threshold to make the chart pass.",
            ],
        }[self.verdict]


def _series(refs: Sequence[ReferenceRun]) -> tuple[np.ndarray, np.ndarray]:
    r = sorted(refs, key=lambda x: x.run_order)
    return (np.array([x.run_order for x in r], dtype=float),
            np.array([x.crf for x in r], dtype=float))


def fit_reference_trend(refs: Sequence[ReferenceRun]):
    """OLS CRF ~ run_order over the reference series -> (slope, intercept, rho)."""
    if refs is None or len(refs) < 2:
        return float("nan"), float("nan"), float("nan")
    ro, y = _series(refs)
    if np.ptp(ro) == 0:
        return float("nan"), float("nan"), float("nan")
    from scipy.stats import linregress, spearmanr
    lr = linregress(ro, y)
    rho = float(spearmanr(ro, y).correlation) if len(ro) >= 3 else float("nan")
    return float(lr.slope), float(lr.intercept), rho


def drift_verdict(refs: Sequence[ReferenceRun],
                  watch_delta: float = WATCH_DELTA,
                  halt_delta: float = HALT_DELTA,
                  k: int = MONOTONE_K) -> DriftVerdict:
    """PASS / WATCH / HALT from the reference series.

    NO REFERENCE IS NOT A PASS. With nothing measured, g(t) is unmeasured, and
    an unmeasured confound is a WATCH, not a clean bill of health.
    """
    n = 0 if refs is None else len(refs)
    v = DriftVerdict(verdict="PASS", n=n)
    if n == 0:
        v.verdict = "WATCH"
        v.reasons.append(
            "No reference run exists, so instrument drift is UNMEASURED. That is "
            "not a pass. Run the t = 0 anchor before anything else.")
        return v

    ro, y = _series(refs)
    v.anchor, v.last = float(y[0]), float(y[-1])
    v.delta = float(y[-1] - y[0])
    if n == 1:
        v.reasons.append(
            f"Anchor only (n = 1): the chart has a zero but no trend yet. The "
            f"next reference is due after {REFERENCE_EVERY} proposals.")
        return v

    slope, intercept, rho = fit_reference_trend(refs)
    v.slope, v.intercept, v.rho = slope, intercept, rho
    v.span = float(ro[-1] - ro[0])
    v.implied_fall = float(-slope * v.span) if np.isfinite(slope) else float("nan")
    v.monotone = bool(n >= k and np.all(np.diff(y[-k:]) < 0))

    if v.delta <= -halt_delta:
        v.verdict = "HALT"
        v.reasons.append(
            f"The last reference is {-v.delta:.2f} CRF BELOW the t=0 anchor "
            f"({v.last:.2f} vs {v.anchor:.2f}); the halt limit is {halt_delta:.2f}.")
    if v.monotone:
        v.verdict = "HALT"
        v.reasons.append(
            f"The last {k} reference runs fall monotonically "
            f"({' > '.join(f'{x:.2f}' for x in y[-k:])}).")
    if v.verdict != "HALT":
        if v.delta <= -watch_delta:
            v.verdict = "WATCH"
            v.reasons.append(
                f"The last reference is {-v.delta:.2f} CRF below the anchor "
                f"(watch limit {watch_delta:.2f}).")
        elif np.isfinite(v.implied_fall) and v.implied_fall >= watch_delta:
            v.verdict = "WATCH"
            v.reasons.append(
                f"The fitted trend is {slope:+.3f} CRF/run, implying a fall of "
                f"{v.implied_fall:.2f} CRF over the {v.span:.0f} runs measured "
                f"so far (watch limit {watch_delta:.2f}).")
    if v.verdict == "PASS":
        v.reasons.append(
            f"The reference is within {abs(v.delta):.2f} CRF of its anchor, shows "
            f"no {k}-run monotone fall, and its trend ({slope:+.3f} CRF/run) "
            f"implies {v.implied_fall:+.2f} CRF over the span measured. Continue.")
    return v


# ── the baseline, and the phantom incumbent ─────────────────────────────────
def drift_baseline(run_order, refs: Sequence[ReferenceRun]) -> np.ndarray:
    """Reference CRF interpolated at arbitrary run-order positions.

    HELD FLAT outside the measured span - np.interp does not extrapolate the
    trend past the last anchor, and neither should we. Between two references
    the baseline is a straight line BY ASSUMPTION: a step change (a new mobile
    phase batch mid-block) is smeared across the interval and every run in it
    is mis-corrected. If you suspect steps, run the reference more often - do
    not interpolate harder.
    """
    q = np.asarray(run_order, dtype=float)
    if refs is None or len(refs) == 0:
        return np.full(q.shape, np.nan, dtype=float)
    ro, y = _series(refs)
    if len(ro) == 1:
        return np.full(q.shape, float(y[0]))
    return np.interp(q, ro, y)


def drift_offset(run_order, refs: Sequence[ReferenceRun]) -> np.ndarray:
    """baseline(t) - baseline(anchor). Zero at the anchor by construction."""
    q = np.asarray(run_order, dtype=float)
    if refs is None or len(refs) == 0:
        return np.zeros(q.shape, dtype=float)
    ro, y = _series(refs)
    return drift_baseline(q, refs) - float(y[0])


def drift_adjust(crf, run_order, refs: Sequence[ReferenceRun]) -> np.ndarray:
    """adjusted = raw - (baseline(t) - baseline(anchor)).

    LEGITIMATE ONLY IF THE DRIFT IS REVERSIBLE, and this function cannot check
    that. Correcting an irreversibly degraded column produces scores the
    instrument can no longer produce - you would be optimising toward a machine
    you no longer own. That is why it is a flag, not a default.

    With no references recorded this is the identity, so every call site works
    unchanged before the monitor has data.
    """
    return np.asarray(crf, dtype=float) - drift_offset(run_order, refs)


@dataclass
class Incumbent:
    n: int
    n_ref: int
    raw_best: float
    raw_method: Optional[str]
    raw_run_order: float
    adj_best: float
    adj_method: Optional[str]
    adj_run_order: float
    adj_of_raw_best: float
    phantom: bool

    def warning(self) -> Optional[str]:
        if not self.phantom:
            return None
        return (f"PHANTOM INCUMBENT: the raw and drift-adjusted winners are "
                f"DIFFERENT methods ({self.raw_method} vs {self.adj_method}). Part "
                f"of the raw winner's lead is WHEN it ran. Re-run the adjusted "
                f"winner now, back to back with the reference, before treating "
                f"either as the incumbent.")


def best_so_far(scores, run_orders, names, refs: Sequence[ReferenceRun]) -> Incumbent:
    """Incumbent, raw and drift-adjusted. Chooses nothing; reports both.

    The phantom flag fires when the two argmaxes name DIFFERENT methods. That
    is the failure that matters: an early run carrying a drift bonus that
    nothing later can beat, so the campaign concludes it has converged against
    a decaying baseline and adopts the second-best chemistry.
    """
    y = np.asarray(scores, dtype=float).ravel()
    ro = np.asarray(run_orders, dtype=float).ravel()
    nm = np.asarray(names, dtype=object).ravel()
    n_ref = 0 if refs is None else len(refs)
    if y.size == 0:
        return Incumbent(0, n_ref, float("nan"), None, float("nan"),
                         float("nan"), None, float("nan"), float("nan"), False)
    i_raw = int(np.nanargmax(y))
    ya = drift_adjust(y, ro, refs) if n_ref else y.copy()
    i_adj = int(np.nanargmax(ya))
    return Incumbent(
        n=int(y.size), n_ref=n_ref,
        raw_best=float(y[i_raw]), raw_method=str(nm[i_raw]),
        raw_run_order=float(ro[i_raw]),
        adj_best=float(ya[i_adj]), adj_method=str(nm[i_adj]),
        adj_run_order=float(ro[i_adj]),
        adj_of_raw_best=float(ya[i_raw]),
        phantom=bool(n_ref >= 2 and nm[i_raw] != nm[i_adj]))


# ── the schedule ────────────────────────────────────────────────────────────
@dataclass
class Schedule:
    next_is: str            # "reference" | "proposal"
    anchor: bool
    n_ref: int
    n_methods: int
    methods_since: int
    overdue: int
    reason: str

    @property
    def blocks_proposing(self) -> bool:
        """ENFORCED, not advised. The cadence is the only thing keeping every
        design run anchored in time; a missed reference means you cannot tell
        which later results were chemistry and which were the clock."""
        return self.next_is == "reference"


def reference_schedule(n_methods_total: int, methods_since_last_reference: int,
                       n_ref: int, every: int = REFERENCE_EVERY) -> Schedule:
    """Is the next thing on the instrument a reference or a proposal?

    Counts PROPOSALS (distinct methods), not runs, so `every = 5` with two
    replicates means one reference per ten design runs.
    """
    if n_ref == 0:
        return Schedule(
            next_is="reference", anchor=True, n_ref=0, n_methods=n_methods_total,
            methods_since=n_methods_total, overdue=0,
            reason="No reference has been run yet. The t = 0 ANCHOR must be "
                   "measured BEFORE the cold-start seeds - without it every later "
                   "reference is a difference from nothing.")
    due = methods_since_last_reference >= every
    return Schedule(
        next_is="reference" if due else "proposal", anchor=False, n_ref=n_ref,
        n_methods=n_methods_total, methods_since=methods_since_last_reference,
        overdue=max(methods_since_last_reference - every, 0),
        reason=f"{methods_since_last_reference} proposal(s) since the last "
               f"reference; the cadence is {every}.")
