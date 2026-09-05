"""space.py - the 4-parameter design space, its constraint, and the feasibility gate.

    u = [start_phi, end_phi, duration_min, T]

      start_phi     starting %B as a fraction   (0.10 = 10 %B)
      end_phi       final   %B as a fraction    (must be >= start_phi)
      duration_min  gradient length in minutes
      T             column temperature, degC

One linear ramp. It maps onto the master sheet's eight absolute columns as a
single-ramp gradient: t1 = duration, t2 = t3 = 0, phi1 = start, phi2 = end,
phi3 = phi4 = 0.

MIN_SPAN IS 0 ON PURPOSE.  end_phi == start_phi is isocratic, and isocratic is
a legitimate member of this design space - being able to explore it is not a
bad thing to show.  `is_simple_gradient` therefore tests phi2 >= phi1, not
phi2 > phi1, or every isocratic row would be invisible downstream.

THE BOX IS INSTRUMENT AND METHOD ENVELOPE, NOT SAMPLE KNOWLEDGE.  The bounds
are what the hardware and the chemistry class allow: compositions the pump can
deliver as a usable gradient, durations that are neither too short to develop a
separation nor longer than a working sequence tolerates, and temperatures a
column and its oven survive. They are the same for any sample run on this
setup, which is the point.

DO NOT NARROW THE BOX TO SUIT A SAMPLE.  It is tempting - the analyst usually
knows roughly where a given sample stops retaining - and it is the fastest way
to turn a general instrument into a single-sample script. A narrowed box
encodes one chemistry into the space every future campaign searches, and the
campaign can no longer discover that the belief was wrong. Let the optimiser
find the dead regions; that is what the budget and the cold start are for.

check4() IS THE ONLY GATE.  Nothing reaches the instrument, and nothing is
recorded, without passing through it - not a cold-start seed, not an
acquisition candidate, not a reference method.  It is cheap and it is the last
thing standing between a boundary-sitting optimiser result and a wasted run.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

P4_NAMES: list[str] = ["start_phi", "end_phi", "duration_min", "T"]
P4_LOWER = np.array([0.02, 0.10, 10.0, 25.0])
P4_UPPER = np.array([0.40, 1.00, 60.0, 60.0])

#: end_phi >= start_phi + MIN_SPAN. Zero keeps isocratic inside the space.
MIN_SPAN = 0.0

PARAM_COLS: list[str] = ["t1", "t2", "t3", "phi1", "phi2", "phi3", "phi4", "T"]


# ── the constraint, in BoTorch's normalised form ────────────────────────────
# BoTorch wants  sum(coef * x[idx]) >= rhs  with x in [0, 1]^d.
#
# Absolute:      end - start >= MIN_SPAN
# Normalised:    end = lo_e + Re * end_n,  start = lo_s + Rs * start_n
#                (lo_e + Re*end_n) - (lo_s + Rs*start_n) >= MIN_SPAN
#                Re*end_n - Rs*start_n >= MIN_SPAN - lo_e + lo_s
#
# With the box above: Rs = 0.38, Re = 0.90, rhs = 0 - 0.10 + 0.02 = -0.08.
#
# The columns touched are 1 (end) and 0 (start) ONLY. That is why the optional
# run-order covariate is APPENDED as column 4 rather than prepended - these
# indices stay correct at 5-D without a second constraint definition.
_RS = float(P4_UPPER[0] - P4_LOWER[0])
_RE = float(P4_UPPER[1] - P4_LOWER[1])
CON_IDX: list[int] = [1, 0]
CON_COEF: list[float] = [_RE, -_RS]
CON_RHS: float = float(MIN_SPAN - P4_LOWER[1] + P4_LOWER[0])


def to_norm(u) -> np.ndarray:
    """Absolute parameters -> the unit cube."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    return (u - P4_LOWER) / (P4_UPPER - P4_LOWER)


def from_norm(x) -> np.ndarray:
    """The unit cube -> absolute parameters."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return P4_LOWER + x * (P4_UPPER - P4_LOWER)


def constraint_slack_norm(x_norm) -> np.ndarray:
    """coef . x[idx] - rhs in normalised space. >= 0 means feasible."""
    x = np.atleast_2d(np.asarray(x_norm, dtype=float))
    return CON_COEF[0] * x[:, CON_IDX[0]] + CON_COEF[1] * x[:, CON_IDX[1]] - CON_RHS


def check4(u, tol: float = 1e-6) -> tuple[bool, str]:
    """(ok, message) for one 4-vector. THE feasibility gate."""
    u = np.asarray(u, dtype=float).ravel()
    if u.size != 4:
        return False, f"expected 4 parameters, got {u.size}"
    if not np.all(np.isfinite(u)):
        return False, "parameters must all be finite"
    msgs: list[str] = []
    if u[1] < u[0] + MIN_SPAN - tol:
        msgs.append(f"end_phi {u[1]:.3f} below start_phi {u[0]:.3f} "
                    f"+ MIN_SPAN {MIN_SPAN:.3f}")
    for i in np.where(u < P4_LOWER - tol)[0]:
        msgs.append(f"{P4_NAMES[i]} {u[i]:.4g} < lower {P4_LOWER[i]:.4g}")
    for i in np.where(u > P4_UPPER + tol)[0]:
        msgs.append(f"{P4_NAMES[i]} {u[i]:.4g} > upper {P4_UPPER[i]:.4g}")
    return (len(msgs) == 0, "; ".join(msgs) or "ok")


def require_feasible(u, what: str = "method") -> np.ndarray:
    """check4 or raise. Use wherever a bad method must not proceed."""
    u = np.asarray(u, dtype=float).ravel()
    ok, msg = check4(u)
    if not ok:
        raise ValueError(f"infeasible {what}: {msg}. Do not run this on the instrument.")
    return u


def is_isocratic(u, tol: float = 1e-6) -> bool:
    u = np.asarray(u, dtype=float).ravel()
    return bool(abs(u[1] - u[0]) <= tol)


# ── mapping to and from the sheet's eight absolute columns ──────────────────
def simple_gradient_abs(u) -> dict[str, float]:
    """4-vector -> the eight absolute master-sheet columns."""
    s, e, dur, T = np.asarray(u, dtype=float).ravel()
    return {"t1": float(dur), "t2": 0.0, "t3": 0.0,
            "phi1": float(s), "phi2": float(e), "phi3": 0.0, "phi4": 0.0,
            "T": float(T)}


def abs_to_4param(row: Mapping[str, Any]) -> np.ndarray:
    """A single-ramp master row (t2 == t3 == 0) -> 4-vector."""
    return np.array([row["phi1"], row["phi2"], row["t1"], row["T"]], dtype=float)


def is_simple_gradient(df):
    """Boolean mask over a DataFrame: genuine single-ramp gradients.

    phi2 >= phi1, not phi2 > phi1 - see the module docstring on MIN_SPAN.
    Rows reusing these columns for a decreasing ramp, or with a phi2 == 0
    isocratic encoding, are still excluded.
    """
    return ((df["t2"] == 0) & (df["t3"] == 0) & (df["phi3"] == 0)
            & (df["phi4"] == 0) & (df["phi2"] >= df["phi1"])
            & (df["phi2"] > 0))


def gradient_table(u) -> list[dict[str, Any]]:
    """The instrument view: what the analyst keys into the HPLC."""
    s, e, dur, T = np.asarray(u, dtype=float).ravel()
    if is_isocratic(u):
        return [{"time_min": 0.0, "pct_b": round(s * 100, 1), "phase": "Start"},
                {"time_min": round(dur, 2), "pct_b": round(e * 100, 1),
                 "phase": "End of hold (ISOCRATIC)"}]
    return [{"time_min": 0.0, "pct_b": round(s * 100, 1), "phase": "Start"},
            {"time_min": round(dur, 2), "pct_b": round(e * 100, 1),
             "phase": "End of gradient"}]


def initial_sample(n: int, seed: int | None = None) -> np.ndarray:
    """Rejection-sample n feasible methods. Cold start only."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > 100000:
            raise RuntimeError("could not generate a spread of runnable methods after many attempts. "
            "This means the design space is almost entirely ruled out by the "
            "constraint, which should not happen with the standard bounds - "
            "check the parameter ranges on the Method page before running "
            "anything.")
        u = P4_LOWER + rng.random(4) * (P4_UPPER - P4_LOWER)
        if check4(u)[0]:
            out.append(u)
    return np.stack(out)


def feasible_fraction(n: int = 200000, seed: int = 0) -> float:
    """Monte-Carlo fraction of the box that satisfies the constraint.

    Useful as a sanity check on the constraint arithmetic: the analytic value
    for this box is 0.868421.
    """
    rng = np.random.default_rng(seed)
    U = P4_LOWER + rng.random((n, 4)) * (P4_UPPER - P4_LOWER)
    return float(np.mean(U[:, 1] >= U[:, 0] + MIN_SPAN))
