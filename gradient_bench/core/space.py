"""space.py - the design spaces, their constraints, and the feasibility gate.

TWO SPACES, CHOSEN PER CAMPAIGN AND LOCKED (version 3).

  ramp4   u = [start_phi, end_phi, duration_min, T]          one linear ramp
  steps8  u = [phi0, dphi1, dphi2, dphi3, dt1, dt2, dt3, T]  three segments

`ramp4` is what every campaign before version 3 searched and remains the
default. `steps8` is a four-node pump timetable written in DELTAS: a starting
composition and three non-negative composition steps, three positive time
steps (the first node is always at t = 0, so it needs no parameter), and the
column temperature. Nodes are the cumulative sums:

    (0, phi0), (dt1, phi0+dphi1), (dt1+dt2, phi0+dphi1+dphi2), (tN, phiN)

Writing the timetable in deltas puts monotonicity into the BOX - every delta
is non-negative - so the only linear constraints left are on the sums: the
final composition may not exceed PHI_MAX and the method may not run longer
than STEPS8_T_MAX. Deliberately NOTHING ELSE: no minimum segment length, no
largest single step. A segment with dphi = 0 is an isocratic hold; a segment
with dt = 0 is an instantaneous step in %B; a segment with both is absent. So
one and the same setup reaches a ramp, a step gradient, a hold-then-ramp, a
two-slope gradient and a three-slope one - which is the point of the space,
and exactly why a shape-aware kernel belongs with it. The one floor that
remains is STEPS8_T_MIN on the TOTAL duration: a method of zero length is not
a method, and the profile is drawn on time divided by the duration.

Both spaces map onto the same eight absolute sheet columns
(t1, t2, t3, phi1, phi2, phi3, phi4, T): a ramp is t1 = duration,
t2 = t3 = 0, phi1 = start, phi2 = end; a steps8 method fills all eight with
node times and node compositions. The row therefore says which space it came
from, and `space_for_row` reads it back.

MIN_SPAN IS 0 ON PURPOSE (ramp4).  end_phi == start_phi is isocratic, and
isocratic is a legitimate member of this design space - being able to explore
it is not a bad thing to show.

THE BOX IS INSTRUMENT AND METHOD ENVELOPE, NOT SAMPLE KNOWLEDGE.  The bounds
are what the hardware and the chemistry class allow. They are the same for any
sample run on this setup, which is the point. DO NOT NARROW THE BOX TO SUIT A
SAMPLE: a narrowed box encodes one chemistry into the space every future
campaign searches. Let the optimiser find the dead regions; that is what the
budget and the cold start are for.

check4() / check8() ARE THE ONLY GATES.  Nothing reaches the instrument, and
nothing is recorded, without passing through `require_feasible` - not a
cold-start seed, not an acquisition candidate, not a reference method.

THE MODULE-LEVEL NAMES ARE THE RAMP.  `P4_NAMES`, `check4`, `to_norm`,
`initial_sample` and friends are the ramp4 space's methods exposed at module
level, because that is how every caller written before version 3 spelled
them. New code should take a `Space` and call its methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

PARAM_COLS: list[str] = ["t1", "t2", "t3", "phi1", "phi2", "phi3", "phi4", "T"]

#: The shared envelope, in absolute units. Both spaces read these.
PHI_MIN = 0.02          # lowest %B the pump delivers as a usable gradient start
PHI_START_MAX = 0.40    # highest starting %B
PHI_LAST_MIN = 0.10     # lowest final %B
PHI_MAX = 1.00
T_LAST_MIN = 10.0       # shortest method, min
T_MAX = 60.0            # longest method, min
TEMP_MIN, TEMP_MAX = 25.0, 60.0

#: steps8 only. The longest method, and the technical floor on its length -
#: not a design choice, a division: the profile is drawn on t / t_N.
STEPS8_T_MAX = 30.0
STEPS8_T_MIN = 1.0


@dataclass(frozen=True)
class Constraint:
    """sum(coef * u[idx]) >= rhs, in ABSOLUTE units. `normalised()` gives the
    same inequality over the unit cube, which is what BoTorch wants."""
    idx: tuple[int, ...]
    coef: tuple[float, ...]
    rhs: float
    label: str

    def normalised(self, lower, upper) -> tuple[list[int], list[float], float]:
        lo = np.asarray(lower, float); hi = np.asarray(upper, float)
        R = hi - lo
        coef_n = [float(c * R[i]) for c, i in zip(self.coef, self.idx)]
        rhs_n = float(self.rhs - sum(c * lo[i] for c, i in zip(self.coef, self.idx)))
        return list(self.idx), coef_n, rhs_n

    def slack(self, U) -> np.ndarray:
        U = np.atleast_2d(np.asarray(U, float))
        return sum(c * U[:, i] for c, i in zip(self.coef, self.idx)) - self.rhs


@dataclass(frozen=True)
class Space:
    key: str
    label: str
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    constraints: tuple[Constraint, ...]
    what: str
    when: str
    tradeoff: str
    recommended: bool = False

    # ── geometry ────────────────────────────────────────────────────────
    @property
    def d(self) -> int:
        return len(self.names)

    def to_norm(self, u) -> np.ndarray:
        u = np.atleast_2d(np.asarray(u, dtype=float))
        return (u - self.lower) / (self.upper - self.lower)

    def from_norm(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return self.lower + x * (self.upper - self.lower)

    def normalised_constraints(self) -> list[tuple[list[int], list[float], float]]:
        return [c.normalised(self.lower, self.upper) for c in self.constraints]

    # ── the gate ────────────────────────────────────────────────────────
    def check(self, u, tol: float = 1e-6) -> tuple[bool, str]:
        u = np.asarray(u, dtype=float).ravel()
        if u.size != self.d:
            return False, f"expected {self.d} parameters, got {u.size}"
        if not np.all(np.isfinite(u)):
            return False, "parameters must all be finite"
        msgs: list[str] = []
        for c in self.constraints:
            s = float(c.slack(u)[0])
            if s < -tol:
                msgs.append(f"{c.label} violated by {-s:.4g}")
        for i in np.where(u < self.lower - tol)[0]:
            msgs.append(f"{self.names[i]} {u[i]:.4g} < lower {self.lower[i]:.4g}")
        for i in np.where(u > self.upper + tol)[0]:
            msgs.append(f"{self.names[i]} {u[i]:.4g} > upper {self.upper[i]:.4g}")
        return (len(msgs) == 0, "; ".join(msgs) or "ok")

    def mask(self, U, tol: float = 1e-6) -> np.ndarray:
        """Vectorised `check` over rows, for grids."""
        U = np.atleast_2d(np.asarray(U, float))
        ok = np.all(U >= self.lower - tol, axis=1) & np.all(U <= self.upper + tol, axis=1)
        for c in self.constraints:
            ok &= c.slack(U) >= -tol
        return ok

    def require_feasible(self, u, what: str = "method") -> np.ndarray:
        u = np.asarray(u, dtype=float).ravel()
        ok, msg = self.check(u)
        if not ok:
            raise ValueError(f"infeasible {what}: {msg}. Do not run this on the instrument.")
        return u

    # ── the method as chemistry ─────────────────────────────────────────
    def profile(self, u) -> tuple[np.ndarray, np.ndarray]:
        """Node times (min, absolute) and node compositions (fraction)."""
        raise NotImplementedError

    def summary(self, u) -> dict[str, float]:
        t, phi = self.profile(u)
        u = np.asarray(u, float).ravel()
        return {"start_phi": float(phi[0]), "end_phi": float(phi[-1]),
                "duration_min": float(t[-1]), "T": float(u[-1]),
                "n_segments": int(len(t) - 1)}

    def is_isocratic(self, u, tol: float = 1e-6) -> bool:
        _, phi = self.profile(u)
        return bool(np.ptp(phi) <= tol)

    def gradient_table(self, u) -> list[dict[str, Any]]:
        """The instrument view: what the analyst keys into the HPLC."""
        t, phi = self.profile(u)
        n = len(t)
        rows = []
        for k in range(n):
            if k == 0:
                phase = "Start"
            elif k == n - 1:
                phase = ("End of hold (ISOCRATIC)" if self.is_isocratic(u)
                         else "End of gradient")
            else:
                phase = f"Node {k}"
            if 0 < k and abs(phi[k] - phi[k - 1]) <= 1e-9 and not self.is_isocratic(u):
                phase += " (hold)"
            rows.append({"time_min": round(float(t[k]), 2),
                         "pct_b": round(float(phi[k]) * 100, 1), "phase": phase})
        return rows

    def to_abs(self, u) -> dict[str, float]:
        """u -> the eight absolute sheet columns."""
        raise NotImplementedError

    def from_abs(self, row: Mapping[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def initial_sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError

    def feasible_fraction(self, n: int = 200000, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        U = self.lower + rng.random((n, self.d)) * (self.upper - self.lower)
        return float(np.mean(self.mask(U)))

    def features(self) -> list[tuple[str, np.ndarray, float, float]]:
        """The endpoint features the shape-aware kernels read beside the
        profile: (name, coefficients over u, lower, upper). Each is a LINEAR
        function of the absolute parameters, so the kernel can compute it
        from normalised inputs with one matrix. The names are the same in
        every space because the quantities are: a steps8 campaign's
        `end_phi` feature is phi0 + dphi1 + dphi2 + dphi3."""
        raise NotImplementedError

    def option(self) -> dict[str, Any]:
        """What the interface shows when the analyst chooses."""
        return {"key": self.key, "label": self.label, "d": self.d,
                "names": list(self.names),
                "lower": [float(x) for x in self.lower],
                "upper": [float(x) for x in self.upper],
                "constraints": [c.label for c in self.constraints],
                "what": self.what, "when": self.when, "tradeoff": self.tradeoff,
                "recommended": self.recommended}


# ── ramp4 ───────────────────────────────────────────────────────────────────
MIN_SPAN = 0.0


class Ramp4(Space):
    def profile(self, u):
        s, e, dur, _ = np.asarray(u, float).ravel()
        return np.array([0.0, dur]), np.array([s, e])

    def to_abs(self, u):
        s, e, dur, T = np.asarray(u, dtype=float).ravel()
        return {"t1": float(dur), "t2": 0.0, "t3": 0.0,
                "phi1": float(s), "phi2": float(e), "phi3": 0.0, "phi4": 0.0,
                "T": float(T)}

    def from_abs(self, row):
        return np.array([row["phi1"], row["phi2"], row["t1"], row["T"]], dtype=float)

    def features(self):
        e = np.eye(4)
        return [("start_phi", e[0], PHI_MIN, PHI_START_MAX),
                ("end_phi", e[1], PHI_LAST_MIN, PHI_MAX),
                ("duration_min", e[2], T_LAST_MIN, T_MAX),
                ("T", e[3], TEMP_MIN, TEMP_MAX)]

    def initial_sample(self, n, seed=None):
        rng = np.random.default_rng(seed)
        out: list[np.ndarray] = []
        guard = 0
        while len(out) < n:
            guard += 1
            if guard > 100000:
                raise RuntimeError(
                    "could not generate a spread of runnable methods after many "
                    "attempts. This means the design space is almost entirely "
                    "ruled out by the constraint, which should not happen with "
                    "the standard bounds - check the parameter ranges on the "
                    "Method page before running anything.")
            u = self.lower + rng.random(4) * (self.upper - self.lower)
            if self.check(u)[0]:
                out.append(u)
        return np.stack(out)


RAMP4 = Ramp4(
    key="ramp4", label="Single ramp (4 parameters)",
    names=("start_phi", "end_phi", "duration_min", "T"),
    lower=np.array([PHI_MIN, PHI_LAST_MIN, T_LAST_MIN, TEMP_MIN]),
    upper=np.array([PHI_START_MAX, PHI_MAX, T_MAX, TEMP_MAX]),
    constraints=(Constraint(idx=(1, 0), coef=(1.0, -1.0), rhs=MIN_SPAN,
                            label=f"end_phi >= start_phi + {MIN_SPAN}"),),
    recommended=True,
    what="One linear ramp: where it starts, where it ends, how long it takes, "
         "and the column temperature. Every method is a straight line on the "
         "%B-against-time chart.",
    when="The default, and what every campaign before version 3 searched. Four "
         "parameters are what a 40-method budget can map; the model learns "
         "quickly and every proposal is easy to read.",
    tradeoff="Cannot express a hold, a shallow start followed by a steep finish, "
             "or any other bend in the gradient. A sample whose analytes cluster "
             "in two composition windows has no method in this space that "
             "serves both.")


# ── steps8 ──────────────────────────────────────────────────────────────────
class Steps8(Space):
    def profile(self, u):
        u = np.asarray(u, float).ravel()
        phi = np.concatenate([[u[0]], u[0] + np.cumsum(u[1:4])])
        t = np.concatenate([[0.0], np.cumsum(u[4:7])])
        return t, phi

    def to_abs(self, u):
        t, phi = self.profile(u)
        return {"t1": float(t[1]), "t2": float(t[2]), "t3": float(t[3]),
                "phi1": float(phi[0]), "phi2": float(phi[1]),
                "phi3": float(phi[2]), "phi4": float(phi[3]),
                "T": float(np.asarray(u, float).ravel()[7])}

    def features(self):
        c = np.zeros((4, 8))
        c[0, 0] = 1.0                 # start = phi0
        c[1, 0:4] = 1.0               # end = phi0 + dphi1 + dphi2 + dphi3
        c[2, 4:7] = 1.0               # duration = dt1 + dt2 + dt3
        c[3, 7] = 1.0                 # T
        return [("start_phi", c[0], PHI_MIN, PHI_START_MAX),
                ("end_phi", c[1], PHI_MIN, PHI_MAX),
                ("duration_min", c[2], STEPS8_T_MIN, STEPS8_T_MAX),
                ("T", c[3], TEMP_MIN, TEMP_MAX)]

    def from_abs(self, row):
        t = np.array([0.0, row["t1"], row["t2"], row["t3"]], float)
        phi = np.array([row["phi1"], row["phi2"], row["phi3"], row["phi4"]], float)
        return np.concatenate([[phi[0]], np.diff(phi), np.diff(t), [float(row["T"])]])

    def initial_sample(self, n, seed=None):
        """The paper's window-relative sampler (Boelrijk et al. 2026, Section
        2.4), then the gate. Sampling the box uniformly would land almost every
        draw outside the sum constraints and, worse, would favour diagonal
        gradients; this samples the endpoints first and the interior inside
        the window they define, so shallow and bent profiles are as reachable
        as steep ones."""
        rng = np.random.default_rng(seed)
        out: list[np.ndarray] = []
        guard = 0
        while len(out) < n:
            guard += 1
            if guard > 200000:
                raise RuntimeError("could not generate a spread of runnable "
                                   "methods after many attempts - check the "
                                   "steps8 bounds on the Method page.")
            phi0 = rng.uniform(PHI_MIN, PHI_START_MAX)
            phiN = rng.uniform(PHI_MIN, PHI_MAX)
            if phiN < phi0:
                continue
            inner = np.sort(rng.uniform(phi0, phiN, 2))
            phi = np.array([phi0, inner[0], inner[1], phiN])
            tN = rng.uniform(STEPS8_T_MIN, STEPS8_T_MAX)
            v = np.sort(rng.uniform(0.0, tN, 2))
            t = np.array([0.0, v[0], v[1], tN])
            T = rng.uniform(TEMP_MIN, TEMP_MAX)
            u = np.concatenate([[phi0], np.diff(phi), np.diff(t), [T]])
            if self.check(u)[0]:
                out.append(u)
        return np.stack(out)


STEPS8 = Steps8(
    key="steps8", label="Three-segment timetable (8 parameters)",
    names=("phi0", "dphi1", "dphi2", "dphi3", "dt1", "dt2", "dt3", "T"),
    lower=np.array([PHI_MIN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, TEMP_MIN]),
    upper=np.array([PHI_START_MAX, PHI_MAX - PHI_MIN, PHI_MAX - PHI_MIN,
                    PHI_MAX - PHI_MIN, STEPS8_T_MAX, STEPS8_T_MAX, STEPS8_T_MAX,
                    TEMP_MAX]),
    constraints=(
        Constraint(idx=(0, 1, 2, 3), coef=(-1.0, -1.0, -1.0, -1.0), rhs=-PHI_MAX,
                   label=f"phi0 + dphi1 + dphi2 + dphi3 <= {PHI_MAX}"),
        Constraint(idx=(4, 5, 6), coef=(-1.0, -1.0, -1.0), rhs=-STEPS8_T_MAX,
                   label=f"dt1 + dt2 + dt3 <= {STEPS8_T_MAX}"),
        Constraint(idx=(4, 5, 6), coef=(1.0, 1.0, 1.0), rhs=STEPS8_T_MIN,
                   label=f"dt1 + dt2 + dt3 >= {STEPS8_T_MIN}"),
    ),
    what="A four-node pump timetable written in steps: a starting %B and three "
         "composition steps, three time steps (the first node is always at "
         "t = 0), and the column temperature. A composition step of zero is a "
         "hold, a time step of zero is an instantaneous jump in %B, so the "
         "same setup reaches ramps, steps, holds and multi-slope gradients.",
    when="A sample the single ramp cannot serve - analytes in two composition "
         "windows, a need for an initial hold, or a shallow start and a steep "
         "finish. Pair it with a shape-aware kernel: many different timetables "
         "draw the same profile, and the shape-aware kernels know that.",
    tradeoff="Eight parameters on the same budget: the model needs more runs "
             "before it can tell which of them matter, and a coordinate kernel "
             "must additionally learn that three collinear nodes are one ramp. "
             "Only the sums are constrained - final %B at most 100, duration "
             "at most 30 min - so a proposal can be a near-vertical step; "
             "check the pump can deliver it.")

SPACES: dict[str, Space] = {RAMP4.key: RAMP4, STEPS8.key: STEPS8}
SPACE_KEYS = tuple(SPACES)
DEFAULT_SPACE = RAMP4.key
SPACE_OPTIONS: list[dict[str, Any]] = [s.option() for s in SPACES.values()]


def space(key: Optional[str]) -> Space:
    k = str(key or DEFAULT_SPACE).strip().lower()
    if k not in SPACES:
        raise ValueError(f"space must be one of {SPACE_KEYS}, not {key!r}")
    return SPACES[k]


def for_vector(u) -> Space:
    """The space a method vector belongs to, read off its length. 4 and 8 do
    not collide, which is the only reason this is allowed."""
    n = np.asarray(u, float).size
    for s in SPACES.values():
        if s.d == n:
            return s
    raise ValueError(f"no design space has {n} parameters")


def space_for_row(row: Mapping[str, Any]) -> Space:
    """The space a sheet row was recorded in: a ramp leaves t2 = t3 = 0."""
    try:
        if float(row.get("t2", 0) or 0) == 0.0 and float(row.get("t3", 0) or 0) == 0.0:
            return RAMP4
    except (TypeError, ValueError):
        return RAMP4
    return STEPS8


def gradient_table(u) -> list[dict[str, Any]]:
    return for_vector(u).gradient_table(u)


def require_feasible(u, what: str = "method") -> np.ndarray:
    return for_vector(u).require_feasible(u, what)


def check(u, tol: float = 1e-6) -> tuple[bool, str]:
    try:
        return for_vector(u).check(u, tol)
    except ValueError as exc:
        return False, str(exc)


# ── the ramp4 space at module level, as every earlier caller spells it ──────
P4_NAMES: list[str] = list(RAMP4.names)
P4_LOWER = RAMP4.lower
P4_UPPER = RAMP4.upper

_RS = float(P4_UPPER[0] - P4_LOWER[0])
_RE = float(P4_UPPER[1] - P4_LOWER[1])
CON_IDX, CON_COEF, CON_RHS = RAMP4.normalised_constraints()[0]


def to_norm(u) -> np.ndarray:
    return RAMP4.to_norm(u)


def from_norm(x) -> np.ndarray:
    return RAMP4.from_norm(x)


def constraint_slack_norm(x_norm) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x_norm, dtype=float))
    return CON_COEF[0] * x[:, CON_IDX[0]] + CON_COEF[1] * x[:, CON_IDX[1]] - CON_RHS


def check4(u, tol: float = 1e-6) -> tuple[bool, str]:
    return RAMP4.check(u, tol)


def check8(u, tol: float = 1e-6) -> tuple[bool, str]:
    return STEPS8.check(u, tol)


def is_isocratic(u, tol: float = 1e-6) -> bool:
    return for_vector(u).is_isocratic(u, tol)


def simple_gradient_abs(u) -> dict[str, float]:
    return RAMP4.to_abs(u)


def abs_to_4param(row: Mapping[str, Any]) -> np.ndarray:
    return RAMP4.from_abs(row)


def is_simple_gradient(df):
    """Boolean mask over a DataFrame: genuine single-ramp gradients.

    phi2 >= phi1, not phi2 > phi1 - see the module docstring on MIN_SPAN.
    Rows reusing these columns for a decreasing ramp, or with a phi2 == 0
    isocratic encoding, are still excluded.
    """
    return ((df["t2"] == 0) & (df["t3"] == 0) & (df["phi3"] == 0)
            & (df["phi4"] == 0) & (df["phi2"] >= df["phi1"])
            & (df["phi2"] > 0))


def initial_sample(n: int, seed: Optional[int] = None) -> np.ndarray:
    return RAMP4.initial_sample(n, seed)


def feasible_fraction(n: int = 200000, seed: int = 0) -> float:
    """Analytic value for the ramp4 box: 0.868421."""
    return RAMP4.feasible_fraction(n, seed)
