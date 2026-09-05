"""uncertainty.py - the fixed evaluation grid, the four slice panels, the audit.

THE GRID IS FIXED AND CACHED.  Every "how uncertain is the model" number is a
statistic over a set of points. If that set moves between runs, the statistic
is not comparable with itself, and the campaign's learning curve becomes an
artefact of resampling. A scrambled Sobol draw, rejection-filtered on the
feasibility gate, generated ONCE and cached: comparable across kernel restarts
and across scipy versions, which a fresh draw would not be.

FOUR PANELS, NOT ONE PICTURE.  You cannot draw a four-dimensional posterior,
and the usual substitutes read as sophisticated while communicating nothing.
`slice_posteriors` sweeps ONE parameter across its range with the other three
pinned at the incumbent, which is a question a chemist can actually answer:
what does the model think about temperature, and how sure is it?
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from . import space as S
from . import optimiser as O

GRID_N = 512
GRID_SEED = 20260805


def make_reference_grid(n: int = GRID_N, seed: int = GRID_SEED) -> np.ndarray:
    """One fixed feasible grid. Four columns, always.

    Run order is NOT part of the grid - the grid is a set of METHODS, not of
    moments. When run order is modelled the posterior is evaluated over this
    grid AT a pinned moment.
    """
    from scipy.stats import qmc
    m = 10
    while True:
        pts = qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(m)
        U = S.P4_LOWER + pts * (S.P4_UPPER - S.P4_LOWER)
        keep = np.array([S.check4(u)[0] for u in U])
        if keep.sum() >= n:
            return U[keep][:n]
        m += 1


@dataclass
class SlicePanel:
    name: str
    values: np.ndarray        # the swept parameter, absolute units
    mean: np.ndarray          # posterior mean, CRF
    sd: np.ndarray            # posterior sd, CRF
    lower: np.ndarray         # mean - 2 sd
    upper: np.ndarray         # mean + 2 sd
    pinned_at: dict[str, float]
    tested: np.ndarray        # this parameter's value for every tested method
    tested_scores: np.ndarray
    infeasible: np.ndarray    # sweep points the constraint rejects


def slice_posteriors(fit, incumbent_u, tested_u=None, tested_y=None,
                     n: int = 120, run_order=None) -> list[SlicePanel]:
    """One panel per parameter: sweep it, pin the other three at the incumbent.

    Sweep points that violate the constraint are marked rather than dropped, so
    the panel shows honestly where the design space actually ends - for a low
    start_phi much of the end_phi axis is reachable, for a high one it is not.
    """
    inc = np.asarray(incumbent_u, dtype=float).ravel()
    S.require_feasible(inc, "incumbent")
    tested_u = None if tested_u is None else np.atleast_2d(np.asarray(tested_u, float))
    tested_y = None if tested_y is None else np.asarray(tested_y, float).ravel()

    panels: list[SlicePanel] = []
    for j, name in enumerate(S.P4_NAMES):
        vals = np.linspace(S.P4_LOWER[j], S.P4_UPPER[j], n)
        U = np.tile(inc, (n, 1))
        U[:, j] = vals
        bad = np.array([not S.check4(u)[0] for u in U])
        # evaluate everywhere; the mask tells the UI what to grey out
        mu, sd = O.posterior(fit, U, run_order=run_order)
        panels.append(SlicePanel(
            name=name, values=vals, mean=mu, sd=sd,
            lower=mu - 2 * sd, upper=mu + 2 * sd,
            pinned_at={nm: float(inc[k]) for k, nm in enumerate(S.P4_NAMES) if k != j},
            tested=(tested_u[:, j] if tested_u is not None else np.array([])),
            tested_scores=(tested_y if tested_y is not None else np.array([])),
            infeasible=bad))
    return panels


@dataclass
class UncertaintyStep:
    n_runs: int
    mean_sd_before: float
    mean_sd_after: float
    max_sd_before: float
    max_sd_after: float
    mean_abs_dmu: float
    sd_incumbent_before: float
    sd_incumbent_after: float
    sd_newpoint_before: float
    sd_newpoint_after: float
    noise_sd_crf: float
    noise_floor_z: float
    sigma_pooled_crf: float
    best_crf: float
    lengthscales: dict[str, float] = field(default_factory=dict)
    run_order: float = float("nan")

    @property
    def d_mean_sd(self) -> float:
        return self.mean_sd_before - self.mean_sd_after

    def reading(self) -> str:
        """The diagnostic the whole audit exists for."""
        if self.d_mean_sd <= 0:
            return ("Mean posterior SD did NOT fall. Either the measurement landed "
                    "where the model was already confident, or it disagreed with "
                    "its neighbours and the noise term absorbed it.")
        return (f"Mean posterior SD fell by {self.d_mean_sd:.4f} CRF. The model "
                f"learned something from this run.")


def uncertainty_step(train_u, train_y, sigma_crf=None, grid=None, new_u=None,
                     run_order=None, model_run_order: bool = False,
                     kernel: str = O.DEFAULT_KERNEL) -> Optional[UncertaintyStep]:
    """Refit on 1..k-1 and 1..k, and report what the k-th measurement bought.

    Both fits use the SAME sigma_crf so a change in the reported numbers comes
    from the data, not from the floor moving underneath them. (floor_z itself
    is recomputed per fit as sigma^2/var(y), because var(y) legitimately
    changes when a point is added - that is the conversion doing its job, not
    an inconsistency.)

    Both posteriors are taken at the SAME moment, so the before/after
    difference is about the data and not about the clock.
    """
    U = np.atleast_2d(np.asarray(train_u, dtype=float))
    y = np.asarray(train_y, dtype=float).ravel()
    k = y.size
    if k < 3:
        return None
    grid = make_reference_grid() if grid is None else grid
    ro = None if run_order is None else np.asarray(run_order, dtype=float).ravel()
    now = float(ro[k - 1]) if ro is not None else None

    before = O.fit_gp(U[:k - 1], y[:k - 1], sigma_crf,
                      None if ro is None else ro[:k - 1], model_run_order,
                      kernel=kernel)
    after = O.fit_gp(U, y, sigma_crf, ro, model_run_order, kernel=kernel)

    mu0, sd0 = O.posterior(before, grid, run_order=now)
    mu1, sd1 = O.posterior(after, grid, run_order=now)

    inc = U[int(np.nanargmax(y))].reshape(1, -1)
    _, si0 = O.posterior(before, inc, run_order=now)
    _, si1 = O.posterior(after, inc, run_order=now)
    nu = U[k - 1:k] if new_u is None else np.atleast_2d(np.asarray(new_u, float))
    _, sn0 = O.posterior(before, nu, run_order=now)
    _, sn1 = O.posterior(after, nu, run_order=now)

    names = O.model_names(model_run_order)
    ls = np.resize(after.lengthscales, len(names))
    return UncertaintyStep(
        n_runs=k,
        mean_sd_before=float(sd0.mean()), mean_sd_after=float(sd1.mean()),
        max_sd_before=float(sd0.max()), max_sd_after=float(sd1.max()),
        mean_abs_dmu=float(np.abs(mu1 - mu0).mean()),
        sd_incumbent_before=float(si0[0]), sd_incumbent_after=float(si1[0]),
        sd_newpoint_before=float(sn0[0]), sd_newpoint_after=float(sn1[0]),
        noise_sd_crf=float(after.noise_sd_crf),
        noise_floor_z=float(after.noise.floor_z),
        sigma_pooled_crf=(float(sigma_crf) if sigma_crf is not None
                          and np.isfinite(sigma_crf) else float("nan")),
        best_crf=float(np.nanmax(y)),
        lengthscales={nm: float(v) for nm, v in zip(names, ls)},
        run_order=(now if now is not None else float(k)))


def campaign_reading(history: Sequence[UncertaintyStep],
                     drift_falling: bool) -> str:
    """The three-signal diagnostic, read together.

    Rising fitted noise + flat posterior SD + a falling reference is drift
    being absorbed as noise. The correct next action is not another BO step.
    """
    if len(history) < 3:
        return "Too few runs to read a trend yet."
    noise = np.array([h.noise_sd_crf for h in history], dtype=float)
    msd = np.array([h.mean_sd_after for h in history], dtype=float)
    noise_rising = noise[-1] > noise[0]
    sd_flat = abs(msd[-1] - msd[0]) < 0.05 * max(abs(msd[0]), 1e-9)
    if noise_rising and sd_flat and drift_falling:
        return ("STOP AND FIX THE INSTRUMENT. Fitted noise is rising, posterior "
                "uncertainty is not falling, and the reference is dropping - that "
                "is drift being absorbed as noise, not a campaign that is learning.")
    if sd_flat and not noise_rising:
        return ("Posterior uncertainty has stopped falling. The model may have "
                "learned what this budget can teach it; consider replicating one "
                "method 4-5 times to pin sigma with real degrees of freedom.")
    if msd[-1] < msd[0]:
        return "Posterior uncertainty is falling run on run. The campaign is learning."
    return "Mixed signals - read the noise, uncertainty and reference charts together."
