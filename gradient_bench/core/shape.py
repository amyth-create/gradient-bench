"""shape.py - the shape-aware gradient representation (version 3).

WHAT IT IS.  A way of saying how similar two gradient methods are that looks at
the gradient PROFILE the pump will actually deliver, rather than at the numbers
typed into the timetable. Two profiles are compared by the area between them
once both are drawn on a normalised time axis (0 = start, 1 = end of the
gradient) with composition as a fraction of the maximum %B. For monotonic,
one-dimensional profiles that area is the Wasserstein-1 distance (Boelrijk,
Detlefsen, Duerholt and Haas, "Shape-aware Gaussian process kernels for
closed-loop Bayesian optimization of gradients in automated LC method
development", ChemRxiv 2026, Eq. 16). It is turned into a covariance with an
absolute-exponential form, k_W = exp(-d_W / l_W) (Eq. 17), and multiplied by
a second kernel over the inputs the time normalisation throws away - the
method duration - and over the column temperature, which the profile does not
contain at all (Eq. 18, extended as the paper's Section 2.3.6 allows). The
"plus" form gives the start and end compositions their own lengthscales as
well (Eq. 19, Wasserstein+). The second factor is a squared exponential in the
paper; this app also offers it as a Matern-5/2, for the reason optimiser.py
gives for preferring Matern over RBF on a chromatographic response.

WHY IT MATTERS MOST IN THE steps8 SPACE.  A four-node timetable can write one
and the same profile many ways: slide a node along the straight line joining
its neighbours and the profile is unchanged while every coordinate moves. A
coordinate kernel has to LEARN that those timetables are the same experiment,
from runs it cannot spare; the Wasserstein distance is zero between them by
construction. In the single-ramp space there is no such redundancy - every
(start, end, duration) is a different profile - and the shape kernel merely
changes what "nearby" means: a ramp from 10 to 50 %B and one from 20 to 40 %B
sweep nearly the same composition window and are 0.05 apart, while 10 -> 50
and 30 -> 70 are 0.20 apart.

THE DISTANCE, EXACTLY.  Both profiles are interpolated onto the union of
their breakpoints. On one interval the two differ by p at its start and by q
at its end; if p and q have the same sign the profiles do not cross and the
area is |p + q| / 2, otherwise it is (p^2 + q^2) / (2 |q - p|). Both are
exact; there is no quadrature and nothing to tune. The torch version below
does this for every pair in a batch at once and is differentiable, which the
acquisition optimiser needs.

POSITIVE DEFINITENESS.  d_W is an L1 distance between functions on [0, 1],
and exp(-d / l) with an L1 distance is the Laplacian kernel, which is
positive definite for any l > 0. The product with an RBF or Matern is
therefore a valid covariance. `test_core.py` checks the Gram matrix
numerically as well.

THE PRIORS ARE STILL BOTORCH'S.  l_W is placed under the same dimension-scaled
LogNormal prior BoTorch gives an RBF lengthscale, with D = 1 (the paper's
choice), and under the same GreaterThan(0.025) constraint. The second factor
is built by BoTorch's own constructor. Both are read off modules BoTorch
itself builds rather than re-typed here: the version pin, not this file, is
what those numbers depend on.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from . import space as S

#: Composition is scaled by the fixed upper bound on %B, as in the paper's
#: Eq. 15. In this app %B is already a fraction, so the scale is 1.0.
PHI_MAX = float(S.PHI_MAX)

#: The name the shape lengthscale is reported under.
SHAPE_NAME = "shape"

#: What the second factor reads. `rest` recovers what the profile comparison
#: discards; `plus` adds the endpoints (the paper's h(x) = [phi_0, phi_N, t_N]
#: with temperature appended).
REST_FEATURES = ("duration_min", "T")
PLUS_FEATURES = ("start_phi", "end_phi", "duration_min", "T")


# ── the distance, in numpy ──────────────────────────────────────────────────
def segment_l1(p, q, width=1.0) -> np.ndarray:
    """Area between two linear segments of the given width whose values
    differ by p at the start and q at the end. Exact; see the module note."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    same = (p * q) >= 0.0
    no_cross = np.abs(p + q) / 2.0
    den = np.where(same, 1.0, np.abs(q - p))
    cross = (p * p + q * q) / (2.0 * den)
    return np.asarray(width, dtype=float) * np.where(same, no_cross, cross)


def normalised_profile(space: S.Space, u) -> tuple[np.ndarray, np.ndarray]:
    """(t_hat, phi_hat) nodes of one method: time divided by the method's own
    duration, composition by the shared PHI_MAX - the paper's Eq. 15. The two
    axes are treated differently on purpose: the time axis is per-curve, the
    composition axis keeps its absolute level."""
    t, phi = space.profile(u)
    return t / float(t[-1]), np.asarray(phi, float) / PHI_MAX


def ramp_nodes(u) -> tuple[np.ndarray, np.ndarray]:
    return normalised_profile(S.RAMP4, u)


def _interp(t, v, g):
    """Piecewise-linear evaluation with NON-DECREASING node times, vertical
    steps allowed: v_0 + sum_k dv_k * clamp((g - t_k) / dt_k, 0, 1)."""
    t = np.asarray(t, float); v = np.asarray(v, float); g = np.asarray(g, float)
    dt = np.maximum(np.diff(t), 1e-300)
    frac = np.clip((g[:, None] - t[None, :-1]) / dt[None, :], 0.0, 1.0)
    return v[0] + (frac * np.diff(v)[None, :]).sum(1)


def profile_w1(t1, v1, t2, v2) -> float:
    """W1 (area) between two piecewise-linear profiles given as node arrays
    on [0, 1]: both interpolated onto the union of their breakpoints, each
    interval integrated exactly.

    The difference is linear INSIDE each interval but may jump AT a
    breakpoint (a vertical step in one profile), so it is read at the
    interval's quarter points, which are always interior, and extrapolated
    to the ends: p = 1.5 d(1/4) - 0.5 d(3/4), q = 1.5 d(3/4) - 0.5 d(1/4).
    Exact for a linear function; no epsilon anywhere.
    """
    t1 = np.asarray(t1, float); v1 = np.asarray(v1, float)
    t2 = np.asarray(t2, float); v2 = np.asarray(v2, float)
    grid = np.union1d(t1, t2)
    lo, hi = grid[:-1], grid[1:]
    g1 = lo + 0.25 * (hi - lo)
    g3 = lo + 0.75 * (hi - lo)
    d1 = _interp(t1, v1, g1) - _interp(t2, v2, g1)
    d3 = _interp(t1, v1, g3) - _interp(t2, v2, g3)
    p = 1.5 * d1 - 0.5 * d3
    q = 1.5 * d3 - 0.5 * d1
    return float(np.sum(segment_l1(p, q, hi - lo)))


def pairwise_w1(space: S.Space, U1, U2) -> np.ndarray:
    """d_W between every method in U1 (n,d) and every method in U2 (m,d)."""
    U1 = np.atleast_2d(np.asarray(U1, float)); U2 = np.atleast_2d(np.asarray(U2, float))
    P1 = [normalised_profile(space, u) for u in U1]
    P2 = [normalised_profile(space, u) for u in U2]
    out = np.zeros((len(U1), len(U2)))
    for i, (ta, va) in enumerate(P1):
        for j, (tb, vb) in enumerate(P2):
            out[i, j] = profile_w1(ta, va, tb, vb)
    return out


def ramp_w1(u1, u2) -> np.ndarray:
    """The exact closed form for the single-ramp space: one segment."""
    U1 = np.atleast_2d(np.asarray(u1, dtype=float))
    U2 = np.atleast_2d(np.asarray(u2, dtype=float))
    p = (U1[:, 0][:, None] - U2[:, 0][None, :]) / PHI_MAX
    q = (U1[:, 1][:, None] - U2[:, 1][None, :]) / PHI_MAX
    return segment_l1(p, q)


def explain_distance(space: S.Space, u1, u2) -> dict[str, Any]:
    """The distance between two methods, with its parts, for the interface."""
    ta, va = normalised_profile(space, u1)
    tb, vb = normalised_profile(space, u2)
    s1, s2 = space.summary(u1), space.summary(u2)
    return {"d_w": profile_w1(ta, va, tb, vb),
            "profile_1": {"t_hat": ta.tolist(), "phi_hat": va.tolist()},
            "profile_2": {"t_hat": tb.tolist(), "phi_hat": vb.tolist()},
            "duration_gap_min": abs(s1["duration_min"] - s2["duration_min"]),
            "temperature_gap_c": abs(s1["T"] - s2["T"])}


# ── the same, in torch ──────────────────────────────────────────────────────
def _torch_segment_l1(p, q):
    import torch
    same = (p * q) >= 0
    no_cross = (p + q).abs() / 2.0
    den = torch.where(same, torch.ones_like(p), (q - p).abs())
    cross = (p * p + q * q) / (2.0 * den)
    return torch.where(same, no_cross, cross)


def _torch_profile(space_key: str, u):
    """Absolute inputs (..., n, d) -> normalised nodes (t_hat, phi_hat), each
    (..., n, K). The torch twin of `normalised_profile`."""
    import torch
    if space_key == S.RAMP4.key:
        t = torch.stack([torch.zeros_like(u[..., 0]), torch.ones_like(u[..., 0])], -1)
        phi = u[..., 0:2] / PHI_MAX
        return t, phi
    if space_key == S.STEPS8.key:
        phi = torch.cumsum(u[..., 0:4], dim=-1) / PHI_MAX
        dt = torch.cat([torch.zeros_like(u[..., 4:5]), u[..., 4:7]], -1)
        t = torch.cumsum(dt, dim=-1)
        return t / t[..., -1:], phi
    raise ValueError(f"no profile map for space {space_key!r}")


def _torch_interp(t, v, g):
    """Piecewise-linear interpolation of a profile with NON-DECREASING node
    times t (..., K) and values v (..., K) at grid points g (..., G).
    v_0 + sum_k (v_{k+1} - v_k) * clamp((g - t_k) / (t_{k+1} - t_k), 0, 1):
    exact, vertical steps allowed, differentiable everywhere it needs to be."""
    import torch
    dt = (t[..., 1:] - t[..., :-1]).clamp_min(1e-300)         # (..., K-1)
    dv = v[..., 1:] - v[..., :-1]
    frac = ((g.unsqueeze(-1) - t[..., None, :-1]) / dt[..., None, :]).clamp(0.0, 1.0)
    return v[..., :1] + (frac * dv[..., None, :]).sum(-1)     # (..., G)


def _torch_w1(t1, v1, t2, v2):
    """Pairwise d_W: t1, v1 (..., n, K); t2, v2 (..., m, K) -> (..., n, m)."""
    import torch
    T1 = t1.unsqueeze(-2); V1 = v1.unsqueeze(-2)                  # (..., n, 1, K)
    T2 = t2.unsqueeze(-3); V2 = v2.unsqueeze(-3)                  # (..., 1, m, K)
    T1, T2 = torch.broadcast_tensors(T1, T2)
    V1, V2 = torch.broadcast_tensors(V1, V2)
    grid, _ = torch.sort(torch.cat([T1, T2], -1), dim=-1)         # (..., n, m, 2K)
    lo, hi = grid[..., :-1], grid[..., 1:]
    # quarter points, then exact linear extrapolation to the interval ends:
    # the difference may jump AT a breakpoint (a vertical step), never inside
    g1 = lo + 0.25 * (hi - lo)
    g3 = lo + 0.75 * (hi - lo)
    d1 = _torch_interp(T1, V1, g1) - _torch_interp(T2, V2, g1)
    d3 = _torch_interp(T1, V1, g3) - _torch_interp(T2, V2, g3)
    p = 1.5 * d1 - 0.5 * d3
    q = 1.5 * d3 - 0.5 * d1
    return (_torch_segment_l1(p, q) * (hi - lo)).sum(-1)


_CLASS_CACHE: dict[str, Any] = {}


def wasserstein_kernel_class():
    """exp(-d_W / l) over the whole input vector: the kernel maps inputs to
    normalised profiles itself, so it works in any space that has a profile
    map. One lengthscale, `has_lengthscale=True`, so gpytorch supplies the
    parameter, its prior and its constraint as for its own kernels. Defined
    inside a function so importing shape.py does not require torch."""
    if "w" in _CLASS_CACHE:
        return _CLASS_CACHE["w"]
    import torch
    from gpytorch.kernels import Kernel

    class WassersteinProfileKernel(Kernel):
        has_lengthscale = True

        def __init__(self, space_key: str, lower, upper, **kwargs):
            super().__init__(**kwargs)
            self.space_key = str(space_key)
            self.register_buffer("_lo", torch.as_tensor(np.asarray(lower, float),
                                                        dtype=torch.double))
            self.register_buffer("_hi", torch.as_tensor(np.asarray(upper, float),
                                                        dtype=torch.double))

        def _absolute(self, x):
            lo = self._lo.to(x); hi = self._hi.to(x)
            return lo + (hi - lo) * x

        def forward(self, x1, x2, diag: bool = False,
                    last_dim_is_batch: bool = False, **params):
            if last_dim_is_batch:
                raise RuntimeError("WassersteinProfileKernel does not support "
                                   "last_dim_is_batch")
            t1, v1 = _torch_profile(self.space_key, self._absolute(x1))
            t2, v2 = _torch_profile(self.space_key, self._absolute(x2))
            if diag:
                d = _torch_w1(t1.unsqueeze(-2), v1.unsqueeze(-2),
                              t2.unsqueeze(-2), v2.unsqueeze(-2))[..., 0, 0]
                return torch.exp(-d / self.lengthscale.squeeze(-1))
            d = _torch_w1(t1, v1, t2, v2)
            return torch.exp(-d / self.lengthscale)

    _CLASS_CACHE["w"] = WassersteinProfileKernel
    return WassersteinProfileKernel


def feature_kernel_class():
    """A kernel over LINEAR FEATURES of the input: f = x @ A + b, then a base
    kernel on f. The features are the endpoints, the duration and the
    temperature in normalised units, which in the ramp4 space are the inputs
    themselves and in steps8 are sums of them. The base kernel keeps its own
    lengthscales, prior and constraint, so the fit sees one lengthscale per
    feature under BoTorch's regularisation."""
    if "f" in _CLASS_CACHE:
        return _CLASS_CACHE["f"]
    import torch
    from gpytorch.kernels import Kernel

    class LinearFeatureKernel(Kernel):
        has_lengthscale = False

        def __init__(self, base_kernel, A, b, feature_names: Sequence[str], **kwargs):
            super().__init__(**kwargs)
            self.base_kernel = base_kernel
            self.feature_names = list(feature_names)
            self.register_buffer("_A", torch.as_tensor(np.asarray(A, float),
                                                       dtype=torch.double))
            self.register_buffer("_b", torch.as_tensor(np.asarray(b, float),
                                                       dtype=torch.double))

        def _f(self, x):
            return x @ self._A.to(x) + self._b.to(x)

        def forward(self, x1, x2, diag: bool = False,
                    last_dim_is_batch: bool = False, **params):
            if last_dim_is_batch:
                raise RuntimeError("LinearFeatureKernel does not support "
                                   "last_dim_is_batch")
            return self.base_kernel.forward(self._f(x1), self._f(x2), diag=diag,
                                            **params)

    _CLASS_CACHE["f"] = LinearFeatureKernel
    return LinearFeatureKernel


def is_shape_kernel(module) -> bool:
    return type(module).__name__ == "WassersteinProfileKernel"


def is_feature_kernel(module) -> bool:
    return type(module).__name__ == "LinearFeatureKernel"


def feature_map(space: S.Space, names: Sequence[str], d: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """(A, b) taking NORMALISED inputs of dimension d to normalised features.

    Feature f = (c . u_abs - lo_f) / (hi_f - lo_f) with u_abs = lo + R x, so
    f = sum_i c_i R_i x_i / R_f + (c . lo - lo_f) / R_f. `run_order`, when
    modelled, is the extra column d-1 passed through unchanged.
    """
    feats = {n: (c, lo, hi) for n, c, lo, hi in space.features()}
    A = np.zeros((d, len(names)))
    b = np.zeros(len(names))
    R = space.upper - space.lower
    for j, n in enumerate(names):
        if n == "run_order":
            A[d - 1, j] = 1.0
            continue
        c, lo_f, hi_f = feats[n]
        Rf = float(hi_f - lo_f)
        A[:space.d, j] = c * R / Rf
        b[j] = (float(c @ space.lower) - float(lo_f)) / Rf
    return A, b


def make_shape_kernel(d: int, space: S.Space = S.RAMP4, plus: bool = False,
                      base: str = "rbf"):
    """The version-3 covariance module: k_W(profile) x k_base(features).

    `plus=False`  features = (duration, T)              - the paper's k_W,time
    `plus=True`   features = (start, end, duration, T)  - Wasserstein+
    `base`        "rbf" (the paper) or "matern52"

    d is the model dimension (space.d, or +1 with run order, which is then
    appended to the features so drift has somewhere to live).
    """
    from gpytorch.kernels import ProductKernel
    from gpytorch.constraints import GreaterThan
    from gpytorch.priors import LogNormalPrior
    from botorch.models.utils.gpytorch_modules import (
        get_covar_module_with_dim_scaled_prior)

    if d not in (space.d, space.d + 1):
        raise ValueError(f"model dimension {d} does not fit the {space.key} space")
    if base not in ("rbf", "matern52"):
        raise ValueError(f"base must be rbf or matern52, not {base!r}")

    # l_W's prior: BoTorch's dimension-scaled LogNormal with D = 1 (the
    # paper's choice), read off a module BoTorch builds.
    ref = get_covar_module_with_dim_scaled_prior(ard_num_dims=1, use_rbf_kernel=True)
    lp = LogNormalPrior(loc=float(ref.lengthscale_prior.loc),
                        scale=float(ref.lengthscale_prior.scale))
    lb = float(ref.raw_lengthscale_constraint.lower_bound)
    KW = wasserstein_kernel_class()
    k_shape = KW(space.key, space.lower, space.upper, active_dims=tuple(range(space.d)),
                 lengthscale_prior=lp,
                 lengthscale_constraint=GreaterThan(lb, transform=None,
                                                    initial_value=lp.mode))

    names = list(PLUS_FEATURES if plus else REST_FEATURES)
    if d == space.d + 1:
        names.append("run_order")
    k_base = get_covar_module_with_dim_scaled_prior(
        ard_num_dims=len(names), use_rbf_kernel=(base == "rbf"))
    A, b = feature_map(space, names, d)
    KF = feature_kernel_class()
    k_rest = KF(k_base, A, b, names)
    return ProductKernel(k_shape, k_rest)


def shape_kernel_factors(cm, input_names: Sequence[str]) -> list[dict[str, Any]]:
    """Describe each factor of a product covariance module, for the Method
    page and the ARD table. `is_shape` marks the Wasserstein term; a feature
    kernel reports its feature names as its lengthscale names."""
    out: list[dict[str, Any]] = []
    for k in getattr(cm, "kernels", [cm]):
        shape = is_shape_kernel(k)
        feat = is_feature_kernel(k)
        inner = k.base_kernel if feat else k
        ad = getattr(k, "active_dims", None)
        idx = (list(range(len(input_names))) if ad is None
               else [int(i) for i in np.atleast_1d(ad.detach().cpu().numpy())])
        inputs = [input_names[i] for i in idx if i < len(input_names)]
        if shape:
            ls_names = [SHAPE_NAME]
        elif feat:
            ls_names = list(k.feature_names)
        else:
            ls_names = inputs
        nu = float(getattr(inner, "nu", float("nan")))
        if shape:
            form = ("exp(-d_W / l): absolute-exponential in the Wasserstein-1 "
                    "distance between the normalised gradient profiles")
        elif np.isfinite(nu):
            form = f"Matern-{nu:.1f} over the features, one lengthscale each"
        else:
            form = "squared exponential over the features, one lengthscale each"
        out.append({
            "module": type(inner).__name__ if feat else type(k).__name__,
            "is_shape": shape, "is_feature": feat,
            "inputs": inputs, "lengthscale_names": ls_names,
            "ard_num_dims": len(ls_names),
            "prior": getattr(inner, "lengthscale_prior", None),
            "constraint": getattr(inner, "raw_lengthscale_constraint", None),
            "nu": nu, "form": form,
        })
    return out


def factor_lengthscales(cm) -> list[np.ndarray]:
    """One array per factor, in `shape_kernel_factors` order."""
    out = []
    for k in getattr(cm, "kernels", [cm]):
        inner = k.base_kernel if is_feature_kernel(k) else k
        ls = getattr(inner, "lengthscale", None)
        out.append(np.array([np.nan]) if ls is None
                   else np.atleast_1d(ls.detach().squeeze().cpu().numpy()).ravel())
    return out
