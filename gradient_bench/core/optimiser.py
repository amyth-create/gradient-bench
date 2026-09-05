"""optimiser.py - the surrogate, the noise model and the acquisition.

Ported from 03b_bayesian_optimisation_4param.ipynb. The notebook's reasoning is
preserved here because it is the reason the code looks the way it does.

FOUR CHOICES WORTH KNOWING
--------------------------------------------------------------------------
1. THE KERNEL IS CHOSEN, BUT THE PRIORS ARE STILL BOTORCH'S.  A hand-built
   `covar_module` silently discards BoTorch's hyperparameter priors and leaves
   the lengthscales at unregularised MLE on a few dozen points - which on this
   corpus sends two lengthscales to infinity and collapses the other two onto
   the floor. That is why the earlier note here said to pass no covar_module
   at all.

   But that was too strong. BoTorch's OWN constructor,
   `get_covar_module_with_dim_scaled_prior`, takes `use_rbf_kernel=False` and
   returns a Matern-5/2 carrying exactly the same priors as the RBF default:
   LogNormal(sqrt(2) + ln(d)/2, sqrt(3)) on the lengthscales, constrained
   GreaterThan(0.025), and no outputscale. So the kernel and the priors are
   separable after all, and the campaign can choose the kernel without giving
   up the regularisation.

   `matern52` is the default for new campaigns, and the reason is a property of
   the class of problem, not of any one sample: chromatographic response is
   generally NOT infinitely smooth in composition or temperature. Retention
   regimes change, peaks merge and separate, and a response can turn over a
   short interval. An RBF assumes infinite differentiability, which forces it
   either to smooth an edge away or to collapse its lengthscale chasing one.
   Matern-5/2 is twice differentiable - the weaker assumption, and the
   long-standing default in the Bayesian-optimisation literature for physical
   responses. Verified against botorch 0.18: the RBF default is a bare ARD
   RBFKernel; OLDER releases defaulted to ScaleKernel(Matern-5/2) with GAMMA
   priors, which is a different thing from what is built here. The installed
   version is recorded with every campaign - see `library_versions()`.

2. THE NOISE FLOOR IS MEASURED, NOT ASSUMED.  Two runs at IDENTICAL x differ
   only by noise, so replicates identify sigma directly. Spatial scatter cannot:
   the same points are explained equally well by "smooth surface, large noise"
   and "wiggly surface, small noise", and a few dozen observations do not
   separate those. BoTorch's default floor of 1e-4 tells the model the data are
   essentially noiseless, and it then explains replicate disagreement with very
   short lengthscales. `noise_floor_std` converts the measured within-method sd
   into Standardize(m=1) units, which is what the likelihood actually sees.

3. qLogNoisyExpectedImprovement, not qNEI.  The incumbent is uncertain when
   observations are noisy, hence *noisy* EI; and the log form fixes vanishing
   gradients under a flat posterior, which is exactly the regime a noisy
   objective creates.

4. RUN ORDER IS A COVARIATE, NOT A KNOB.  A GP models f(x) + iid noise. Given
   f(x) + g(t) it has nowhere to put g except sigma^2, which flattens the
   posterior everywhere and pushes proposals to the boundary. Appending run
   order as a fifth input gives g somewhere to live. But run order is OBSERVED,
   not chosen: the acquisition PINS it at the next run's value via
   FixedFeatureAcquisitionFunction and searches the four controllable
   dimensions only, so the optimize_acqf call, its bounds and its constraint
   triple are identical in both modes. Off by default; see the note in
   `suggest_next` about what it does not fix.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from . import space as S

# torch and botorch are imported lazily so the sheet, drift and picker layers
# stay usable (and testable) without them.
_TK = {"dtype": None, "device": None}


def _torch():
    import torch
    if _TK["dtype"] is None:
        _TK["dtype"] = torch.double
        _TK["device"] = torch.device("cpu")
    return torch


def tkwargs() -> dict:
    _torch()
    return {"dtype": _TK["dtype"], "device": _TK["device"]}


def library_versions() -> dict[str, str]:
    """Recorded in every campaign's Config sheet.

    BoTorch's default kernel and priors have changed between releases, so a
    campaign that does not record its version cannot be reproduced.
    """
    out: dict[str, str] = {}
    for name in ("torch", "botorch", "gpytorch", "numpy", "scipy"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = "not installed"
    return out


# ── the noise model ─────────────────────────────────────────────────────────
#: Placeholder, NOT a measurement: one clean peak is the quantum of an
#: integer-count objective and the smallest disagreement it can express.
#: Replaced by pooled_within_sd as soon as one method has two runs.
FALLBACK_SIGMA_CRF = 1.0
MIN_NOISE_VAR = 1e-6


def pooled_within_sd(scores_by_method) -> tuple[float, int, int]:
    """Pooled WITHIN-METHOD sd of the objective, in CRF units.

    `scores_by_method` maps a method base name to its list of replicate scores.
    Returns (sd, n_groups_used, dof). Returns (nan, 0, 0) when no method has
    two or more runs - this number only exists once replicates are actually
    run, which is the whole reason for running them.

    A pooled sd of EXACTLY ZERO is a real outcome, not a degenerate one: the
    objective is driven by an integer peak count, so two replicates that pick
    the same number of peaks over the same hump fraction tie exactly. It means
    sigma is smaller than the objective's own quantum, not that it is zero.
    Callers must treat sd == 0 as "unmeasured" - see `noise_floor_std`.
    """
    ss, dof, groups = 0.0, 0, 0
    for _, ys in (scores_by_method or {}).items():
        y = np.asarray([v for v in ys if v is not None and np.isfinite(v)], dtype=float)
        if y.size < 2:
            continue
        ss += float(((y - y.mean()) ** 2).sum())
        dof += y.size - 1
        groups += 1
    if dof == 0:
        return float("nan"), 0, 0
    return float(np.sqrt(ss / dof)), groups, dof


def floor_is_dominated(floor_z: float) -> bool:
    """floor_z >= 1: the measured replicate noise is as large as the whole
    campaign's spread. The posterior will be nearly flat and the acquisition
    nearly uninformative.

    THIS IS A FINDING, NOT A BUG, and the floor must never be lowered to make
    the plot look better. Lowering it tells the model the runs are more precise
    than the replicates showed, and the methods it then proposes are chasing
    differences the instrument cannot resolve. The two honest responses are to
    run more replicates, which sharpens sigma, or to explore further apart,
    which grows the spread the noise is being compared against.

    A pure function of the stored number, not a property of a live fit, so a
    floor_z read back from `uncertainty_history.csv` months later reads the
    same way as one that has just been computed - the same reason
    `lengthscale_reading` is a pure function (HANDOFF section 15).
    """
    try:
        return float(floor_z) >= 1.0
    except (TypeError, ValueError):
        return False


def floor_reading(floor_z: float) -> str:
    """The number in words. Bands, not a formula: the exact value carries less
    than which side of 1 it sits on and by how far."""
    try:
        z = float(floor_z)
    except (TypeError, ValueError):
        return "not recorded"
    if z >= 1.0:
        return "noise dominates - the model cannot tell these methods apart"
    if z >= 0.5:
        return "noise is over half the spread - differences are marginal"
    if z >= 0.2:
        return "noise is a real share of the spread"
    return "spread is well above the noise"


@dataclass
class NoiseFloor:
    """The likelihood floor, with every intermediate kept so the arithmetic can
    be shown rather than asserted."""
    sigma_crf: float
    used_fallback: bool
    sd_y: float
    var_y: float
    floor_z: float
    floor_sd_crf: float
    ratio: float

    @property
    def dominated(self) -> bool:
        return floor_is_dominated(self.floor_z)

    @property
    def reading(self) -> str:
        return floor_reading(self.floor_z)

    def explain(self) -> list[str]:
        src = "FALLBACK constant" if self.used_fallback else "measured within-method sd"
        lines = [
            f"sigma_y = {self.sigma_crf:.3f} CRF ({src});  "
            f"sd(train_y) = {self.sd_y:.3f} CRF",
            f"floor_z = sigma_y^2 / var(y) = {self.sigma_crf:.3f}^2 / "
            f"{self.var_y:.4f} = {self.floor_z:.5f}   "
            f"(= {self.floor_sd_crf:.3f} CRF as an sd)",
        ]
        if self.dominated:
            lines.append(
                "WARNING: floor_z >= 1. The measured replicate noise is as large "
                "as the whole campaign's spread. The posterior will be nearly "
                "flat and the acquisition nearly uninformative. That is the "
                "finding, not a bug - do not lower the floor.")
        return lines


def noise_floor_std(y, sigma_crf: Optional[float] = None) -> NoiseFloor:
    """Likelihood noise-variance floor, in STANDARDIZED units.

    Standardize(m=1) divides the targets by their unbiased sd, so a floor
    quoted in CRF units means nothing to the likelihood until it is divided by
    var(y). That conversion is the whole function:

        floor_z = sigma_crf^2 / var(train_y)

    A sigma that is missing, non-finite, or exactly zero (see
    `pooled_within_sd`) counts as unmeasured and falls back.
    """
    y = np.asarray(y, dtype=float).ravel()
    used_fallback = (sigma_crf is None or not np.isfinite(sigma_crf) or sigma_crf <= 0)
    sigma = FALLBACK_SIGMA_CRF if used_fallback else float(sigma_crf)

    sd_y = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    var_y = sd_y ** 2
    floor_z = MIN_NOISE_VAR if var_y <= 0 else max(sigma ** 2 / var_y, MIN_NOISE_VAR)
    return NoiseFloor(
        sigma_crf=sigma, used_fallback=used_fallback, sd_y=sd_y, var_y=var_y,
        floor_z=floor_z,
        floor_sd_crf=float(np.sqrt(floor_z) * sd_y) if sd_y > 0 else float("nan"),
        ratio=float(sigma / sd_y) if sd_y > 0 else float("inf"))


# ── run order as an optional fifth input ────────────────────────────────────
#: A UNIT CHOICE, not a prior. It only fixes what "lengthscale 1.0" means in
#: the ARD table: roughly one full campaign.
RUN_ORDER_SPAN = 40.0
P5_NAMES = S.P4_NAMES + ["run_order"]
P5_LOWER = np.append(S.P4_LOWER, 0.0)
P5_UPPER = np.append(S.P4_UPPER, RUN_ORDER_SPAN)


def run_order_to_abs(run_order):
    """1-based instrument position -> the fifth absolute coordinate."""
    return np.clip(np.asarray(run_order, dtype=float) - 1.0, 0.0, RUN_ORDER_SPAN)


def run_order_to_norm(run_order):
    return run_order_to_abs(run_order) / RUN_ORDER_SPAN


def augment_run_order(U, run_order=None, model_run_order: bool = False) -> np.ndarray:
    """(N,4) -> (N,5) when run order is modelled; pass-through when not.

    Idempotent: already-5-column input is returned unchanged.
    """
    U = np.atleast_2d(np.asarray(U, dtype=float))
    if not model_run_order:
        if U.shape[1] != 4:
            raise ValueError(f"run order is off but got {U.shape[1]} columns")
        return U
    if U.shape[1] == 5:
        return U
    if run_order is None:
        raise ValueError("run order is modelled: every training row needs its "
                         "instrument position")
    t = np.asarray(run_order, dtype=float).ravel()
    if t.size == 1:
        # .item(), not float(): numpy 2 refuses float() on a 1-element array
        t = np.full(len(U), t.item())
    if t.size != len(U):
        raise ValueError(f"{t.size} run-order values for {len(U)} rows")
    return np.column_stack([U, run_order_to_abs(t)])


def model_dim(model_run_order: bool) -> int:
    return 5 if model_run_order else 4


def model_names(model_run_order: bool) -> list[str]:
    return list(P5_NAMES) if model_run_order else list(S.P4_NAMES)


def _model_bounds(model_run_order: bool):
    torch = _torch()
    lo, hi = (P5_LOWER, P5_UPPER) if model_run_order else (S.P4_LOWER, S.P4_UPPER)
    return torch.stack([torch.tensor(lo, **tkwargs()), torch.tensor(hi, **tkwargs())])


def _candidate_bounds():
    torch = _torch()
    return torch.stack([torch.tensor(S.P4_LOWER, **tkwargs()),
                        torch.tensor(S.P4_UPPER, **tkwargs())])


# ── the surrogate ───────────────────────────────────────────────────────────
@dataclass
class FitResult:
    model: Any
    noise: NoiseFloor
    n: int
    d: int
    model_run_order: bool
    lengthscales: np.ndarray = field(default_factory=lambda: np.array([]))
    noise_sd_crf: float = float("nan")
    kernel: str = ""

    def lengthscale_table(self) -> list[tuple[str, float, str]]:
        """(name, value, reading) per input."""
        names = model_names(self.model_run_order)
        return [(nm, float(v), lengthscale_reading(nm, v))
                for nm, v in zip(names, np.resize(self.lengthscales, len(names)))]


def _make_likelihood(floor_z: float):
    """GaussianLikelihood with the measured floor, PRIOR INTACT.

    Dropping the noise prior here would repeat, on the likelihood, exactly the
    mistake the module docstring avoids on the covariance module. So the
    likelihood is built by BoTorch's own helper and only its lower BOUND is
    replaced: `register_constraint` is public API and swaps the constraint
    module in place, leaving the prior untouched. Reaching into the prior to
    copy it out would depend on gpytorch attribute names that have moved
    between versions.
    """
    from gpytorch.constraints import GreaterThan
    floor_z = float(floor_z)
    try:
        from botorch.models.utils.gpytorch_modules import (
            get_gaussian_likelihood_with_lognormal_prior)
        lik = get_gaussian_likelihood_with_lognormal_prior()
        lik.noise_covar.register_constraint("raw_noise", GreaterThan(floor_z))
        lik._prior_source = "BoTorch default noise prior, floor replaced"
    except Exception as exc:
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.priors import GammaPrior
        lik = GaussianLikelihood(noise_prior=GammaPrior(1.1, 0.05),
                                 noise_constraint=GreaterThan(floor_z))
        lik._prior_source = f"GammaPrior(1.1, 0.05) fallback ({type(exc).__name__})"
    return lik


#: Below this an ARD lengthscale has hit BoTorch's own constraint and stopped
#: being a measurement of anything.
LS_FLOOR = 0.0255
#: Above this the input spans far more than the box, so the score barely moves
#: along it within the space being searched.
LS_INERT = 20.0


def lengthscale_reading(name: str, v: float) -> str:
    """One ARD lengthscale, in words.

    A PURE FUNCTION ON PURPOSE. The same reading has to be available for a live
    fit and for a lengthscale read back out of a campaign's uncertainty
    history months later, and a diagnostic that can only be produced by
    refitting is one nobody looks at.

    Normalised units: the box is mapped to [0, 1] before fitting, so a
    lengthscale near 1 means the score changes over about the whole range of
    that parameter, and one near 0.1 means it changes ten times faster.
    """
    v = float(v)
    if not np.isfinite(v):
        return "unreadable"
    if name == "run_order":
        # Run order is the only input whose short lengthscale is bad news
        # rather than information: it means the response decorrelates in TIME.
        if v < 0.25:
            return "SHORT - strong drift, the response decorrelates in time"
        if v > 2.0:
            return "long - the time axis is nearly inert"
        return "intermediate - drift comparable with the campaign length"
    if v <= LS_FLOOR:
        return ("at the floor - collapsed, so this is not a measurement: the "
                "fit could not identify a scale along this parameter")
    if v > LS_INERT:
        return "inert - this parameter barely moves the score inside the box"
    if v < 0.25:
        return "short - the score changes quickly along this parameter"
    if v > 2.0:
        return "long - the score changes slowly across the whole range"
    return "active"


#: The kernel is a per-campaign setting, recorded in Config. `matern52` is the
#: default; `matern32` is the same construction one notch rougher; `rbf`
#: reproduces BoTorch 0.18's own default; `default` passes no covar_module at
#: all, which is whatever the installed version does.
#:
#: VERSION 2: the choice is made ONCE, at campaign creation, on the Campaign
#: tab, and it is locked afterwards. A campaign fitted under one kernel and
#: refitted under another is not the same campaign, so there is no toggle.
KERNELS = ("matern52", "matern32", "rbf", "default")
DEFAULT_KERNEL = "matern52"

#: What the interface shows when the analyst chooses. Every entry says what the
#: choice IS, when it is the right one, and what it costs - the same three
#: questions the glossary answers - so the decision can be made by a chemist
#: who has never fitted a Gaussian process. Text lives here, beside the code it
#: describes, so it cannot go stale against the implementation.
SURROGATE_OPTIONS: list[dict[str, Any]] = [
    {"key": "matern52", "label": "Matérn-5/2", "recommended": True,
     "family": "Gaussian process, ARD",
     "smoothness": "twice differentiable",
     "what": "A Gaussian process whose covariance assumes the score is smooth "
             "but not infinitely so: it can turn over a short interval without "
             "the fit collapsing a lengthscale to chase it.",
     "when": "The default for a chromatographic response. Retention regimes "
             "change, peaks merge and separate, and the CRF steps when a peak "
             "crosses the prominence gate - all of which a Matérn kernel "
             "tolerates.",
     "tradeoff": "Slightly wider posterior than an RBF on the same data, "
                 "because it promises less. That is the honest width."},
    {"key": "matern32", "label": "Matérn-3/2", "recommended": False,
     "family": "Gaussian process, ARD",
     "smoothness": "once differentiable",
     "what": "The same construction as Matérn-5/2, one notch rougher: the "
             "response may have kinks, and the model does not smooth across "
             "them.",
     "when": "A sample known to behave erratically across the space - sharp "
             "regime changes, or a picker that flips a peak between categories "
             "over a narrow composition range.",
     "tradeoff": "Learns local structure quickly and generalises less far from "
                 "each run. On a 40-method budget it can spend proposals "
                 "re-checking neighbourhoods a smoother model would trust."},
    {"key": "rbf", "label": "Squared exponential (RBF)", "recommended": False,
     "family": "Gaussian process, ARD",
     "smoothness": "infinitely differentiable",
     "what": "BoTorch's own default kernel, with the same priors. It assumes "
             "the score varies smoothly everywhere.",
     "when": "A response you expect to be gentle across the whole box - a "
             "simple mixture with well-separated components whose CRF changes "
             "slowly and steadily with the gradient.",
     "tradeoff": "Where the response has an edge, an RBF must either smooth it "
                 "away or collapse a lengthscale to fit it; both make the "
                 "proposals worse than a Matérn's on the same data."},
    {"key": "default", "label": "Installed BoTorch default", "recommended": False,
     "family": "Gaussian process",
     "smoothness": "whatever the installed version builds",
     "what": "No covariance module is passed at all; BoTorch's SingleTaskGP "
             "builds whatever the installed release builds.",
     "when": "Only when reproducing a campaign that was created this way, or "
             "auditing what the library does on its own.",
     "tradeoff": "Cannot go stale against a future BoTorch, but also cannot be "
                 "described in advance: the Method page reports what it turned "
                 "out to be."},
]

#: Acquisition functions. `qlognei` is the default and the one the science was
#: validated with; the rest are offered because a campaign with a clear brief
#: sometimes wants a different balance of exploring and exploiting, and the
#: choice is recorded so a reader can tell which campaign ran which.
ACQUISITIONS = ("qlognei", "qlogei", "ucb", "logpi")
DEFAULT_ACQUISITION = "qlognei"
ACQ_PARAM_DEFAULTS: dict[str, float] = {"ucb_beta": 2.0}

ACQUISITION_OPTIONS: list[dict[str, Any]] = [
    {"key": "qlognei", "label": "Log noisy expected improvement",
     "short": "qLogNEI", "recommended": True,
     "class": "qLogNoisyExpectedImprovement",
     "what": "Scores every candidate by how much it is expected to improve on "
             "the best result so far, treating that best result as the noisy "
             "measurement it is, and working in log space so the score keeps a "
             "gradient under a flat posterior.",
     "when": "Any campaign with replicates and a measured sigma - which is "
             "every campaign this app is built for.",
     "tradeoff": "The most careful of the four and the slowest to evaluate; "
                 "a proposal takes a few seconds rather than one.",
     "params": []},
    {"key": "qlogei", "label": "Log expected improvement",
     "short": "qLogEI", "recommended": False,
     "class": "qLogExpectedImprovement",
     "what": "The same expected improvement, but against the best OBSERVED "
             "score rather than a noise-aware incumbent.",
     "when": "A very quiet instrument, where sigma is small against the "
             "spread of scores and a single lucky run is unlikely to become an "
             "incumbent nothing can beat.",
     "tradeoff": "A noisy high reading sets a bar the model cannot clear, and "
                 "the campaign stalls around it. The noisy form exists to avoid "
                 "exactly that.",
     "params": []},
    {"key": "ucb", "label": "Upper confidence bound",
     "short": "UCB", "recommended": False,
     "class": "qUpperConfidenceBound",
     "what": "Proposes wherever the posterior mean plus beta times the "
             "posterior spread is highest. beta sets how much unexplored "
             "territory is worth against a known good region.",
     "when": "A scouting campaign whose brief is to map the space rather than "
             "to converge - a high beta keeps it exploring - or a late-stage "
             "confirmation at low beta.",
     "tradeoff": "beta is a dial the analyst has to set and defend; there is "
                 "no measured quantity that chooses it. It does not account "
                 "for the incumbent being noisy.",
     "params": [{"key": "ucb_beta", "label": "beta", "default": 2.0,
                 "min": 0.1, "max": 10.0, "step": 0.1,
                 "note": "2.0 is a conventional balance. Below 1 the search "
                         "sits on the best known region; above 4 it roams."}]},
    {"key": "logpi", "label": "Log probability of improvement",
     "short": "LogPI", "recommended": False,
     "class": "LogProbabilityOfImprovement",
     "what": "Proposes wherever the model thinks ANY improvement over the best "
             "observed score is most probable, regardless of how large.",
     "when": "Late in a campaign, when the question is whether a small "
             "refinement near the incumbent exists at all.",
     "tradeoff": "Greedy: it prefers a near-certain tiny gain to a possible "
                 "large one, so it hugs the incumbent and under-explores. Not "
                 "a good way to spend the first half of a budget.",
     "params": []},
]


def surrogate_option(key: str) -> dict[str, Any]:
    for o in SURROGATE_OPTIONS:
        if o["key"] == key:
            return o
    raise ValueError(f"kernel must be one of {KERNELS}, not {key!r}")


def acquisition_option(key: str) -> dict[str, Any]:
    for o in ACQUISITION_OPTIONS:
        if o["key"] == key:
            return o
    raise ValueError(f"acquisition must be one of {ACQUISITIONS}, not {key!r}")


def resolve_acq_params(acquisition: str,
                       params: Optional[dict] = None) -> dict[str, float]:
    """Only the parameters the chosen acquisition actually reads, at their
    defaults where unset, validated against the option's own bounds."""
    opt = acquisition_option(acquisition)
    out: dict[str, float] = {}
    given = dict(params or {})
    for p in opt["params"]:
        v = given.get(p["key"], p["default"])
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{p['label']} must be a number, not {v!r}")
        if not (p["min"] <= v <= p["max"]):
            raise ValueError(f"{p['label']} must be between {p['min']} and "
                             f"{p['max']}, not {v}")
        out[p["key"]] = v
    return out


def describe_acquisition(acquisition: str,
                         params: Optional[dict] = None) -> str:
    """One line for the Config sheet and the Method page."""
    opt = acquisition_option(acquisition)
    rp = resolve_acq_params(acquisition, params)
    extra = ", ".join(f"{k}={v:g}" for k, v in rp.items())
    if acquisition == "qlognei":
        extra = "prune_baseline=True"
    return f"{opt['class']}" + (f" ({extra})" if extra else "")


def make_covar_module(d: int, kernel: str = DEFAULT_KERNEL):
    """The covariance module, with BoTorch's priors intact.

    Returns None for `default`, meaning "pass nothing and let the installed
    version decide" - kept because it is the only setting that cannot go stale
    against a future BoTorch.

    The prior comes from BoTorch's constructor, not from here. That is the
    whole point: choosing Matern over RBF is a statement about how smooth
    retention is, and it should not cost the regularisation that stops a
    twenty-point fit from collapsing.
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, not {kernel!r}")
    if kernel == "default":
        return None
    from botorch.models.utils.gpytorch_modules import (
        get_covar_module_with_dim_scaled_prior)
    try:
        cm = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=d, use_rbf_kernel=(kernel == "rbf"))
        if kernel == "matern32":
            # The same priors and the same constraint, read off the module
            # BoTorch just built rather than re-typed here, so this stays in
            # step with the installed release. Only nu changes.
            from gpytorch.kernels import MaternKernel
            from gpytorch.constraints import GreaterThan
            from gpytorch.priors import LogNormalPrior
            prior = cm.lengthscale_prior
            lp = LogNormalPrior(loc=float(prior.loc), scale=float(prior.scale))
            lb = float(cm.raw_lengthscale_constraint.lower_bound)
            cm = MaternKernel(nu=1.5, ard_num_dims=d, lengthscale_prior=lp,
                              lengthscale_constraint=GreaterThan(
                                  lb, transform=None, initial_value=lp.mode))
        return cm
    except TypeError as exc:
        # A BoTorch without `use_rbf_kernel` cannot give us a Matern with these
        # priors. Refuse rather than silently fitting a different model than
        # the campaign's Config claims.
        raise RuntimeError(
            f"the installed BoTorch's get_covar_module_with_dim_scaled_prior "
            f"does not accept use_rbf_kernel ({exc}), so a {kernel} kernel "
            f"with BoTorch's own priors cannot be built. Install the pinned "
            f"version, or set the campaign's kernel to 'default'.") from exc


def describe_kernel(kernel: str, d: int) -> dict[str, Any]:
    """What the campaign's kernel choice actually builds. For the Method page."""
    out = {"kernel": kernel, "d": d, "available": False}
    try:
        cm = make_covar_module(d, kernel)
    except Exception as exc:                                  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["available"] = True
    if cm is None:
        out["module"] = "none passed - the installed BoTorch's own default"
        return out
    base = getattr(cm, "base_kernel", cm)
    out["module"] = type(cm).__name__
    out["base"] = type(base).__name__
    out["nu"] = float(getattr(base, "nu", float("nan")))
    out["ard_num_dims"] = int(getattr(base, "ard_num_dims", d) or d)
    out["outputscale"] = hasattr(cm, "outputscale")
    return out


def _ard_lengthscales(model, d: int) -> np.ndarray:
    """Works whether the covariance module is a bare kernel (BoTorch's modern
    default) or wrapped in a ScaleKernel (older defaults)."""
    cm = getattr(model, "covar_module", None)
    k = getattr(cm, "base_kernel", cm)
    ls = getattr(k, "lengthscale", None)
    if ls is None:
        return np.full(d, np.nan)
    return np.atleast_1d(ls.detach().squeeze().cpu().numpy()).ravel()


def fit_gp(train_u, train_y, sigma_crf: Optional[float] = None,
           run_order=None, model_run_order: bool = False,
           kernel: str = DEFAULT_KERNEL) -> FitResult:
    """Fit the surrogate on one row per RUN (replicates repeat x)."""
    torch = _torch()
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from botorch.utils.transforms import normalize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    U = np.atleast_2d(np.asarray(train_u, dtype=float))
    y = np.asarray(train_y, dtype=float).ravel()
    if len(U) != len(y):
        raise ValueError(f"{len(U)} inputs vs {len(y)} targets")
    if len(y) < 2:
        raise ValueError("need at least 2 observations to fit")

    nf = noise_floor_std(y, sigma_crf)
    X = augment_run_order(U, run_order, model_run_order)
    bounds = _model_bounds(model_run_order)
    train_x = normalize(torch.as_tensor(X, **tkwargs()), bounds)
    ty = torch.as_tensor(y, **tkwargs()).unsqueeze(-1)

    covar = make_covar_module(X.shape[1], kernel)
    model = SingleTaskGP(train_x, ty,
                         likelihood=_make_likelihood(nf.floor_z),
                         outcome_transform=Standardize(m=1),
                         **({} if covar is None else {"covar_module": covar}))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

    scale = float(np.atleast_1d(
        model.outcome_transform.stdvs.detach().cpu().numpy()).ravel()[0])
    noise_sd = float(model.likelihood.noise.detach().sqrt().mean()) * scale
    return FitResult(model=model, noise=nf, n=len(y), d=X.shape[1],
                     model_run_order=model_run_order,
                     lengthscales=_ard_lengthscales(model, X.shape[1]),
                     noise_sd_crf=noise_sd, kernel=kernel)


def posterior(fit: FitResult, U, observation_noise: bool = False,
              run_order=None) -> tuple[np.ndarray, np.ndarray]:
    """Posterior (mean, sd) at absolute-unit points, in CRF units.

    observation_noise=False gives the SURROGATE's own uncertainty - the part a
    new measurement can reduce, and the part the uncertainty audit tracks. The
    noise part is reported separately by `FitResult.noise_sd_crf`.

    With run order modelled, this is a posterior AT A MOMENT: pass the run
    order the question is being asked about.
    """
    torch = _torch()
    from botorch.utils.transforms import normalize
    X = augment_run_order(np.atleast_2d(np.asarray(U, dtype=float)),
                          run_order, fit.model_run_order)
    x = normalize(torch.as_tensor(X, **tkwargs()), _model_bounds(fit.model_run_order))
    fit.model.eval()
    with torch.no_grad():
        post = fit.model.posterior(x, observation_noise=observation_noise)
        mu = post.mean.squeeze(-1).cpu().numpy().ravel()
        sd = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy().ravel()
    return mu, sd


@dataclass
class Suggestion:
    u: np.ndarray
    acq_value: float
    posterior_mean: float
    posterior_sd: float
    fit: FitResult
    phase: str            # "cold_start" | "optimise"
    reason: str
    acquisition: str = DEFAULT_ACQUISITION
    acq_params: dict[str, float] = field(default_factory=dict)


def _build_acquisition(fit: FitResult, train_x, train_y, acquisition: str,
                       params: dict[str, float]):
    """The acquisition object for one proposal, plus the sentence the Run tab
    shows beside the method it produced."""
    from botorch.acquisition.logei import (qLogExpectedImprovement,
                                           qLogNoisyExpectedImprovement)
    torch = _torch()
    if acquisition == "qlognei":
        return (qLogNoisyExpectedImprovement(model=fit.model, X_baseline=train_x,
                                             prune_baseline=True),
                "model-driven proposal: highest expected improvement over the "
                "best result so far, allowing for that result being a noisy "
                "measurement")
    y = np.asarray(train_y, dtype=float).ravel()
    best_f = float(np.nanmax(y))
    if acquisition == "qlogei":
        return (qLogExpectedImprovement(model=fit.model, best_f=best_f),
                f"model-driven proposal: highest expected improvement over the "
                f"best observed score ({best_f:.3f} CRF)")
    if acquisition == "ucb":
        from botorch.acquisition.monte_carlo import qUpperConfidenceBound
        beta = float(params.get("ucb_beta", ACQ_PARAM_DEFAULTS["ucb_beta"]))
        return (qUpperConfidenceBound(model=fit.model, beta=beta),
                f"model-driven proposal: highest upper confidence bound "
                f"(posterior mean + {beta:g} x posterior spread)")
    if acquisition == "logpi":
        from botorch.acquisition.analytic import LogProbabilityOfImprovement
        return (LogProbabilityOfImprovement(model=fit.model, best_f=best_f),
                f"model-driven proposal: highest probability of beating the "
                f"best observed score ({best_f:.3f} CRF)")
    raise ValueError(f"acquisition must be one of {ACQUISITIONS}, "
                     f"not {acquisition!r}")


def suggest_next(train_u, train_y, sigma_crf: Optional[float] = None,
                 run_order=None, next_run_order=None,
                 model_run_order: bool = False, seed: Optional[int] = None,
                 num_restarts: int = 10, raw_samples: int = 512,
                 fit: Optional[FitResult] = None,
                 kernel: str = DEFAULT_KERNEL,
                 acquisition: str = DEFAULT_ACQUISITION,
                 acq_params: Optional[dict] = None) -> Suggestion:
    """Fit and return the next method to run.

    NOTE on the fifth input, if it is ever switched on: X_baseline carries each
    row's HISTORICAL run order, so the improvement threshold qLogNEI compares
    against is still max_i f(x_i, t_i). Modelling run order removes drift from
    the posterior surface but leaves it in the yardstick. That is a deliberate
    limitation, not an oversight - see the phantom-incumbent handling in
    drift.py for the other half.
    """
    torch = _torch()
    from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
    from botorch.optim import optimize_acqf
    from botorch.utils.transforms import normalize, unnormalize

    if seed is not None:
        torch.manual_seed(int(seed))
    if fit is None:
        fit = fit_gp(train_u, train_y, sigma_crf, run_order, model_run_order,
                     kernel=kernel)

    Xb = augment_run_order(np.atleast_2d(train_u), run_order, model_run_order)
    train_x = normalize(torch.as_tensor(Xb, **tkwargs()),
                        _model_bounds(model_run_order))

    params = resolve_acq_params(acquisition, acq_params)
    acq, reason = _build_acquisition(fit, train_x, train_y, acquisition, params)
    if model_run_order:
        if next_run_order is None:
            raise ValueError(
                "run order is modelled but next_run_order was not given. "
                "Refusing to guess when 'now' is.")
        t_pin = float(run_order_to_norm(next_run_order))
        acq = FixedFeatureAcquisitionFunction(acq_function=acq, d=5,
                                              columns=[4], values=[t_pin])

    norm_bounds = torch.stack([torch.zeros(4, **tkwargs()),
                               torch.ones(4, **tkwargs())])
    ineq = [(torch.tensor(S.CON_IDX),
             torch.tensor(S.CON_COEF, **tkwargs()),
             S.CON_RHS)]
    cand_x, acq_val = optimize_acqf(
        acq_function=acq, bounds=norm_bounds, q=1,
        num_restarts=num_restarts, raw_samples=raw_samples,
        inequality_constraints=ineq)
    cand = unnormalize(cand_x, _candidate_bounds()).detach().cpu().numpy()

    # the gate again: nothing leaves this function unchecked
    u = S.require_feasible(np.atleast_2d(cand)[0], "acquisition candidate")
    mu, sd = posterior(fit, u.reshape(1, -1), run_order=next_run_order)
    return Suggestion(u=u, acq_value=float(acq_val),
                      posterior_mean=float(mu[0]), posterior_sd=float(sd[0]),
                      fit=fit, phase="optimise", reason=reason,
                      acquisition=acquisition, acq_params=params)


def cold_start_methods(n: int, seed: Optional[int] = None) -> np.ndarray:
    """n feasible seed methods, before any model exists."""
    return S.initial_sample(int(n), seed=seed)
