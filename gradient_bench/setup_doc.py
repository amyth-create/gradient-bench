"""setup_doc.py - the page you print when someone asks what you did.

NOT A SUMMARY OF THE CONFIGURATION.  A summary is what a reader can already
guess. This assembles the whole method: every kernel prior with its
distribution and its parameters, the constraint in both the absolute and the
normalised form, the acquisition settings, the objective's formula, the
noise-floor derivation WITH THIS CAMPAIGN'S OWN NUMBERS SUBSTITUTED IN, every
picker field with the campaign's value beside the picker's default, and every
library version - the one recorded when the campaign was created and the one
installed right now, because those disagreeing is the single most likely way a
campaign quietly stops being reproducible.

WHY IT IS COMPUTED, NOT WRITTEN DOWN.  Everything here is read from the running
code and the campaign's own workbook. A page maintained by hand describes the
app as it was when someone last remembered to update the page.

WHAT IT DOES NOT DO.  It does not fit anything. A reader opening this page
should not trigger a GP fit, so the kernel section introspects BoTorch's
default module constructors directly and reports what they would build. If
torch is not installed it says so plainly rather than guessing.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pandas as pd

from .core import budget as B
from .core import crf as CRF
from .core import drift as D
from .core import optimiser as O
from .core import picker as P
from .core import space as S
from .store import sheet as SH
from .store.campaign import Campaign


def _row(label: str, value: Any, note: str = "",
         term: str = "") -> dict[str, Any]:
    """One line of the Method page.

    `term` names a glossary entry, so a row rendered from Python can still
    carry its definition where it appears - the rule is dense explanation, and
    this page holds the densest technical vocabulary in the app.
    """
    if isinstance(value, float) and not math.isfinite(value):
        value = None
    return {"label": label, "value": value, "note": note, "term": term}


def _sec(key: str, title: str, blurb: str = "", **kw) -> dict[str, Any]:
    out = {"key": key, "title": title, "blurb": blurb, "rows": [],
           "tables": [], "warnings": [], "math": []}
    out.update(kw)
    return out


# ── the surrogate's priors, introspected without fitting ────────────────────
def _describe_prior(prior) -> str:
    """`LogNormal(loc=1.11, scale=1.73)` from a live gpytorch prior object."""
    if prior is None:
        return "none"
    name = type(prior).__name__
    parts = []
    for attr in ("loc", "scale", "concentration", "rate", "a", "b"):
        v = getattr(prior, attr, None)
        if v is None:
            continue
        try:
            arr = np.atleast_1d(np.asarray(v.detach().cpu().numpy()
                                           if hasattr(v, "detach") else v,
                                           dtype=float))
            shown = (f"{float(arr.ravel()[0]):.4g}" if arr.size == 1
                     else "[" + ", ".join(f"{float(x):.4g}"
                                          for x in arr.ravel()[:4]) + "]")
            parts.append(f"{attr}={shown}")
        except Exception:                                   # noqa: BLE001
            continue
    return f"{name}({', '.join(parts)})" if parts else name


def _describe_constraint(con) -> str:
    if con is None:
        return "none"
    lb = getattr(con, "lower_bound", None)
    try:
        lb = float(lb)
    except (TypeError, ValueError):
        lb = None
    return (f"{type(con).__name__}({lb:.6g})" if lb is not None
            else type(con).__name__)


def kernel_report(d: int, floor_z: float,
                  kernel: str = "matern52") -> dict[str, Any]:
    """What THIS CAMPAIGN'S kernel setting actually builds, read from BoTorch.

    It must be handed the campaign's kernel. Reporting BoTorch's default here
    while `fit_gp` passes a Matern module would print `base kernel: RBFKernel`
    on a Matern campaign - on the one page whose whole premise is that it is
    assembled from the running code rather than maintained by hand.

    A hand-built covariance module would discard BoTorch's hyperparameter
    priors; one built by BoTorch's own constructor keeps them, which is what
    `optimiser.make_covar_module` uses.
    """
    out: dict[str, Any] = {"available": False, "rows": [], "warnings": []}
    try:
        import botorch                                       # noqa: F401
    except Exception as exc:                                 # noqa: BLE001
        out["warnings"].append(
            f"BoTorch is not importable here ({type(exc).__name__}), so the "
            f"kernel and its priors cannot be read from the installed code. "
            f"The campaign's Config sheet records which versions it was "
            f"created under; install them to see this section.")
        return out

    out["available"] = True
    try:
        cov = O.make_covar_module(d, kernel)
        if cov is None:
            out["rows"].append(_row(
                "covariance module", "none passed",
                "this campaign is set to `default`, so the installed BoTorch's "
                "own default is used and recorded by version"))
            return out
        base = getattr(cov, "base_kernel", cov)
        out["rows"] += [
            _row("covariance module", type(cov).__name__,
                 "built by BoTorch's own get_covar_module_with_dim_scaled_prior, "
                 "so the priors below are BoTorch's - a hand-built module would "
                 "discard them"),
            _row("base kernel", type(base).__name__,
                 f"nu = {float(getattr(base, 'nu', float('nan'))):.1f}"
                 if hasattr(base, "nu") else "no smoothness parameter (RBF)"),
            _row("ARD dimensions", int(getattr(base, "ard_num_dims", d) or d),
                 "one lengthscale per input, so the model can find that the "
                 "score changes faster along one parameter than another"),
            _row("lengthscale prior",
                 _describe_prior(getattr(base, "lengthscale_prior", None)),
                 "on the log scale; the location grows with the number of "
                 "inputs, which is what 'dim scaled' means"),
            _row("lengthscale constraint",
                 _describe_constraint(getattr(base, "raw_lengthscale_constraint",
                                              None)), ""),
            _row("outputscale",
                 ("present: " + type(cov).__name__) if hasattr(cov, "outputscale")
                 else "none",
                 "absent by design in this constructor; an outputscale is "
                 "partly degenerate with the noise term"),
        ]
    except Exception as exc:                                 # noqa: BLE001
        out["warnings"].append(
            f"the campaign's kernel ({kernel}) could not be built from the "
            f"installed BoTorch ({type(exc).__name__}: {exc}). This is exactly "
            f"the kind of change between releases that the version pin exists "
            f"to prevent.")

    try:
        lik = O._make_likelihood(float(floor_z))
        nc = getattr(lik, "noise_covar", None)
        out["rows"] += [
            _row("likelihood", type(lik).__name__, ""),
            _row("noise prior", _describe_prior(getattr(nc, "noise_prior", None)),
                 getattr(lik, "_prior_source", "")),
            _row("noise constraint",
                 _describe_constraint(getattr(nc, "raw_noise_constraint", None)),
                 "the floor is this campaign's measured replicate noise, "
                 "converted into standardized units - see the noise model"),
        ]
    except Exception as exc:                                 # noqa: BLE001
        out["warnings"].append(f"likelihood could not be built: "
                               f"{type(exc).__name__}: {exc}")
    return out


# ── the document ────────────────────────────────────────────────────────────
def build(c: Campaign) -> dict[str, Any]:
    info = c.info()
    cfg = c.config()
    d = c.design()
    y = (pd.to_numeric(d["CRF"], errors="coerce").to_numpy(float)
         if len(d) else np.zeros(0))
    sd, groups, dof = c.sigma_crf()
    n_rep = c.n_replicates()
    sections: list[dict[str, Any]] = []

    # 1 ── identity ─────────────────────────────────────────────────────
    s = _sec("identity", "Campaign",
             "Who ran it, on what, and against which sample. A campaign is a "
             "folder; that folder is its identity.")
    for key, label in [("name", "Campaign name"), ("analyst", "Analyst"),
                       ("sample", "Sample"), ("column", "Column"),
                       ("mobile_phase", "Mobile phase"),
                       ("flow_rate", "Flow rate (mL/min)"),
                       ("detector", "Detector / wavelength"),
                       ("instrument", "Instrument / serial"),
                       ("created", "Date started"), ("notes", "Notes")]:
        v = info.get(key, "")
        s["rows"].append(_row(label, "" if v is None else v))
    s["rows"].append(_row("Folder", c.root))
    hist = c.metadata_history()
    if hist:
        s["tables"].append({
            "title": "Metadata changes since creation",
            "note": "Correcting how a column is written down is routine. "
                    "Changing WHICH column or sample was on the instrument "
                    "means the earlier runs and the later ones are not the "
                    "same experiment - this table is how a reader tells.",
            "columns": ["when", "runs recorded", "field", "from", "to", "why"],
            "rows": [[h.get("at", ""), h.get("runs_recorded", ""), ch["field"],
                      ch["from"], ch["to"], h.get("reason", "")]
                     for h in hist for ch in h.get("changes", [])]})
    sections.append(s)

    # 2 ── the objective ────────────────────────────────────────────────
    s = _sec("objective", "The objective",
             "One number, maximised. Frozen at version 1.0.0: chosen by a "
             "56-candidate study against 20 expert-ranked chromatograms, where "
             "it finished first on every metric. A moving objective makes "
             "campaigns incomparable, so it does not move.")
    s["math"] = ["CRF = n_clean_peaks x (1 - hump_time_fraction) ^ 2"]
    s["rows"] += [
        _row("Formula", "n_clean_peaks * (1 - hump_time_fraction) ** 2",
             term="CRF"),
        _row("Version", CRF.CRF_VERSION,
             "stamped on every row, so a re-derived score can be matched to "
             "the definition that produced it"),
        _row("Inputs", ", ".join(CRF.CRF_COLUMNS),
             "the ONLY two measurements that may reach the objective"),
        _row("Direction", "maximised"),
        _row("Why squared", "the penalty bites only once the hump takes a real "
                            "share of the run"),
    ]
    s["warnings"].append(
        "The 15 trace-health descriptors are recorded on every run and are "
        "structurally barred from the objective: `assert_objective_is_clean` "
        "runs on EVERY analysis and fails loudly if one of them ever reaches "
        "the score. They diagnose the instrument; they must never score the "
        "chemistry.")
    sections.append(s)

    # 3 ── the design space ─────────────────────────────────────────────
    s = _sec("space", "The design space",
             "Four controllable parameters, a box, and one linear constraint.")
    s["tables"].append({
        "title": "Parameters",
        "columns": ["parameter", "lower", "upper", "unit"],
        "rows": [[n, float(lo), float(hi), u] for n, lo, hi, u in zip(
            S.P4_NAMES, S.P4_LOWER, S.P4_UPPER,
            ["fraction B", "fraction B", "minutes", "degrees C"])]})
    s["math"] = [
        f"absolute:    end_phi >= start_phi + {S.MIN_SPAN}",
        f"normalised:  {S.CON_COEF[0]:+.2f} * end_n {S.CON_COEF[1]:+.2f} * "
        f"start_n >= {S.CON_RHS:+.2f}"
        f"   (indices {S.CON_IDX})",
    ]
    s["rows"] += [
        _row("MIN_SPAN", S.MIN_SPAN,
             "zero ON PURPOSE: end_phi == start_phi is isocratic, and isocratic "
             "is a legitimate method that must stay inside the space"),
        _row("Feasible fraction", 0.868421,
             "the analytic share of the box the constraint admits. Stated "
             "rather than sampled: a Monte Carlo estimate on every page load "
             "would print a slightly different 'fact' each time"),
        _row("Enforced at", "cold-start seeds, acquisition output, and recording",
             "one gate, `space.check4`, called at all three - a method that "
             "cannot be run cannot be proposed OR written down"),
    ]
    sections.append(s)

    # 4 ── replicates and the noise model ───────────────────────────────
    s = _sec("noise", "Replicates and the noise model",
             "Two runs at identical settings differ only by noise. That is the "
             "only way to measure sigma directly - spatial scatter cannot do "
             "it, because the same points are explained equally well by a "
             "smooth surface with large noise and a wiggly one with small.")
    s["rows"] += [
        _row("Replicates per method", n_rep, term="replicate"),
        _row("Methods with >= 2 runs", groups),
        _row("Degrees of freedom", dof),
        _row("Measured sigma (CRF)",
             (None if not np.isfinite(sd) else round(float(sd), 4)),
             ("pooled within-method sd" if np.isfinite(sd)
              else "not yet measurable - no method has two recorded runs"),
             term="sigma"),
        _row("Fallback sigma (CRF)", O.FALLBACK_SIGMA_CRF,
             "used until a replicate pair lands. NOT a measurement: one clean "
             "peak is the quantum of an integer-count objective, the smallest "
             "disagreement it can express"),
    ]
    if y.size > 1:
        nf = O.noise_floor_std(y, sd if np.isfinite(sd) and sd > 0 else None)
        s["math"] = nf.explain()
        s["rows"] += [
            _row("sd(train_y)", round(nf.sd_y, 4), "this campaign's own spread"),
            _row("var(train_y)", round(nf.var_y, 4)),
            _row("floor_z", round(nf.floor_z, 6),
                 "sigma^2 / var(y) - Standardize(m=1) divides the targets by "
                 "their sd, so a floor quoted in CRF units means nothing to "
                 "the likelihood until it is converted", term="floor_z"),
            _row("floor as an sd (CRF)", (None if not np.isfinite(nf.floor_sd_crf)
                                          else round(nf.floor_sd_crf, 4))),
            _row("sigma / sd(y)", (None if not np.isfinite(nf.ratio)
                                   else round(nf.ratio, 4))),
        ]
        if nf.dominated:
            s["warnings"].append(
                "floor_z >= 1: the measured replicate noise is as large as the "
                "whole campaign's spread. The posterior will be nearly flat and "
                "the acquisition nearly uninformative. That is the finding, not "
                "a fault - do not lower the floor to make the plot look better.")
        if sd == 0 and groups:
            s["warnings"].append(
                "A replicate pair tied EXACTLY. With an integer peak count that "
                "is a real outcome, not evidence of zero noise: sigma is smaller "
                "than one clean peak and this pair cannot resolve it. It is "
                "treated as unmeasured. Replicating one method four or five "
                "times would pin it properly.")
    else:
        s["warnings"].append(
            "Fewer than two design runs are recorded, so the noise floor cannot "
            "be derived yet. It appears here with the campaign's own numbers as "
            "soon as it can.")
    sections.append(s)

    # 5 ── the surrogate ────────────────────────────────────────────────
    dim = O.model_dim(str(cfg.get("model_run_order", "false")).lower() == "true")
    floor_z = 1e-4
    if y.size > 1:
        floor_z = O.noise_floor_std(y, sd if np.isfinite(sd) and sd > 0
                                    else None).floor_z
    s = _sec("surrogate", "The surrogate",
             "A Gaussian process over the design space. The covariance module "
             "is built by BoTorch's own constructor, so choosing the kernel "
             "does not cost the hyperparameter priors - only a module built by "
             "hand would.")
    kern = c.kernel()
    kd = O.describe_kernel(kern, dim)
    ko = O.surrogate_option(kern)
    s["rows"] += [
        _row("Model", "botorch.models.SingleTaskGP", term="GP"),
        _row("Kernel", f"{ko['label']} ({kern})", term="kernel",
             note=ko["what"] + " " + ko["when"]),
        _row("Chosen at creation", "locked for the life of the campaign",
             "a campaign fitted under one kernel and refitted under another "
             "is not the same campaign"),
        _row("Covariance module built by",
             ("BoTorch's own get_covar_module_with_dim_scaled_prior"
              "(use_rbf_kernel=False)" if kern == "matern52" else
              "BoTorch's own get_covar_module_with_dim_scaled_prior, then a "
              "MaternKernel(nu=1.5) carrying the same prior and constraint"
              if kern == "matern32" else
              "BoTorch's own get_covar_module_with_dim_scaled_prior"
              if kern == "rbf" else
              "nothing - no covar_module is passed"),
             "the priors below are BoTorch's, not this app's: choosing the "
             "kernel does not cost the regularisation that stops a "
             "twenty-point fit collapsing"),
        _row("Built as", (kd.get("base") or kd.get("module") or "-")
             + (f", nu = {kd['nu']:.1f}" if np.isfinite(kd.get("nu", float("nan")))
                else "")
             + (f", ARD over {kd['ard_num_dims']} dims" if kd.get("ard_num_dims")
                else ""),
             kd.get("error", "")),
        _row("Outcome transform", "Standardize(m=1)",
             "targets are divided by their unbiased sd, which is why the noise "
             "floor has to be converted"),
        _row("Inputs", ", ".join(O.model_names(dim == 5)) + f"  (d = {dim})"),
        _row("Run order as a covariate",
             "on" if dim == 5 else "off (default)",
             "it works, but earns no interface until a leave-one-out "
             "comparison on a recorded campaign says it should"),
        _row("Fitted with", "fit_gpytorch_mll (exact marginal log likelihood)"),
    ]
    kr = kernel_report(dim, floor_z, kern)
    s["rows"] += kr["rows"]
    s["warnings"] += kr["warnings"]
    sections.append(s)

    # 6 ── the acquisition ──────────────────────────────────────────────
    acq_key = c.acquisition()
    ao = O.acquisition_option(acq_key)
    ap = c.acq_params()
    s = _sec("acquisition", "The acquisition function",
             "Where to run next. Chosen once, at creation, and locked for the "
             "life of the campaign - the same discipline as the kernel.")
    s["rows"] += [
        _row("Acquisition", f"{ao['label']} ({ao['short']})",
             ao["what"], term="acquisition function"),
        _row("BoTorch class", O.describe_acquisition(acq_key, ap),
             ("NOISY, because the incumbent is itself a noisy measurement; "
              "LOG, because plain EI has vanishing gradients under a flat "
              "posterior - exactly the regime a noisy objective creates")
             if acq_key == "qlognei" else ao["tradeoff"]),
        _row("Chosen because", ao["when"]),
    ]
    for p in ao["params"]:
        s["rows"].append(_row(p["label"], ap.get(p["key"], p["default"]),
                              p["note"]))
    s["rows"] += [
        _row("Restarts", cfg.get("acq_num_restarts", 10)),
        _row("Raw samples", cfg.get("acq_raw_samples", 512)),
        _row("Constraint passed to the optimiser",
             f"inequality triple: indices {S.CON_IDX}, coefficients "
             f"[{S.CON_COEF[0]:+.2f}, {S.CON_COEF[1]:+.2f}], rhs {S.CON_RHS:+.2f}",
             "the same constraint as above, in the normalised units "
             "optimize_acqf works in"),
        _row("Cold start", f"{c.n_seed()} spread-out methods before any model "
                           f"is fitted",
             "recorded as measured; no model exists yet to be wrong"),
    ]
    sections.append(s)

    # 7 ── the drift monitor ────────────────────────────────────────────
    u = c.reference_method()
    v = c.verdict()
    s = _sec("drift", "The instrument-check method and the drift monitor",
             "One method held fixed and re-run on a schedule. Its movement can "
             "only be the instrument, which is the only thing in the loop that "
             "separates drift from chemistry.")
    if u is not None:
        s["tables"].append({
            "title": "The reference method",
            "columns": ["time (min)", "%B", "phase"],
            "rows": [[g["time_min"], g["pct_b"], g["phase"]]
                     for g in S.gradient_table(u)]})
        s["rows"].append(_row("Temperature", f"{float(u[3]):.1f} C"))
    s["rows"] += [
        _row("Cadence", f"every {int(float(cfg.get('reference_every', D.REFERENCE_EVERY)))} proposals",
             "ENFORCED, not advised: the schedule blocks proposing",
             term="cadence"),
        _row("WATCH limit", cfg.get("drift_watch_delta", D.WATCH_DELTA),
             "CRF below the anchor"),
        _row("HALT limit", cfg.get("drift_halt_delta", D.HALT_DELTA),
             "CRF below the anchor; blocks proposing outright"),
        _row("Monotone rule", f"{int(float(cfg.get('drift_monotone_k', D.MONOTONE_K)))} "
                              f"consecutive falls -> HALT"),
        _row("References recorded", len(c.references())),
        _row("Current verdict", v.verdict, " ".join(v.reasons)),
        _row("Drift-adjusted scores",
             "on" if str(cfg.get("use_drift_adjusted", "false")).lower() == "true"
             else "off (default)",
             "adjusting assumes the drift is REVERSIBLE and cannot check it; "
             "correcting an irreversibly degraded column optimises toward a "
             "machine you no longer own"),
    ]
    sections.append(s)

    # 8 ── the stopping rule ────────────────────────────────────────────
    bud = c.budget()
    s = _sec("stopping", "The stopping rule",
             "A campaign ends at its run budget, or when the analyst judges "
             "the separation good enough and closes it.")
    cost = bud.cost
    s["rows"] += [
        _row("Run budget", f"{bud.limit} methods",
             "UNIQUE METHODS - a repeat costs no budget, and neither does an "
             "instrument check"),
        _row("Original budget", f"{bud.original} methods"),
        _row("Methods used", bud.used),
        _row("Injections recorded", bud.injections,
             "what the campaign has actually cost the instrument"),
        _row("Budget in injections", cost["injections"],
             f"if the whole budget is spent: {cost['design_injections']} design "
             f"runs at {cost['n_replicates']} repeats, plus about "
             f"{cost['instrument_checks']} instrument checks - roughly "
             f"{cost['hours']} hours at a 30-minute gradient"),
        _row("Extensions", bud.n_extensions),
        _row("State", bud.state, bud.reason()),
    ]
    if bud.history:
        s["tables"].append({
            "title": "Budget history",
            "columns": ["when", "event", "from", "to", "at method", "why"],
            "rows": [[h.get("at", ""), h.get("event", ""), h.get("from", ""),
                      h.get("to", ""), h.get("used", ""), h.get("reason", "")]
                     for h in bud.history]})
    sections.append(s)

    # 9 ── the picker ───────────────────────────────────────────────────
    camp_pick = c.picker_config()
    defaults = P.picker_defaults()
    s = _sec("picker", "The peak picker",
             "What turns a chromatogram into the two numbers the objective "
             "eats. Every field is listed, not only the ones this campaign "
             "changed, because a reader cannot tell an untouched setting from "
             "an absent one.")
    s["rows"] += [
        _row("Picker version", cfg.get("picker_version",
                                       P._find_hplc_picker().__version__)),
        _row("Settings the campaign states", len(camp_pick)),
        _row("Settings at the picker's own default",
             len(defaults) - len(set(camp_pick) & set(defaults))),
    ]
    s["tables"].append({
        "title": "Every picker setting",
        "note": "`campaign` is what this campaign uses. Rows marked SET are "
                "stated by the campaign; the rest sit at the picker's default.",
        "columns": ["setting", "campaign", "picker default", "stated"],
        "rows": [[k, camp_pick.get(k, defaults[k]), defaults[k],
                  "SET" if k in camp_pick else ""]
                 for k in sorted(defaults)]})
    from . import rescore as RS
    ph = RS.history(c)
    if ph:
        s["tables"].append({
            "title": "Configuration changes, and the re-scores they forced",
            "note": "Changing the picker mid-campaign re-measures every stored "
                    "trace, so the CRF column keeps meaning one thing.",
            "columns": ["when", "changed", "rows moved", "why"],
            "rows": [[h.get("at", ""),
                      ", ".join(f"{k}: {a} -> {b}"
                                for k, (a, b) in (h.get("changed") or {}).items()),
                      f"{h.get('n_moved', '')}/{h.get('n_rows', '')}",
                      h.get("reason", "")] for h in ph]})
    sections.append(s)

    # 10 ── versions ────────────────────────────────────────────────────
    s = _sec("versions", "Library versions",
             "BoTorch's default kernel and priors have changed between "
             "releases. A campaign fitted under one and refitted under another "
             "is not the same campaign, which is why the versions are recorded "
             "per campaign and compared here against what is installed now.")
    now = O.library_versions()
    rows, drifted = [], []
    for lib in sorted(set(now) | {k[4:] for k in cfg if k.startswith("lib_")}):
        was = cfg.get(f"lib_{lib}", "")
        is_ = now.get(lib, "not installed")
        same = (str(was) == str(is_))
        rows.append([lib, was or "not recorded", is_,
                     "" if same else "DIFFERENT"])
        if was and not same:
            drifted.append(f"{lib} {was} -> {is_}")
    s["tables"].append({"title": "Recorded at creation vs installed now",
                        "columns": ["library", "at creation", "installed now",
                                    ""], "rows": rows})
    if drifted:
        s["warnings"].append(
            "The environment has moved since this campaign was created: "
            + "; ".join(drifted) + ". Refitting under a different BoTorch "
            "means a different default kernel and different priors, so the "
            "posterior and every proposal from it are not comparable with the "
            "ones this campaign already recorded. Pin the recorded versions "
            "before continuing it.")
    s["rows"].append(_row("Requirement pin", "botorch>=0.18,<0.19",
                          "deliberate; see requirements.txt"))
    sections.append(s)

    # 11 ── the record ──────────────────────────────────────────────────
    s = _sec("record", "The record",
             "The workbook is the campaign. Everything else can be rebuilt "
             "from it; it cannot be rebuilt from anything else.")
    s["rows"] += [
        _row("Schema version", cfg.get("schema_version", SH.SCHEMA_VERSION)),
        _row("Data columns", len(SH.data_columns()),
             "identity + 8 absolute gradient columns + every chromatographic "
             "feature + 15 trace-health descriptors + provenance"),
        _row("Rows", f"{c.n_injections()} instrument runs "
                     f"({len(d)} design, {len(c.references())} reference)"),
        _row("Workbook", os.path.basename(c.workbook)),
        _row("Writes", "atomic (temp file + os.replace), with a timestamped "
                       "backup taken first, last 20 kept"),
        _row("Traces kept", "every uploaded file, in traces/",
             "which is what makes re-scoring possible at all"),
    ]
    sections.append(s)

    return {
        "campaign": c.summary(),
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "sections": sections,
    }
