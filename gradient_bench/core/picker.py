"""picker.py - the adapter over hplc_picker.

TWO LEVELS OF CONFIGURATION, AND THE DIFFERENCE MATTERS.

The CAMPAIGN configuration is the default every run starts from, stored once in
the campaign's Config sheet. A PER-RUN override is the analyst correcting a
failed estimate on one chromatogram.

That is not a change of ruler. The picker's settings do not DEFINE the
measurement; they ESTIMATE a physical truth - the peaks that are really there,
the region that is really unresolved. When the estimator fails, an analyst
correcting it moves the number toward truth, which reduces error rather than
introducing inconsistency. Per-run tuning on design runs is therefore
unrestricted, and the interface should make it easy.

ONE EXCEPTION, AND IT IS NOT ABOUT THE OBJECTIVE.  Reference runs are a
DIFFERENTIAL measurement: their whole job is that the only thing changing
between ref01 and ref06 is the instrument. Retune the picker on one reference
and its movement becomes a picking artefact wearing the costume of drift - a
false HALT, or worse a real drift masked. `resolve_config` refuses a per-run
override on a reference run for that reason. If a reference genuinely needs
different treatment, re-score the WHOLE reference series.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .crf import CRF_COLUMNS, CRF_VERSION, assert_objective_is_clean, compute_crf


def _find_hplc_picker():
    """The peak picker. Vendored INTO the package, deliberately.

    It used to be a loose module beside `gradient_bench/`, found by searching
    the disk, with a second copy in `v3/` that the notebook was regenerated
    from. Searching meant the app could silently bind to whichever copy
    happened to be on sys.path first, and two copies of the file that defines
    what a peak IS could drift apart without anything noticing - the app and
    the notebook disagreeing about a chromatogram, with no record of it.

    Now there is one file, it lives here, and this import cannot resolve to
    anything else. `hplc_picker.py` at the project root is a three-line shim
    that hands back this same module object, so old imports keep working.
    """
    from . import hplc_picker
    return hplc_picker


#: The campaign default. snr = 5.0 is not a guess: tested against all twenty
#: adjudicated traces it gets 19/20 right, where 4.0 gets 11/20, 3.0 gets 5/20
#: and 2.5 gets 5/20. Lowering it to rescue one trace would break fifteen.
CAMPAIGN_DEFAULTS: dict[str, Any] = {
    "snr": 5.0,
    "hump_min_floor_ratio": 0.95,
    "hump_min_span_abs": 1.0,
    "arpls_lam": 1e5,
    "detection": "hybrid_fit",
}

#: The three controls the review screen exposes first, with the guidance text
#: shown beside each. Everything else in PickerConfig sits behind "advanced".
PRIMARY_CONTROLS = [
    {"key": "snr", "label": "Signal-to-noise gate",
     "term": "signal-to-noise gate", "default": 5.0,
     "raise_it": "Stricter. Fewer peaks - small real peaks start being rejected.",
     "lower_it": "More permissive. More peaks, including noise dressed as peaks.",
     "note": "The prominence gate is snr x the noise estimate. This is the "
             "control that decides what counts as a peak at all."},
    {"key": "hump_min_floor_ratio", "label": "Hump floor ratio",
     "term": "hump floor ratio", "default": 0.95,
     "raise_it": "FEWER humps. The region must be more completely filled in to "
                 "qualify.",
     "lower_it": "MORE humps. Partially filled regions start qualifying.",
     "note": "How elevated the signal's lower envelope must stay relative to the "
             "signal itself before a region counts as unresolved."},
    {"key": "hump_min_span_abs", "label": "Minimum hump span (min)",
     "term": "hump", "default": 1.0,
     "raise_it": "FEWER humps - only broad regions qualify.",
     "lower_it": "MORE humps - short stretches start qualifying.",
     "note": "The shortest stretch of time allowed to be called a hump."},
]

#: Everything else worth exposing, behind a disclosure. Grouped by what the
#: setting acts on, because the failure an analyst is looking at is physical -
#: "the hump was not found", "the baseline ate it", "that shoulder was missed" -
#: and the group is the fastest route from the symptom to the knob.
#:
#: These are DELIBERATELY second-class. The three primary controls plus drawing
#: a region by hand settle almost every disagreement; reaching in here means
#: changing how the estimator works rather than where its threshold sits, and
#: every value below was measured against the twenty adjudicated traces at its
#: current setting. The `note` on each one says what it is for; the campaign
#: default is the value that made 19/20 come out right.
ADVANCED_GROUPS = [
    {
        "key": "humps",
        "label": "Unresolved regions",
        "why": "How a hump is found, grown and merged, once the floor ratio and "
               "the minimum span have decided a candidate exists at all. Reach "
               "for these when a region you can SEE is unresolved is not being "
               "called one - though drawing it by hand on the trace is usually "
               "faster and always more honest, because it annotates one "
               "chromatogram instead of moving a threshold under all of them.",
        "controls": [
            {"key": "hump_level_frac", "label": "Hump level fraction",
             "default": 0.005, "step": 0.001,
             "raise_it": "FEWER humps. The floor must sit higher above zero "
                         "before the region counts as filled in.",
             "lower_it": "MORE humps. Very low raised floors start qualifying.",
             "note": "The height, as a fraction of the trace maximum, at which "
                     "the region's floor is measured. It is powerless against a "
                     "baseline fitter that has already absorbed the hump: if "
                     "the baseline bent into the raised floor, there is nothing "
                     "left here to measure. Draw the region by hand instead."},
            {"key": "hump_min_span_factor", "label": "Minimum span, in peak widths",
             "default": 4.0, "step": 0.5,
             "raise_it": "FEWER humps - the region must be wider relative to the "
                         "peaks in it.",
             "lower_it": "MORE humps - narrower regions qualify.",
             "note": "The other minimum-span gate, expressed in typical peak "
                     "widths rather than minutes. A region must clear BOTH this "
                     "and the absolute minimum span, so on a trace with broad "
                     "peaks this one is what actually binds."},
            {"key": "hump_min_real_peaks", "label": "Minimum peaks inside a hump",
             "default": 1, "step": 1,
             "raise_it": "FEWER humps. A raised region with only one peak in it "
                         "stops counting as unresolved.",
             "lower_it": "MORE humps - a raised floor alone can qualify.",
             "note": "A hump is a stretch where several things co-elute. Setting "
                     "this to 0 lets a plain baseline rise be called a hump, "
                     "which penalises the score for something that is not a "
                     "separation failure."},
            {"key": "hump_merge_gap", "label": "Merge gap (min)",
             "default": 4.5, "step": 0.5,
             "raise_it": "Neighbouring regions merge across bigger gaps - fewer, "
                         "longer humps, and a LARGER hump time fraction.",
             "lower_it": "Regions stay separate - more, shorter humps.",
             "note": "Two unresolved stretches separated by less than this are "
                     "treated as one. It moves the score directly: the objective "
                     "penalises hump TIME, and merging counts the clean gap "
                     "between two humps as hump."},
        ],
    },
    {
        "key": "baseline",
        "label": "Baseline",
        "why": "What the trace is measured against. Everything downstream - "
               "noise, prominence, the hump floor ratio - is computed after the "
               "baseline is subtracted, so a baseline that follows the signal "
               "too closely quietly removes the very structure being looked for.",
        "controls": [
            {"key": "arpls_lam", "label": "Baseline stiffness (arPLS lambda)",
             "term": "arPLS",
             "default": 1e5, "step": 1e5,
             "raise_it": "STIFFER. The baseline flattens out and follows broad "
                         "structure less, so humps survive it.",
             "lower_it": "More flexible. The baseline bends into broad features "
                         "and can absorb a hump entirely.",
             "note": "Stiffening this does not reliably recover a hump the "
                     "fitter has already absorbed, and a very stiff baseline "
                     "breaks traces that were being read correctly. The default "
                     "was chosen against a 20-trace validation set for that "
                     "reason. If a hump is being eaten here, draw it by hand."},
            {"key": "baseline_method", "label": "Baseline method",
             "term": "baseline",
             "default": "arpls", "type": "choice", "choices": ["arpls", "snip"],
             "raise_it": "", "lower_it": "",
             "note": "arPLS is asymmetric least squares - it fits a smooth "
                     "baseline that sits under the peaks. SNIP is a rank filter. "
                     "Changing this changes every number downstream, so it is a "
                     "whole-campaign decision, never a per-run one."},
        ],
    },
    {
        "key": "detection",
        "label": "Peak detection",
        "why": "What counts as a peak, beyond the signal-to-noise gate: how "
               "close two peaks may sit before they are one, how narrow a peak "
               "may be, and how hard the second-derivative search looks for "
               "shoulders riding on a neighbour's flank.",
        "controls": [
            {"key": "min_width_min", "label": "Minimum peak width (min)",
             "default": 0.05, "step": 0.01,
             "raise_it": "FEWER peaks - narrow spikes are rejected as noise.",
             "lower_it": "MORE peaks, including sharp detector artefacts.",
             "note": "The narrowest thing allowed to be a peak. On a fast "
                     "acquisition a real peak can be only a few points wide, so "
                     "this and the sampling rate interact."},
            {"key": "merge_min_sep_min", "label": "Minimum separation (min)",
             "default": 0.0667, "step": 0.01,
             "raise_it": "FEWER peaks - close pairs merge into one.",
             "lower_it": "MORE peaks - close pairs stay separate.",
             "note": "Two picks closer together than this become one. It acts on "
                     "the CLEAN count, which is what the objective counts."},
            {"key": "d2_snr", "label": "Shoulder search gate",
             "default": 3.0, "step": 0.5,
             "raise_it": "Fewer shoulders found.",
             "lower_it": "More shoulders found, including curvature that is noise.",
             "note": "The second-derivative search that finds partial separation. "
                     "Shoulders are RECORDED but not counted by the objective, so "
                     "this changes the diagnostics rather than the score - unless "
                     "a pick moves between the shoulder and clean categories."},
            {"key": "hybrid_shoulder_snr", "label": "Shoulder acceptance gate",
             "default": 5.0, "step": 0.5,
             "raise_it": "Stricter - fewer shoulder candidates are kept.",
             "lower_it": "More permissive.",
             "note": "How strong a shoulder candidate must be before the hybrid "
                     "detector keeps it alongside the prominence picks."},
        ],
    },
]


def advanced_controls() -> list[dict[str, Any]]:
    """Flat list of every advanced control, for validation."""
    return [c for g in ADVANCED_GROUPS for c in g["controls"]]


def known_settings() -> list[str]:
    """Every picker field name, so the API can refuse a typo loudly."""
    return sorted(_find_hplc_picker().DEFAULT_CONFIG.__dataclass_fields__)


def picker_defaults() -> dict[str, Any]:
    """Every picker field and the value it holds when nobody has set it.

    The campaign's Config sheet stores only the settings the campaign states
    explicitly, so a setting absent from it is not UNSET - it is sitting at the
    picker's own default. Anything reporting a change needs this, or a first
    edit to an advanced setting reads as `None -> 2.0` when the truth is
    `4.5 -> 2.0`.
    """
    cfg = _find_hplc_picker().DEFAULT_CONFIG
    return {f: getattr(cfg, f) for f in cfg.__dataclass_fields__}


@dataclass
class PickResult:
    """Everything the review screen and the record need from one analysis."""
    name: str
    n_clean_peaks: int
    n_shoulder: int
    n_on_hump: int
    hump_time_fraction: float
    humps: list[tuple[float, float]]
    crf: float
    features: dict[str, Any]
    config_used: dict[str, Any]
    deviates: bool
    manual_humps: bool
    picker_version: str
    crf_version: str = CRF_VERSION
    result: Any = field(default=None, repr=False)   # the raw PickerResult


def resolve_config(campaign: Optional[dict] = None,
                   overrides: Optional[dict] = None,
                   is_reference: bool = False) -> tuple[Any, dict, bool]:
    """Build the PickerConfig for one analysis.

    Returns (cfg, effective_settings, deviates_from_campaign).
    Raises if a per-run override is attempted on a reference run.
    """
    hp = _find_hplc_picker()
    base = dict(CAMPAIGN_DEFAULTS)
    base.update(campaign or {})
    ov = dict(overrides or {})
    if ov and is_reference:
        raise ValueError(
            "Refusing a per-run picker override on a REFERENCE run. The "
            "reference series is a differential measurement of the instrument; "
            "changing how one of its runs is processed makes a picking artefact "
            "look like drift. Re-score the whole reference series instead.")
    eff = dict(base)
    eff.update(ov)
    # `deviates` means THE PICKER SETTINGS WERE TUNED for this run, and a
    # hand-drawn hump is not that. They are different acts with different
    # justifications - a tune moves a threshold, a drawn region annotates one
    # chromatogram - and collapsing them into one flag makes the record unable
    # to say which happened. The drawn region has its own column.
    deviates = bool(set(ov) - {"hump_override"})

    cfg = hp.DEFAULT_CONFIG
    known = {f for f in cfg.__dataclass_fields__}
    unknown = sorted(set(eff) - known)
    if unknown:
        raise ValueError(
            f"unknown picker setting(s): {unknown}. Every valid name is listed "
            f"on the Setup tab under the campaign's picker settings, and on "
            f"the Method page. Check the spelling against one of those.")
    cfg = cfg.replace(**eff)
    return cfg, eff, deviates


def analyse(path: str, campaign: Optional[dict] = None,
            overrides: Optional[dict] = None,
            manual_humps: Optional[Sequence[Sequence[float]]] = None,
            is_reference: bool = False) -> PickResult:
    """One trace -> the score and everything the record keeps.

    `manual_humps` is the analyst drawing the region by hand. It REPLACES
    automatic detection for this trace, which is the right semantics: it
    annotates one chromatogram rather than moving a threshold, so it does not
    imply anything about any other run.
    """
    hp = _find_hplc_picker()
    ov = dict(overrides or {})
    if manual_humps:
        spans = [(float(a), float(b)) for a, b in manual_humps]
        for a, b in spans:
            if not (b > a):
                raise ValueError(
                    f"the drawn region {a}-{b} does not run forwards in time. "
                    f"Drag across the trace from left to right, or use the "
                    f"region's x to remove it and draw it again.")
        ov["hump_override"] = spans
    cfg, eff, deviates = resolve_config(campaign, ov, is_reference)

    res = hp.analyse(path, cfg)
    ft = dict(res.features)
    assert_objective_is_clean(hp.TRACE_HEALTH_KEYS)
    missing = [c for c in CRF_COLUMNS if c not in ft]
    if missing:
        raise RuntimeError(f"picker did not produce {missing}")

    return PickResult(
        name=res.name,
        n_clean_peaks=int(ft["n_clean_peaks"]),
        n_shoulder=int(ft["n_shoulder_fronting"]),
        n_on_hump=int(ft["n_on_hump_peaks"]),
        hump_time_fraction=float(ft["hump_time_fraction"]),
        humps=[(float(h["t_start"]), float(h["t_end"])) for h in res.humps],
        crf=float(compute_crf(ft)),
        features=ft,
        config_used={k: v for k, v in eff.items()},
        deviates=deviates,
        manual_humps=bool(manual_humps),
        picker_version=getattr(hp, "__version__", "unknown"),
        result=res)


def record_row(pick: PickResult) -> dict[str, Any]:
    """The wide row a recorded run keeps: every feature, plus provenance.

    Every chromatographic feature and the whole trace-health block, because one
    analyse() pass already produced them and a trace that is later moved or
    re-exported cannot be re-measured.
    """
    hp = _find_hplc_picker()
    row = {k: pick.features[k] for k in hp.RECORD_KEYS if k in pick.features}
    row.update({
        "CRF": round(pick.crf, 4),
        "crf_version": pick.crf_version,
        "picker_version": pick.picker_version,
        "picker_config": repr(sorted(pick.config_used.items())),
        "picker_deviates": bool(pick.deviates),
        "manual_humps": (";".join(f"{a}-{b}" for a, b in pick.humps)
                         if pick.manual_humps else ""),
    })
    return row
