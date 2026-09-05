"""health.py - what the instrument is doing, channel by channel.

THE CONTROL CHART SAYS *THAT* SOMETHING MOVED.  A reference CRF falling from
10 to 6 is a fact about the instrument and a reason to stop, but it does not
say what to go and fix. The 15 trace-health descriptors are already recorded on
every run; each one implicates a different piece of hardware, and read together
they turn "the reference is failing" into "the column is broadening" or "the
mobile phase batch changed at run 12".

READ ON THE REFERENCE SERIES ONLY, AND THAT IS NOT A DETAIL.  A design run's
peak widths change because the METHOD changed - a 60-minute gradient at 25 C
produces broader peaks than an 18-minute one at 48 C, and nothing is wrong with
the instrument. Only the reference holds the method fixed, so only the
reference's movement can be attributed to hardware. `channel_reports` takes the
reference rows and there is no way to hand it design rows by accident.

THESE ARE HYPOTHESES, NOT DIAGNOSES.  Every reading here is "consistent with",
and the panels say so. Peak width and tailing both rise on a column that is
dying, but they also rise on a column that is merely dirty, and the honest
output is the number, the trend, and the short list of things that produce it.
The chemist has the instrument in front of them; this module does not.

A STEP AND A SLOPE MEAN DIFFERENT THINGS.  A smooth trend is something wearing
out - gradual loss of stationary phase, an oven settling. A step is something
that was CHANGED: a new mobile-phase batch, a new vial tray, a column swapped
between runs. Distinguishing them is most of the diagnostic value, so it is
computed explicitly rather than left to the eye.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

#: Relative movement, against the anchor, that counts as real for each channel.
#: Set from what the descriptor physically does, not from a common default:
#: a 20% wider peak is a column worth looking at, whereas detector noise is
#: naturally jumpy and needs to half or double before it means anything.
DEFAULT_TOLERANCE = 0.15

#: Fraction of the total absolute movement that a SINGLE jump must account for
#: before the series is called a step rather than a trend. Needs at least 3
#: points to be meaningful - with two, every change is trivially one jump.
STEP_SHARE = 0.70
STEP_MIN_POINTS = 3


CHANNELS: list[dict[str, Any]] = [
    {
        "key": "width",
        "label": "Peak width",
        "implicates": "the column",
        "why": "Peaks that broaden while the method is held fixed are the "
               "classic signature of a column losing efficiency - voids "
               "forming at the head, stationary phase eroding, particles "
               "fracturing. It hits the score directly: broader peaks overlap, "
               "and overlapping peaks stop being counted as clean.",
        "then": ["Wash the column by its manufacturer's protocol.",
                 "Check the guard column, if there is one.",
                 "Replace the column and RE-ANCHOR - the runs before and after "
                 "a column change belong to different blocks."],
        "tolerance": 0.20,
        "descriptors": [
            {"key": "trace_mean_peak_width_min", "label": "mean peak width",
             "unit": "min", "rising": "broadening"},
            {"key": "trace_median_peak_width_min", "label": "median peak width",
             "unit": "min", "rising": "broadening"},
        ],
    },
    {
        "key": "tailing",
        "label": "Peak tailing",
        "implicates": "active sites, or a void",
        "why": "Tailing is asymmetry: the peak drags on its trailing edge "
               "because some of the analyte is being held back by exposed "
               "silanols or metal sites, or is passing through a void. Rising "
               "tailing with steady widths points at surface chemistry rather "
               "than at efficiency.",
        "then": ["Check the mobile-phase pH and buffer strength - tailing of "
                 "basic compounds is usually a pH problem before it is a "
                 "column problem.",
                 "Consider a column wash; consider whether the sample solvent "
                 "is stronger than the starting mobile phase."],
        "tolerance": 0.15,
        "descriptors": [
            {"key": "trace_mean_tailing", "label": "mean tailing factor",
             "unit": "", "rising": "more tailing"},
        ],
    },
    {
        "key": "retention",
        "label": "Retention",
        "implicates": "mobile phase, or temperature",
        "why": "Where the peaks come out. THE SHAPE OF THE CHANGE IS THE "
               "DIAGNOSIS: a step means something was swapped between two runs "
               "- most often a fresh mobile-phase batch mixed slightly "
               "differently. A smooth drift, especially one that reverses, is "
               "usually thermal: an oven settling, or an ambient swing the "
               "oven is not fully rejecting.",
        "then": ["If it stepped: check when the mobile phase was last made up, "
                 "and remake both lines from the same batch.",
                 "If it drifted smoothly: let the oven equilibrate longer, and "
                 "check the laboratory's own temperature over the same period.",
                 "Either way the METHOD has not changed - only where it puts "
                 "the peaks."],
        "tolerance": 0.02,
        "descriptors": [
            {"key": "trace_first_peak_rt", "label": "first peak", "unit": "min",
             "rising": "later"},
            {"key": "trace_last_peak_rt", "label": "last peak", "unit": "min",
             "rising": "later"},
        ],
    },
    {
        "key": "response",
        "label": "Signal and area",
        "implicates": "injection, or sample age",
        "why": "Total area is how much analyte reached the detector. Falling "
               "area on a fixed method means less is arriving - a partly "
               "blocked needle, a leaking seal, evaporation from an uncapped "
               "vial, or a reference vial that has simply been sitting on the "
               "tray for a fortnight. Rising area with growing widths can be "
               "overload.",
        "then": ["Make up a FRESH reference vial - the cheapest test, and the "
                 "most common cause.",
                 "Check the injector for leaks and the needle for blockage.",
                 "Compare injection volume against the method."],
        "tolerance": 0.15,
        "descriptors": [
            {"key": "trace_total_area", "label": "total area", "unit": "",
             "rising": "more analyte"},
            {"key": "trace_max_signal", "label": "tallest peak", "unit": "",
             "rising": "taller"},
        ],
    },
    {
        "key": "detector",
        "label": "Baseline and noise",
        "implicates": "the detector, or the mobile phase",
        "why": "Noise and baseline offset are the detector's own condition: "
               "lamp age, a dirty or bubbled flow cell, absorbing impurities "
               "in a fresh solvent batch. This channel matters more than its "
               "size suggests, because the peak picker's threshold is a "
               "MULTIPLE of the noise estimate - when noise rises, small real "
               "peaks stop being counted and the score falls without anything "
               "chromatographic having changed.",
        "then": ["Check lamp hours and flow-cell cleanliness.",
                 "Degas, and look for bubbles.",
                 "If noise has risen sharply, the CRF fall may be a picking "
                 "artefact rather than a separation loss - look at the "
                 "chromatogram before believing the number."],
        "tolerance": 0.50,
        "descriptors": [
            {"key": "trace_noise_sd", "label": "noise (sd)", "unit": "",
             "rising": "noisier"},
            {"key": "trace_baseline_offset", "label": "baseline offset",
             "unit": "", "rising": "higher"},
            {"key": "trace_signal_span", "label": "signal span", "unit": "",
             "rising": "wider"},
        ],
    },
    {
        "key": "acquisition",
        "label": "Acquisition",
        "implicates": "the data system, not the chemistry",
        "why": "Run length and sampling rate. THIS ONE IS NOT DRIFT AND IS NOT "
               "GRADED ON A TOLERANCE: if the sampling rate or the run length "
               "changed, the traces are not comparable measurements of the "
               "same thing, and every other channel on this page is reading "
               "across a discontinuity. A slower sampling rate alone lowers "
               "the peak count, because narrow peaks stop having enough points "
               "to be found.",
        "then": ["Find out what changed in the acquisition method and put it "
                 "back.",
                 "Treat runs on either side of the change as SEPARATE BLOCKS. "
                 "Re-anchor; do not correct across it."],
        "tolerance": 0.0,          # exact: any change is a change
        "descriptors": [
            {"key": "trace_pts_per_min", "label": "sampling rate",
             "unit": "points/min", "rising": "faster"},
            {"key": "trace_tmax_min", "label": "run length", "unit": "min",
             "rising": "longer"},
            {"key": "trace_n_points", "label": "points in the trace", "unit": "",
             "rising": "more"},
        ],
    },
]


@dataclass
class DescriptorReading:
    key: str
    label: str
    unit: str
    values: list[float] = field(default_factory=list)
    run_order: list[float] = field(default_factory=list)
    anchor: float = float("nan")
    last: float = float("nan")
    rel_change: float = float("nan")     # against the anchor
    slope: float = float("nan")          # per run
    step_at: Optional[float] = None      # run_order of the jump, if it is one
    step_size: float = float("nan")
    monotone: bool = False
    n: int = 0

    def as_dict(self) -> dict[str, Any]:
        def f(v):
            return None if v is None or not np.isfinite(v) else float(v)
        return {"key": self.key, "label": self.label, "unit": self.unit,
                "values": [f(v) for v in self.values],
                "run_order": [f(v) for v in self.run_order],
                "anchor": f(self.anchor), "last": f(self.last),
                "rel_change": f(self.rel_change), "slope": f(self.slope),
                "step_at": f(self.step_at), "step_size": f(self.step_size),
                "monotone": self.monotone, "n": self.n}


@dataclass
class ChannelReading:
    key: str
    label: str
    implicates: str
    why: str
    then: list[str]
    descriptors: list[DescriptorReading] = field(default_factory=list)
    verdict: str = "unmeasured"     # steady | drifting | step | changed | unmeasured
    headline: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label,
                "implicates": self.implicates, "why": self.why,
                "then": self.then, "verdict": self.verdict,
                "headline": self.headline,
                "descriptors": [d.as_dict() for d in self.descriptors]}


def _read_descriptor(key: str, label: str, unit: str,
                     run_order: Sequence[float],
                     values: Sequence[float]) -> DescriptorReading:
    ro = np.asarray(run_order, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(ro) & np.isfinite(v)
    ro, v = ro[ok], v[ok]
    order = np.argsort(ro)
    ro, v = ro[order], v[order]

    d = DescriptorReading(key=key, label=label, unit=unit,
                          values=[float(x) for x in v],
                          run_order=[float(x) for x in ro], n=int(v.size))
    if v.size == 0:
        return d
    d.anchor, d.last = float(v[0]), float(v[-1])
    if d.anchor != 0:
        d.rel_change = float((v[-1] - v[0]) / abs(v[0]))
    elif v[-1] != 0:
        d.rel_change = float("inf")
    else:
        d.rel_change = 0.0
    if v.size >= 2 and np.ptp(ro) > 0:
        d.slope = float(np.polyfit(ro, v, 1)[0])
    if v.size >= 2:
        diffs = np.diff(v)
        total = float(np.abs(diffs).sum())
        j = int(np.argmax(np.abs(diffs)))
        # One jump carrying most of the movement is a CHANGE; movement spread
        # evenly across the series is wear.
        if (v.size >= STEP_MIN_POINTS and total > 0
                and abs(diffs[j]) / total >= STEP_SHARE):
            d.step_at, d.step_size = float(ro[j + 1]), float(diffs[j])
        d.monotone = bool(np.all(diffs > 0) or np.all(diffs < 0))
    return d


def _pct(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x * 100:+.1f}%"


def channel_reports(rows: Sequence[dict[str, Any]]) -> list[ChannelReading]:
    """Read every channel over the REFERENCE runs.

    `rows` is one dict per reference run, each carrying `run_order` and the
    trace-health descriptors. Design runs must never be passed: their
    descriptors move because the method moved.
    """
    ro = [r.get("run_order") for r in rows]
    out: list[ChannelReading] = []
    for spec in CHANNELS:
        ch = ChannelReading(key=spec["key"], label=spec["label"],
                            implicates=spec["implicates"], why=spec["why"],
                            then=list(spec["then"]))
        for dspec in spec["descriptors"]:
            ch.descriptors.append(_read_descriptor(
                dspec["key"], dspec["label"], dspec.get("unit", ""), ro,
                [r.get(dspec["key"]) for r in rows]))

        usable = [d for d in ch.descriptors if d.n >= 2]
        if not usable:
            ch.verdict = "unmeasured"
            ch.headline = ("Needs at least two reference runs before anything "
                           "can be said." if len(rows) < 2
                           else "Not recorded on these runs.")
            out.append(ch)
            continue

        tol = float(spec["tolerance"])
        if spec["key"] == "acquisition":
            # Not a tolerance question. Either the acquisition is identical
            # across the series or the series is not one measurement.
            moved = [d for d in usable
                     if np.isfinite(d.rel_change) and d.rel_change != 0.0]
            if moved:
                ch.verdict = "changed"
                names = ", ".join(f"{d.label} {d.anchor:g} -> {d.last:g}"
                                  for d in moved)
                ch.headline = (
                    f"THE ACQUISITION CHANGED ({names}). The reference runs are "
                    f"no longer measurements of the same thing, so every other "
                    f"panel here is reading across a discontinuity.")
            else:
                ch.verdict = "steady"
                ch.headline = ("Identical across every reference run - the "
                               "traces are comparable.")
            out.append(ch)
            continue

        worst = max(usable, key=lambda d: (abs(d.rel_change)
                                           if np.isfinite(d.rel_change) else 0))
        stepped = [d for d in usable if d.step_at is not None]
        if abs(worst.rel_change) < tol:
            ch.verdict = "steady"
            ch.headline = (f"Steady: {worst.label} has moved {_pct(worst.rel_change)} "
                           f"from the anchor, inside the {_pct(tol)} that counts "
                           f"as real for this channel.")
        elif stepped:
            s = stepped[0]
            ch.verdict = "step"
            ch.headline = (
                f"STEP at run {s.step_at:g}: {s.label} jumped "
                f"{s.step_size:+.4g} {s.unit}".strip()
                + f" in one run ({_pct(s.rel_change)} from the anchor overall). "
                  f"A step is something that was CHANGED between two runs, not "
                  f"something wearing out.")
        else:
            ch.verdict = "drifting"
            direction = "rising" if worst.rel_change > 0 else "falling"
            ch.headline = (
                f"Drifting: {worst.label} is {direction}, {_pct(worst.rel_change)} "
                f"from the anchor over {worst.n} reference runs"
                + (f", monotonically" if worst.monotone else "")
                + f" ({worst.slope:+.4g} {worst.unit} per run).".replace("  ", " "))
        out.append(ch)
    return out


def summary(readings: Sequence[ChannelReading]) -> dict[str, Any]:
    """One line for the top of the page, and the channels worth looking at."""
    flagged = [r for r in readings if r.verdict in ("drifting", "step", "changed")]
    if any(r.verdict == "changed" for r in readings):
        line = ("The acquisition changed part-way through the reference series. "
                "Resolve that before reading anything else here.")
    elif not flagged:
        line = ("Every channel is steady. If the reference CRF has moved, it is "
                "not from any of the six causes these descriptors can see - "
                "look at the chromatogram itself.")
    else:
        line = ("Flagged: " + ", ".join(f"{r.label} ({r.verdict}, "
                                        f"{r.implicates})" for r in flagged) + ".")
    return {"flagged": [r.key for r in flagged], "line": line,
            "n_flagged": len(flagged)}
