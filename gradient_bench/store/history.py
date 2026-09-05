"""history.py - uncertainty_history.csv, one row per recorded method.

WHAT IT IS FOR.  After each method is recorded the model is refitted on the
data before and after it, and the two posteriors are compared over the fixed
reference grid. That comparison is the only direct answer to "did this run
teach the model anything", and it is worthless unless it is kept: it cannot be
reconstructed later, because reproducing the "before" fit needs the campaign as
it was at that moment.

WHY A CSV BESIDE THE WORKBOOK AND NOT A SHEET IN IT.  The workbook is the
record of what the INSTRUMENT did; this is a record of what the MODEL did with
it. Keeping them apart means a corrupted or refitted diagnostic can never
threaten the measurements, and it means this file can be deleted and
regenerated - the run rows cannot.

THE PAGE READS THIS FILE, NOT A FRESH FIT.  Every number here was expensive to
compute; recomputing it to draw a chart would make opening a tab cost a GP fit,
and a diagnostic nobody opens is not a diagnostic. It also means the whole
diagnostics view works with torch absent - the fit happens at record time or
not at all.

APPEND-ONLY, AND TOLERANT.  A row that cannot be written must never take a
recorded run down with it, so every caller here is expected to be wrapped. A
malformed or truncated file is read as far as it parses rather than raising.
"""
from __future__ import annotations

import csv
import datetime as _dt
import os
from typing import Any, Optional

import numpy as np

FILENAME = "uncertainty_history.csv"

#: Fixed leading columns. Lengthscales are appended as `ls_<name>` because the
#: input set depends on whether run order is modelled, and a header that
#: changed shape mid-campaign would be unreadable.
BASE_COLUMNS = [
    "recorded_at", "run_order", "n_runs", "method",
    "mean_sd_before", "mean_sd_after", "d_mean_sd",
    "max_sd_before", "max_sd_after", "mean_abs_dmu",
    "sd_incumbent_before", "sd_incumbent_after",
    "sd_newpoint_before", "sd_newpoint_after",
    "noise_sd_crf", "noise_floor_z", "sigma_pooled_crf", "best_crf",
    "kernel", "model_dim", "reading",
]


def path_for(root: str) -> str:
    return os.path.join(root, FILENAME)


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def row_from_step(step, *, method: str = "", kernel: str = "",
                  model_dim: int = 4) -> dict[str, Any]:
    """One `UncertaintyStep` as a flat row."""
    row: dict[str, Any] = {
        "recorded_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "run_order": _num(step.run_order), "n_runs": int(step.n_runs),
        "method": method,
        "mean_sd_before": _num(step.mean_sd_before),
        "mean_sd_after": _num(step.mean_sd_after),
        "d_mean_sd": _num(step.d_mean_sd),
        "max_sd_before": _num(step.max_sd_before),
        "max_sd_after": _num(step.max_sd_after),
        "mean_abs_dmu": _num(step.mean_abs_dmu),
        "sd_incumbent_before": _num(step.sd_incumbent_before),
        "sd_incumbent_after": _num(step.sd_incumbent_after),
        "sd_newpoint_before": _num(step.sd_newpoint_before),
        "sd_newpoint_after": _num(step.sd_newpoint_after),
        "noise_sd_crf": _num(step.noise_sd_crf),
        "noise_floor_z": _num(step.noise_floor_z),
        "sigma_pooled_crf": _num(step.sigma_pooled_crf),
        "best_crf": _num(step.best_crf),
        "kernel": kernel, "model_dim": int(model_dim),
        "reading": step.reading(),
    }
    for name, v in (step.lengthscales or {}).items():
        row[f"ls_{name}"] = _num(v)
    return row


def append(root: str, row: dict[str, Any]) -> str:
    """Append one row, widening the header if new lengthscale columns appear.

    Widening rewrites the file. That is acceptable here and nowhere else in
    this app: the file is small, it is a derived artefact, and the alternative
    - silently dropping a column - would lose the very number somebody went
    looking for.
    """
    p = path_for(root)
    os.makedirs(root, exist_ok=True)
    existing, cols = read(root)
    has_file = os.path.exists(p) and os.path.getsize(p) > 0
    new_cols = [c for c in row if c not in cols]

    # THE HEADER ON DISK IS THE AUTHORITY, and appending under a header that
    # does not match writes values into the wrong columns - silently, and
    # unrecoverably once more rows follow. So: append only when the existing
    # header already covers this row. Any new column, or a file whose header
    # could not be read, means rewrite the whole file with a header that fits.
    if has_file and cols and not new_cols:
        with open(p, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=cols,
                           extrasaction="ignore").writerow(row)
        return p

    if not has_file:
        cols = _columns_for(row)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerow(row)
        return p

    cols = (cols + new_cols) if cols else _columns_for(row)
    tmp = p + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in existing + [row]:
            w.writerow(r)
    os.replace(tmp, p)
    return p


#: Lengthscale columns follow the MODEL INPUT order, not the alphabet, so the
#: ARD table reads in the order the four parameters are always written in.
LS_ORDER = ["start_phi", "end_phi", "duration_min", "T", "run_order"]


def _columns_for(row: dict[str, Any]) -> list[str]:
    present = [k for k in row if k.startswith("ls_")]
    ordered = [f"ls_{n}" for n in LS_ORDER if f"ls_{n}" in present]
    return BASE_COLUMNS + ordered + sorted(set(present) - set(ordered))


def read(root: str) -> tuple[list[dict[str, Any]], list[str]]:
    """(rows, columns). A missing file is an empty history, not an error."""
    p = path_for(root)
    if not os.path.exists(p):
        return [], []
    rows: list[dict[str, Any]] = []
    cols: list[str] = []
    try:
        with open(p, newline="", encoding="utf-8") as fh:
            r = csv.DictReader(fh)
            cols = list(r.fieldnames or [])
            # Read as far as it parses, and KEEP what was read. Returning
            # nothing on a bad final line would throw away every earlier fit
            # and, worse, leave `append` with no header to respect.
            for raw in r:
                out: dict[str, Any] = {}
                for k, v in raw.items():
                    if k is None:
                        continue
                    if k in ("recorded_at", "method", "kernel", "reading"):
                        out[k] = v or ""
                    else:
                        out[k] = _num(v)
                rows.append(out)
    except (OSError, csv.Error):
        pass
    return rows, cols


def lengthscale_names(cols) -> list[str]:
    return [c[3:] for c in cols if c.startswith("ls_")]


def _is_fit(row: dict[str, Any]) -> bool:
    """A row that records an actual fit, rather than a damaged line."""
    return row.get("mean_sd_after") is not None


def latest(root: str) -> Optional[dict[str, Any]]:
    """The most recent real FIT, which is not always the last line.

    A truncated write or a stray line appended by something else would
    otherwise become "the latest fit" and blank every number on the page.
    """
    rows, _ = read(root)
    for r in reversed(rows):
        if _is_fit(r):
            return r
    return None


def trend(root: str) -> dict[str, Any]:
    """The series the uncertainty chart draws, plus what it is doing.

    Mean posterior SD over the fixed grid, per recorded method. A band
    narrowing run after run IS the campaign learning, and needs no further
    explanation; a flat one is the finding that matters.
    """
    rows, cols = read(root)
    if not rows:
        return {"n": 0, "points": [], "names": [], "reading":
                "No fit has been recorded yet. The history starts once three "
                "design runs exist, which is the fewest a before/after "
                "comparison can be made from."}
    # A row without a usable mean SD carries no point - a truncated write, or
    # a line something else appended. Skipped rather than drawn as a gap.
    pts = [{"run_order": r.get("run_order"), "n_runs": r.get("n_runs"),
            "method": r.get("method", ""),
            "mean_sd": r.get("mean_sd_after"),
            "mean_sd_before": r.get("mean_sd_before"),
            "d_mean_sd": r.get("d_mean_sd"),
            "noise_sd_crf": r.get("noise_sd_crf"),
            "best_crf": r.get("best_crf")}
           for r in rows if r.get("mean_sd_after") is not None]
    if not pts:
        return {"n": 0, "points": [], "names": lengthscale_names(cols),
                "reading": "The history file holds no readable fits."}
    first = next((p["mean_sd"] for p in pts if p["mean_sd"] is not None), None)
    last = next((p["mean_sd"] for p in reversed(pts)
                 if p["mean_sd"] is not None), None)
    reading = "Not enough recorded fits to read a trend yet."
    if first is not None and last is not None and len(pts) >= 2:
        if last < first:
            reading = (f"Mean posterior SD has fallen from {first:.3f} to "
                       f"{last:.3f} CRF across {len(pts)} recorded fits. The "
                       f"campaign is learning.")
        elif last > first:
            reading = (f"Mean posterior SD has RISEN from {first:.3f} to "
                       f"{last:.3f} CRF. Added data that disagrees with its "
                       f"neighbours widens the posterior; check the noise term "
                       f"and the instrument before adding more methods.")
        else:
            reading = (f"Mean posterior SD is unchanged at {last:.3f} CRF. The "
                       f"runs are landing where the model was already confident.")
    return {"n": len(pts), "points": pts, "names": lengthscale_names(cols),
            "reading": reading}


def ard_table(root: str) -> list[dict[str, Any]]:
    """The last recorded lengthscales, in words. No fit required."""
    from ..core import optimiser as O
    rows, cols = read(root)
    # The most recent row that actually CARRIES lengthscales - a fit may have
    # been recorded without them, and a damaged trailing line has none at all.
    last = next((r for r in reversed(rows)
                 if any(r.get(f"ls_{n}") is not None
                        for n in lengthscale_names(cols))), None)
    if last is None:
        return []
    out = []
    for name in lengthscale_names(cols):
        v = last.get(f"ls_{name}")
        if v is None:
            continue
        out.append({"name": name, "value": float(v),
                    "reading": O.lengthscale_reading(name, float(v))})
    return out


def reading_steps(root: str) -> list[Any]:
    """The stored history in the shape `uncertainty.campaign_reading` expects.

    That function reads three fields per step. Handing it light stand-ins
    built from the file - rather than refitting to rebuild real objects - is
    the entire reason the history is kept, and it keeps the adapter in the
    store rather than in the API.
    """
    from types import SimpleNamespace
    rows, _ = read(root)
    out = []
    for r in rows:
        if r.get("noise_sd_crf") is None or r.get("mean_sd_after") is None:
            continue
        before = r.get("mean_sd_before")
        out.append(SimpleNamespace(
            noise_sd_crf=float(r["noise_sd_crf"]),
            mean_sd_after=float(r["mean_sd_after"]),
            mean_sd_before=(float(before) if before is not None
                            else float("nan"))))
    return out
