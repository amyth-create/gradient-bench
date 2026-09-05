"""rescore.py - re-measuring every stored trace under a changed configuration.

WHY THIS EXISTS.  A campaign's picker configuration is the ruler its scores are
read against. Change it halfway and the sheet becomes a mixture of two regimes:
runs 1-10 measured one way, runs 11-20 another, with a single `CRF` column
pretending they are comparable. The optimiser then fits a surface to two
different measurements of the same objective.

Every uploaded trace is kept, so there is a way out: re-measure all of them
under the new configuration and rewrite the affected columns. That makes the
column mean ONE thing again. This module is that operation, and it is the only
sanctioned route to changing a campaign's picker settings mid-campaign.

WHAT IS PRESERVED, AND WHY IT MUST BE.  A per-run deviation is not noise in the
record - it is an analyst looking at a chromatogram and correcting a failed
estimate. Wiping those on a re-score would silently undo human judgement and
push several traces back to a number a person already rejected. So each row's
deviation is recovered as a DELTA against the old campaign configuration and
re-applied on top of the new one; drawn hump regions are re-applied verbatim.
A re-score moves the baseline every run is measured from, and leaves the
corrections sitting on top of it.

REFERENCE RUNS ARE RE-SCORED TOO, AND ONLY ALL TOGETHER.  `picker.resolve_config`
refuses a per-run override on a single reference because that would dress a
picking artefact as drift. A campaign-wide re-score is the exception the
docstring there points at: the whole series moves at once, so the DIFFERENCES
between references - which is all the drift monitor reads - stay honest.

PREVIEW BEFORE COMMIT.  Nothing is written until the analyst has seen every CRF
that would move. The preview is the same computation as the commit, so what is
shown is what happens.

A MISSING TRACE STOPS THE WHOLE THING.  Re-scoring some rows and not others
produces exactly the mixture this operation exists to remove, and it would look
identical to a clean sheet afterwards. If a trace file cannot be found the
commit refuses and names the rows.
"""
from __future__ import annotations

import ast
import datetime as _dt
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from .core import picker as P
from .store import sheet as SH
from .store.campaign import Campaign

#: Config-sheet key holding the log of configuration changes.
HISTORY_FIELD = "picker_config_history"

#: Columns a re-score is allowed to touch. Identity, the gradient parameters,
#: run order and notes are facts about what the instrument did and are never
#: rewritten by re-measuring a file.
def rewritable_columns() -> list[str]:
    return SH.measurement_cols() + [
        "CRF", "crf_version", "picker_version", "picker_config",
        "picker_deviates", "manual_humps"]


@dataclass
class RowPlan:
    """What re-scoring one recorded run would do to it."""
    method: str
    source: str
    run_order: float
    trace: str = ""
    found: bool = False
    old_crf: float = float("nan")
    new_crf: float = float("nan")
    old_peaks: int = -1
    new_peaks: int = -1
    deviates: bool = False
    manual: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def delta(self) -> float:
        return self.new_crf - self.old_crf

    @property
    def moved(self) -> bool:
        return bool(np.isfinite(self.delta) and abs(self.delta) > 5e-4)

    def as_dict(self) -> dict[str, Any]:
        d = {"method": self.method, "source": self.source,
             "run_order": self.run_order, "trace": os.path.basename(self.trace),
             "found": self.found, "old_crf": self.old_crf,
             "new_crf": self.new_crf, "old_peaks": self.old_peaks,
             "new_peaks": self.new_peaks, "deviates": self.deviates,
             "manual": self.manual, "overrides": self.overrides,
             "error": self.error,
             "delta": (None if not np.isfinite(self.delta) else self.delta),
             "moved": self.moved}
        return d


@dataclass
class RescorePlan:
    old_config: dict[str, Any]
    new_config: dict[str, Any]
    rows: list[RowPlan] = field(default_factory=list)

    @property
    def changed_keys(self) -> list[str]:
        keys = sorted(set(self.old_config) | set(self.new_config))
        return [k for k in keys
                if self.old_config.get(k) != self.new_config.get(k)]

    @property
    def missing(self) -> list[RowPlan]:
        return [r for r in self.rows if not r.found]

    @property
    def failed(self) -> list[RowPlan]:
        return [r for r in self.rows if r.found and r.error]

    @property
    def moved(self) -> list[RowPlan]:
        return [r for r in self.rows if r.moved]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.failed

    def summary(self) -> str:
        if not self.changed_keys:
            return ("Nothing would change: the settings given are the ones the "
                    "campaign already uses.")
        n = len(self.rows)
        m = len(self.moved)
        parts = [f"{len(self.changed_keys)} setting(s) changed "
                 f"({', '.join(self.changed_keys)}).",
                 f"{m} of {n} recorded run(s) would change score."]
        if self.missing:
            parts.append(
                f"{len(self.missing)} trace file(s) are missing, so this CANNOT "
                f"be committed: re-scoring some rows and not others rebuilds the "
                f"very mixture the operation exists to remove, and the sheet "
                f"would look clean afterwards.")
        if self.failed:
            parts.append(f"{len(self.failed)} trace(s) failed to re-analyse.")
        return " ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "old_config": self.old_config, "new_config": self.new_config,
            "changed_keys": self.changed_keys,
            "rows": [r.as_dict() for r in self.rows],
            "n_rows": len(self.rows), "n_moved": len(self.moved),
            "n_missing": len(self.missing), "n_failed": len(self.failed),
            "ok": self.ok, "summary": self.summary(),
        }


# ── recovering what a row was measured with ─────────────────────────────────
def _int_or(v: Any, default: int) -> int:
    """int(v) if it is a real number, else `default`. NaN counts as absent."""
    n = pd.to_numeric(v, errors="coerce")
    try:
        f = float(n)
    except (TypeError, ValueError):
        return default
    return int(f) if np.isfinite(f) else default


def parse_stamped_config(raw: Any) -> dict[str, Any]:
    """Read back the `picker_config` stamp written by `picker.record_row`.

    It is stored as `repr(sorted(dict.items()))`, so a literal_eval of a list of
    pairs recovers it. Anything unreadable returns empty, which the caller
    treats as "no recoverable deviation" rather than guessing.
    """
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return {}
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return {}
    try:
        val = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {}
    if isinstance(val, dict):
        return dict(val)
    if isinstance(val, (list, tuple)):
        try:
            return {str(k): v for k, v in val}
        except (TypeError, ValueError):
            return {}
    return {}


def recover_overrides(row_config: dict[str, Any],
                      old_campaign: dict[str, Any]) -> dict[str, Any]:
    """The per-run deviation, as a delta against the OLD campaign settings.

    The row stamps the EFFECTIVE configuration, not the delta, so the delta has
    to be reconstructed by comparing against what the campaign default was at
    the time. Anything that differs was a deliberate correction by the analyst
    and is carried forward onto the new defaults.
    """
    out = {}
    for k, v in row_config.items():
        if k == "hump_override":
            continue                       # drawn regions travel separately
        if old_campaign.get(k) != v:
            out[k] = v
    return out


def parse_manual_humps(raw: Any) -> list[list[float]]:
    """`"6.74-13.85;19.25-22.39"` -> [[6.74, 13.85], [19.25, 22.39]]."""
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return []
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return []
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            a, b = part.split("-")
            out.append([float(a), float(b)])
        except ValueError:
            continue
    return out


def find_trace(c: Campaign, row: pd.Series) -> str:
    """The file that produced this row.

    Prefers the recorded `trace_file`. Rows written before that column existed
    fall back to the naming convention, which is deterministic but a guess -
    hence the preference, and hence `found` being reported per row.
    """
    name = row.get("trace_file")
    if isinstance(name, str) and name.strip():
        p = os.path.join(c.traces, name.strip())
        if os.path.exists(p):
            return p
    hits = sorted(glob.glob(os.path.join(c.traces, f"{row['method']}.*")))
    return hits[0] if hits else ""


# ── the plan ────────────────────────────────────────────────────────────────
def plan(c: Campaign, new_config: dict[str, Any]) -> RescorePlan:
    """Work out what re-scoring would do. Writes nothing."""
    want = dict(new_config or {})
    unknown = sorted(set(want) - set(P.known_settings()))
    if unknown:
        raise ValueError(f"unknown picker setting(s): {unknown}")

    # A setting the campaign never stated is not unset - it is sitting at the
    # picker's own default, and saying so is the difference between reporting
    # `4.5 -> 2.0` and the meaningless `None -> 2.0`.
    hp = P.picker_defaults()
    old = dict(P.CAMPAIGN_DEFAULTS)
    old.update(c.picker_config())
    for k in want:
        old.setdefault(k, hp.get(k))
    new = dict(old)
    new.update(want)

    df = c.data()
    out = RescorePlan(old_config=old, new_config=new)
    if not len(df):
        return out

    for _, row in df.sort_values("run_order").iterrows():
        is_ref = row["source"] == SH.SOURCE_REFERENCE
        stamped = parse_stamped_config(row.get("picker_config"))
        humps = parse_manual_humps(row.get("manual_humps"))
        ov = {} if is_ref else recover_overrides(stamped, old)
        rp = RowPlan(
            method=str(row["method"]), source=str(row["source"]),
            run_order=float(pd.to_numeric(row.get("run_order"),
                                          errors="coerce")),
            old_crf=float(pd.to_numeric(row.get("CRF"), errors="coerce")),
            # NOT `int(x or -1)`: NaN is truthy, so a blank peak count - which
            # `read_data` produces for any column an older workbook lacks -
            # would raise and take the whole preview down, and a legitimate
            # zero would silently become -1.
            old_peaks=_int_or(row.get("n_clean_peaks"), -1),
            deviates=bool(ov), manual=bool(humps and not is_ref),
            overrides=ov)
        rp.trace = find_trace(c, row)
        rp.found = bool(rp.trace)
        if not rp.found:
            rp.error = "trace file not found in the campaign's traces/ folder"
            out.rows.append(rp)
            continue
        try:
            res = P.analyse(rp.trace, campaign=new,
                            overrides=ov or None,
                            manual_humps=(humps or None) if not is_ref else None,
                            is_reference=is_ref)
            rp.new_crf = float(res.crf)
            rp.new_peaks = int(res.n_clean_peaks)
        except Exception as exc:                      # noqa: BLE001
            rp.error = f"{type(exc).__name__}: {exc}"
        out.rows.append(rp)
    return out


# ── the commit ──────────────────────────────────────────────────────────────
def commit(c: Campaign, new_config: dict[str, Any], *,
           reason: str = "", plan_: Optional[RescorePlan] = None) -> dict[str, Any]:
    """Re-measure every stored trace and rewrite the affected columns.

    The workbook is backed up first by the atomic writer, and the whole Data
    sheet is replaced in ONE write, so a crash halfway cannot leave half the
    campaign on the new ruler and half on the old.
    """
    pl = plan_ if plan_ is not None else plan(c, new_config)
    if not pl.changed_keys:
        raise ValueError(
            "These are the settings the campaign already uses - there is "
            "nothing to re-score.")
    if pl.missing:
        names = ", ".join(r.method for r in pl.missing[:8])
        raise RuntimeError(
            f"REFUSING TO RE-SCORE: {len(pl.missing)} trace file(s) are missing "
            f"({names}). Re-scoring the rest would leave the campaign measured "
            f"two different ways with nothing in the sheet to say so. Restore "
            f"the files into the campaign's traces/ folder, or accept that this "
            f"campaign's configuration cannot be changed.")
    if pl.failed:
        first = pl.failed[0]
        raise RuntimeError(
            f"REFUSING TO RE-SCORE: {len(pl.failed)} trace(s) failed to "
            f"re-analyse under these settings, starting with {first.method} "
            f"({first.error}). Fix the settings, or the trace.")

    frames = SH.read_all(c.workbook)
    df = frames[SH.DATA_SHEET].copy()
    for col in SH.data_columns():
        if col not in df.columns:
            df[col] = np.nan
    by_method = {str(m): i for i, m in enumerate(df["method"].astype(str))}
    cols = set(rewritable_columns())
    # A column that has only ever held NaN comes back as float64, and writing a
    # string into it is deprecated in pandas and will raise. These four are
    # text or flags by definition, so say so before touching them.
    for col in ("picker_config", "manual_humps", "crf_version",
                "picker_version", "picker_deviates"):
        if col in df.columns:
            df[col] = df[col].astype(object)

    for rp in pl.rows:
        i = by_method.get(rp.method)
        if i is None:
            continue
        is_ref = rp.source == SH.SOURCE_REFERENCE
        humps = parse_manual_humps(df.at[i, "manual_humps"])
        res = P.analyse(rp.trace, campaign=pl.new_config,
                        overrides=rp.overrides or None,
                        manual_humps=(humps or None) if not is_ref else None,
                        is_reference=is_ref)
        row = P.record_row(res)
        for k, v in row.items():
            if k in cols and k in df.columns:
                df.at[i, k] = v

    frames[SH.DATA_SHEET] = df[SH.data_columns()]

    cfg = c.config()
    hist = _history(cfg)
    hist.append({
        "at": _dt.datetime.now().isoformat(timespec="seconds"),
        "changed": {k: [pl.old_config.get(k), pl.new_config.get(k)]
                    for k in pl.changed_keys},
        "n_rows": len(pl.rows), "n_moved": len(pl.moved),
        "reason": str(reason or ""),
    })
    cfg["picker_config"] = json.dumps(pl.new_config, sort_keys=True)
    cfg[HISTORY_FIELD] = json.dumps(hist)
    frames[SH.CONFIG_SHEET] = pd.DataFrame(
        {"field": [k for k in sorted(cfg)], "value": [cfg[k] for k in sorted(cfg)]})

    SH._atomic_write(c.workbook, frames)
    return {"rescored": len(pl.rows), "moved": len(pl.moved),
            "changed_keys": pl.changed_keys, "history": hist,
            "summary": pl.summary(), "rows": [r.as_dict() for r in pl.rows]}


def _history(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    raw = cfg.get(HISTORY_FIELD)
    try:
        out = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
    except Exception:
        out = []
    return [e for e in out if isinstance(e, dict)]


def history(c: Campaign) -> list[dict[str, Any]]:
    return _history(c.config())


def analyse_as_recorded(c: Campaign, method: str):
    """Re-analyse a stored trace UNDER THE SETTINGS STAMPED ON ITS ROW.

    Not under the campaign's current settings. The chromatogram shown beside a
    score has to be the one that PRODUCED that score - including any per-run
    tuning and any hand-drawn hump - or the picture and the number in the same
    panel disagree, and the number is the one that is right.

    Returns (PickResult, row, exact). `exact` is False when the row carries no
    readable configuration stamp - an older or hand-edited workbook - in which
    case the campaign's CURRENT settings are used and the replayed score may
    not be the recorded one. The caller must say so rather than drawing a
    chromatogram that silently contradicts the number printed beside it.
    """
    df = c.data()
    hit = df[df["method"].astype(str) == str(method)]
    if not len(hit):
        raise ValueError(
            f"{method} is not a run in this campaign. The Results table lists "
            f"every run by name; if it was there a moment ago, reopen the "
            f"campaign to reload the sheet.")
    row = hit.iloc[0]
    trace = find_trace(c, row)
    if not trace:
        raise FileNotFoundError(
            f"the trace file for {method} is not in the campaign's traces "
            f"folder, so the chromatogram behind this score cannot be shown")

    is_ref = row["source"] == SH.SOURCE_REFERENCE
    stamped = parse_stamped_config(row.get("picker_config"))
    humps = parse_manual_humps(row.get("manual_humps"))
    exact = bool(stamped)
    # The stamp is the EFFECTIVE config, so it is passed as the campaign-level
    # settings and nothing is layered on top - that reproduces the recorded
    # measurement exactly, whatever the campaign has been changed to since.
    res = P.analyse(trace, campaign=(stamped or c.picker_config()),
                    manual_humps=(humps or None) if not is_ref else None,
                    is_reference=is_ref)
    return res, row, exact
