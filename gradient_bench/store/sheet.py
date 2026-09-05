"""sheet.py - the workbook. Three sheets, one source of truth.

    Data      one row per instrument RUN (not per method - see `replicate`)
    Campaign  who, what, which instrument      one row per field
    Config    the scientific configuration     one row per setting

THE SCHEMA IS COMPLETE FROM THE FIRST RELEASE.  run_order, the reference-run
columns and the provenance block are all written from day one even though the
interfaces that surface them arrive in later phases. A campaign recorded
without them cannot be repaired afterwards: you cannot reconstruct which run
happened when, and the drift monitor's whole clock depends on it.

WRITES ARE ATOMIC AND BACKED UP.  The workbook IS the campaign; losing it loses
the campaign. Every write goes to a temporary file and is moved into place with
os.replace, so a crash mid-write leaves the previous workbook intact, and a
timestamped copy is kept first.
"""
from __future__ import annotations

import datetime as _dt
import os
import shutil
import tempfile
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0.0"
DATA_SHEET = "Data"
CAMPAIGN_SHEET = "Campaign"
CONFIG_SHEET = "Config"

#: The eight absolute gradient columns, kept for compatibility with the
#: historic sheets and with any future multi-ramp parameterisation.
PARAM_COLS = ["t1", "t2", "t3", "phi1", "phi2", "phi3", "phi4", "T"]

#: Identity and bookkeeping.
ID_COLS = ["method", "replicate", "source", "run_order", "recorded_at"]

#: Provenance: what produced this number. Stamped per RUN, not per campaign,
#: because configuration legitimately changes mid-campaign and a sheet that
#: only records the campaign-level setting silently becomes a mixture of two
#: regimes with no record of the switch.
#: `trace_file` was added after the first release. It is additive and safe:
#: `read_data` fills any missing column with NaN, so an older workbook still
#: opens, and re-scoring falls back to finding the trace by the run's name. It
#: earns its place because re-scoring a stored trace needs to know WHICH file
#: produced the row, and reconstructing that from a naming convention is a
#: guess that will eventually be wrong.
PROVENANCE_COLS = ["CRF", "crf_version", "picker_version", "picker_config",
                   "picker_deviates", "manual_humps", "trace_file",
                   "model_dim", "notes"]

SOURCE_DESIGN = "design"
SOURCE_REFERENCE = "reference"


def truthy(v: Any) -> bool:
    """How a boolean column in the sheet reads. ONE definition.

    openpyxl round-trips these as True, "True", 1, 1.0, "1", "" or NaN
    depending on how they were written, and the same rule was being spelled
    three different ways in three layers.
    """
    if v is None:
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(float(v)) and float(v) != 0.0)
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def has_text(v: Any) -> bool:
    """A text column that actually holds something. NaN and "nan" do not."""
    s = str(v or "").strip()
    return bool(s) and s.lower() != "nan"


def measurement_cols() -> list[str]:
    """Every chromatographic feature plus the trace-health block."""
    from ..core.picker import _find_hplc_picker
    return list(_find_hplc_picker().RECORD_KEYS)


def data_columns() -> list[str]:
    cols = ID_COLS + PARAM_COLS + measurement_cols() + PROVENANCE_COLS
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ── atomic write ────────────────────────────────────────────────────────────
def _atomic_write(path: str, frames: dict[str, pd.DataFrame],
                  keep_backups: int = 20) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    if os.path.exists(path):
        bdir = os.path.join(d, "backups")
        os.makedirs(bdir, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        shutil.copy2(path, os.path.join(
            bdir, f"{os.path.splitext(os.path.basename(path))[0]}.{stamp}.xlsx"))
        old = sorted(os.listdir(bdir))
        for name in old[:max(0, len(old) - keep_backups)]:
            try:
                os.remove(os.path.join(bdir, name))
            except OSError:
                pass
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".xlsx")
    os.close(fd)
    try:
        with pd.ExcelWriter(tmp, engine="openpyxl") as w:
            for name, df in frames.items():
                df.to_excel(w, sheet_name=name, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_new(path: str, frames: dict[str, pd.DataFrame]) -> str:
    """Write a workbook that is NOT the campaign record.

    Deliberately separate from `_atomic_write`. That function keeps the
    campaign's crash-recovery backups, and it derives the backup directory from
    the path it is writing - so calling it for an export INTO the campaign
    folder points its retention logic at the campaign's own `backups/`. With
    `keep_backups=0` that deletes every one of them: the record survives, its
    only recovery net does not. An export must never touch the record's
    history, so it does not go through the record's writer.
    """
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".xlsx")
    os.close(fd)
    try:
        with pd.ExcelWriter(tmp, engine="openpyxl") as w:
            for name, df in frames.items():
                df.to_excel(w, sheet_name=name[:31], index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def read_all(path: str) -> dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    return {n: pd.read_excel(xl, sheet_name=n) for n in xl.sheet_names}


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=DATA_SHEET)
    for c in data_columns():
        if c not in df.columns:
            df[c] = np.nan
    if len(df):
        df["replicate"] = (pd.to_numeric(df["replicate"], errors="coerce")
                           .fillna(1).astype(int))
    return df


def _kv(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or not len(df):
        return {}
    return {str(k): v for k, v in zip(df.iloc[:, 0], df.iloc[:, 1])
            if isinstance(k, str) and k}


def read_campaign(path: str) -> dict[str, Any]:
    try:
        return _kv(pd.read_excel(path, sheet_name=CAMPAIGN_SHEET))
    except Exception:
        return {}


def read_config(path: str) -> dict[str, Any]:
    try:
        return _kv(pd.read_excel(path, sheet_name=CONFIG_SHEET))
    except Exception:
        return {}


def _frame(pairs: Sequence[tuple[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame({"field": [k for k, _ in pairs],
                         "value": [v for _, v in pairs]})


def create(path: str, campaign: dict[str, Any], config: dict[str, Any]) -> None:
    """Create an empty workbook with the full schema in place."""
    _atomic_write(path, {
        DATA_SHEET: pd.DataFrame(columns=data_columns()),
        CAMPAIGN_SHEET: _frame(sorted(campaign.items())),
        CONFIG_SHEET: _frame(sorted(config.items())),
    })


def write_campaign(path: str, campaign: dict[str, Any]) -> None:
    frames = read_all(path)
    frames[CAMPAIGN_SHEET] = _frame(sorted(campaign.items()))
    _atomic_write(path, frames)


def write_config(path: str, config: dict[str, Any]) -> None:
    frames = read_all(path)
    frames[CONFIG_SHEET] = _frame(sorted(config.items()))
    _atomic_write(path, frames)


# ── run order ───────────────────────────────────────────────────────────────
def next_run_order(df: pd.DataFrame) -> int:
    """The 1-based position the NEXT instrument run will occupy.

    Counts DESIGN and REFERENCE runs together, because the column does not care
    which is which and neither does whatever is drifting.

    Defensive against a partially stamped sheet: taking nanmax alone would hand
    the next run a position the instrument has already used - with an anchor at
    1 and ten unstamped rows it returns 2, not 12, and every later position is
    ten short, silently corrupting the drift interpolation, the cadence counter
    and the fifth GP input. Rows always outnumber or equal positions, so take
    whichever is larger.
    """
    if df is None or not len(df):
        return 1
    ro = pd.to_numeric(df.get("run_order"), errors="coerce").to_numpy(float)
    n = len(df)
    if np.isfinite(ro).any():
        return int(max(np.nanmax(ro), n)) + 1
    return n + 1


def method_base(name: str) -> str:
    """`m03r2` -> `m03`. This is what pairs replicates of the same method."""
    import re
    m = re.match(r"^(.*?)r(\d+)$", str(name))
    return m.group(1) if m else str(name)


def run_name(idx: int, rep: int) -> str:
    return f"m{int(idx):02d}r{int(rep)}"


def reference_name(idx: int) -> str:
    """`ref03`. Cannot collide with `m03r1`, and method_base leaves it alone,
    so a reference is never paired with anything as if it were a replicate."""
    return f"ref{int(idx):02d}"


def next_method_index(df: pd.DataFrame) -> int:
    if df is None or not len(df):
        return 1
    nums = (df.loc[df["source"] == SOURCE_DESIGN, "method"].astype(str)
            .str.extract(r"^m(\d+)r\d+$")[0].dropna().astype(int))
    return int(nums.max()) + 1 if len(nums) else 1


def next_reference_index(df: pd.DataFrame) -> int:
    if df is None or not len(df):
        return 1
    nums = (df["method"].astype(str).str.extract(r"^ref(\d+)$")[0]
            .dropna().astype(int))
    return int(nums.max()) + 1 if len(nums) else 1


# ── appending a run ─────────────────────────────────────────────────────────
def append_run(path: str, *, method_name: str, source: str, replicate: int,
               params_abs: dict[str, float], row: dict[str, Any],
               run_order: Optional[int] = None, notes: str = "",
               model_dim: int = 4,
               trace_file: str = "") -> dict[str, Any]:
    """Record ONE measured run. Returns the row as written."""
    frames = read_all(path)
    df = frames.get(DATA_SHEET, pd.DataFrame(columns=data_columns()))
    for c in data_columns():
        if c not in df.columns:
            df[c] = np.nan
    if len(df) and (df["method"].astype(str) == str(method_name)).any():
        raise ValueError(
            f"{method_name} already exists in this campaign, so writing it "
            f"again would create two rows claiming to be the same run. If you "
            f"meant to re-measure it, change the picker settings on the Setup "
            f"tab and re-score - that rewrites the row instead of adding one.")

    ro = int(next_run_order(df) if run_order is None else run_order)
    rec = {c: np.nan for c in data_columns()}
    rec.update({
        "method": method_name, "replicate": int(replicate), "source": source,
        "run_order": ro,
        "recorded_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "notes": notes, "model_dim": int(model_dim),
        "trace_file": str(trace_file or ""),
    })
    rec.update({c: float(params_abs[c]) for c in PARAM_COLS})
    for k, v in row.items():
        if k in rec:
            rec[k] = v

    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    df = df[data_columns()]
    frames[DATA_SHEET] = df
    _atomic_write(path, frames)
    return rec


def replicate_groups(df: pd.DataFrame) -> dict[str, list[float]]:
    """Design-run scores grouped by method base - the input to the noise model."""
    out: dict[str, list[float]] = {}
    if df is None or not len(df):
        return out
    d = df[df["source"] == SOURCE_DESIGN]
    for _, r in d.iterrows():
        out.setdefault(method_base(r["method"]), []).append(
            float(pd.to_numeric(pd.Series([r["CRF"]]), errors="coerce").iloc[0]))
    return out
