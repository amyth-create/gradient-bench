"""migrate.py - bringing a campaign folder written by an older build forward.

THE PRINCIPLE: A CAMPAIGN FOLDER IS A RECORD, NOT A CACHE.  It holds
measurements the instrument made, and those cannot be regenerated. So migration
only ever ADDS - a missing column, a missing default, a marker file. It never
rewrites a measured value, never drops a column it does not recognise, and
never deletes anything. If a folder cannot be brought forward safely, it is
reported and left exactly as it is.

WHY THIS IS SEPARATE FROM READING.  `read_data` already tolerates a missing
column by filling it with NaN, so an old folder OPENS without any of this. That
tolerance is what makes the app usable; it is not a migration, because nothing
is written back and the gap reappears on the next read. This module makes the
change durable, once, with a backup taken first - and, just as importantly,
tells the analyst what was missing rather than papering over it.

WHAT CANNOT BE BACKFILLED, AND MUST NOT BE GUESSED.  `trace_file` on a row
recorded before that column existed is genuinely unknown: the naming convention
makes it recoverable in practice, but a convention is not a record. The
migration fills it only where a file of exactly the expected name is sitting in
the folder, marks it as inferred in the report, and leaves it blank otherwise.
A guess written into a provenance column is worse than a gap, because a gap is
visibly a gap.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import sheet as SH
from ..core import budget as B
from ..core import picker as P

#: Bumped when a change here is required to read a folder correctly.
CURRENT_SCHEMA = SH.SCHEMA_VERSION

MIGRATION_FIELD = "migrations"


@dataclass
class Finding:
    kind: str            # column | campaign_field | config_field | marker | inferred
    name: str
    detail: str
    rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name,
                "detail": self.detail, "rows": self.rows}


@dataclass
class MigrationReport:
    root: str
    from_version: str = ""
    to_version: str = CURRENT_SCHEMA
    findings: list[Finding] = field(default_factory=list)
    applied: bool = False
    backup: str = ""

    @property
    def needed(self) -> bool:
        return bool(self.findings)

    def summary(self) -> str:
        if not self.findings:
            return (f"This folder is already at schema {self.to_version}. "
                    f"Nothing to do.")
        by = {}
        for f in self.findings:
            by.setdefault(f.kind, []).append(f.name)
        bits = [f"{len(v)} {k.replace('_', ' ')}(s): {', '.join(v[:6])}"
                for k, v in by.items()]
        head = (f"Written by an older build (schema "
                f"{self.from_version or 'unrecorded'}). ")
        return head + "; ".join(bits) + "."

    def as_dict(self) -> dict[str, Any]:
        return {"root": self.root, "from_version": self.from_version,
                "to_version": self.to_version, "needed": self.needed,
                "applied": self.applied, "backup": self.backup,
                "findings": [f.as_dict() for f in self.findings],
                "summary": self.summary()}


def _marker_version(root: str) -> str:
    p = os.path.join(root, ".gradient_bench")
    try:
        with open(p) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def inspect(root: str) -> MigrationReport:
    """What an older folder is missing. Writes nothing."""
    from .campaign import (Campaign, BUDGET_FIELD, BUDGET_ORIGINAL_FIELD,
                           BUDGET_HISTORY_FIELD, FINISHED_FIELD)
    c = Campaign(os.path.abspath(root))
    rep = MigrationReport(root=c.root, from_version=_marker_version(c.root))

    frames = SH.read_all(c.workbook)
    df = frames.get(SH.DATA_SHEET, pd.DataFrame())
    for col in SH.data_columns():
        if col not in df.columns:
            n = 0
            if col == "trace_file" and len(df):
                n = sum(1 for _, r in df.iterrows()
                        if glob.glob(os.path.join(c.traces, f"{r['method']}.*")))
            rep.findings.append(Finding(
                "column", col,
                "missing from the Data sheet; added empty"
                + (f", {n} value(s) recoverable from the traces folder"
                   if n else ""), rows=n))

    camp = SH.read_campaign(c.workbook)
    for fld, why in (
            (BUDGET_FIELD, f"no run budget recorded; defaults to "
                           f"{B.DEFAULT_RUN_BUDGET}"),
            (BUDGET_ORIGINAL_FIELD, "no original budget recorded"),
            (BUDGET_HISTORY_FIELD, "no budget history recorded"),
            (FINISHED_FIELD, "no open/closed state recorded; treated as open")):
        if fld not in camp:
            rep.findings.append(Finding("campaign_field", fld, why))

    cfg = SH.read_config(c.workbook)
    if "picker_config" not in cfg:
        rep.findings.append(Finding(
            "config_field", "picker_config",
            "no picker configuration recorded; the campaign defaults are "
            "written in, which is what it was already being scored with"))
    if "schema_version" not in cfg:
        rep.findings.append(Finding("config_field", "schema_version",
                                    "not recorded"))
    if _marker_version(c.root) != CURRENT_SCHEMA:
        rep.findings.append(Finding(
            "marker", ".gradient_bench",
            f"marker says {_marker_version(c.root) or 'nothing'}, "
            f"current is {CURRENT_SCHEMA}"))
    return rep


def apply(root: str, *, infer_trace_files: bool = True) -> MigrationReport:
    """Bring the folder forward. Additive only, and backed up first."""
    from .campaign import (Campaign, BUDGET_FIELD, BUDGET_ORIGINAL_FIELD,
                           BUDGET_HISTORY_FIELD, FINISHED_FIELD,
                           ENDED_AT_FIELD, END_REASON_FIELD)
    rep = inspect(root)
    c = Campaign(os.path.abspath(root))
    if not rep.needed:
        return rep

    frames = SH.read_all(c.workbook)
    df = frames.get(SH.DATA_SHEET, pd.DataFrame())
    for col in SH.data_columns():
        if col not in df.columns:
            df[col] = ""  if col in ("trace_file", "notes") else pd.NA

    # Only where a file of exactly the expected name is present. A convention
    # is not a record, so anything else is left blank and reported as a gap.
    if infer_trace_files and len(df):
        df["trace_file"] = df["trace_file"].astype(object)
        for i, r in df.iterrows():
            have = str(r.get("trace_file") or "").strip()
            if have:
                continue
            hits = sorted(glob.glob(os.path.join(c.traces, f"{r['method']}.*")))
            if len(hits) == 1:
                df.at[i, "trace_file"] = os.path.basename(hits[0])
    frames[SH.DATA_SHEET] = df[SH.data_columns()]

    camp = SH.read_campaign(c.workbook)
    camp.setdefault(BUDGET_FIELD, B.DEFAULT_RUN_BUDGET)
    camp.setdefault(BUDGET_ORIGINAL_FIELD, camp[BUDGET_FIELD])
    camp.setdefault(FINISHED_FIELD, "false")
    camp.setdefault(ENDED_AT_FIELD, "")
    camp.setdefault(END_REASON_FIELD, "")
    camp.setdefault(BUDGET_HISTORY_FIELD, json.dumps([B.event(
        B.EVENT_CREATED, used=int(len(df)),
        to=int(float(camp[BUDGET_FIELD])),
        reason="budget added when the folder was migrated; the runs already "
               "recorded were made before any budget existed")]))
    hist = camp.get(MIGRATION_FIELD)
    try:
        hist = json.loads(hist) if isinstance(hist, str) and hist.strip() else []
    except Exception:
        hist = []
    hist.append({"at": _dt.datetime.now().isoformat(timespec="seconds"),
                 "from": rep.from_version or "unrecorded",
                 "to": CURRENT_SCHEMA,
                 "findings": [f.as_dict() for f in rep.findings]})
    camp[MIGRATION_FIELD] = json.dumps(hist)
    frames[SH.CAMPAIGN_SHEET] = pd.DataFrame(
        {"field": sorted(camp), "value": [camp[k] for k in sorted(camp)]})

    cfg = SH.read_config(c.workbook)
    cfg.setdefault("picker_config",
                   json.dumps(dict(P.CAMPAIGN_DEFAULTS), sort_keys=True))
    cfg["schema_version"] = CURRENT_SCHEMA
    frames[SH.CONFIG_SHEET] = pd.DataFrame(
        {"field": sorted(cfg), "value": [cfg[k] for k in sorted(cfg)]})

    SH._atomic_write(c.workbook, frames)      # takes the backup itself
    with open(os.path.join(c.root, ".gradient_bench"), "w") as fh:
        fh.write(CURRENT_SCHEMA + "\n")

    bdir = os.path.join(c.root, "backups")
    backups = sorted(glob.glob(os.path.join(bdir, "*.xlsx")))
    rep.backup = os.path.basename(backups[-1]) if backups else ""
    rep.applied = True
    return rep
