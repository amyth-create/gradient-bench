"""campaign.py - a campaign is a folder the user chose the location of.

    <wherever they chose>/<slug>/
        master_methods.xlsx      Data | Campaign | Config - the source of truth
        traces/                  every uploaded ASCII file, kept
        backups/                 timestamped workbook copies
        reference_grid.npy       the fixed evaluation grid, cached
        uncertainty_history.csv  one row per recorded run
        state.json               the in-progress loop, for resume
        .gradient_bench          marker: schema version

Self-contained and portable: copy the folder to a USB stick and it opens on
another machine with its full history intact.

CAMPAIGN IDENTITY IS THE FOLDER, NOT A TAG.  The historic sheets tag every row
`bo_4param` - the same string a new campaign would use - so pointing the old
notebook at one silently read twenty historical runs as the live campaign. Here
`source` is a label (design vs reference) and the folder is the identity, which
kills that class of bug outright.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import sheet as SH
from ..core import budget as B
from ..core import drift as D
from ..core import optimiser as O
from ..core import picker as P
from ..core import space as S
from ..core.crf import CRF_COLUMNS, CRF_VERSION
from .. import APP_VERSION

MARKER = ".gradient_bench"
WORKBOOK = "master_methods.xlsx"
REGISTRY = os.path.join(os.path.expanduser("~"), ".gradient_bench", "recent.json")

CAMPAIGN_FIELDS = [
    ("analyst", "Analyst"), ("name", "Campaign name"), ("sample", "Sample"),
    ("column", "Column"), ("mobile_phase", "Mobile phase"),
    ("flow_rate", "Flow rate (mL/min)"), ("detector", "Detector / wavelength"),
    ("instrument", "Instrument / serial"), ("created", "Date started"),
    ("notes", "Notes"),
]

#: Lifecycle lives on the Campaign sheet, not Config.  Config records the
#: SCIENTIFIC settings - what the model and the picker were told to do - and a
#: reader comparing two campaigns needs that block to mean the same thing in
#: both. How long a campaign was allowed to run is a fact about the plan, not
#: about the method, and it changes during the campaign's life while the
#: scientific configuration is meant not to.
#: Metadata stays EDITABLE after runs are recorded, and every change is kept.
#: A typo in a column part number should cost ten seconds, not a campaign - but
#: `column` and `sample` are what make the runs comparable to each other, so a
#: change to one of those is either a correction of the record or an admission
#: that the campaign is no longer a single experiment. The log cannot tell the
#: difference; it makes sure a reader can.
EDIT_LOG_FIELD = "metadata_history"

#: Changing one of these is not a typo fix - it changes what the numbers are
#: measurements OF. Flagged in the interface and in the log.
MATERIAL_FIELDS = {"sample", "column", "mobile_phase", "instrument",
                   "flow_rate", "detector"}

BUDGET_FIELD = "run_budget"
BUDGET_ORIGINAL_FIELD = "budget_original"
BUDGET_HISTORY_FIELD = "budget_history"
FINISHED_FIELD = "finished"
ENDED_AT_FIELD = "ended_at"
END_REASON_FIELD = "end_reason"


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def _runs_phrase(n_design: int, n_inj: int) -> str:
    """How many runs a campaign has, said the way an analyst counts them.

    Design runs are "runs". Instrument checks are injections that measure the
    instrument rather than the separation, so they are named separately rather
    than folded into the total - and never omitted, because they are still
    bench time under whatever value is being changed.
    """
    checks = max(0, n_inj - n_design)
    runs = f"{n_design} recorded run{'' if n_design == 1 else 's'}"
    if not checks:
        return runs
    chk = f"{checks} instrument check{'' if checks == 1 else 's'}"
    return chk + " and no runs yet" if not n_design else f"{runs} (plus {chk})"


def slugify(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", (name or "campaign").lower()).strip("-")
    return base or "campaign"


@dataclass
class Campaign:
    root: str

    # ── paths ───────────────────────────────────────────────────────────
    @property
    def workbook(self) -> str: return os.path.join(self.root, WORKBOOK)
    @property
    def traces(self) -> str: return os.path.join(self.root, "traces")
    @property
    def state_path(self) -> str: return os.path.join(self.root, "state.json")
    @property
    def grid_path(self) -> str: return os.path.join(self.root, "reference_grid.npy")
    @property
    def history_path(self) -> str: return os.path.join(self.root, "uncertainty_history.csv")
    @property
    def slug(self) -> str: return os.path.basename(os.path.abspath(self.root))

    # ── reading ─────────────────────────────────────────────────────────
    def data(self) -> pd.DataFrame: return SH.read_data(self.workbook)
    def info(self) -> dict[str, Any]: return SH.read_campaign(self.workbook)
    def config(self) -> dict[str, Any]: return SH.read_config(self.workbook)

    def picker_config(self) -> dict[str, Any]:
        cfg = self.config()
        raw = cfg.get("picker_config")
        try:
            out = json.loads(raw) if isinstance(raw, str) else dict(P.CAMPAIGN_DEFAULTS)
        except Exception:
            out = dict(P.CAMPAIGN_DEFAULTS)
        return out

    def reference_method(self) -> Optional[np.ndarray]:
        raw = self.config().get("reference_method")
        if raw in (None, "", float("nan")):
            return None
        try:
            return np.asarray(json.loads(raw), dtype=float)
        except Exception:
            return None

    def n_replicates(self) -> int:
        try:
            return max(1, int(float(self.config().get("n_replicates", 2))))
        except Exception:
            return 2

    def kernel(self) -> str:
        """The covariance kernel this campaign is fitted with.

        Recorded per campaign and never changed underneath one: a campaign
        fitted under Matern and refitted under RBF is not the same campaign,
        for the same reason the BoTorch version is pinned. Folders written
        before this setting existed get the value they were actually fitted
        with, which was BoTorch's own default.
        """
        k = str(self.config().get("kernel", "") or "").strip().lower()
        return k if k in O.KERNELS else "default"

    def n_seed(self) -> int:
        try:
            return max(1, int(float(self.config().get("n_seed", 5))))
        except Exception:
            return 5

    def acquisition(self) -> str:
        """The acquisition function this campaign proposes with.

        Chosen once, at creation, and locked - the same reasoning as the
        kernel. Folders written by earlier builds carry a descriptive string here
        ("qLogNoisyExpectedImprovement (prune_baseline=True)") rather than a
        key; they were all proposed with qLogNEI, so that is what they read as.
        """
        a = str(self.config().get("acquisition", "") or "").strip().lower()
        if a in O.ACQUISITIONS:
            return a
        return O.DEFAULT_ACQUISITION

    def acq_params(self) -> dict[str, float]:
        raw = self.config().get("acq_params")
        try:
            given = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
        except Exception:
            given = {}
        try:
            return O.resolve_acq_params(self.acquisition(), given)
        except Exception:
            return O.resolve_acq_params(self.acquisition(), None)

    def acq_settings(self) -> dict[str, int]:
        return {"num_restarts": int(self._cfgnum("acq_num_restarts", 10)),
                "raw_samples": int(self._cfgnum("acq_raw_samples", 512))}

    def optimiser_summary(self) -> dict[str, Any]:
        """The model choices, for the status bar and the cards."""
        k, a = self.kernel(), self.acquisition()
        try:
            ko = O.surrogate_option(k)
        except Exception:
            ko = {"label": k}
        ao = O.acquisition_option(a)
        return {"kernel": k, "kernel_label": ko.get("label", k),
                "acquisition": a, "acquisition_label": ao["label"],
                "acquisition_short": ao["short"],
                "acq_params": self.acq_params(),
                "n_seed": self.n_seed(), "n_replicates": self.n_replicates(),
                "description": O.describe_acquisition(a, self.acq_params())}

    # ── lifecycle: the run budget and the ending ────────────────────────
    def n_injections(self) -> int:
        """Every recorded run - design and reference together.

        One row is one injection, so the row count IS the number of runs the
        instrument performed, with no need to reason about replicate grouping.
        This is BENCH TIME, and since 2026-08-30 it is no longer what the
        budget spends - the budget counts distinct methods (HANDOFF section 24).
        It is still reported everywhere the budget is, because the budget is
        the plan and this is what the plan costs.

        Named for what it counts, because the old name was `n_runs` and
        `summary()` also published an `n_runs` meaning the OTHER thing. The
        status bar read 4 while the metadata banner read 5 on the same
        campaign, and both were calling something `n_runs`. See HANDOFF §23.
        """
        return int(len(self.data()))

    def n_design_runs(self) -> int:
        """Runs that measure the SAMPLE - instrument checks excluded.

        This is what "runs" means to the analyst, and it is what the status
        bar, the campaign cards and the metadata banner all show. An
        instrument check is an injection and it spends budget, but it is a
        measurement of the instrument, not of the separation.
        """
        return int(len(self.design()))

    def run_budget(self) -> int:
        """The current ceiling. A folder written before budgets existed has no
        field, and gets the default rather than an error - an old campaign is
        not a broken one."""
        try:
            return B.validate_budget(self.info().get(BUDGET_FIELD,
                                                     B.DEFAULT_RUN_BUDGET))
        except Exception:
            return B.DEFAULT_RUN_BUDGET

    def budget_original(self) -> int:
        try:
            return B.validate_budget(self.info().get(BUDGET_ORIGINAL_FIELD,
                                                     self.run_budget()))
        except Exception:
            return self.run_budget()

    def budget_history(self) -> list[dict[str, Any]]:
        raw = self.info().get(BUDGET_HISTORY_FIELD)
        try:
            out = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
        except Exception:
            out = []
        return [e for e in out if isinstance(e, dict)]

    def is_finished(self) -> bool:
        return _truthy(self.info().get(FINISHED_FIELD, "false"))

    def budget(self, need: Optional[int] = None) -> B.BudgetStatus:
        """Where the campaign stands against its budget.

        `need` is how many METHODS the next thing on the instrument costs:
        1 for a design method whatever its replicate count, 0 for an instrument
        check. The caller passes it because the caller already knows what is
        next; left out, it assumes a design method, which is the costly case.
        """
        info = self.info()
        if need is None:
            need = 1
        end_reason = info.get(END_REASON_FIELD, "")
        ended_at = info.get(ENDED_AT_FIELD, "")
        # METHODS, not injections. `n_methods()` counts distinct design methods,
        # so replicates cost nothing extra and instrument checks cost nothing at
        # all. `injections` rides along for reporting and never gates anything.
        return B.BudgetStatus(
            used=self.n_methods(), limit=self.run_budget(),
            original=self.budget_original(), n_replicates=self.n_replicates(),
            need=int(need), injections=self.n_injections(),
            finished=self.is_finished(),
            end_reason=("" if not isinstance(end_reason, str) else end_reason),
            ended_at=("" if not isinstance(ended_at, str) else ended_at),
            history=self.budget_history())

    # ── metadata, and the record of it changing ─────────────────────────
    def metadata_history(self) -> list[dict[str, Any]]:
        raw = self.info().get(EDIT_LOG_FIELD)
        try:
            out = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
        except Exception:
            out = []
        return [e for e in out if isinstance(e, dict)]

    def edit_info(self, updates: dict[str, Any], reason: str = "") -> dict[str, Any]:
        """Change campaign metadata, keeping what it said before.

        Editable at any time - a wrong part number should cost ten seconds.
        But `column` and `sample` are what make a campaign's runs comparable
        with each other, so every change records the old value, when it moved,
        how many runs were already recorded against the old value, and why.
        A reader can then see whether run 3 and run 30 were measuring the same
        thing; the app cannot decide that for them.
        """
        info = self.info()
        editable = {k for k, _ in CAMPAIGN_FIELDS}
        bad = sorted(set(updates) - editable)
        if bad:
            raise ValueError(
                f"not editable campaign metadata: {bad}. The budget, the "
                f"schema version and the campaign's history are changed "
                f"through their own operations, not by editing a field.")

        # Design runs, not injections: an instrument check is a measurement of
        # the instrument, so it does not tally towards "runs recorded against
        # this value". The injection count goes into the log beside it, so the
        # record still knows how much bench time sat under the old value.
        n_runs = self.n_design_runs()
        n_inj = self.n_injections()
        changes = []
        for k, v in updates.items():
            old = info.get(k, "")
            old = "" if old is None or (isinstance(old, float) and old != old) else old
            if str(old) == str(v):
                continue
            changes.append({"field": k, "from": str(old), "to": str(v),
                            "material": k in MATERIAL_FIELDS})
        if not changes:
            return info

        material = [c["field"] for c in changes if c["material"]]
        # The GATE is on injections, not design runs. An instrument check does
        # not tally towards the count an analyst reads, but it was still run on
        # this column and this sample, and it is the drift baseline for the
        # whole campaign - so changing a material field after one still needs a
        # reason. Displayed count and protected scope are different questions.
        if material and n_inj and not str(reason or "").strip():
            raise ValueError(
                f"Changing {', '.join(material)} on a campaign with "
                f"{_runs_phrase(n_runs, n_inj)} needs a reason. These fields "
                f"say what the runs are measurements OF: correcting how one is "
                f"written down is routine, but changing which column or which "
                f"sample was on the instrument means the earlier runs and the "
                f"later ones are not the same experiment. Say which of the two "
                f"this is.")

        hist = self.metadata_history()
        hist.append({
            "at": _dt.datetime.now().isoformat(timespec="seconds"),
            "runs_recorded": n_runs, "injections_recorded": n_inj,
            "changes": changes,
            "reason": str(reason or ""),
            "material": bool(material),
        })
        info.update({k: v for k, v in updates.items()})
        info[EDIT_LOG_FIELD] = json.dumps(hist)
        SH.write_campaign(self.workbook, info)
        return info

    def _update_info(self, updates: dict[str, Any]) -> None:
        info = self.info()
        info.update(updates)
        SH.write_campaign(self.workbook, info)

    def extend_budget(self, n: int, reason: str) -> B.BudgetStatus:
        """Raise the ceiling by `n` runs, on the record."""
        st = self.budget()
        new_limit, ev = B.plan_extension(st, n, reason)
        self._update_info({
            BUDGET_FIELD: int(new_limit),
            BUDGET_ORIGINAL_FIELD: int(st.original),
            BUDGET_HISTORY_FIELD: json.dumps(self.budget_history() + [ev]),
        })
        return self.budget()

    def finish(self, reason: str) -> B.BudgetStatus:
        """Close the campaign. Nothing is deleted and nothing is locked: the
        workbook, the traces and the export all keep working. It stops the
        loop serving new methods, and records that a person decided to stop."""
        ev = B.plan_finish(self.budget(), reason)
        self._update_info({
            FINISHED_FIELD: "true",
            ENDED_AT_FIELD: ev["at"],
            END_REASON_FIELD: ev.get("reason", ""),
            BUDGET_HISTORY_FIELD: json.dumps(self.budget_history() + [ev]),
        })
        return self.budget()

    def reopen(self, reason: str) -> B.BudgetStatus:
        """Reopen a closed campaign. The ending stays in the history - it
        happened - but the live fields clear, because the campaign is running
        again and a stale `ended_at` in the header would read as a lie."""
        ev = B.plan_reopen(self.budget(), reason)
        self._update_info({
            FINISHED_FIELD: "false", ENDED_AT_FIELD: "", END_REASON_FIELD: "",
            BUDGET_HISTORY_FIELD: json.dumps(self.budget_history() + [ev]),
        })
        return self.budget()

    # ── derived campaign state ──────────────────────────────────────────
    def design(self) -> pd.DataFrame:
        df = self.data()
        return df[df["source"] == SH.SOURCE_DESIGN] if len(df) else df

    def references(self) -> list[D.ReferenceRun]:
        df = self.data()
        if not len(df):
            return []
        r = df[df["source"] == SH.SOURCE_REFERENCE]
        return [D.ReferenceRun(float(x["run_order"]), float(x["CRF"]),
                               str(x["method"]))
                for _, x in r.sort_values("run_order").iterrows()]

    def training_set(self) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        """(U, y, names, run_order) over DESIGN runs only.

        Reference runs are excluded structurally - they can never reach the GP.

        THE TARGETS ARE ALWAYS RAW, and `use_drift_adjusted` deliberately does
        not reach this function. Adjusting assumes the drift is REVERSIBLE and
        cannot check it; feeding adjusted scores to the model would bake that
        unverifiable assumption into every future proposal, and proposals are
        what cost bench time. The toggle changes what is REPORTED as best - a
        reading of the record the analyst can weigh - not what the model is
        fitted to. See `incumbent()`.
        """
        d = self.design()
        if not len(d):
            e = np.zeros((0, 4))
            return e, np.zeros(0), [], np.zeros(0)
        U = np.array([S.abs_to_4param(r) for _, r in d.iterrows()])
        y = pd.to_numeric(d["CRF"], errors="coerce").to_numpy(float)
        ro = pd.to_numeric(d["run_order"], errors="coerce").to_numpy(float)
        return U, y, [str(m) for m in d["method"]], ro

    def sigma_crf(self) -> tuple[float, int, int]:
        return O.pooled_within_sd(SH.replicate_groups(self.data()))

    def methods_since_last_reference(self) -> int:
        df = self.data()
        if not len(df):
            return 0
        refs = df[df["source"] == SH.SOURCE_REFERENCE]
        last = float(pd.to_numeric(refs["run_order"], errors="coerce").max()) \
            if len(refs) else -1.0
        d = df[(df["source"] == SH.SOURCE_DESIGN)
               & (pd.to_numeric(df["run_order"], errors="coerce") > last)]
        return int(d["method"].map(SH.method_base).nunique()) if len(d) else 0

    def n_methods(self) -> int:
        d = self.design()
        return int(d["method"].map(SH.method_base).nunique()) if len(d) else 0

    def schedule(self) -> D.Schedule:
        return D.reference_schedule(
            n_methods_total=self.n_methods(),
            methods_since_last_reference=self.methods_since_last_reference(),
            n_ref=len(self.references()),
            every=self.drift_limits()["every"])

    def reference_rows(self) -> list[dict[str, Any]]:
        """Reference runs, in instrument order, as plain dicts.

        The trace-health channels must be read on the REFERENCE series only -
        a design run's peaks broaden because the METHOD changed. That gate
        lives here, so no caller can hand design rows to `health` by accident.
        """
        df = self.data()
        if not len(df):
            return []
        r = df[df["source"] == SH.SOURCE_REFERENCE].sort_values("run_order")
        return r.to_dict(orient="records")

    def methods(self) -> list[dict[str, Any]]:
        """Design runs grouped by METHOD, which is the unit the optimiser works
        in - the sheet stores one row per RUN because that is what the
        instrument did, but a two-replicate method is one decision and one
        score, and a table that lists it twice invites reading the pair as two
        results that happen to agree.
        """
        d = self.design()
        if not len(d):
            return []
        adj = self.adjusted_scores()
        out = []
        for base, g in d.groupby(d["method"].map(SH.method_base), sort=False):
            y = pd.to_numeric(g["CRF"], errors="coerce").to_numpy(float)
            ro = pd.to_numeric(g["run_order"], errors="coerce").to_numpy(float)
            u = S.abs_to_4param(g.iloc[0])
            n_scored = int(np.isfinite(y).sum())
            out.append({
                "method": str(base),
                # runs that produced a SCORE, not rows: a blank CRF is not a
                # measurement, and counting it makes the sd below meaningless
                "n": n_scored, "n_rows": int(len(g)),
                "runs": [str(m) for m in g["method"]],
                "run_order": (float(np.nanmin(ro)) if np.isfinite(ro).any()
                              else None),
                "crf_mean": float(np.nanmean(y)) if y.size else None,
                # sd over the replicates of ONE method: the only direct read on
                # noise this campaign has. Undefined for a single run, and that
                # is reported as unmeasured rather than as zero.
                "crf_sd": (float(np.nanstd(y, ddof=1)) if n_scored > 1 else None),
                "crf_min": float(np.nanmin(y)) if y.size else None,
                "crf_max": float(np.nanmax(y)) if y.size else None,
                "tied": bool(n_scored > 1 and np.ptp(y[np.isfinite(y)]) == 0),
                "crf_adjusted": (float(np.nanmean([adj[m] for m in g["method"]
                                                   if m in adj]))
                                 if any(m in adj for m in g["method"]) else None),
                "start_phi": float(u[0]), "end_phi": float(u[1]),
                "duration_min": float(u[2]), "T": float(u[3]),
                "n_clean_peaks": float(pd.to_numeric(
                    g["n_clean_peaks"], errors="coerce").mean()),
                "hump_time_fraction": float(pd.to_numeric(
                    g["hump_time_fraction"], errors="coerce").mean()),
                "deviates": bool(any(SH.truthy(x) for x in g["picker_deviates"])),
                "manual_humps": bool(any(SH.has_text(x)
                                         for x in g["manual_humps"])),
                "notes": " ".join(sorted({str(n) for n in g["notes"]
                                          if isinstance(n, str) and n.strip()})),
            })
        return out

    def adjusted_scores(self) -> dict[str, float]:
        """Drift-adjusted CRF per design run, keyed by method name."""
        d = self.design()
        if not len(d):
            return {}
        refs = self.references()
        y = pd.to_numeric(d["CRF"], errors="coerce").to_numpy(float)
        ro = pd.to_numeric(d["run_order"], errors="coerce").to_numpy(float)
        adj = D.drift_adjust(y, ro, refs)
        return {str(m): float(v) for m, v in zip(d["method"], adj)}

    def _cfgnum(self, key: str, default: float) -> float:
        try:
            return float(self.config().get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def drift_limits(self) -> dict[str, float]:
        """The limits THIS campaign is judged against.

        Read from Config, not from the module constants. They were being
        written at creation, printed on the Method page and drawn on the
        control chart while `drift_verdict` was still using the defaults - so a
        campaign with a Config halt limit of 5.0 got a chart banded at 5.0 and
        a HALT that fired at 3.0. The chart and the gate must be the same
        numbers or neither can be trusted.
        """
        return {
            "watch": self._cfgnum("drift_watch_delta", D.WATCH_DELTA),
            "halt": self._cfgnum("drift_halt_delta", D.HALT_DELTA),
            "monotone_k": int(self._cfgnum("drift_monotone_k", D.MONOTONE_K)),
            "every": int(self._cfgnum("reference_every", D.REFERENCE_EVERY)),
        }

    def verdict(self) -> D.DriftVerdict:
        lim = self.drift_limits()
        return D.drift_verdict(self.references(), watch_delta=lim["watch"],
                               halt_delta=lim["halt"], k=lim["monotone_k"])

    def use_drift_adjusted(self) -> bool:
        return _truthy(self.config().get("use_drift_adjusted", "false"))

    def incumbent(self) -> D.Incumbent:
        """The best method so far, raw and drift-adjusted, reported together.

        `best_so_far` chooses nothing - it returns both and flags when they
        name DIFFERENT methods. That flag is the phantom incumbent, and it is
        the failure that matters most in this whole app: an early run carrying
        a drift bonus that nothing later can beat, so the campaign concludes it
        has converged and adopts the second-best chemistry.
        """
        U, y, names, ro = self.training_set()
        return D.best_so_far(y, ro, names, self.references())

    #: Settings the interface may flip directly. Everything else on the Config
    #: sheet is scientific configuration, changed through an operation that
    #: knows what else has to move with it - the picker through a re-score,
    #: the budget through an extension.
    TOGGLES = {"use_drift_adjusted"}

    def set_toggle(self, key: str, value: bool) -> dict[str, Any]:
        if key not in self.TOGGLES:
            raise ValueError(
                f"{key} is not a toggle. Picker settings change through a "
                f"re-score, and the run budget through an extension, because "
                f"both have to bring something else with them.")
        cfg = self.config()
        cfg[key] = "true" if value else "false"
        SH.write_config(self.workbook, cfg)
        return cfg

    def summary(self) -> dict[str, Any]:
        d = self.design()
        y = (pd.to_numeric(d["CRF"], errors="coerce").to_numpy(float)
             if len(d) else np.zeros(0))
        info = self.info()
        sd, groups, dof = self.sigma_crf()
        sch = self.schedule()
        inc = self.incumbent()
        bud = self.budget(need=(0 if sch.next_is == "reference" else 1))
        return {
            "slug": self.slug, "root": self.root,
            "name": info.get("name", self.slug), "analyst": info.get("analyst", ""),
            "sample": info.get("sample", ""), "column": info.get("column", ""),
            "created": info.get("created", ""),
            "n_runs": int(len(d)), "n_methods": self.n_methods(),
            "n_references": len(self.references()),
            # `best_crf` FOLLOWS THE TOGGLE. It is the number shown as "best"
            # in the status bar and on the cards, so if the toggle did not
            # reach it, switching drift adjustment on would change a caption
            # and nothing else. The raw value travels beside it either way.
            "best_crf": ((float(inc.adj_best) if self.use_drift_adjusted()
                          else float(np.nanmax(y))) if y.size else None),
            "best_crf_raw": (float(np.nanmax(y)) if y.size else None),
            "best_method": (inc.adj_method if self.use_drift_adjusted()
                            else inc.raw_method),
            "drift_adjusted": self.use_drift_adjusted(),
            "phantom": bool(inc.phantom),
            "sigma_crf": (None if not np.isfinite(sd) else float(sd)),
            "sigma_dof": dof,
            "drift": self.verdict().verdict,
            "next_is": sch.next_is,
            # the budget counts every instrument run, so it is deliberately
            # NOT `n_runs` above, which counts design runs only
            "runs_used": bud.used, "run_budget": bud.limit,
            "runs_left": bud.remaining, "methods_left": bud.methods_left,
            "budget_state": bud.state, "finished": bud.finished,
            "archived": SH.truthy(info.get("archived", "false")),
            "schema_version": str(info.get("schema_version", "")),
            "optimiser": self.optimiser_summary(),
        }

    def grid(self) -> np.ndarray:
        from ..core import uncertainty as UNC
        if os.path.exists(self.grid_path):
            g = np.load(self.grid_path)
            if g.shape[1] == 4 and all(S.check4(u)[0] for u in g):
                return g
        g = UNC.make_reference_grid()
        np.save(self.grid_path, g)
        return g


# ── create / open / list ────────────────────────────────────────────────────
def create(parent_dir: str, info: dict[str, Any], *, reference_method,
           n_replicates: int = 2, n_seed: int = 5,
           picker_config: Optional[dict] = None,
           reference_every: int = D.REFERENCE_EVERY,
           run_budget: int = B.DEFAULT_RUN_BUDGET,
           kernel: str = O.DEFAULT_KERNEL,
           acquisition: str = O.DEFAULT_ACQUISITION,
           acq_params: Optional[dict] = None,
           acq_num_restarts: int = 10, acq_raw_samples: int = 512) -> Campaign:
    """Create a campaign folder under `parent_dir` and write its workbook."""
    u = S.require_feasible(reference_method, "reference method")
    if kernel not in O.KERNELS:
        raise ValueError(f"kernel must be one of {O.KERNELS}, not {kernel!r}")
    if acquisition not in O.ACQUISITIONS:
        raise ValueError(f"acquisition must be one of {O.ACQUISITIONS}, "
                         f"not {acquisition!r}")
    acq_params = O.resolve_acq_params(acquisition, acq_params)
    acq_num_restarts = max(1, int(acq_num_restarts))
    acq_raw_samples = max(16, int(acq_raw_samples))
    # The budget is in METHODS and the anchor is free, so one is enough to be a
    # campaign; `validate_budget` already refuses less. The old guard here asked
    # whether the budget could pay for the anchor plus one replicated method,
    # which was the right question while the budget was in injections.
    run_budget = B.validate_budget(run_budget)
    if S.is_isocratic(u):
        raise ValueError("an isocratic reference has little structure to lose; "
                         "choose a gradient the instrument can degrade")
    name = (info or {}).get("name") or "Untitled campaign"
    root = os.path.join(parent_dir, slugify(name))
    n = 2
    while os.path.exists(root):
        root = os.path.join(parent_dir, f"{slugify(name)}-{n}")
        n += 1
    os.makedirs(os.path.join(root, "traces"), exist_ok=True)

    created = (info or {}).get("created") or _dt.date.today().isoformat()
    camp = {k: (info or {}).get(k, "") for k, _ in CAMPAIGN_FIELDS}
    camp["created"] = created
    camp["schema_version"] = SH.SCHEMA_VERSION
    camp[BUDGET_FIELD] = int(run_budget)
    camp[BUDGET_ORIGINAL_FIELD] = int(run_budget)
    camp[FINISHED_FIELD] = "false"
    camp[ENDED_AT_FIELD] = ""
    camp[END_REASON_FIELD] = ""
    camp[BUDGET_HISTORY_FIELD] = json.dumps(
        [B.event(B.EVENT_CREATED, used=0, to=int(run_budget),
                 reason=f"campaign created with a {run_budget}-method budget")])

    pcfg = dict(P.CAMPAIGN_DEFAULTS)
    pcfg.update(picker_config or {})
    cfg = {
        "surrogate": "BoTorch SingleTaskGP",
        "kernel": kernel,
        "kernel_note":
            "Built by BoTorch's own get_covar_module_with_dim_scaled_prior, so "
            "the hyperparameter priors are BoTorch's and not ours: "
            "LogNormal(sqrt(2)+ln(d)/2, sqrt(3)) on the lengthscales, "
            "GreaterThan(0.025), no outputscale. Matern-5/2 is twice "
            "differentiable where an RBF is infinitely smooth, which is the "
            "weaker assumption for a chromatographic response in general - not "
            "a statement about any particular sample. Chosen at creation and "
            "locked: " + O.surrogate_option(kernel)["what"],
        "outcome_transform": "Standardize(m=1)",
        "acquisition": acquisition,
        "acquisition_note": O.describe_acquisition(acquisition, acq_params),
        "acq_params": json.dumps(acq_params, sort_keys=True),
        "acq_num_restarts": int(acq_num_restarts),
        "acq_raw_samples": int(acq_raw_samples),
        "app_version": APP_VERSION,
        "parameters": json.dumps(
            {n_: [float(lo), float(hi)] for n_, lo, hi
             in zip(S.P4_NAMES, S.P4_LOWER, S.P4_UPPER)}),
        "constraint": f"end_phi >= start_phi + {S.MIN_SPAN}  (normalised: "
                      f"{S.CON_COEF[0]:+.2f}*end_n {S.CON_COEF[1]:+.2f}*start_n "
                      f">= {S.CON_RHS:+.2f})",
        "crf_formula": "n_clean_peaks * (1 - hump_time_fraction) ** 2",
        "crf_version": CRF_VERSION,
        "crf_columns": json.dumps(CRF_COLUMNS),
        "n_replicates": int(n_replicates),
        "n_seed": int(n_seed),
        "reference_method": json.dumps([float(x) for x in u]),
        "reference_every": int(reference_every),
        "drift_watch_delta": D.WATCH_DELTA,
        "drift_halt_delta": D.HALT_DELTA,
        "drift_monotone_k": D.MONOTONE_K,
        "model_run_order": "false",
        "use_drift_adjusted": "false",
        "picker_config": json.dumps(pcfg, sort_keys=True),
        "picker_version": P._find_hplc_picker().__version__,
        "schema_version": SH.SCHEMA_VERSION,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    cfg.update({f"lib_{k}": v for k, v in O.library_versions().items()})

    c = Campaign(root)
    SH.create(c.workbook, camp, cfg)
    with open(os.path.join(root, MARKER), "w") as f:
        f.write(SH.SCHEMA_VERSION + "\n")
    remember(root)
    return c


def duplicate_settings(src: Campaign, parent_dir: str,
                       info: dict[str, Any]) -> Campaign:
    """A new campaign carrying this one's SETTINGS and none of its data.

    The common case after a campaign ends: same instrument, same column, a new
    sample, and no wish to re-derive twenty-nine configuration decisions. What
    travels is the configuration - reference method, replicate count, seed
    count, cadence, picker settings, run budget. What does NOT travel is every
    measurement, the run order, the reference series and the drift history,
    because those belong to the instrument on the days it made them.

    The new campaign starts with its own anchor, deliberately: an anchor is the
    zero later references are differences from, and inheriting one from another
    campaign would measure this instrument against another week's machine.
    """
    cfg = src.config()
    u = src.reference_method()
    if u is None:
        raise ValueError(
            "the source campaign has no reference method recorded, so there "
            "is nothing to carry over")
    merged = {k: (info or {}).get(k, src.info().get(k, ""))
              for k, _ in CAMPAIGN_FIELDS}
    merged["name"] = (info or {}).get("name") or f"{src.info().get('name', src.slug)} (copy)"
    merged["created"] = _dt.date.today().isoformat()
    merged["notes"] = ((info or {}).get("notes")
                       or f"settings copied from {src.slug}")
    return create(
        parent_dir, merged, reference_method=u,
        n_replicates=src.n_replicates(), n_seed=src.n_seed(),
        picker_config=src.picker_config(),
        reference_every=int(float(cfg.get("reference_every", D.REFERENCE_EVERY))),
        run_budget=src.budget_original(),
        kernel=src.kernel(), acquisition=src.acquisition(),
        acq_params=src.acq_params(),
        acq_num_restarts=src.acq_settings()["num_restarts"],
        acq_raw_samples=src.acq_settings()["raw_samples"])


def archive(c: Campaign, archived: bool = True) -> dict[str, Any]:
    """Hide a campaign from the recent list. NOTHING IS MOVED OR DELETED.

    Archiving is a view preference, not an operation on the record: the folder
    stays exactly where the analyst put it, and opening it by path still works.
    Moving folders on someone's behalf is how records get lost.
    """
    info = c.info()
    info["archived"] = "true" if archived else "false"
    SH.write_campaign(c.workbook, info)
    return info


def is_archived(c: Campaign) -> bool:
    return _truthy(c.info().get("archived", "false"))


def is_campaign(path: str) -> bool:
    return (os.path.isdir(path)
            and os.path.exists(os.path.join(path, WORKBOOK))
            and os.path.exists(os.path.join(path, MARKER)))


def open_campaign(path: str) -> Campaign:
    if not is_campaign(path):
        raise ValueError(f"{path} is not a Gradient Bench campaign folder")
    remember(path)
    return Campaign(os.path.abspath(path))


# ── the recent-campaigns registry ───────────────────────────────────────────
def _load_registry() -> list[str]:
    try:
        with open(REGISTRY) as f:
            return [p for p in json.load(f).get("recent", []) if isinstance(p, str)]
    except Exception:
        return []


def remember(path: str) -> None:
    path = os.path.abspath(path)
    recent = [p for p in _load_registry() if p != path]
    recent.insert(0, path)
    os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
    tmp = REGISTRY + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"recent": recent[:40]}, f, indent=2)
    os.replace(tmp, REGISTRY)


def recent() -> list[dict[str, Any]]:
    """Campaign cards for the project manager, freshest first."""
    out = []
    for p in _load_registry():
        if not is_campaign(p):
            continue
        try:
            out.append(open_campaign(p).summary())
        except Exception as exc:
            out.append({"slug": os.path.basename(p), "root": p,
                        "error": f"{type(exc).__name__}: {exc}"})
    return out
