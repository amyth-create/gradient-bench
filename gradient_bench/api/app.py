"""app.py - the HTTP layer. Thin on purpose.

Every rule lives in `loop.py` and `core/`; these endpoints translate JSON and
nothing else. If a decision is being made here, it is in the wrong place.
"""
from __future__ import annotations

import os
import tempfile
import traceback
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from .. import APP_VERSION, VERSION_TAG, jobs
from .. import loop as L
from .. import rescore as RS
from .. import export as EXP
from .. import setup_doc as SETUP
from ..core import budget as B
from ..core import drift as D
from ..core import health as H
from ..core import optimiser as O
from ..core import uncertainty as UNC
from ..store import history as HIST
from ..core import picker as P
from ..core import space as S
from ..store import campaign as C
from ..store import sheet as SH
from ..store import migrate as MIG
from ..store import state as ST

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _json_safe(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _torch_available() -> bool:
    try:
        import torch                                          # noqa: F401
        import botorch                                        # noqa: F401
        return True
    except Exception:                                         # noqa: BLE001
        return False


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["OPEN"] = {"campaign": None}

    def current() -> C.Campaign:
        c = app.config["OPEN"]["campaign"]
        if c is None:
            raise RuntimeError(
                "No campaign is open. Go to the Campaign tab and open one, or "
                "create a new one - every other screen reads from the open "
                "campaign's folder.")
        return c

    def ok(payload: Any = None, **extra):
        body = {"ok": True}
        if payload is not None:
            body.update(payload if isinstance(payload, dict) else {"result": payload})
        body.update(extra)
        return jsonify(_json_safe(body))

    @app.errorhandler(Exception)
    def _err(exc):
        # HTTP errors keep their own status. Without this a missing favicon
        # comes back as a 500, which sends anyone reading the log hunting for
        # a server fault that is not there.
        from werkzeug.exceptions import HTTPException
        if isinstance(exc, HTTPException):
            return jsonify({"ok": False, "error": exc.description,
                            "kind": exc.name}), exc.code
        code = 400 if isinstance(exc, (ValueError, RuntimeError)) else 500
        app.logger.error(traceback.format_exc())
        return jsonify({"ok": False, "error": str(exc),
                        "kind": type(exc).__name__}), code

    # ── the page ────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        if os.path.exists(os.path.join(STATIC, "index.html")):
            return send_from_directory(STATIC, "index.html")
        return ("<h1>Gradient Bench</h1><p>The frontend has not been built. "
                "Run <code>npm run build</code> in <code>frontend/</code>.</p>")

    # The mark is one SVG served for both names: browsers that ask for the
    # legacy .ico get the same file, and an unbuilt frontend still 204s
    # rather than 500ing on a missing path.
    @app.route("/favicon.ico")
    @app.route("/favicon.svg")
    def favicon():
        if os.path.exists(os.path.join(STATIC, "favicon.svg")):
            return send_from_directory(STATIC, "favicon.svg",
                                       mimetype="image/svg+xml")
        return ("", 204)

    @app.route("/assets/<path:fname>")
    def assets(fname):
        return send_from_directory(os.path.join(STATIC, "assets"), fname)

    @app.get("/api/version")
    def version():
        return ok({"version": APP_VERSION, "tag": VERSION_TAG,
                   "libraries": O.library_versions(),
                   "torch": _torch_available()})

    @app.get("/api/optimiser/options")
    def optimiser_options():
        """Every surrogate and acquisition the app can run, described for a
        chemist, with the defaults marked. Read by the new-campaign screen;
        the choice it produces is written once and locked."""
        return ok({"surrogates": O.SURROGATE_OPTIONS,
                   "acquisitions": O.ACQUISITION_OPTIONS,
                   "default_kernel": O.DEFAULT_KERNEL,
                   "default_acquisition": O.DEFAULT_ACQUISITION,
                   "acq_param_defaults": O.ACQ_PARAM_DEFAULTS,
                   "acq_num_restarts": 10, "acq_raw_samples": 512,
                   "reference_presets": {
                       k: v for k, v in D.REFERENCE_PRESETS.items()
                       if v.get("u")},
                   "space": {"names": S.P4_NAMES,
                             "lower": [float(x) for x in S.P4_LOWER],
                             "upper": [float(x) for x in S.P4_UPPER]},
                   "budget_default": B.DEFAULT_RUN_BUDGET,
                   "torch": _torch_available()})

    # ── campaigns ───────────────────────────────────────────────────────
    @app.get("/api/campaigns")
    def list_campaigns():
        show_archived = request.args.get("archived") == "1"
        cards = C.recent()
        n_arch = sum(1 for c in cards if c.get("archived"))
        if not show_archived:
            cards = [c for c in cards if not c.get("archived")]
        return ok({"campaigns": cards, "n_archived": n_arch,
                   "open": (current().summary()
                            if app.config["OPEN"]["campaign"] else None)})

    @app.get("/api/browse")
    def browse():
        """Server-side directory listing - the app and the files are on the
        same machine, and a browser cannot open a native folder picker."""
        path = request.args.get("path") or os.path.expanduser("~")
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(path):
            raise ValueError(
                f"{path} is not a directory - it may have been moved or "
                f"renamed. Step up with the arrow and pick the folder again.")
        entries = []
        for name in sorted(os.listdir(path)):
            if name.startswith("."):
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full):
                entries.append({"name": name, "path": full,
                                "campaign": C.is_campaign(full)})
        return ok({"path": path, "parent": os.path.dirname(path),
                   "entries": entries, "is_campaign": C.is_campaign(path)})

    @app.post("/api/campaigns/create")
    def create_campaign():
        b = request.get_json(silent=True) or {}
        # No second gate here: C.create calls S.require_feasible, which is THE
        # feasibility gate. Two gates with different wording is two places to
        # keep in step.
        u = [float(x) for x in b.get("reference_method", [])]
        c = C.create(b["parent_dir"], b.get("info", {}), reference_method=u,
                     n_replicates=int(b.get("n_replicates", 2)),
                     n_seed=int(b.get("n_seed", 5)),
                     picker_config=b.get("picker_config"),
                     run_budget=b.get("run_budget", B.DEFAULT_RUN_BUDGET),
                     reference_every=int(b.get("reference_every",
                                               D.REFERENCE_EVERY)),
                     kernel=str(b.get("kernel") or O.DEFAULT_KERNEL),
                     acquisition=str(b.get("acquisition")
                                     or O.DEFAULT_ACQUISITION),
                     acq_params=b.get("acq_params") or {},
                     acq_num_restarts=int(b.get("acq_num_restarts", 10)),
                     acq_raw_samples=int(b.get("acq_raw_samples", 512)))
        app.config["OPEN"]["campaign"] = c
        return ok({"campaign": c.summary()})

    @app.post("/api/campaigns/open")
    def open_campaign():
        b = request.get_json(silent=True) or {}
        c = C.open_campaign(b["path"])
        app.config["OPEN"]["campaign"] = c
        return ok({"campaign": c.summary()})

    @app.post("/api/campaigns/duplicate")
    def duplicate_campaign():
        """Same settings, no data. The new campaign runs its own anchor."""
        b = request.get_json(silent=True) or {}
        src = (C.open_campaign(b["path"]) if b.get("path") else current())
        c = C.duplicate_settings(src, b.get("parent_dir") or
                                 os.path.dirname(src.root), b.get("info") or {})
        app.config["OPEN"]["campaign"] = c
        return ok({"campaign": c.summary()})

    @app.post("/api/campaigns/archive")
    def archive_campaign():
        b = request.get_json(silent=True) or {}
        c = (C.open_campaign(b["path"]) if b.get("path") else current())
        C.archive(c, bool(b.get("archived", True)))
        return ok({"campaign": c.summary()})

    @app.post("/api/campaigns/reveal")
    def reveal_campaign():
        """Open the campaign folder in the desktop file manager.

        The app and the files are on the same machine - that is the whole
        premise - so this is a local convenience, not a remote action. It
        refuses anything that is not a campaign folder.
        """
        import subprocess
        import sys as _sys
        b = request.get_json(silent=True) or {}
        path = os.path.abspath(b.get("path") or current().root)
        if not C.is_campaign(path):
            raise ValueError(
                f"{path} is not a Gradient Bench campaign folder. One holds a "
                f"master_methods.xlsx and a traces/ folder - you may be one "
                f"level too high or too low. Step up and look for the folder "
                f"marked with a filled square.")
        cmd = ({"darwin": ["open"], "win32": ["explorer"]}
               .get(_sys.platform, ["xdg-open"])) + [path]
        try:
            subprocess.Popen(cmd)
        except Exception as exc:                              # noqa: BLE001
            raise RuntimeError(
                f"could not open a file manager ({type(exc).__name__}). The "
                f"folder is at {path}.")
        return ok({"path": path})

    @app.get("/api/state")
    def state():
        if app.config["OPEN"]["campaign"] is None:
            return ok({"no_campaign": True})
        c = current()
        pend = ST.load(c.state_path)
        v = c.verdict()
        sch = c.schedule()
        bud = c.budget(need=(0 if sch.next_is == "reference" else 1))
        return ok({
            "budget": bud.as_dict(),
            "campaign": c.summary(),
            "info": c.info(),
            "config": c.config(),
            "next": L.next_step(c).__dict__,
            "optimiser": c.optimiser_summary(),
            "version": APP_VERSION,
            "pending": (None if pend is None else {
                "kind": pend.kind, "u": pend.u, "phase": pend.phase,
                "reason": pend.reason, "n_replicates": pend.n_replicates,
                "uploads": [os.path.basename(p) for p in pend.uploads],
                "picked": pend.picked, "step": pend.step(),
                "posterior_mean": pend.posterior_mean,
                "posterior_sd": pend.posterior_sd,
                "acq_value": pend.acq_value,
                "seed_index": pend.seed_index,
                "manual_humps": pend.manual_humps,
                "picker_overrides": pend.picker_overrides,
                "gradient": (S.gradient_table(pend.u) if pend.u else None),
            }),
            "drift": {"verdict": v.verdict, "reasons": v.reasons,
                      "actions": v.actions(), "n": v.n,
                      "anchor": v.anchor, "last": v.last, "delta": v.delta,
                      "blocks": v.blocks_proposing},
            "schedule": {"next_is": sch.next_is, "anchor": sch.anchor,
                         "since": sch.methods_since, "overdue": sch.overdue,
                         "reason": sch.reason, "blocks": sch.blocks_proposing},
        })

    # ── the loop ────────────────────────────────────────────────────────
    @app.post("/api/suggest")
    def suggest():
        c = current()
        def _work(prog):
            prog("fitting the model")
            pend = L.suggest(c)
            return _json_safe({"pending": pend.__dict__,
                               "gradient": S.gradient_table(pend.u),
                               "step": pend.step()})
        job = jobs.submit("Working out the next method", _work)
        return ok({"job": job.as_dict()})

    @app.post("/api/upload")
    def upload():
        c = current()
        f = request.files.get("file")
        if f is None or not f.filename:
            raise ValueError(
                "No file was uploaded. Drop the instrument's exported ASCII "
                "file (.txt) onto the upload area in step 2, or click it to "
                "choose the file.")
        fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(f.filename)[1])
        os.close(fd)
        f.save(tmp)
        try:
            pend = L.add_upload(c, tmp, f.filename)
        finally:
            os.remove(tmp)
        return ok({"uploads": [os.path.basename(p) for p in pend.uploads],
                   "n_replicates": pend.n_replicates,
                   "ready": pend.ready_to_pick})

    @app.post("/api/pick")
    def pick():
        c = current()
        b = request.get_json(silent=True) or {}
        def _work(prog):
            prog("running the pipeline")
            return L.pick(c, overrides=b.get("overrides"),
                          manual_humps=b.get("manual_humps"))
        job = jobs.submit("Picking peaks", _work)
        return ok({"job": job.as_dict()})

    @app.post("/api/record")
    def record():
        c = current()
        b = request.get_json(silent=True) or {}
        def _work(prog):
            prog("writing the workbook")
            return L.record(c, notes=(b.get("notes") or ""))
        job = jobs.submit("Recording", _work)
        return ok({"job": job.as_dict()})

    @app.post("/api/discard")
    def discard():
        L.discard(current())
        return ok({"next": L.next_step(current()).__dict__})

    # ── the budget and the ending ───────────────────────────────────────
    @app.get("/api/budget")
    def budget():
        return ok({"budget": current().budget().as_dict(),
                   "default": B.DEFAULT_RUN_BUDGET})

    @app.post("/api/budget/extend")
    def extend_budget():
        b = request.get_json(silent=True) or {}
        st = current().extend_budget(b.get("n"), b.get("reason"))
        return ok({"budget": st.as_dict(),
                   "next": L.next_step(current()).__dict__})

    @app.post("/api/campaign/finish")
    def finish_campaign():
        b = request.get_json(silent=True) or {}
        st = current().finish(b.get("reason"))
        return ok({"budget": st.as_dict(),
                   "next": L.next_step(current()).__dict__})

    @app.post("/api/campaign/reopen")
    def reopen_campaign():
        b = request.get_json(silent=True) or {}
        st = current().reopen(b.get("reason"))
        return ok({"budget": st.as_dict(),
                   "next": L.next_step(current()).__dict__})

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        j = jobs.get(job_id)
        if j is None:
            raise ValueError(
                "That background task is no longer running - the app was "
                "probably restarted. Nothing was lost; start the action again.")
        return ok({"job": j.as_dict()})

    # ── results ─────────────────────────────────────────────────────────
    @app.get("/api/runs")
    def runs():
        c = current()
        df = c.data()
        cols = ["method", "replicate", "source", "run_order", "CRF",
                "n_clean_peaks", "n_shoulder_fronting", "n_on_hump_peaks",
                "hump_time_fraction", "total_time_min", "phi1", "phi2", "t1",
                "T", "picker_deviates", "manual_humps", "notes", "recorded_at"]
        cols = [x for x in cols if x in df.columns]
        rows = [] if not len(df) else df[cols].to_dict(orient="records")

        # The drift-adjusted score travels beside the raw one, ALWAYS, whatever
        # the toggle says. The toggle decides which is reported as best; it
        # never decides which numbers the analyst is allowed to see. The
        # arithmetic is Campaign's - this only attaches it to the rows.
        adjusted = c.adjusted_scores()
        inc = c.incumbent()
        summary = c.summary()
        for r in rows:
            key = str(r.get("method"))
            if key in adjusted:
                r["CRF_adjusted"] = adjusted[key]
        return ok({"rows": _json_safe(rows), "columns": cols,
                   "best": summary.get("best_crf"),
                   "best_raw": inc.raw_best, "best_raw_method": inc.raw_method,
                   "best_adjusted": inc.adj_best,
                   "best_adjusted_method": inc.adj_method,
                   "phantom": inc.phantom, "phantom_warning": inc.warning(),
                   "use_drift_adjusted": c.use_drift_adjusted()})

    @app.post("/api/export")
    def export_campaign():
        """Write the workbook and/or the PDF into the campaign folder.

        They go into the campaign's own folder deliberately: the export is part
        of the record, and a file written somewhere else is one nobody finds
        when they open the folder six months later.
        """
        b = request.get_json(silent=True) or {}
        c = current()
        want = b.get("what") or ["workbook", "pdf"]

        def _work(prog):
            made = []
            if "workbook" in want:
                prog("writing the standalone workbook")
                made.append(EXP.workbook(c))
            if "pdf" in want:
                prog("drawing the report")
                made.append(EXP.pdf(c))
            return {"files": [{"name": os.path.basename(p), "path": p,
                               "bytes": os.path.getsize(p)} for p in made],
                    "folder": c.root}

        job = jobs.submit("Exporting", _work)
        return ok({"job": job.as_dict()})

    @app.get("/api/methods")
    def methods():
        """Design runs grouped by method - the unit the optimiser works in."""
        c = current()
        return ok({"methods": c.methods(), "summary": c.summary(),
                   "use_drift_adjusted": c.use_drift_adjusted()})

    @app.get("/api/trace/<string:method>")
    def stored_trace(method):
        """The chromatogram behind one recorded run, as it was measured."""
        c = current()
        res, row, exact = RS.analyse_as_recorded(c, method)
        payload = L._review_payload(res, str(row.get("trace_file") or method))
        payload.update({
            "method": str(row["method"]), "source": str(row["source"]),
            "run_order": row.get("run_order"),
            "recorded_crf": row.get("CRF"),
            "gradient": S.gradient_table(S.abs_to_4param(row)),
            "u": [float(x) for x in S.abs_to_4param(row)],
            "deviates": SH.truthy(row.get("picker_deviates")),
            "hump_drawn": bool(str(row.get("manual_humps") or "").strip()
                               not in ("", "nan")),
            "exact": exact,
            "exact_note": ("" if exact else
                           "this run carries no picker configuration stamp, so "
                           "it is shown under the campaign's CURRENT settings - "
                           "the trace may not reproduce the recorded score"),
            "notes": (row.get("notes") if isinstance(row.get("notes"), str)
                      else ""),
        })
        return ok({"trace": payload})

    @app.get("/api/reference")
    def reference_series():
        c = current()
        refs = c.references()
        v = c.verdict()
        lim = c.drift_limits()
        return ok({"series": [{"run_order": r.run_order, "crf": r.crf,
                               "name": r.name} for r in refs],
                   "verdict": v.verdict, "anchor": v.anchor,
                   "watch": lim["watch"], "halt": lim["halt"],
                   "slope": v.slope, "reasons": v.reasons})

    # ── campaign metadata, method setup, migration ──────────────────────
    @app.get("/api/campaign/info")
    def campaign_info():
        c = current()
        return ok({"info": c.info(), "fields": C.CAMPAIGN_FIELDS,
                   "material": sorted(C.MATERIAL_FIELDS),
                   "history": c.metadata_history(),
                   "n_runs": c.n_design_runs(),
                   "n_injections": c.n_injections()})

    @app.post("/api/campaign/info")
    def edit_campaign_info():
        c = current()
        b = request.get_json(silent=True) or {}
        info = c.edit_info(dict(b.get("updates") or {}), b.get("reason") or "")
        return ok({"info": info, "history": c.metadata_history(),
                   "campaign": c.summary()})

    @app.get("/api/setup")
    def setup_document():
        return ok(SETUP.build(current()))

    @app.get("/api/migrate")
    def migration_report():
        path = request.args.get("path")
        root = path or (current().root if app.config["OPEN"]["campaign"] else None)
        if not root:
            raise ValueError("No campaign is open.")
        return ok({"report": MIG.inspect(root).as_dict()})

    @app.post("/api/migrate")
    def migrate_apply():
        b = request.get_json(silent=True) or {}
        root = b.get("path") or (current().root
                                 if app.config["OPEN"]["campaign"] else None)
        if not root:
            raise ValueError("No campaign is open.")
        rep = MIG.apply(root)
        return ok({"report": rep.as_dict()})

    @app.get("/api/model")
    def model_page():
        """The model diagnostics, read from the campaign's uncertainty history.

        Deliberately does NOT fit anything: every number here was computed when
        a run was recorded, and recomputing it to draw a chart would make
        opening a tab cost a GP fit. It also means this page works with torch
        absent.
        """
        c = current()
        t = HIST.trend(c.root)
        ard = HIST.ard_table(c.root)
        last = HIST.latest(c.root) or {}
        sd, groups, dof = c.sigma_crf()
        v = c.verdict()
        steps = HIST.reading_steps(c.root)
        return ok({
            "trend": t, "ard": ard, "kernel": c.kernel(),
            "optimiser": c.optimiser_summary(),
            "n_design_runs": int(len(c.design())),
            "noise": {
                "sigma_pooled_crf": last.get("sigma_pooled_crf"),
                "noise_sd_crf": last.get("noise_sd_crf"),
                "noise_floor_z": last.get("noise_floor_z"),
                # The threshold is the optimiser's to define, not the
                # frontend's - it is read here off the stored value so a
                # history row from months ago reads the same as a live fit.
                "floor_dominated": O.floor_is_dominated(last.get("noise_floor_z")),
                "floor_reading": O.floor_reading(last.get("noise_floor_z")),
                "sigma_measured": (None if not np.isfinite(sd) else float(sd)),
                "sigma_groups": groups, "sigma_dof": dof,
                "fallback": O.FALLBACK_SIGMA_CRF,
            },
            "campaign_reading": UNC.campaign_reading(
                steps, drift_falling=v.falling),
            "drift": {"verdict": v.verdict, "delta": v.delta},
            "last": last,
        })

    @app.get("/api/drift")
    def drift_page():
        """Everything the instrument page needs, in one call."""
        c = current()
        refs = c.references()
        v = c.verdict()
        inc = c.incumbent()
        readings = H.channel_reports(c.reference_rows())
        return ok({
            "series": [{"run_order": r.run_order, "crf": r.crf, "name": r.name}
                       for r in refs],
            "verdict": {"verdict": v.verdict, "reasons": v.reasons,
                        "actions": v.actions(), "n": v.n, "anchor": v.anchor,
                        "last": v.last, "delta": v.delta, "slope": v.slope,
                        "intercept": v.intercept, "span": v.span,
                        "implied_fall": v.implied_fall, "monotone": v.monotone,
                        "rho": v.rho, "blocks": v.blocks_proposing},
            "limits": c.drift_limits(),
            "schedule": {"next_is": c.schedule().next_is,
                         "since": c.schedule().methods_since,
                         "reason": c.schedule().reason},
            "incumbent": {
                "n": inc.n, "n_ref": inc.n_ref,
                "raw_best": inc.raw_best, "raw_method": inc.raw_method,
                "raw_run_order": inc.raw_run_order,
                "adj_best": inc.adj_best, "adj_method": inc.adj_method,
                "adj_run_order": inc.adj_run_order,
                "adj_of_raw_best": inc.adj_of_raw_best,
                "phantom": inc.phantom, "warning": inc.warning()},
            "channels": [r.as_dict() for r in readings],
            "channel_summary": H.summary(readings),
            "use_drift_adjusted": c.use_drift_adjusted(),
        })

    @app.post("/api/toggle")
    def set_toggle():
        b = request.get_json(silent=True) or {}
        c = current()
        c.set_toggle(str(b.get("key", "")), bool(b.get("value")))
        return ok({"config": c.config(), "campaign": c.summary()})

    @app.get("/api/picker/controls")
    def controls():
        c = current()
        return ok({"controls": P.PRIMARY_CONTROLS,
                   "groups": P.ADVANCED_GROUPS,
                   "campaign": c.picker_config(),
                   "defaults": P.CAMPAIGN_DEFAULTS,
                   "history": RS.history(c)})

    @app.post("/api/picker/rescore")
    def rescore():
        """Preview by default. `commit: true` writes, and only after a preview
        computed on the same settings has been shown - the two calls run the
        same code, so what the analyst saw is what happens."""
        c = current()
        b = request.get_json(silent=True) or {}
        cfg = dict(b.get("config") or {})
        do_commit = bool(b.get("commit"))

        def _work(prog):
            prog("re-measuring every stored trace")
            pl = RS.plan(c, cfg)
            if not do_commit:
                return _json_safe(pl.as_dict())
            prog("rewriting the workbook")
            out = RS.commit(c, cfg, reason=(b.get("reason") or ""), plan_=pl)
            out["campaign"] = c.summary()
            return _json_safe(out)

        job = jobs.submit("Re-scoring the campaign" if do_commit
                          else "Working out what would change", _work)
        return ok({"job": job.as_dict()})

    return app


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5051)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", dest="open_path", default=None,
                    help="open this campaign folder at startup")
    a = ap.parse_args()
    app = create_app()
    if a.open_path:
        app.config["OPEN"]["campaign"] = C.open_campaign(a.open_path)
    print(f"\n  Gradient Bench {VERSION_TAG} ({APP_VERSION})  ->  "
          f"http://{a.host}:{a.port}\n")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
