"""loop.py - the engine. What happens next, and what happens when it does.

Every gate the plan describes lives here, in one place, so the API is a thin
translation layer and the rules cannot drift between endpoints:

    anchor first   a new campaign cannot serve a single design method until the
                   reference has been measured - it is the zero every later
                   comparison is made against and cannot be added afterwards
    cold start     until the seeds are done, serve pre-generated spread-out
                   methods and simply record what they score
    cadence        every N proposals the next thing on the instrument is the
                   reference, ENFORCED rather than suggested
    halt           if the reference has left its limits, proposing is blocked
    budget         the campaign stops when its runs are spent or the analyst
                   closes it; both block proposing, neither blocks recording

A RUN ALREADY ON THE INSTRUMENT ALWAYS FINISHES.  Every gate below sits behind
the pending-run check, so a campaign that is closed or out of budget can still
upload, pick and record the traces it already paid for. Blocking those would
throw away bench time that has already been spent and leave the workbook
disagreeing with what the instrument actually did.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .core import drift as D
from .core import optimiser as O
from .core import picker as P
from .core import space as S
from .store import sheet as SH
from .store import state as ST
from .store.campaign import Campaign


@dataclass
class NextStep:
    action: str          # anchor | seed | proposal | reference | halted | pick | upload | record
    kind: str            # design | reference
    title: str
    detail: str
    blocked: bool = False
    seed_index: Optional[int] = None
    seed_total: Optional[int] = None


def next_step(c: Campaign) -> NextStep:
    pend = ST.load(c.state_path)
    if pend is not None and pend.u is not None:
        st = pend.step()
        return NextStep(action=st, kind=pend.kind,
                        title={"upload": "Upload the chromatogram",
                               "pick": "Check the peaks",
                               "record": "Record the result"}.get(st, st),
                        detail=pend.reason)

    sch = c.schedule()

    # The budget is checked against what is ACTUALLY next, because a reference
    # costs one run and a design method costs one per replicate. Asking "is
    # anything left" instead would serve a two-replicate method on a one-run
    # budget - an injection the chemist cannot record.
    # A method costs 1 whatever its replicate count; an instrument check
    # spends bench time and no budget, so it costs 0.
    need = 0 if sch.next_is == "reference" else 1
    bud = c.budget(need=need)
    v = c.verdict()
    if bud.finished:
        return NextStep(action="finished", kind="design", blocked=True,
                        title="This campaign is closed",
                        detail=bud.reason())
    if bud.exhausted:
        detail = bud.reason()
        if v.blocks_proposing:
            # Say it now rather than after the extension is spent: buying runs
            # for an instrument that is failing its own reference buys nothing.
            detail += (" The instrument is also failing its reference check, "
                       "so extending the budget will not release the loop "
                       "until it has been serviced and re-anchored.")
        return NextStep(
            action="budget",
            kind=("reference" if sch.next_is == "reference" else "design"),
            blocked=True,
            title="Out of runs - extend the budget or finish the campaign",
            detail=detail)

    if sch.next_is == "reference":
        return NextStep(
            action="anchor" if sch.anchor else "reference", kind="reference",
            title=("Run the instrument-check method (anchor)" if sch.anchor
                   else "Run the instrument-check method"),
            detail=sch.reason)

    if v.blocks_proposing:
        return NextStep(action="halted", kind="design", blocked=True,
                        title="Stopped - the instrument needs attention",
                        detail=" ".join(v.reasons))

    n_done = c.n_methods()
    n_seed = c.n_seed()
    if n_done < n_seed:
        return NextStep(action="seed", kind="design",
                        title=f"Run exploring method {n_done + 1} of {n_seed}",
                        detail="Cold start: spread-out methods, recorded as measured. "
                               "No model is fitted yet.",
                        seed_index=n_done + 1, seed_total=n_seed)
    U, y, _, _ = c.training_set()
    if len(y) < 2:
        return NextStep(action="seed", kind="design",
                        title="Run another exploring method",
                        detail="At least two scored runs are needed before a model "
                               "can be fitted.")
    return NextStep(action="proposal", kind="design",
                    title="Get the next method from the model",
                    detail="The model proposes where the biggest expected "
                           "improvement is.")


def suggest(c: Campaign, seed: Optional[int] = None) -> ST.PendingRun:
    """Decide and persist the next method to put on the instrument."""
    step = next_step(c)
    if step.blocked:
        raise RuntimeError(step.detail)
    pend = ST.load(c.state_path)
    if pend is not None and pend.u is not None:
        return pend

    n_rep = c.n_replicates()
    if step.kind == "reference":
        u = c.reference_method()
        pend = ST.PendingRun(kind="reference", u=[float(x) for x in u],
                             phase="anchor" if step.action == "anchor" else "reference",
                             reason=step.detail, n_replicates=1)
    elif step.action == "seed":
        idx = (step.seed_index or 1) - 1
        seeds = S.initial_sample(c.n_seed(),
                                 seed=int(c.config().get("seed_rng", 20260805)))
        pend = ST.PendingRun(kind="design", u=[float(x) for x in seeds[idx]],
                             phase="cold_start", reason=step.detail,
                             seed_index=idx + 1, n_replicates=n_rep)
    else:
        U, y, _, ro = c.training_set()
        sd, _, _ = c.sigma_crf()
        acq = c.acq_settings()
        sug = O.suggest_next(U, y, sigma_crf=(sd if np.isfinite(sd) else None),
                             seed=seed, kernel=c.kernel(),
                             acquisition=c.acquisition(),
                             acq_params=c.acq_params(),
                             num_restarts=acq["num_restarts"],
                             raw_samples=acq["raw_samples"])
        pend = ST.PendingRun(kind="design", u=[float(x) for x in sug.u],
                             phase="optimise", reason=sug.reason,
                             n_replicates=n_rep,
                             posterior_mean=sug.posterior_mean,
                             posterior_sd=sug.posterior_sd,
                             acq_value=sug.acq_value)
    S.require_feasible(pend.u, "suggested method")
    ST.save(c.state_path, pend)
    return pend


def add_upload(c: Campaign, src_path: str, filename: str) -> ST.PendingRun:
    """Take an uploaded trace into the campaign folder and remember it."""
    pend = ST.load(c.state_path)
    if pend is None or pend.u is None:
        raise RuntimeError(
            "No method has been suggested yet. Press the button in step 1 to "
            "get one - the upload has nothing to attach to until then.")
    if len(pend.uploads) >= pend.n_replicates:
        raise RuntimeError(
            f"This method already has {pend.n_replicates} trace(s). "
            "Record it, or discard the run.")
    os.makedirs(c.traces, exist_ok=True)
    idx = len(pend.uploads) + 1
    stem = _pending_name(c, pend, idx)
    dest = os.path.join(c.traces, f"{stem}{os.path.splitext(filename)[1] or '.TXT'}")
    shutil.copy(src_path, dest)
    pend.uploads.append(dest)
    pend.picked = False
    ST.save(c.state_path, pend)
    return pend


def _pending_name(c: Campaign, pend: ST.PendingRun, rep: int) -> str:
    df = c.data()
    if pend.kind == "reference":
        return SH.reference_name(SH.next_reference_index(df))
    return SH.run_name(SH.next_method_index(df), rep)


def pick(c: Campaign, overrides: Optional[dict] = None,
         manual_humps: Optional[list] = None) -> dict[str, Any]:
    """Analyse every uploaded trace and return the review payload."""
    pend = ST.load(c.state_path)
    if pend is None or not pend.uploads:
        raise RuntimeError(
            "Upload a chromatogram first - step 2. The picker needs a trace "
            "before it can find peaks in one.")
    if overrides is not None:
        pend.picker_overrides = dict(overrides)
    if manual_humps is not None:
        pend.manual_humps = [[float(a), float(b)] for a, b in manual_humps]

    results = []
    for path in pend.uploads:
        res = P.analyse(path, campaign=c.picker_config(),
                        overrides=pend.picker_overrides or None,
                        manual_humps=pend.manual_humps or None,
                        is_reference=(pend.kind == "reference"))
        results.append(res)
    pend.picked = True
    ST.save(c.state_path, pend)

    crfs = [r.crf for r in results]
    return {
        "replicates": [_review_payload(r, p) for r, p in zip(results, pend.uploads)],
        "crf_mean": float(np.mean(crfs)),
        "crf_values": crfs,
        "crf_spread": (float(np.max(crfs) - np.min(crfs)) if len(crfs) > 1 else 0.0),
        "tied": bool(len(crfs) > 1 and np.ptp(crfs) == 0),
        "deviates": bool(pend.picker_overrides),
        "manual_humps": pend.manual_humps,
        "campaign_picker_config": c.picker_config(),
        "controls": P.PRIMARY_CONTROLS,
        "groups": P.ADVANCED_GROUPS,
        "is_reference": pend.kind == "reference",
    }


def _review_payload(res: P.PickResult, path: str, max_pts: int = 2200) -> dict:
    r = res.result
    t = np.asarray(r.t)
    step = max(1, int(np.ceil(len(t) / max_pts)))
    det = r.detection_hybrid
    # Which picks the PROMINENCE pass found. The clean rule is "prominence
    # found it AND it lies outside every unresolved region", so with this flag
    # the review screen can show a provisional score the instant a region is
    # drawn, before the server confirms it - the same rule, applied locally.
    prom_idx = set(int(i) for i in np.asarray(
        getattr(r, "detection_prominence", {}).get("idx", [])).tolist())
    return {
        "name": os.path.basename(path),
        "t": [round(float(x), 4) for x in t[::step]],
        "y": [round(float(x), 2) for x in np.asarray(r.smoothed)[::step]],
        "raw": [round(float(x), 2) for x in np.asarray(r.y_raw)[::step]],
        "baseline": [round(float(x), 2) for x in np.asarray(r.baseline)[::step]],
        "peaks": [{"t": round(float(det["t"][k]), 3),
                   "h": round(float(det["height"][k]), 2),
                   "w": round(float(det["fwhm_t"][k]), 3),
                   "cat": str(r.categories[k]),
                   "prom": bool(int(det["idx"][k]) in prom_idx)}
                  for k in range(len(det["idx"]))],
        "humps": [[round(a, 3), round(b, 3)] for a, b in res.humps],
        "n_clean": res.n_clean_peaks, "n_shoulder": res.n_shoulder,
        "n_on_hump": res.n_on_hump,
        "hump_time_fraction": res.hump_time_fraction,
        "crf": round(res.crf, 4),
        "noise": round(float(r.noise), 4),
        "pts_per_min": round(float(r.pts_per_min), 1),
        "tmax": round(float(t[-1]), 2),
    }


def record(c: Campaign, notes: str = "") -> dict[str, Any]:
    """Write every replicate to the sheet, update the model, clear the loop."""
    pend = ST.load(c.state_path)
    if pend is None or pend.u is None:
        raise RuntimeError(
            "Nothing to record. Get a method in step 1, run it, upload the "
            "trace in step 2 and check the peaks in step 3 first.")
    # FEASIBILITY FIRST, ahead of the workflow guards. A missing replicate is
    # an incomplete workflow and the message tells you how to finish it; an
    # infeasible method means the pending state is CORRUPT - that method could
    # never have been proposed, so it could never have been run, so there is no
    # workflow to finish. `space.py` claims check4 is the only gate and that
    # nothing is recorded without it; until this, that held only transitively
    # because `suggest` checked before persisting, and anything reaching
    # state.json another way wrote unrunnable parameters into the workbook.
    u = S.require_feasible(pend.u, "the method being recorded")

    # then the replicate guard: of the workflow guards it is the more
    # actionable message, and it used to fire second - telling an analyst to
    # pick peaks when the real problem was a missing run.
    if len(pend.uploads) < pend.n_replicates:
        raise RuntimeError(
            f"REFUSING TO RECORD: {len(pend.uploads)} of {pend.n_replicates} "
            "traces uploaded. Both runs of a method must be recorded together - "
            "a single observation at a new method leaves the noise term "
            "unidentifiable. Re-run the missing replicate, or lower the "
            "replicate count for this campaign and accept that sigma becomes "
            "unmeasurable.")
    if not pend.picked:
        raise RuntimeError(
            "Check the peaks before recording - step 3. The score comes from "
            "that check, so it has to happen before the run is written down.")

    params = S.simple_gradient_abs(u)
    df = c.data()
    ro0 = SH.next_run_order(df)
    is_ref = pend.kind == "reference"
    idx = (SH.next_reference_index(df) if is_ref else SH.next_method_index(df))

    written = []
    for rep, path in enumerate(pend.uploads, start=1):
        res = P.analyse(path, campaign=c.picker_config(),
                        overrides=pend.picker_overrides or None,
                        manual_humps=pend.manual_humps or None,
                        is_reference=is_ref)
        name = SH.reference_name(idx) if is_ref else SH.run_name(idx, rep)
        row = P.record_row(res)
        written.append(SH.append_run(
            c.workbook, method_name=name,
            source=SH.SOURCE_REFERENCE if is_ref else SH.SOURCE_DESIGN,
            replicate=rep, params_abs=params, row=row,
            run_order=ro0 + rep - 1, notes=notes, model_dim=4,
            trace_file=os.path.basename(path)))

    ST.clear(c.state_path)
    unc = _record_uncertainty(c, written)
    crfs = [float(w["CRF"]) for w in written]
    out = {
        "recorded": [{"method": w["method"], "run_order": int(w["run_order"]),
                      "crf": float(w["CRF"])} for w in written],
        "crf_mean": float(np.mean(crfs)),
        "within_sd": (float(np.std(crfs, ddof=1)) if len(crfs) > 1 else None),
        "tied": bool(len(crfs) > 1 and np.ptp(crfs) == 0),
    }
    # EVERYTHING BELOW THIS LINE IS A CONVENIENCE, AND THE RUN IS ALREADY ON
    # DISK. `summary` and `next_step` re-open the workbook and evaluate the
    # budget, the schedule and the verdict; if one of them raised, the analyst
    # would be told the record FAILED while the rows sat in the sheet and the
    # pending state was already cleared - unretryable, and a lie.
    try:
        out["summary"] = c.summary()
        out["next"] = next_step(c).__dict__
    except Exception as exc:                                   # noqa: BLE001
        out["post_write_failed"] = f"{type(exc).__name__}: {exc}"
        out["note"] = ("the run IS recorded and the workbook is intact - only "
                       "the follow-on summary could not be computed. Reopen "
                       "the campaign to refresh it.")
    if unc is not None:
        out["uncertainty"] = unc
    if out["tied"]:
        out["tie_note"] = (
            "Both repeats returned an identical score. With an integer peak "
            "count that is a tie, not evidence of zero noise: sigma is smaller "
            "than one clean peak and this pair cannot resolve it. Replicating "
            "one method four or five times would pin it properly.")
    return out


def _record_uncertainty(c: Campaign, written: list) -> Optional[dict[str, Any]]:
    """What the run just recorded taught the model. Never raises.

    THIS RUNS AFTER THE WORKBOOK IS WRITTEN, AND ITS FAILURE IS NOT AN ERROR.
    It refits the GP twice, which needs torch, takes seconds, and can fail for
    reasons that have nothing to do with the measurement - a missing library, a
    fit that will not converge, a degenerate design. None of that is a reason
    to lose a run the instrument actually performed. The measurement is in the
    sheet before this is attempted, and anything that goes wrong here is
    reported alongside the record rather than instead of it.

    Reference runs are skipped: they never reach the model, so there is nothing
    for them to have taught it.
    """
    try:
        from .core import uncertainty as UNC
        from .store import history as H

        if all(w.get("source") == SH.SOURCE_REFERENCE for w in written):
            return None
        U, y, names, ro = c.training_set()
        if len(y) < 3:
            return {"skipped": "fewer than three design runs - a before/after "
                                "comparison needs one to compare against"}
        sd, _, _ = c.sigma_crf()
        step = UNC.uncertainty_step(
            U, y, sigma_crf=(sd if np.isfinite(sd) and sd > 0 else None),
            grid=c.grid(), run_order=ro, kernel=c.kernel())
        if step is None:
            return {"skipped": "not enough runs for a before/after comparison"}
        row = H.row_from_step(step, method=str(names[-1] if names else ""),
                              kernel=c.kernel(), model_dim=4)
        H.append(c.root, row)
        return {"reading": step.reading(),
                "mean_sd_before": step.mean_sd_before,
                "mean_sd_after": step.mean_sd_after,
                "d_mean_sd": step.d_mean_sd,
                "noise_sd_crf": step.noise_sd_crf,
                "lengthscales": step.lengthscales}
    except Exception as exc:                                   # noqa: BLE001
        # Deliberately broad. The run is already recorded; this is a diagnostic.
        return {"failed": f"{type(exc).__name__}: {exc}",
                "note": "the run is recorded and the workbook is intact - only "
                        "the uncertainty diagnostic could not be computed"}


def discard(c: Campaign) -> None:
    ST.clear(c.state_path)
