"""export.py - the campaign as something you can hand to someone else.

TWO ARTEFACTS, ONE RULE: EACH MUST STATE ITS OWN PROVENANCE.  A report that
gives a winning method and a score, without saying what the score means, which
picker measured it, whether the instrument was drifting, or how many times the
budget was extended, is a number without a chain of custody. Somebody will read
it six months from now with none of this conversation available.

THE WORKBOOK EXPORT IS A COPY, NOT A REWRITE.  It carries the Data, Campaign
and Config sheets through unchanged and ADDS derived sheets beside them. The
original is never the thing being edited: an export that could corrupt the
record it exports is worse than no export.

THE PDF IS BUILT WITH MATPLOTLIB, which is already pinned. Nothing new has to
install on a lab machine that may have no internet - the whole reason it is not
a typeset document.

WHAT IS DELIBERATELY NOT HERE.  No conclusions. The report shows the best
method, the runs behind it, the instrument's condition and the model's
uncertainty, and stops. Deciding whether a separation is good enough is the
analyst's, and a report that made that call would be read as having made it.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from .core import space as S
from .store import history as HIST
from .store import sheet as SH
from .store.campaign import Campaign

#: Matplotlib's default page, in inches. A4 rather than Letter: the instrument
#: and the analyst are not in the United States.
PAGE = (8.27, 11.69)


def _stamp() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _safe(name: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_ " else "-" for ch in name)
    return "-".join(keep.split()) or "campaign"


def default_name(c: Campaign, ext: str) -> str:
    info = c.info()
    return f"{_safe(str(info.get('name') or c.slug))}-{_dt.date.today().isoformat()}.{ext}"


# ── the standalone workbook ─────────────────────────────────────────────────
def workbook(c: Campaign, out_path: Optional[str] = None) -> str:
    """A copy of the record with the derived sheets written in beside it.

    Everything a reader needs is in one file: what ran, on what, under which
    configuration, how the budget moved, how the metadata was corrected, and
    what each run taught the model.
    """
    out_path = out_path or os.path.join(c.root, default_name(c, "xlsx"))
    if os.path.abspath(out_path) == os.path.abspath(c.workbook):
        raise ValueError(
            "refusing to export over the campaign's own workbook - the export "
            "is a copy of the record, never the record itself")
    frames = SH.read_all(c.workbook)          # Data | Campaign | Config, verbatim

    methods = c.methods()
    if methods:
        frames["Methods"] = pd.DataFrame(methods).drop(columns=["runs"],
                                                       errors="ignore")

    bud = c.budget()
    if bud.history:
        frames["Budget history"] = pd.DataFrame(bud.history)
    meta = c.metadata_history()
    if meta:
        frames["Metadata history"] = pd.DataFrame(
            [{"at": h.get("at"), "runs_recorded": h.get("runs_recorded"),
              "field": ch.get("field"), "from": ch.get("from"),
              "to": ch.get("to"), "material": ch.get("material"),
              "reason": h.get("reason")}
             for h in meta for ch in h.get("changes", [])])
    from . import rescore as RS
    pick_hist = RS.history(c)
    if pick_hist:
        frames["Picker history"] = pd.DataFrame(
            [{"at": h.get("at"), "changed": json.dumps(h.get("changed", {})),
              "n_rows": h.get("n_rows"), "n_moved": h.get("n_moved"),
              "reason": h.get("reason")} for h in pick_hist])
    rows, _ = HIST.read(c.root)
    if rows:
        frames["Uncertainty history"] = pd.DataFrame(rows)

    refs = c.references()
    if refs:
        # The verdict goes in a COLUMN, not in DataFrame.attrs - `to_excel`
        # does not write attrs, so that would have been provenance that is not
        # actually in the file.
        v = c.verdict()
        frames["Reference series"] = pd.DataFrame(
            [{"name": r.name, "run_order": r.run_order, "crf": r.crf,
              "delta_from_anchor": r.crf - refs[0].crf,
              "verdict_at_export": v.verdict} for r in refs])

    lim = c.drift_limits()
    frames["Export"] = pd.DataFrame({"field": [
        "exported_at", "exported_from", "schema_version", "verdict",
        "runs_used", "run_budget", "injections", "budget_injections",
        "best_method", "best_crf",
        "drift_adjusted_reporting", "kernel",
        "drift_watch_delta", "drift_halt_delta", "drift_monotone_k",
        "reference_every", "reference_method",
    ], "value": [
        _stamp(), c.root, SH.SCHEMA_VERSION, c.verdict().verdict,
        # `runs_used` and `run_budget` are in METHODS. The two injection
        # figures travel with them because a method budget says nothing about
        # bench time on its own, and this sheet is read without the app.
        bud.used, bud.limit, bud.injections, bud.cost["injections"],
        c.summary().get("best_method"),
        c.summary().get("best_crf"), c.use_drift_adjusted(), c.kernel(),
        lim["watch"], lim["halt"], lim["monotone_k"], lim["every"],
        json.dumps([round(float(x), 6) for x in (c.reference_method()
                                                 if c.reference_method() is not None
                                                 else [])]),
    ]})

    # Every picker field, not only the ones the campaign states - a reader
    # cannot tell an untouched setting from an absent one, which is the same
    # reasoning the Method page uses.
    from .core import picker as P
    defaults = P.picker_defaults()
    stated = c.picker_config()
    frames["Picker settings"] = pd.DataFrame(
        [{"setting": k, "campaign": stated.get(k, defaults[k]),
          "picker_default": defaults[k], "stated_by_campaign": k in stated}
         for k in sorted(defaults)])

    SH.write_new(out_path, frames)
    return out_path


# ── the PDF report ──────────────────────────────────────────────────────────
def _fig_text(pdf, title: str, blocks: list[tuple[str, list[str]]],
              footer: str = "") -> None:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=PAGE)
    fig.text(0.07, 0.955, title, fontsize=15, weight="bold", va="top")
    y = 0.915
    for heading, lines in blocks:
        if y < 0.08:
            break
        if heading:
            fig.text(0.07, y, heading.upper(), fontsize=8.5, weight="bold",
                     color="#55635B", va="top")
            y -= 0.022
        for ln in lines:
            if y < 0.06:
                fig.text(0.07, y, "... continues in the workbook export",
                         fontsize=8, color="#8B978F", va="top")
                break
            fig.text(0.07, y, ln, fontsize=8.6, va="top",
                     family="monospace" if ln.startswith("  ") else None)
            y -= 0.0185
        y -= 0.012
    if footer:
        fig.text(0.07, 0.03, footer, fontsize=7.5, color="#8B978F")
    pdf.savefig(fig)
    plt.close(fig)


def _wrap(label: str, value: Any, width: int = 92) -> list[str]:
    import textwrap
    text = f"{label}: {value}"
    return textwrap.wrap(text, width) or [text]


def _noise_lines(c: Campaign, sd, groups, dof) -> list[str]:
    """The noise floor with THIS campaign's numbers substituted in - the same
    derivation the Method page shows. Without floor_z a reader cannot tell
    whether the posterior behind the report was informative at all."""
    from .core import optimiser as O
    out = [*_wrap("Measured sigma",
                  "not measurable - no method has two recorded runs"
                  if not np.isfinite(sd) else
                  f"{sd:.4f} CRF over {groups} method(s), {dof} dof"),
           *_wrap("Fallback sigma", f"{O.FALLBACK_SIGMA_CRF} CRF, used until a "
                                    f"replicate pair lands")]
    d = c.design()
    y = (pd.to_numeric(d["CRF"], errors="coerce").to_numpy(float)
         if len(d) else np.zeros(0))
    if y.size > 1:
        nf = O.noise_floor_std(y, sd if np.isfinite(sd) and sd > 0 else None)
        out += ["  " + ln for ln in nf.explain()]
        if nf.dominated:
            out += _wrap("WARNING", "floor_z >= 1: the replicate noise is as "
                                    "large as the whole campaign's spread, so "
                                    "the posterior is nearly flat. That is the "
                                    "finding, not a fault.", 92)
    out += _wrap("A tie", "two replicates scoring identically means sigma is "
                          "smaller than one clean peak, not that it is zero")
    return out


def pdf(c: Campaign, out_path: Optional[str] = None) -> str:
    """The report. Five pages, no conclusions."""
    import matplotlib
    matplotlib.use("Agg")                      # no display on a lab PC
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_path = out_path or os.path.join(c.root, default_name(c, "pdf"))
    info, cfg = c.info(), c.config()
    summary = c.summary()
    lim = c.drift_limits()
    bud = c.budget()
    v = c.verdict()
    sd, groups, dof = c.sigma_crf()
    methods = c.methods()
    inc = c.incumbent()

    with PdfPages(out_path) as pp:
        # ── 1. identity and provenance ──────────────────────────────
        _fig_text(pp, str(info.get("name") or c.slug), [
            ("the campaign", [
                *_wrap("Analyst", info.get("analyst", "")),
                *_wrap("Sample", info.get("sample", "")),
                *_wrap("Column", info.get("column", "")),
                *_wrap("Mobile phase", info.get("mobile_phase", "")),
                *_wrap("Flow rate", info.get("flow_rate", "")),
                *_wrap("Detector", info.get("detector", "")),
                *_wrap("Instrument", info.get("instrument", "")),
                *_wrap("Started", info.get("created", "")),
                *_wrap("Folder", c.root),
            ]),
            ("what was optimised", [
                "  CRF = n_clean_peaks x (1 - hump_time_fraction) ^ 2",
                *_wrap("Objective version", cfg.get("crf_version", "")),
                *_wrap("Design space", ", ".join(
                    f"{n} [{lo:g}, {hi:g}]" for n, lo, hi
                    in zip(S.P4_NAMES, S.P4_LOWER, S.P4_UPPER))),
                *_wrap("Constraint", f"end_phi >= start_phi + {S.MIN_SPAN}"),
                *_wrap("Surrogate", f"SingleTaskGP, {c.kernel()} kernel, "
                                    f"BoTorch priors, Standardize(m=1)"),
                *_wrap("Acquisition", cfg.get("acquisition", "")),
                *_wrap("Replicates per method", c.n_replicates()),
            ]),
            ("how it was measured", [
                *_wrap("Peak picker", f"hplc_picker {cfg.get('picker_version','')}"),
                *_wrap("Picker settings stated by this campaign",
                       json.dumps(c.picker_config(), sort_keys=True)),
                *_wrap("Libraries at creation", ", ".join(
                    f"{k[4:]} {v}" for k, v in sorted(cfg.items())
                    if k.startswith("lib_"))),
            ]),
            ("the instrument check", [
                *_wrap("Reference method", ", ".join(
                    f"{n} {val:g}" for n, val in zip(
                        S.P4_NAMES, (c.reference_method()
                                     if c.reference_method() is not None
                                     else [float("nan")] * 4)))),
                *_wrap("Cadence", f"one reference every {lim['every']} proposals, "
                                  f"enforced"),
                *_wrap("Limits", f"WATCH at {lim['watch']} CRF below the anchor, "
                                 f"HALT at {lim['halt']}, or "
                                 f"{lim['monotone_k']} consecutive falls"),
                *_wrap("Verdict", f"{v.verdict} - {' '.join(v.reasons)[:200]}"),
            ]),
            ("how it ended", [
                *_wrap("Runs used", f"{bud.used} of {bud.limit} "
                                    f"({bud.n_extensions} extension(s))"),
                *_wrap("State", bud.state),
                *_wrap("Best method reported", f"{summary.get('best_method')} at "
                       f"{summary.get('best_crf')} CRF, selected on "
                       + ("DRIFT-ADJUSTED scores - the raw winner is "
                          f"{inc.raw_method} at {inc.raw_best:.3f}. Adjusting "
                          "assumes the drift is reversible and nothing can "
                          "check that."
                          if c.use_drift_adjusted() else
                          "RAW scores, which is the default. The drift-adjusted "
                          f"winner would be {inc.adj_method} at "
                          f"{inc.adj_best:.3f}.")),
            ]),
        ], footer=f"Generated {_stamp()} by Gradient Bench. "
                  f"Schema {SH.SCHEMA_VERSION}.")

        # ── 2. the best separation ──────────────────────────────────
        best = summary.get("best_method")
        fig = plt.figure(figsize=PAGE)
        fig.text(0.07, 0.955, "Best separation", fontsize=15, weight="bold",
                 va="top")
        drawn = False
        if best:
            try:
                from . import rescore as RS
                res, row, exact = RS.analyse_as_recorded(c, str(best))
                r = res.result
                ax = fig.add_axes([0.09, 0.55, 0.85, 0.34])
                ax.plot(r.t, r.smoothed, lw=0.9, color="#16201C")
                for a, b in res.humps:
                    ax.axvspan(a, b, color="#B4531F", alpha=0.16)
                det = r.detection_hybrid
                for k in range(len(det["idx"])):
                    ax.plot(det["t"][k], det["height"][k], "o", ms=3.4,
                            mfc="white", mec="#0E6E5B", mew=1.2)
                ax.set_xlabel("retention time (min)", fontsize=8.5)
                ax.set_ylabel("signal", fontsize=8.5)
                ax.tick_params(labelsize=7.5)
                ax.set_title(f"{best} - CRF {float(row.get('CRF')):.3f}, "
                             f"{res.n_clean_peaks} clean peaks",
                             fontsize=9.5)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                drawn = True
                grad = S.gradient_table(S.abs_to_4param(row))
                lines = [f"  {g['time_min']:>6.2f} min   {g['pct_b']:>6.2f} %B   "
                         f"{g['phase']}" for g in grad]
                u = S.abs_to_4param(row)
                fig.text(0.07, 0.47, "THE METHOD THAT PRODUCED IT", fontsize=8.5,
                         weight="bold", color="#55635B", va="top")
                y = 0.445
                extra = [f"  temperature   {float(u[3]):.1f} C",
                         f"  recorded at run {row.get('run_order')}",
                         f"  picker tuned for this run: "
                         f"{SH.truthy(row.get('picker_deviates'))}",
                         f"  hump region drawn by hand: "
                         f"{SH.has_text(row.get('manual_humps'))}"]
                if not exact:
                    extra += ["  NOTE: no configuration stamp on this row - it is",
                              "        shown under the campaign's CURRENT settings"]
                for ln in lines + extra:
                    fig.text(0.07, y, ln, fontsize=8.6, family="monospace",
                             va="top")
                    y -= 0.0185
                if inc.phantom:
                    fig.text(0.07, y - 0.02, "PHANTOM INCUMBENT: " +
                             (inc.warning() or "")[:400], fontsize=8,
                             color="#A32F27", va="top", wrap=True)
            except Exception as exc:                          # noqa: BLE001
                fig.text(0.07, 0.9, f"The winning chromatogram could not be "
                                    f"drawn: {type(exc).__name__}: {exc}",
                         fontsize=8.6, va="top", wrap=True)
        if not drawn and not best:
            fig.text(0.07, 0.9, "No design run has been recorded yet.",
                     fontsize=9, va="top")
        pp.savefig(fig)
        plt.close(fig)

        # ── 3. the charts ───────────────────────────────────────────
        fig = plt.figure(figsize=PAGE)
        fig.text(0.07, 0.955, "How the campaign went", fontsize=15,
                 weight="bold", va="top")

        ax = fig.add_axes([0.10, 0.66, 0.84, 0.22])
        y = [m["crf_mean"] for m in methods if m["crf_mean"] is not None]
        if y:
            run = np.maximum.accumulate(y)
            ax.plot(range(1, len(y) + 1), y, "o", ms=3.5, color="#8B978F",
                    label="each method")
            ax.plot(range(1, len(run) + 1), run, "-", lw=1.6, color="#0E6E5B",
                    label="best so far")
            ax.legend(fontsize=7, frameon=False)
        ax.set_title("Score, method by method", fontsize=9.5)
        ax.set_xlabel("method", fontsize=8.5); ax.set_ylabel("CRF", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        ax = fig.add_axes([0.10, 0.37, 0.84, 0.20])
        refs = c.references()
        lim = c.drift_limits()
        if refs:
            ax.plot([r.run_order for r in refs], [r.crf for r in refs], "o-",
                    lw=1.5, ms=4, color="#16201C")
            a = refs[0].crf
            ax.axhline(a, color="#55635B", lw=1, label=f"anchor {a:.2f}")
            ax.axhline(a - lim["watch"], color="#B4531F", ls="--", lw=1,
                       label=f"WATCH -{lim['watch']:g}")
            ax.axhline(a - lim["halt"], color="#A32F27", ls="--", lw=1,
                       label=f"HALT -{lim['halt']:g}")
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        ax.set_title(f"Instrument check - {v.verdict}", fontsize=9.5)
        ax.set_xlabel("run order", fontsize=8.5); ax.set_ylabel("CRF", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        ax = fig.add_axes([0.10, 0.08, 0.84, 0.20])
        t = HIST.trend(c.root)
        pts = [p["mean_sd"] for p in t["points"] if p["mean_sd"] is not None]
        if pts:
            ax.plot(range(1, len(pts) + 1), pts, "o-", lw=1.5, ms=4,
                    color="#0E6E5B")
        ax.set_title("Model uncertainty, run after run", fontsize=9.5)
        ax.set_xlabel("recorded fit", fontsize=8.5)
        ax.set_ylabel("mean posterior SD (CRF)", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.text(0.07, 0.035, t["reading"][:170], fontsize=7.5, color="#55635B")
        pp.savefig(fig)
        plt.close(fig)

        # ── 4. every method ─────────────────────────────────────────
        head = (f"  {'method':<8}{'CRF':>8}{'sd':>7}{'n':>3}  "
                f"{'start%B':>8}{'end%B':>7}{'min':>6}{'degC':>6}  flags")
        lines = [head, "  " + "-" * (len(head) - 2)]
        for m in sorted(methods, key=lambda x: -(x["crf_mean"] or -1)):
            # A tuned picker and a hand-drawn region are DIFFERENT acts with
            # different justifications - see HANDOFF 3 and 11 - so they are
            # never collapsed into one flag.
            flags = "".join([" tuned" if m["deviates"] else "",
                             " hump-drawn" if m["manual_humps"] else "",
                             " tied" if m["tied"] else ""])
            lines.append(
                f"  {m['method']:<8}{(m['crf_mean'] or 0):>8.3f}"
                f"{(m['crf_sd'] if m['crf_sd'] is not None else float('nan')):>7.3f}"
                f"{m['n']:>3}  {m['start_phi']*100:>8.1f}{m['end_phi']*100:>7.1f}"
                f"{m['duration_min']:>6.1f}{m['T']:>6.1f} {flags}")
        _fig_text(pp, "Every method", [
            ("design methods, best first", lines),
            ("noise", _noise_lines(c, sd, groups, dof)),
        ], footer="sd is the within-method spread over replicates; blank where "
                  "a method has only one run.")

        # ── 5. the histories ────────────────────────────────────────
        blocks: list[tuple[str, list[str]]] = []
        if bud.history:
            blocks.append(("run budget", [
                f"  {h.get('at','')[:16]}  {h.get('event',''):<9} "
                f"{str(h.get('from','')):>4} -> {str(h.get('to','')):<4} "
                f"at method {h.get('used','')}  {h.get('reason','')[:60]}"
                for h in bud.history]))
        meta = c.metadata_history()
        if meta:
            lines = []
            for h in meta:
                for ch in h.get("changes", []):
                    lines.append(
                        f"  {h.get('at','')[:16]}  {ch['field']}"
                        f"{'  MATERIAL' if ch.get('material') else ''}"
                        f"  at {h.get('runs_recorded','?')} run(s)")
                    lines += [f"      {x}" for x in _wrap(
                        "from", f"{ch['from']!r}  ->  {ch['to']!r}", 84)]
                    if h.get("reason"):
                        # The reason is the WHOLE point of this table: it is
                        # what separates "corrected how it is written down"
                        # from "a different column was on the instrument".
                        lines += [f"      {x}" for x in _wrap(
                            "why", h["reason"], 84)]
            blocks.append(("metadata corrections", lines))
        from . import rescore as RS
        ph = RS.history(c)
        if ph:
            blocks.append(("picker configuration changes", [
                f"  {h.get('at','')[:16]}  "
                f"{', '.join(f'{k}: {a} -> {b}' for k, (a, b) in (h.get('changed') or {}).items())[:70]}"
                f"  ({h.get('n_moved','')}/{h.get('n_rows','')} moved)"
                for h in ph]))
        if not blocks:
            blocks = [("", ["Nothing was extended, corrected or re-scored - the "
                            "campaign ran as it was created."])]
        from .core import optimiser as O
        now = O.library_versions()
        drift_lines, moved = [], False
        for libname in sorted(set(now) | {k[4:] for k in cfg
                                          if k.startswith("lib_")}):
            was = str(cfg.get(f"lib_{libname}", "") or "not recorded")
            is_ = str(now.get(libname, "not installed"))
            drift_lines.append(f"  {libname:<10} at creation {was:<12} "
                               f"now {is_:<12}"
                               f"{'  DIFFERENT' if was != is_ and was != 'not recorded' else ''}")
            moved = moved or (was != is_ and was != "not recorded")
        if moved:
            drift_lines += _wrap(
                "NOTE", "the environment has moved since this campaign was "
                        "created. BoTorch's kernel and priors have changed "
                        "between releases, so a refit here is not comparable "
                        "with what is recorded above.", 92)
        blocks.append(("library versions (pin: botorch>=0.18,<0.19)", drift_lines))
        blocks.append(("what this report does not say", [
            "Whether the separation is good enough. That is a judgement the",
            "analyst makes by looking at the chromatogram; a report that made",
            "it would be read as having made it.",
        ]))
        _fig_text(pp, "Provenance", blocks,
                  footer="Every run, every setting and every change is in the "
                         "workbook export beside this file.")

        d = pp.infodict()
        d["Title"] = f"Gradient Bench - {info.get('name') or c.slug}"
        d["Author"] = str(info.get("analyst") or "")
        d["Subject"] = (f"HPLC method development campaign, "
                        f"{bud.used} methods, best CRF "
                        f"{summary.get('best_crf')}")
        d["CreationDate"] = _dt.datetime.now()
    return out_path
