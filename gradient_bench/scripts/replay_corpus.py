#!/usr/bin/env python3
"""Replay the real trace corpus through the whole library, headless.

This is phase 0's acceptance test: no Flask, no React, no notebook. If this
runs clean, the science layer is ready for an API to sit on top of it.

    python -m gradient_bench.scripts.replay_corpus [--trace-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from gradient_bench.core import space as S
from gradient_bench.core import optimiser as O
from gradient_bench.core import drift as D
from gradient_bench.core import picker as P
from gradient_bench.core import uncertainty as UNC

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rule(t):
    print("\n" + "=" * 74); print(t); print("=" * 74)


def _require_modelling_stack() -> None:
    """Refuse up front rather than 60 lines in.

    The replay fits a GP, so it needs the modelling stack. It used to print its
    environment banner - including "torch not installed" - and then die with a
    raw ModuleNotFoundError much later, which reads as a crash rather than a
    missing dependency.
    """
    try:
        import torch, botorch, gpytorch                    # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"\n  This replay fits the model, so it needs the modelling stack, "
            f"and it is not installed ({exc}).\n"
            f"  Install it with:  pip install -r gradient_bench/requirements.txt\n"
            f"  Then check the whole machine with:  "
            f"python -m gradient_bench.scripts.verify_install\n\n"
            f"  Everything that does NOT need the model - scoring traces, "
            f"recording runs,\n  reading campaigns, re-scoring and exporting - "
            f"works without it.\n")


def main() -> int:
    _require_modelling_stack()
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir",
                    default="/mnt/user-data/uploads/exp result/4d exp output")
    a = ap.parse_args()
    gold = json.load(open(os.path.join(PKG, "golden_files.json")))

    rule("1 - ENVIRONMENT")
    for k, v in O.library_versions().items():
        print(f"   {k:<10} {v}")
    print(f"   picker     {P._find_hplc_picker().__version__}")

    rule("2 - SCORE EVERY TRACE FROM ITS CHROMATOGRAM")
    import pandas as pd
    from gradient_bench.tests.test_core import find_corpus_sheet
    sheet = pd.read_excel(find_corpus_sheet(a.trace_dir),
                          sheet_name="Data").set_index("method")
    runs = []
    for m in sorted(sheet.index, key=lambda s: int(s.replace("method", ""))):
        g = gold["settled"].get(m, {})
        cfg = dict(g.get("config", {}))
        manual = cfg.pop("hump_override", None) or (g.get("hump_spans") or None)
        pick = P.analyse(os.path.join(a.trace_dir, m + ".TXT"),
                         overrides=cfg or None, manual_humps=manual)
        u = S.abs_to_4param(sheet.loc[m])
        ok, why = S.check4(u)
        runs.append(dict(name=m, u=u, crf=pick.crf, peaks=pick.n_clean_peaks,
                         feasible=ok, why=why, row=P.record_row(pick)))
        tag = "" if ok else "  [outside the design box]"
        tune = ("  tuned: " + ", ".join(
            f"{k}={v}" for k, v in sorted(g.get("config", {}).items()))) if g.get("config") else ""
        print(f"   {m:<10} {pick.n_clean_peaks:>3} peaks  hump "
              f"{pick.hump_time_fraction:6.4f}  CRF {pick.crf:7.3f}{tune}{tag}")

    inside = [r for r in runs if r["feasible"]]
    print(f"\n   {len(inside)}/{len(runs)} runs are inside the 4-parameter box")
    print(f"   record row carries {len(runs[0]['row'])} columns")

    rule("3 - THE OBJECTIVE IS SEALED")
    hp = P._find_hplc_picker()
    from gradient_bench.core.crf import CRF_COLUMNS, assert_objective_is_clean
    assert_objective_is_clean(hp.TRACE_HEALTH_KEYS)
    print(f"   CRF_COLUMNS = {CRF_COLUMNS}")
    print(f"   {len(hp.TRACE_HEALTH_KEYS)} trace-health descriptors recorded, "
          f"0 of them readable by the objective")

    rule("4 - FIT THE SURROGATE")
    U = np.array([r["u"] for r in inside])
    Y = np.array([r["crf"] for r in inside])
    fit = O.fit_gp(U, Y, sigma_crf=None)
    print(f"   {fit.n} runs, {fit.d}-D, fitted noise sd {fit.noise_sd_crf:.3f} CRF")
    for line in fit.noise.explain():
        print(f"   {line}")
    print("   ARD lengthscales:")
    for nm, v, read in fit.lengthscale_table():
        print(f"      {nm:<14}{v:8.3f}   {read}")

    rule("5 - PROPOSE THE NEXT METHOD")
    sug = O.suggest_next(U, Y, sigma_crf=None, seed=20260805, fit=fit)
    print(f"   {sug.u[0]*100:.2f} -> {sug.u[1]*100:.2f} %B over {sug.u[2]:.2f} min "
          f"at {sug.u[3]:.1f} C")
    print(f"   feasible={S.check4(sug.u)[0]}  logqNEI={sug.acq_value:.4f}  "
          f"posterior {sug.posterior_mean:.2f} +- {sug.posterior_sd:.2f} CRF")
    for row in S.gradient_table(sug.u):
        print(f"      {row['time_min']:>6.2f} min   {row['pct_b']:>5.1f} %B   {row['phase']}")

    rule("6 - WHAT EACH MEASUREMENT BOUGHT")
    grid = UNC.make_reference_grid()
    hist = []
    for k in range(3, len(Y) + 1):
        st = UNC.uncertainty_step(U[:k], Y[:k], sigma_crf=None, grid=grid)
        if st:
            hist.append(st)
    print(f"   {'runs':>5} {'mean SD':>9} {'change':>9} {'noise SD':>9} {'best':>7}")
    for st in hist:
        print(f"   {st.n_runs:>5} {st.mean_sd_after:9.4f} {-st.d_mean_sd:+9.4f} "
              f"{st.noise_sd_crf:9.3f} {st.best_crf:7.2f}")
    print(f"\n   {UNC.campaign_reading(hist, drift_falling=False)}")

    rule("7 - THE FOUR POSTERIOR PANELS")
    inc = U[int(np.argmax(Y))]
    panels = UNC.slice_posteriors(fit, inc, tested_u=U, tested_y=Y)
    print(f"   incumbent: {inc[0]*100:.2f} -> {inc[1]*100:.2f} %B, "
          f"{inc[2]:.2f} min, {inc[3]:.1f} C  (CRF {Y.max():.2f})")
    for p in panels:
        widest = int(np.argmax(p.sd))
        print(f"   {p.name:<14} sd {p.sd.min():.2f}-{p.sd.max():.2f} CRF, "
              f"least certain at {p.values[widest]:.3g}, "
              f"{int(p.infeasible.sum())}/{len(p.values)} sweep points infeasible")

    rule("8 - DRIFT")
    sch = D.reference_schedule(n_methods_total=len(inside),
                              methods_since_last_reference=len(inside), n_ref=0)
    print(f"   schedule: next is {sch.next_is.upper()} (anchor={sch.anchor}, "
          f"blocks proposing={sch.blocks_proposing})")
    print(f"   {sch.reason}")
    v = D.drift_verdict([])
    print(f"   verdict with no reference: {v.verdict} - {v.reasons[0][:64]}...")

    rule("REPLAY COMPLETE")
    print(f"   {len(runs)} traces scored from raw chromatograms")
    print(f"   {len(inside)} inside the box, fitted, {fit.d}-D surrogate")
    print(f"   {len(hist)} uncertainty steps, 4 posterior panels, drift gate live")
    print("   no notebook, no Flask, no React - the science layer stands alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
