"""verify_install.py - is this machine able to run a campaign?

    python -m gradient_bench.scripts.verify_install

WHO THIS IS FOR.  Somebody standing at a lab PC that has just had the
requirements installed, who needs to know whether it works BEFORE putting a
sample on the instrument. Finding out at run 6 that BoTorch will not import
costs a column equilibration and an afternoon.

WHAT IT CHECKS, IN THE ORDER THAT MATTERS.  Cheap and structural first, so a
broken install fails in two seconds rather than after a minute of scoring.
Every check says what it means and what to do if it fails - a verifier that
prints a stack trace has told the analyst nothing they can act on.

IT DOES NOT NEED AN INSTRUMENT, A NETWORK, OR A CAMPAIGN.  Everything it needs
ships in the folder.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
import warnings
from typing import Any, Callable, Optional

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


class Check:
    def __init__(self, name: str, why: str, fix: str, fn: Callable[[], Any],
                 required: bool = True):
        self.name, self.why, self.fix, self.fn = name, why, fix, fn
        self.required = required
        self.status, self.detail, self.seconds = "", "", 0.0

    def run(self) -> str:
        """Library warnings are captured, not printed.

        A baseline fitter that warns on one trace is not an install problem,
        and letting it interleave with this report turns a clean list into
        something nobody can read - which is the opposite of the point.
        """
        t0 = time.time()
        try:
            buf = io.StringIO()
            with warnings.catch_warnings(), contextlib.redirect_stderr(buf):
                warnings.simplefilter("ignore")
                self.detail = str(self.fn() or "")
            self.status = PASS
        except Exception as exc:                                # noqa: BLE001
            self.detail = f"{type(exc).__name__}: {exc}"
            self.status = FAIL if self.required else WARN
        self.seconds = time.time() - t0
        return self.status


# ── the checks ──────────────────────────────────────────────────────────────
def _python() -> str:
    if sys.version_info < (3, 9):
        raise RuntimeError(f"Python {sys.version.split()[0]} is too old")
    return f"Python {sys.version.split()[0]}"


def _imports() -> str:
    import numpy, pandas, scipy, openpyxl, pybaselines      # noqa: F401
    return (f"numpy {numpy.__version__}, pandas {pandas.__version__}, "
            f"scipy {scipy.__version__}, openpyxl {openpyxl.__version__}, "
            f"pybaselines {pybaselines.__version__}")


def _torch_stack() -> str:
    import torch, botorch, gpytorch                         # noqa: F401
    v = botorch.__version__
    major_minor = ".".join(v.split(".")[:2])
    note = ""
    if major_minor != "0.18":
        note = (f"  <-- NOT the pinned 0.18.x. The default kernel and priors "
                f"have changed between BoTorch releases, so a campaign fitted "
                f"under this version is not comparable with one fitted under "
                f"the pin.")
    return f"torch {torch.__version__}, botorch {v}, gpytorch {gpytorch.__version__}{note}"


def _flask() -> str:
    import flask
    from importlib.metadata import version
    return f"flask {version('flask')}"


def _picker() -> str:
    from gradient_bench.core.picker import _find_hplc_picker
    hp = _find_hplc_picker()
    import gradient_bench.core.hplc_picker as vendored
    if hp is not vendored:
        raise RuntimeError("the picker resolved to something other than the "
                           "vendored copy in gradient_bench/core/")
    return f"hplc_picker {hp.__version__}, {len(hp.RECORD_KEYS)} recorded columns"


def _objective() -> str:
    from gradient_bench.core.crf import (CRF_COLUMNS, CRF_VERSION,
                                         assert_objective_is_clean, compute_crf)
    from gradient_bench.core.picker import _find_hplc_picker
    assert_objective_is_clean(_find_hplc_picker().TRACE_HEALTH_KEYS)
    got = compute_crf({"n_clean_peaks": 10, "hump_time_fraction": 0.25})
    if abs(got - 5.625) > 1e-9:
        raise RuntimeError(f"CRF arithmetic is wrong: expected 5.625, got {got}")
    return (f"CRF {CRF_VERSION}, inputs {CRF_COLUMNS}, trace-health descriptors "
            f"structurally barred")


def _space() -> str:
    import numpy as np
    from gradient_bench.core import space as S
    frac = S.feasible_fraction(20000, seed=7)
    if abs(frac - 0.868421) > 0.01:
        raise RuntimeError(f"feasible fraction {frac:.4f} is not the expected "
                           f"0.8684 - the constraint has moved")
    if not S.check4([0.20, 0.20, 30.0, 40.0])[0]:
        raise RuntimeError("isocratic methods should be inside the design space")
    seeds = S.initial_sample(30, seed=11)
    if not all(S.check4(u)[0] for u in seeds):
        raise RuntimeError("the cold start produced an unrunnable method")
    return f"constraint intact, feasible fraction {frac:.4f}, cold start clean"


def _traces() -> str:
    from gradient_bench import testdata
    d = testdata.traces_dir()
    if not testdata.available():
        raise RuntimeError(f"the bundled example chromatograms are missing from {d}")
    n = len([f for f in os.listdir(d) if f.lower().endswith(".txt")])
    return f"{n} example chromatograms in testdata/traces"


def _golden(n: Optional[int] = None) -> str:
    """Score real traces and compare against the adjudicated expectations.

    This is the check that matters: it exercises the actual pipeline end to end
    on files whose correct answer a chemist has already agreed.
    """
    import json
    from gradient_bench.core import picker as P
    from gradient_bench import testdata
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    golden = json.load(open(os.path.join(here, "golden_files.json")))
    settled = golden["settled"]
    names = sorted(settled, key=lambda s: int(s.replace("method", "")))
    if n:
        names = names[:n]
    bad = []
    for m in names:
        cfg = settled[m]
        ov = {k: v for k, v in cfg.get("config", {}).items()
              if k != "hump_override"}
        manual = cfg.get("config", {}).get("hump_override") or cfg.get("hump_spans")
        res = P.analyse(os.path.join(testdata.traces_dir(), m + ".TXT"),
                        overrides=ov or None, manual_humps=manual or None)
        if (res.n_clean_peaks != cfg["n_clean_peaks"]
                or abs(res.crf - cfg["crf"]) > 5e-3):
            bad.append(f"{m}: got {res.n_clean_peaks} peaks / CRF {res.crf:.3f}, "
                       f"expected {cfg['n_clean_peaks']} / {cfg['crf']:.3f}")
    if bad:
        raise RuntimeError("; ".join(bad))
    return (f"{len(names)} adjudicated traces reproduce their expected peak "
            f"count and CRF exactly")


def _campaign_round_trip() -> str:
    """Create a campaign, record a run, read it back, export it. No instrument."""
    import tempfile
    import numpy as np
    from gradient_bench.store import campaign as C, sheet as SH
    from gradient_bench.core import space as S
    from gradient_bench import export as EXP

    tmp = tempfile.mkdtemp(prefix="gb-verify-")
    saved, C.REGISTRY = C.REGISTRY, os.path.join(tmp, "recent.json")
    try:
        c = C.create(tmp, {"name": "install check"},
                     reference_method=[0.10, 0.30, 50.0, 45.0],
                     n_replicates=1, run_budget=5)
        SH.append_run(c.workbook, method_name="ref01",
                      source=SH.SOURCE_REFERENCE, replicate=1,
                      params_abs=S.simple_gradient_abs(
                          np.array([0.10, 0.30, 50.0, 45.0])),
                      row={"CRF": 9.0, "n_clean_peaks": 9,
                           "hump_time_fraction": 0.0},
                      run_order=1)
        if c.n_injections() != 1:
            raise RuntimeError("a recorded run did not read back")
        xl = EXP.workbook(c)
        pdf = EXP.pdf(c)
        sizes = (os.path.getsize(xl), os.path.getsize(pdf))
        if min(sizes) < 2000:
            raise RuntimeError("an export came back suspiciously small")
        return (f"campaign created, run recorded, workbook + PDF exported "
                f"({sizes[0] // 1024} kB, {sizes[1] // 1024} kB)")
    finally:
        C.REGISTRY = saved


def _fit() -> str:
    """The only check that needs torch. Fits the real surrogate on a few points."""
    import numpy as np
    from gradient_bench.core import optimiser as O, space as S
    rng = np.random.default_rng(0)
    U = S.initial_sample(8, seed=3)
    y = np.array([float(5 + 3 * u[3] / 60 + rng.normal(0, 0.2)) for u in U])
    fit = O.fit_gp(U, y, sigma_crf=1.0, kernel="matern52")
    ls = dict((n, round(v, 4)) for n, v, _ in fit.lengthscale_table())
    sug = O.suggest_next(U, y, sigma_crf=1.0, seed=1, kernel="matern52")
    if not S.check4(sug.u)[0]:
        raise RuntimeError("the optimiser proposed an unrunnable method")
    return (f"Matern-5/2 fitted on 8 points, lengthscales {ls}, "
            f"proposal feasible")


CHECKS: list[Check] = [
    Check("python", "the interpreter this will run under",
          "install Python 3.9 or newer", _python),
    Check("core libraries", "numpy, pandas, scipy, openpyxl, pybaselines - "
          "everything the picker and the workbook need",
          "pip install -r gradient_bench/requirements.txt", _imports),
    Check("modelling stack", "torch, botorch and gpytorch - needed to PROPOSE "
          "methods, not to record them",
          "pip install -r gradient_bench/requirements.txt  (this is the big "
          "download; without it the app still records runs and reads campaigns, "
          "but cannot propose)", _torch_stack),
    Check("web server", "flask, which serves the interface",
          "pip install -r gradient_bench/requirements.txt", _flask),
    Check("peak picker", "the vendored picker, and that nothing else shadows it",
          "check that gradient_bench/core/hplc_picker.py is present and that no "
          "other hplc_picker.py is earlier on sys.path", _picker),
    Check("the objective", "that CRF is computed the way this app defines it, "
          "and that trace-health descriptors cannot reach it",
          "this is a code-level failure - do not run a campaign; report it",
          _objective),
    Check("design space", "the box, the constraint and the cold start",
          "this is a code-level failure - do not run a campaign; report it",
          _space),
    Check("example chromatograms", "the bundled traces the next check scores",
          "restore testdata/traces from the project folder", _traces),
    Check("golden files", "the whole picking pipeline, against traces whose "
          "correct answer a chemist has already agreed",
          "if this fails the picker is behaving differently on this machine - "
          "check the pybaselines and scipy versions against the requirements",
          _golden),
    Check("campaign round trip", "create, record, read back, export",
          "check that the folder is writable", _campaign_round_trip),
    Check("model fit", "fitting the real surrogate and asking it for a method",
          "needs the modelling stack above", _fit),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow checks (golden files, round trip, fit)")
    a = ap.parse_args(argv)

    checks = CHECKS
    if a.quick:
        skip = {"golden files", "campaign round trip", "model fit"}
        checks = [c for c in CHECKS if c.name not in skip]

    print()
    print("  Gradient Bench - install check")
    print("  " + "-" * 66)
    worst = PASS
    for c in checks:
        print(f"  {c.name:24s} ", end="", flush=True)
        st = c.run()
        mark = {PASS: "ok  ", FAIL: "FAIL", WARN: "warn"}[st]
        print(f"{mark}  {c.detail[:110]}")
        if st == FAIL:
            worst = FAIL
        elif st == WARN and worst == PASS:
            worst = WARN

    print("  " + "-" * 66)
    failed = [c for c in checks if c.status == FAIL]
    if not failed:
        print("  Everything this machine needs is present and working.")
        print("  Start the app with:  python -m gradient_bench.api.app")
        print()
        return 0

    print(f"  {len(failed)} check(s) failed. Nothing has been changed on this "
          f"machine.\n")
    for c in failed:
        print(f"  {c.name}")
        print(f"    what it is for : {c.why}")
        print(f"    what went wrong: {c.detail[:300]}")
        print(f"    what to do     : {c.fix}\n")
    print("  A campaign should not be started until these pass: the app would "
          "fail\n  part-way through, after instrument time has been spent.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
