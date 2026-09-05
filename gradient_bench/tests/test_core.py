"""The phase-0 test suite.

Every assertion the notebook made inline is a test here, plus the golden-file
expectations adjudicated against the real traces.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from gradient_bench.core import space as S            # noqa: E402
from gradient_bench.core import drift as D            # noqa: E402
from gradient_bench.core.crf import (                 # noqa: E402
    CRF_COLUMNS, compute_crf, assert_objective_is_clean)

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
GOLDEN = json.load(open(os.path.join(PKG, "golden_files.json")))
def _default_trace_dir() -> str:
    """The bundled corpus, found relative to the package.

    This used to default to a machine-specific absolute path, so `pytest`
    without GB_TRACE_DIR skipped 32 tests - every golden file and the
    exports and the re-scores - and printed a confident "86 passed". The docs
    then told the reader that a pass meant the science layer was intact. Same
    class of bug as the corpus-sheet path in HANDOFF 7.8; that fix was only
    half applied.
    """
    return os.path.join(os.path.dirname(PKG), "testdata", "traces")


TRACE_DIR = os.environ.get("GB_TRACE_DIR") or _default_trace_dir()
HAVE_TRACES = os.path.isdir(TRACE_DIR)

def find_corpus_sheet(trace_dir: str):
    """Locate the corpus workbook beside a trace directory.

    The original folder layout put it at `<traces>/../4d/data/master_methods.xlsx`.
    Do not assume that: the project should survive being moved. GB_SHEET wins if set.
    """
    env = os.environ.get("GB_SHEET")
    if env and os.path.exists(env):
        return env
    for rel in (os.path.join("..", "4d", "data", "master_methods.xlsx"),
                os.path.join("..", "master_methods.xlsx"),
                os.path.join("..", "data", "master_methods.xlsx"),
                "master_methods.xlsx"):
        p = os.path.abspath(os.path.join(trace_dir, rel))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"corpus workbook not found near {trace_dir}; set GB_SHEET to point at it")



# ── the objective ───────────────────────────────────────────────────────────
def test_crf_formula():
    assert compute_crf({"n_clean_peaks": 10, "hump_time_fraction": 0.25}) == 5.625
    assert compute_crf({"n_clean_peaks": 7, "hump_time_fraction": 0.0}) == 7.0


def test_trace_health_cannot_reach_the_objective():
    from gradient_bench.core.picker import _find_hplc_picker
    assert_objective_is_clean(_find_hplc_picker().TRACE_HEALTH_KEYS)
    with pytest.raises(AssertionError):
        assert_objective_is_clean(["n_clean_peaks"])


# ── the design space and its constraint ─────────────────────────────────────
def test_constraint_constants():
    assert S.CON_IDX == [1, 0]
    assert S.CON_COEF == pytest.approx([0.90, -0.38])
    assert S.CON_RHS == pytest.approx(-0.08)


def test_normalised_constraint_matches_the_absolute_definition():
    rng = np.random.default_rng(0)
    U = S.P4_LOWER + rng.random((20000, 4)) * (S.P4_UPPER - S.P4_LOWER)
    assert np.array_equal(U[:, 1] >= U[:, 0] + S.MIN_SPAN - 1e-9,
                          S.constraint_slack_norm(S.to_norm(U)) >= -1e-9)


def test_feasible_fraction_matches_the_analytic_value():
    assert S.feasible_fraction(200000, seed=1) == pytest.approx(0.868421, abs=2e-3)


def test_norm_round_trip():
    rng = np.random.default_rng(3)
    U = S.P4_LOWER + rng.random((500, 4)) * (S.P4_UPPER - S.P4_LOWER)
    assert np.allclose(S.from_norm(S.to_norm(U)), U)


@pytest.mark.parametrize("u,ok", [
    ([0.14, 0.62, 18.0, 48.0], True),      # method4-ish, reference preset A
    ([0.10, 0.30, 50.0, 45.0], True),      # reference preset B
    ([0.20, 0.20, 30.0, 40.0], True),      # isocratic is INSIDE the space
    ([0.30, 0.10, 20.0, 40.0], False),     # end below start
    ([0.50, 0.90, 20.0, 40.0], False),     # start above the box
    ([0.14, 0.62, 5.0, 48.0], False),      # too short
    ([0.14, 0.62, 18.0, 70.0], False),     # too hot
])
def test_check4(u, ok):
    assert S.check4(u)[0] is ok


def test_require_feasible_raises():
    with pytest.raises(ValueError):
        S.require_feasible([0.30, 0.10, 20.0, 40.0])


def test_cold_start_is_always_feasible():
    seeds = S.initial_sample(300, seed=11)
    assert all(S.check4(u)[0] for u in seeds)


def test_abs_round_trip():
    u = np.array([0.1398, 0.6338, 17.95, 47.79])
    assert np.allclose(S.abs_to_4param(S.simple_gradient_abs(u)), u)


# ── the noise model ─────────────────────────────────────────────────────────
def test_noise_floor_unit_conversion():
    from gradient_bench.core import optimiser as O
    y = np.array([2.0, 4.0, 6.0, 8.0])
    sd = float(np.std(y, ddof=1))
    assert O.noise_floor_std(y, sd).floor_z == pytest.approx(1.0)
    assert O.noise_floor_std(y, sd / 2).floor_z == pytest.approx(0.25)
    assert O.noise_floor_std(y, 2 * sd).floor_z == pytest.approx(4.0)
    assert O.noise_floor_std(y, 2 * sd).dominated


def test_tied_replicates_count_as_unmeasured():
    """An integer-count objective makes an exact tie a real outcome. sigma == 0
    is finite and must not be used as a measurement."""
    from gradient_bench.core import optimiser as O
    sd, groups, dof = O.pooled_within_sd({"m01": [13.0, 13.0]})
    assert (sd, groups, dof) == (0.0, 1, 1)
    assert O.noise_floor_std(np.array([2.0, 4.0, 6.0]), sd).used_fallback


def test_pooled_within_sd():
    from gradient_bench.core import optimiser as O
    sd, groups, dof = O.pooled_within_sd(
        {"m01": [13.0, 13.0], "m02": [7.0, 9.0], "m03": [5.0]})
    assert (groups, dof) == (2, 2)
    assert sd == pytest.approx(1.0)          # ss = 0 + 2 over 2 dof
    assert np.isnan(O.pooled_within_sd({"m01": [5.0]})[0])


# ── drift ───────────────────────────────────────────────────────────────────
def test_drift_arithmetic_hand_checked():
    refs = [D.ReferenceRun(1.0, 10.0), D.ReferenceRun(11.0, 6.0)]
    assert float(D.drift_baseline(6.0, refs)) == pytest.approx(8.0)
    assert float(D.drift_offset(6.0, refs)) == pytest.approx(-2.0)
    assert float(D.drift_adjust(7.0, 6.0, refs)) == pytest.approx(9.0)
    assert float(D.drift_adjust(5.0, 1.0, refs)) == pytest.approx(5.0)


def test_drift_baseline_is_held_flat_not_extrapolated():
    refs = [D.ReferenceRun(1.0, 10.0), D.ReferenceRun(11.0, 6.0)]
    assert float(D.drift_baseline(99.0, refs)) == pytest.approx(6.0)


@pytest.mark.parametrize("refs,want", [
    ([], "WATCH"),                                              # unmeasured != pass
    ([(1, 10)], "PASS"),
    ([(1, 10), (11, 9.2)], "PASS"),
    ([(1, 10), (11, 8.2)], "WATCH"),
    ([(1, 10), (11, 6.5)], "HALT"),
    ([(1, 10), (6, 9.5), (11, 9.0), (16, 8.6)], "HALT"),        # 3 monotone falls
])
def test_drift_verdicts(refs, want):
    v = D.drift_verdict([D.ReferenceRun(float(a), float(b)) for a, b in refs])
    assert v.verdict == want
    assert v.blocks_proposing == (want == "HALT")


def test_phantom_incumbent():
    refs = [D.ReferenceRun(1, 10.0), D.ReferenceRun(30, 6.0)]
    inc = D.best_so_far([15.0, 13.0], [3.0, 25.0], ["A", "B"], refs)
    assert inc.raw_method == "A" and inc.adj_method == "B" and inc.phantom
    assert "PHANTOM INCUMBENT" in inc.warning()


def test_no_references_means_no_phantom_claim():
    inc = D.best_so_far([15.0, 13.0], [3.0, 25.0], ["A", "B"], [])
    assert inc.phantom is False


def test_reference_cadence_is_enforced():
    assert D.reference_schedule(0, 0, 0).blocks_proposing      # anchor first
    assert not D.reference_schedule(3, 3, 1).blocks_proposing
    assert D.reference_schedule(7, 5, 1).blocks_proposing      # cadence, ENFORCED
    assert D.reference_schedule(9, 7, 1).overdue == 2


def test_reference_presets_are_feasible():
    u = D.REFERENCE_PRESETS["shallow_survey"]["u"]
    assert S.check4(u)[0]
    assert not S.is_isocratic(u)


# ── the golden files ────────────────────────────────────────────────────────
def _expected():
    import pandas as pd
    out = {}
    for m, g in GOLDEN["settled"].items():
        out[m] = (g["n_clean_peaks"], g["crf"], g.get("config", {}),
                  g.get("hump_spans", []))
    sheet = pd.read_excel(find_corpus_sheet(TRACE_DIR),
                          sheet_name="Data").set_index("method")
    for m in GOLDEN["agree_at_defaults"]:
        out[m] = (int(sheet.loc[m, "n_clean_peaks"]),
                  float(sheet.loc[m, "CRF"]), {}, [])
    return out


@pytest.mark.skipif(not HAVE_TRACES, reason="trace corpus not available")
@pytest.mark.parametrize("method", sorted(
    list(GOLDEN["settled"]) + GOLDEN["agree_at_defaults"],
    key=lambda s: int(s.replace("method", ""))))
def test_golden_trace(method):
    """Every trace must reproduce its adjudicated peak count and CRF."""
    from gradient_bench.core import picker as P
    exp = _expected()[method]
    want_peaks, want_crf, cfg, spans = exp
    overrides = {k: v for k, v in cfg.items() if k != "hump_override"}
    manual = cfg.get("hump_override") or (spans or None)
    got = P.analyse(os.path.join(TRACE_DIR, method + ".TXT"),
                    overrides=overrides or None, manual_humps=manual)
    assert got.n_clean_peaks == want_peaks
    assert got.crf == pytest.approx(want_crf, abs=5e-3)


@pytest.mark.skipif(not HAVE_TRACES, reason="trace corpus not available")
def test_campaign_default_snr_is_the_best_choice():
    """snr 5.0 gets 19/20; lowering it to rescue method9 would break fifteen."""
    from gradient_bench.core import picker as P
    exp = _expected()
    hits = 0
    for m, (want, _crf, cfg, spans) in exp.items():
        manual = cfg.get("hump_override") or (spans or None)
        got = P.analyse(os.path.join(TRACE_DIR, m + ".TXT"), manual_humps=manual)
        hits += (got.n_clean_peaks == want)
    assert hits == 19, f"snr 5.0 should get 19/20 at defaults, got {hits}"


# ── the stopping rule ───────────────────────────────────────────────────────
from gradient_bench.core import budget as B                # noqa: E402


def _status(**kw):
    base = dict(used=0, limit=40, original=40, n_replicates=2, need=1)
    base.update(kw)
    return B.BudgetStatus(**base)


def test_the_budget_counts_unique_methods_not_injections():
    """Reversed on 2026-08-30: one distinct set of parameters costs one,
    however many times it is injected. Replicates and instrument checks are
    bench time, not budget. See HANDOFF section 24."""
    s = _status(used=30)
    assert s.remaining == 10
    assert s.methods_left == 10             # methods, not runs divided by reps
    # and the replicate count does NOT change what is left, which is the whole
    # point of the change - it changes what those methods COST
    assert _status(used=30, n_replicates=3).methods_left == 10
    assert _status(used=30, n_replicates=2).injections_left == 20
    assert _status(used=30, n_replicates=3).injections_left == 30


def test_the_injection_cost_of_a_budget_is_stated_not_hidden():
    """A method budget hides the scarce thing, so the arithmetic is done once
    here and shown wherever a budget is set or reported."""
    c = B.injection_cost(40, 2)
    assert c["design_injections"] == 80          # 40 methods x 2 repeats
    assert c["instrument_checks"] == 9           # the anchor, then one per 5
    assert c["injections"] == 89
    assert c["hours"] == 44                      # at a 30-minute gradient

    # three repeats is the same 40 methods and half again the bench time
    assert B.injection_cost(40, 3)["injections"] == 129
    # and an empty budget costs nothing, including no anchor
    assert B.injection_cost(0, 2)["injections"] == 0


def test_a_method_always_fits_whatever_its_replicate_count():
    """Under an injection budget, one run left could not pay for a
    two-replicate method and the gate had to refuse it. A method costs one
    whatever it takes to run, so that class of refusal is gone."""
    nearly = _status(used=39)
    assert nearly.remaining == 1
    assert nearly.fits and not nearly.exhausted
    assert _status(used=39, n_replicates=5).fits
    # an instrument check costs no budget at all, even with none left
    assert _status(used=40, need=0).fits


def test_running_out_of_budget_is_not_a_halt():
    """A HALT says the numbers may be wrong. An exhausted budget says the plan
    is spent and every number already recorded is exactly as good as it was."""
    spent = _status(used=40)
    assert spent.state == "EXHAUSTED" and spent.blocks_proposing
    assert "HALT" not in spent.reason()
    assert any("Finish the campaign" in a for a in spent.actions())


def test_over_budget_is_exhausted_not_an_error():
    """Reachable by lowering a budget below the methods already run. The runs
    happened; the data is real."""
    s = _status(used=45)
    assert s.over and s.exhausted and s.remaining == 0


def test_the_warning_no_longer_scales_with_the_replicate_count():
    """It used to have to: the budget was in runs and the question was in
    methods, so at three replicates the warning had to arrive earlier. Now the
    budget is already in methods and the threshold is the same either way -
    what changes with the replicate count is the injections those methods
    cost, which the warning states."""
    assert _status(used=37).warn                       # 3 methods left
    assert not _status(used=36).warn                   # 4 left
    assert _status(used=37, n_replicates=3).warn       # same, at 3 repeats
    assert not _status(used=36, n_replicates=3).warn
    assert "6 more injections" in _status(used=37).reason()
    assert "9 more injections" in _status(used=37, n_replicates=3).reason()


def test_an_extension_must_say_why():
    s = _status(used=40)
    for blank in ("", "   ", None):
        with pytest.raises(ValueError):
            B.plan_extension(s, 6, blank)
    with pytest.raises(ValueError):
        B.plan_extension(s, 0, "zero is not an extension")


def test_an_extension_records_where_it_happened():
    """Forty runs extended at run 39 is a campaign that ran out of road; the
    same extension at run 12 is a change of plan. The history must tell them
    apart."""
    limit, ev = B.plan_extension(_status(used=39), 6, "two candidates still tied")
    assert limit == 46
    assert (ev["from"], ev["to"], ev["n"], ev["used"]) == (40, 46, 6, 39)
    assert ev["reason"] == "two candidates still tied"


def test_a_closed_campaign_is_reopened_not_extended():
    """Otherwise reopening could be smuggled in as an extension and the record
    would not show that the ending was undone."""
    closed = _status(used=20, finished=True)
    with pytest.raises(ValueError):
        B.plan_extension(closed, 5, "more runs")
    with pytest.raises(ValueError):
        B.plan_finish(closed, "again")
    assert B.plan_reopen(closed, "closed by mistake")["event"] == "reopened"
    with pytest.raises(ValueError):
        B.plan_reopen(_status(used=20), "not closed")


# ── the stopping rule, against a real campaign folder ───────────────────────
REF_METHOD = [0.10, 0.30, 50.0, 45.0]


def _campaign(tmp_path, monkeypatch, **kw):
    from gradient_bench.store import campaign as C
    # never touch the user's real recent-campaigns registry from a test
    monkeypatch.setattr(C, "REGISTRY", str(tmp_path / "recent.json"))
    opts = dict(n_replicates=2, n_seed=5, run_budget=40)
    opts.update(kw)
    return C.create(str(tmp_path), {"name": "budget test", "analyst": "t"},
                    reference_method=REF_METHOD, **opts)


def _add_run(c, name, source, replicate, crf, run_order):
    """Write a run straight to the sheet - the picker is not what is under
    test here, and the budget counts rows regardless of what produced them."""
    from gradient_bench.store import sheet as SH
    return SH.append_run(
        c.workbook, method_name=name, source=source, replicate=replicate,
        params_abs=S.simple_gradient_abs(np.asarray(REF_METHOD)),
        row={"CRF": float(crf), "n_clean_peaks": int(crf),
             "hump_time_fraction": 0.0},
        run_order=run_order)


def test_a_new_campaign_records_its_budget(tmp_path, monkeypatch):
    c = _campaign(tmp_path, monkeypatch, run_budget=12)
    s = c.budget()
    assert (s.limit, s.original, s.used, s.finished) == (12, 12, 0, False)
    assert s.history[0]["event"] == "created" and s.history[0]["to"] == 12


def test_an_older_campaign_folder_gets_the_default_budget(tmp_path, monkeypatch):
    """A folder written before budgets existed is not a broken folder."""
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch)
    info = {k: v for k, v in c.info().items()
            if not str(k).startswith(("run_budget", "budget_", "finished",
                                      "ended_at", "end_reason"))}
    SH.write_campaign(c.workbook, info)
    assert c.run_budget() == B.DEFAULT_RUN_BUDGET
    assert c.budget().history == [] and not c.is_finished()


def test_a_budget_of_one_method_is_a_campaign_but_zero_is_not(tmp_path, monkeypatch):
    """Under an injection budget the anchor had to be paid for, so a two-run
    budget could not buy a two-replicate method and was refused at creation.
    The anchor is now free and a method costs one, so one method IS a campaign
    - a very short one - and only a budget that cannot afford a single method
    is refused."""
    c = _campaign(tmp_path, monkeypatch, run_budget=1, n_replicates=2)
    assert c.run_budget() == 1
    for bad in (0, -1, "", None, "many"):
        with pytest.raises(ValueError):
            _campaign(tmp_path / str(bad), monkeypatch, run_budget=bad,
                      n_replicates=2)


def test_the_gate_blocks_proposing_but_never_the_record(tmp_path, monkeypatch):
    from gradient_bench import loop as L
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch, run_budget=1)   # ONE method

    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 9.0, 1)
    # The instrument check spent no budget, so the one method is still there.
    assert L.next_step(c).action == "seed"

    _add_run(c, "m01r1", SH.SOURCE_DESIGN, 1, 7.0, 2)
    _add_run(c, "m01r2", SH.SOURCE_DESIGN, 2, 7.0, 3)

    step = L.next_step(c)                            # the one method is spent
    assert step.action == "budget" and step.blocked
    with pytest.raises(RuntimeError):
        L.suggest(c)

    # The workbook is NOT locked. A method already on the instrument records
    # both its replicates even though that ends the campaign one method OVER
    # its ceiling - the injections happened, and a sheet that disagreed with
    # what the instrument did would be worse than a budget overshoot.
    _add_run(c, "m02r1", SH.SOURCE_DESIGN, 1, 8.0, 4)
    _add_run(c, "m02r2", SH.SOURCE_DESIGN, 2, 8.0, 5)
    assert c.n_injections() == 5      # five rows on the instrument
    assert c.n_design_runs() == 4     # the anchor is not one of them
    assert c.n_methods() == 2         # and they are two distinct methods
    s = c.budget()
    assert s.over and s.exhausted and s.remaining == 0


def test_extending_releases_the_loop_and_stays_on_the_record(tmp_path, monkeypatch):
    from gradient_bench import loop as L
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch, run_budget=1)
    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 9.0, 1)
    _add_run(c, "m01r1", SH.SOURCE_DESIGN, 1, 7.0, 2)
    _add_run(c, "m01r2", SH.SOURCE_DESIGN, 2, 7.0, 3)
    assert L.next_step(c).blocked

    s = c.extend_budget(4, "two candidates still tied")
    assert (s.limit, s.original, s.extended_by, s.n_extensions) == (5, 1, 4, 1)
    assert s.history[-1]["used"] == 1        # extended at method 1, not at 0
    assert L.next_step(c).action == "seed"       # released


def test_finishing_and_reopening_are_both_recorded(tmp_path, monkeypatch):
    from gradient_bench import loop as L
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch, run_budget=40)
    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 9.0, 1)

    s = c.finish("separation is good enough for the assay")
    assert s.finished and s.state == "FINISHED" and s.ended_at
    step = L.next_step(c)
    assert step.action == "finished" and step.blocked
    with pytest.raises(RuntimeError):
        L.suggest(c)

    s = c.reopen("one more confirmation method")
    assert not s.finished and not c.is_finished()
    assert s.ended_at == "" and s.end_reason == ""
    assert [e["event"] for e in s.history] == ["created", "finished", "reopened"]
    assert L.next_step(c).action == "seed"


# ── re-scoring a campaign under changed picker settings ─────────────────────
def _campaign_with_traces(tmp_path, monkeypatch, methods, budget=40):
    """A campaign whose traces/ folder holds real chromatograms, recorded the
    way the loop records them."""
    import shutil
    from gradient_bench.core import picker as P
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch, run_budget=budget)
    os.makedirs(c.traces, exist_ok=True)
    for i, (src, source, overrides, humps) in enumerate(methods, start=1):
        is_ref = source == SH.SOURCE_REFERENCE
        name = SH.reference_name(i) if is_ref else SH.run_name(i, 1)
        dest = os.path.join(c.traces, f"{name}.TXT")
        shutil.copy(os.path.join(TRACE_DIR, src + ".TXT"), dest)
        res = P.analyse(dest, campaign=c.picker_config(), overrides=overrides or None,
                        manual_humps=humps or None, is_reference=is_ref)
        SH.append_run(c.workbook, method_name=name, source=source, replicate=1,
                      params_abs=S.simple_gradient_abs(np.asarray(REF_METHOD)),
                      row=P.record_row(res), run_order=i,
                      trace_file=os.path.basename(dest))
    return c


@pytest.mark.skipif(not HAVE_TRACES, reason="trace corpus not available")
def test_rescore_moves_the_scores_it_says_it_will(tmp_path, monkeypatch):
    from gradient_bench import rescore as RS
    from gradient_bench.store import sheet as SH
    c = _campaign_with_traces(tmp_path, monkeypatch, [
        ("method9", SH.SOURCE_DESIGN, None, None),
        ("method4", SH.SOURCE_DESIGN, None, None)])

    # method9 is the trace that needs snr 2.5 to reach its adjudicated 13 peaks
    pl = RS.plan(c, {"snr": 2.5})
    assert pl.changed_keys == ["snr"] and pl.ok
    by = {r.method: r for r in pl.rows}
    assert by["m01r1"].new_peaks == 13 and by["m01r1"].old_peaks == 8
    assert by["m01r1"].moved

    out = RS.commit(c, {"snr": 2.5}, reason="method9 needs a lower gate")
    assert out["rescored"] == 2
    df = c.data().set_index("method")
    assert int(df.loc["m01r1", "n_clean_peaks"]) == 13
    assert float(df.loc["m01r1", "CRF"]) == pytest.approx(13.0, abs=5e-3)
    # and the campaign's own default moved with it, so later runs agree
    assert c.picker_config()["snr"] == 2.5
    assert RS.history(c)[-1]["changed"]["snr"] == [5.0, 2.5]


@pytest.mark.skipif(not HAVE_TRACES, reason="trace corpus not available")
def test_rescore_keeps_a_hand_tuned_run_tuned(tmp_path, monkeypatch):
    """A per-run deviation is an analyst correcting a failed estimate. Wiping it
    on a re-score would silently undo human judgement."""
    from gradient_bench import rescore as RS
    from gradient_bench.store import sheet as SH
    c = _campaign_with_traces(tmp_path, monkeypatch, [
        ("method9", SH.SOURCE_DESIGN, {"snr": 2.5}, None),      # tuned by hand
        ("method2", SH.SOURCE_DESIGN, None, [[6.74, 13.85]])])  # hump drawn

    pl = RS.plan(c, {"hump_min_span_abs": 1.5})
    by = {r.method: r for r in pl.rows}
    assert by["m01r1"].overrides == {"snr": 2.5}     # recovered as a delta
    assert by["m01r1"].new_peaks == 13               # still tuned
    assert by["m02r1"].manual                        # drawn region travels too
    assert by["m02r1"].new_crf == pytest.approx(2.206, abs=5e-3)

    RS.commit(c, {"hump_min_span_abs": 1.5}, reason="test")
    df = c.data().set_index("method")
    assert int(df.loc["m01r1", "n_clean_peaks"]) == 13
    assert bool(df.loc["m01r1", "picker_deviates"])
    assert float(df.loc["m02r1", "CRF"]) == pytest.approx(2.206, abs=5e-3)


@pytest.mark.skipif(not HAVE_TRACES, reason="trace corpus not available")
def test_rescore_refuses_when_a_trace_is_missing(tmp_path, monkeypatch):
    """Re-scoring some rows and not others rebuilds the very mixture the
    operation exists to remove - and the sheet would look clean afterwards."""
    from gradient_bench import rescore as RS
    from gradient_bench.store import sheet as SH
    c = _campaign_with_traces(tmp_path, monkeypatch, [
        ("method4", SH.SOURCE_DESIGN, None, None),
        ("method7", SH.SOURCE_DESIGN, None, None)])
    os.remove(os.path.join(c.traces, "m02r1.TXT"))

    pl = RS.plan(c, {"snr": 4.0})
    assert not pl.ok and [r.method for r in pl.missing] == ["m02r1"]
    before = c.data().set_index("method")["CRF"].to_dict()
    with pytest.raises(RuntimeError, match="REFUSING TO RE-SCORE"):
        RS.commit(c, {"snr": 4.0}, reason="test")
    assert c.data().set_index("method")["CRF"].to_dict() == before   # nothing written


@pytest.mark.skipif(not HAVE_TRACES, reason="trace corpus not available")
def test_rescore_moves_the_whole_reference_series_together(tmp_path, monkeypatch):
    """A single reference may never be retuned - that dresses a picking artefact
    as drift. Moving all of them at once keeps the DIFFERENCES honest, which is
    all the drift monitor reads."""
    from gradient_bench import rescore as RS
    from gradient_bench.core import picker as P
    from gradient_bench.store import sheet as SH
    c = _campaign_with_traces(tmp_path, monkeypatch, [
        ("method1", SH.SOURCE_REFERENCE, None, None),
        ("method1", SH.SOURCE_REFERENCE, None, None)])
    with pytest.raises(ValueError, match="REFERENCE run"):
        P.analyse(os.path.join(c.traces, "ref01.TXT"), campaign=c.picker_config(),
                  overrides={"snr": 3.0}, is_reference=True)

    before = [r.crf for r in c.references()]
    RS.commit(c, {"snr": 3.0}, reason="whole series")
    after = [r.crf for r in c.references()]
    assert after != before
    assert after[0] == after[1]          # same trace, same settings, same score
    assert (after[1] - after[0]) == (before[1] - before[0])   # the delta is intact


def test_rescore_reads_back_what_the_picker_stamped():
    from gradient_bench import rescore as RS
    stamp = repr(sorted({"snr": 2.5, "arpls_lam": 1e5}.items()))
    assert RS.parse_stamped_config(stamp) == {"snr": 2.5, "arpls_lam": 1e5}
    assert RS.parse_stamped_config(float("nan")) == {}
    assert RS.parse_stamped_config("not a config") == {}
    assert RS.recover_overrides({"snr": 2.5, "arpls_lam": 1e5},
                                {"snr": 5.0, "arpls_lam": 1e5}) == {"snr": 2.5}
    assert RS.parse_manual_humps("6.74-13.85;19.25-22.39") == [[6.74, 13.85],
                                                              [19.25, 22.39]]
    assert RS.parse_manual_humps("") == []


def test_every_advanced_control_names_a_real_picker_setting():
    """A control whose key is a typo would be silently ignored at the point it
    matters most - the analyst turning it and nothing happening."""
    from gradient_bench.core import picker as P
    known = set(P.known_settings())
    for c in P.PRIMARY_CONTROLS + P.advanced_controls():
        assert c["key"] in known, c["key"]
        assert c["note"], c["key"]
    keys = [c["key"] for c in P.PRIMARY_CONTROLS + P.advanced_controls()]
    assert len(keys) == len(set(keys)), "a control is listed twice"


# ── phase 3: one picker, migration, metadata, the method page ───────────────
def test_the_picker_has_exactly_one_source_of_truth():
    """It used to sit beside the package with a second copy in v3/. Two copies
    of the file that defines what a peak IS can drift apart silently, and the
    app and the notebook then disagree about a chromatogram."""
    import hplc_picker
    from gradient_bench.core import hplc_picker as vendored
    from gradient_bench.core.picker import _find_hplc_picker
    assert hplc_picker is vendored, "the root shim must hand back the package module"
    assert _find_hplc_picker() is vendored
    root = os.path.dirname(PKG)
    assert not os.path.exists(os.path.join(root, "v3", "hplc_picker.py"))
    # the shim is a shim, not a copy
    assert os.path.getsize(os.path.join(root, "hplc_picker.py")) < 4000
    assert os.path.getsize(os.path.join(PKG, "core", "hplc_picker.py")) > 50000


def test_migration_only_ever_adds(tmp_path, monkeypatch):
    """A campaign folder is a record, not a cache: the measurements in it
    cannot be regenerated, so migration adds and never rewrites."""
    from gradient_bench.store import migrate as MIG, sheet as SH
    c = _campaign(tmp_path, monkeypatch, run_budget=12)
    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 9.0, 1)
    _add_run(c, "m01r1", SH.SOURCE_DESIGN, 1, 7.0, 2)
    before = c.data().set_index("method")["CRF"].to_dict()

    # wind the folder back to something an older build would have written
    frames = SH.read_all(c.workbook)
    frames["Data"] = frames["Data"].drop(columns=["trace_file"])
    camp = frames["Campaign"]
    frames["Campaign"] = camp[~camp["field"].isin(
        ["run_budget", "budget_original", "budget_history", "finished"])]
    SH._atomic_write(c.workbook, frames)
    with open(os.path.join(c.root, ".gradient_bench"), "w") as fh:
        fh.write("0.9.0\n")

    rep = MIG.inspect(c.root)
    assert rep.needed and rep.from_version == "0.9.0"
    kinds = {f.kind for f in rep.findings}
    assert {"column", "campaign_field", "marker"} <= kinds

    rep = MIG.apply(c.root)
    assert rep.applied and rep.backup
    assert not MIG.inspect(c.root).needed              # idempotent
    assert c.data().set_index("method")["CRF"].to_dict() == before   # untouched
    assert c.run_budget() == B.DEFAULT_RUN_BUDGET      # a default, not a guess
    assert "trace_file" in c.data().columns


def test_metadata_is_editable_but_a_material_change_is_recorded(tmp_path, monkeypatch):
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch)
    c.edit_info({"analyst": "A. Joseph"})              # no runs, no ceremony
    assert c.info()["analyst"] == "A. Joseph"

    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 9.0, 1)
    with pytest.raises(ValueError, match="needs a reason"):
        c.edit_info({"column": "XBridge C18 2.1x50"})
    c.edit_info({"column": "XBridge C18 2.1x50"}, reason="full part number")

    h = c.metadata_history()[-1]
    # An instrument check is not a "run": the count shown is 0, the injection
    # count is 1, and the reason was still required - the gate protects the
    # anchor even though the anchor does not tally. See HANDOFF §23.
    assert h["runs_recorded"] == 0 and h["injections_recorded"] == 1
    assert h["material"] is True
    assert h["changes"][0]["from"] == "" and h["reason"] == "full part number"
    with pytest.raises(ValueError, match="not editable"):
        c.edit_info({"run_budget": 99})


def test_duplicate_carries_settings_and_no_data(tmp_path, monkeypatch):
    """Same instrument, same column, new sample - and its own anchor, because
    an anchor from another campaign measures another week's machine."""
    from gradient_bench.store import campaign as C, sheet as SH
    c = _campaign(tmp_path, monkeypatch, run_budget=25)
    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 9.0, 1)
    d = C.duplicate_settings(c, str(tmp_path), {"name": "next sample",
                                                "sample": "mix B"})
    assert d.n_injections() == 0 and not d.references()
    assert list(map(float, d.reference_method())) == list(map(float, c.reference_method()))
    assert d.picker_config() == c.picker_config()
    assert (d.n_replicates(), d.n_seed(), d.run_budget()) == (
        c.n_replicates(), c.n_seed(), 25)
    assert d.info()["sample"] == "mix B"
    from gradient_bench import loop as L
    assert L.next_step(d).action == "anchor"           # its own zero


def test_archiving_moves_nothing(tmp_path, monkeypatch):
    from gradient_bench.store import campaign as C
    c = _campaign(tmp_path, monkeypatch)
    root, wb = c.root, c.workbook
    C.archive(c, True)
    assert C.is_archived(c) and c.summary()["archived"]
    assert os.path.isdir(root) and os.path.exists(wb)
    C.archive(c, False)
    assert not C.is_archived(c)


def test_the_method_page_states_the_method(tmp_path, monkeypatch):
    """It is the page you print when someone asks what you did, so the claims
    that matter must actually be in it."""
    from gradient_bench import setup_doc as SETUP
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch)
    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 9.0, 1)
    _add_run(c, "m01r1", SH.SOURCE_DESIGN, 1, 7.0, 2)
    _add_run(c, "m01r2", SH.SOURCE_DESIGN, 2, 11.0, 3)

    doc = SETUP.build(c)
    keys = [s["key"] for s in doc["sections"]]
    assert keys == ["identity", "objective", "space", "noise", "surrogate",
                    "acquisition", "drift", "stopping", "picker", "versions",
                    "record"]
    by = {s["key"]: s for s in doc["sections"]}

    # the constraint in BOTH forms, with the real constants
    assert any("end_phi >= start_phi" in m for m in by["space"]["math"])
    assert any("+0.90" in m and "-0.38" in m and "-0.08" in m
               for m in by["space"]["math"])

    # the noise floor derived with THIS campaign's numbers, not in the abstract
    sd, _, _ = c.sigma_crf()
    assert np.isfinite(sd) and sd > 0
    math_txt = " ".join(by["noise"]["math"])
    assert "floor_z" in math_txt and f"{sd:.3f}" in math_txt
    assert "measured within-method sd" in math_txt

    # every picker field, marked for whether the campaign states it
    tbl = by["picker"]["tables"][0]
    assert len(tbl["rows"]) == len(P_defaults())
    assert sum(1 for r in tbl["rows"] if r[3] == "SET") == len(c.picker_config())

    # versions recorded at creation vs installed now
    libs = [r[0] for r in by["versions"]["tables"][0]["rows"]]
    assert {"torch", "botorch", "gpytorch", "numpy", "scipy"} <= set(libs)


def P_defaults():
    from gradient_bench.core import picker as P
    return P.picker_defaults()


# ── phase 4: drift, surfaced ────────────────────────────────────────────────
from gradient_bench.core import health as H                 # noqa: E402


def _ref_rows(**series):
    """Reference rows from {descriptor: [values]}, run orders 1, 6, 11, ..."""
    n = len(next(iter(series.values())))
    return [dict({"run_order": 1.0 + 5 * i},
                 **{k: v[i] for k, v in series.items()}) for i in range(n)]


def test_a_smooth_trend_and_a_step_are_told_apart():
    """A slope is something wearing out. A step is something that was CHANGED
    between two runs. Distinguishing them is most of the diagnostic value."""
    worn = H.channel_reports(_ref_rows(
        trace_mean_peak_width_min=[0.10, 0.13, 0.16, 0.19],
        trace_median_peak_width_min=[0.10, 0.13, 0.16, 0.19]))
    width = {c.key: c for c in worn}["width"]
    assert width.verdict == "drifting"
    assert width.descriptors[0].monotone and width.descriptors[0].step_at is None
    assert "column" in width.implicates

    swapped = H.channel_reports(_ref_rows(
        trace_first_peak_rt=[3.00, 3.01, 4.20, 4.21],
        trace_last_peak_rt=[20.0, 20.0, 21.2, 21.2]))
    ret = {c.key: c for c in swapped}["retention"]
    assert ret.verdict == "step"
    assert ret.descriptors[0].step_at == 11.0        # the third run
    assert "STEP at run 11" in ret.headline


def test_a_steady_channel_says_so_rather_than_hunting_for_a_cause():
    r = {c.key: c for c in H.channel_reports(_ref_rows(
        trace_mean_tailing=[1.20, 1.21, 1.19, 1.22],
        trace_total_area=[1000.0, 1005.0, 995.0, 1002.0]))}
    assert r["tailing"].verdict == "steady" and r["response"].verdict == "steady"
    assert H.summary(list(r.values()))["n_flagged"] == 0


def test_a_changed_acquisition_is_not_graded_on_a_tolerance():
    """If the sampling rate or run length changed, the traces are not
    comparable measurements and every other panel reads across a break."""
    r = {c.key: c for c in H.channel_reports(_ref_rows(
        trace_pts_per_min=[60.0, 60.0, 30.0],
        trace_tmax_min=[25.0, 25.0, 25.0],
        trace_n_points=[1500.0, 1500.0, 750.0]))}
    ch = r["acquisition"]
    assert ch.verdict == "changed"
    assert "THE ACQUISITION CHANGED" in ch.headline
    assert H.summary(list(r.values()))["line"].startswith("The acquisition changed")
    # a 1.5% wobble in retention would be "steady", but any acquisition change counts
    steady = {c.key: c for c in H.channel_reports(_ref_rows(
        trace_pts_per_min=[60.0, 60.0], trace_tmax_min=[25.0, 25.0],
        trace_n_points=[1500.0, 1500.0]))}
    assert steady["acquisition"].verdict == "steady"


def test_channels_need_two_reference_runs_before_saying_anything():
    r = {c.key: c for c in H.channel_reports(_ref_rows(
        trace_mean_peak_width_min=[0.10]))}
    assert r["width"].verdict == "unmeasured"
    assert "two reference runs" in r["width"].headline


def test_the_phantom_incumbent_reaches_the_campaign(tmp_path, monkeypatch):
    """An early run carrying a drift bonus that nothing later can beat: the
    campaign concludes it has converged and adopts the second-best chemistry."""
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch, run_budget=40)
    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 12.0, 1)
    _add_run(c, "m01r1", SH.SOURCE_DESIGN, 1, 11.0, 2)     # early, on a good day
    _add_run(c, "ref02", SH.SOURCE_REFERENCE, 1, 6.0, 20)  # instrument decayed
    _add_run(c, "m02r1", SH.SOURCE_DESIGN, 1, 10.0, 21)    # later, on a bad day

    inc = c.incumbent()
    assert inc.raw_method == "m01r1"          # raw says the early one
    assert inc.adj_method == "m02r1"          # adjusted says the later one
    assert inc.phantom and "PHANTOM INCUMBENT" in inc.warning()


def test_drift_adjustment_never_reaches_the_model(tmp_path, monkeypatch):
    """The toggle changes what is REPORTED as best. Feeding adjusted scores to
    the GP would bake an unverifiable assumption into every future proposal."""
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch)
    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 12.0, 1)
    _add_run(c, "m01r1", SH.SOURCE_DESIGN, 1, 11.0, 2)
    _add_run(c, "ref02", SH.SOURCE_REFERENCE, 1, 6.0, 20)
    _add_run(c, "m02r1", SH.SOURCE_DESIGN, 1, 10.0, 21)

    _, raw_before, _, _ = c.training_set()
    assert not c.use_drift_adjusted()
    c.set_toggle("use_drift_adjusted", True)
    assert c.use_drift_adjusted()
    _, raw_after, _, _ = c.training_set()
    assert np.array_equal(raw_before, raw_after)
    assert sorted(raw_after.tolist()) == [10.0, 11.0]      # raw, both times

    with pytest.raises(ValueError, match="is not a toggle"):
        c.set_toggle("snr", True)


def test_every_health_descriptor_is_one_the_picker_actually_records():
    """A channel reading a descriptor the picker never produces would sit at
    'unmeasured' for ever and nobody would know why."""
    from gradient_bench.core.picker import _find_hplc_picker
    recorded = set(_find_hplc_picker().RECORD_KEYS)
    used = {d["key"] for ch in H.CHANNELS for d in ch["descriptors"]}
    assert used <= recorded, sorted(used - recorded)
    # and every channel names what it implicates and what to do about it
    for ch in H.CHANNELS:
        assert ch["implicates"] and ch["why"] and ch["then"]


# ── the kernel, and what the chemist already knows ──────────────────────────
def test_a_new_campaign_records_matern52(tmp_path, monkeypatch):
    """The kernel is a per-campaign fact. A campaign fitted under Matern and
    refitted under RBF is not the same campaign."""
    from gradient_bench.core import optimiser as O
    c = _campaign(tmp_path, monkeypatch)
    assert c.kernel() == "matern52" == O.DEFAULT_KERNEL
    assert c.config()["kernel"] == "matern52"
    with pytest.raises(ValueError):
        _campaign(tmp_path, monkeypatch, kernel="squared-exponential")


def test_a_campaign_written_before_the_setting_reads_as_default(tmp_path, monkeypatch):
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch)
    cfg = c.config(); cfg.pop("kernel")
    SH.write_config(c.workbook, cfg)
    assert c.kernel() == "default"       # what it was actually fitted with


def test_the_kernel_choice_is_validated_not_guessed():
    from gradient_bench.core import optimiser as O
    assert set(O.KERNELS) == {"matern52", "matern32", "rbf", "default"}
    with pytest.raises(ValueError):
        O.make_covar_module(4, "matern")
    assert O.make_covar_module(4, "default") is None      # no torch needed


def test_matern_keeps_botorchs_own_priors():
    """The point of using BoTorch's constructor rather than building a kernel
    by hand: choosing Matern must not cost the regularisation."""
    pytest.importorskip("botorch")
    from gradient_bench.core import optimiser as O
    cm = O.make_covar_module(4, "matern52")
    assert type(cm).__name__ == "MaternKernel" and float(cm.nu) == 2.5
    assert int(cm.ard_num_dims) == 4
    assert not hasattr(cm, "outputscale")          # same as the RBF default
    prior = cm.lengthscale_prior
    assert type(prior).__name__ == "LogNormalPrior"
    assert float(prior.loc) == pytest.approx(np.sqrt(2) + np.log(4) / 2, abs=1e-6)
    assert float(prior.scale) == pytest.approx(np.sqrt(3), abs=1e-6)
    assert float(cm.raw_lengthscale_constraint.lower_bound) == pytest.approx(0.025)
    rbf = O.make_covar_module(4, "rbf")
    assert type(rbf).__name__ == "RBFKernel"


# ── phase 5: the model diagnostics ──────────────────────────────────────────
from gradient_bench.store import history as HIST                # noqa: E402


class _Step:
    """The fields `row_from_step` reads. Stands in for a real fit so the
    history layer is testable without torch."""
    def __init__(self, n, before, after, noise=1.0, ls=None, run_order=None):
        self.n_runs = n
        self.mean_sd_before, self.mean_sd_after = before, after
        self.max_sd_before, self.max_sd_after = before * 1.4, after * 1.4
        self.mean_abs_dmu = 0.2
        self.sd_incumbent_before, self.sd_incumbent_after = before * .6, after * .6
        self.sd_newpoint_before, self.sd_newpoint_after = before * 1.2, after * .4
        self.noise_sd_crf, self.noise_floor_z = noise, 0.08
        self.sigma_pooled_crf, self.best_crf = 1.2, 11.0
        self.lengthscales = ls or {"start_phi": .55, "end_phi": .73,
                                   "duration_min": .62, "T": .189}
        self.run_order = float(n if run_order is None else run_order)

    @property
    def d_mean_sd(self):
        return self.mean_sd_before - self.mean_sd_after

    def reading(self):
        return ("Mean posterior SD fell." if self.d_mean_sd > 0
                else "Mean posterior SD did NOT fall.")


def test_the_history_round_trips(tmp_path):
    root = str(tmp_path)
    assert HIST.read(root) == ([], [])          # a missing file is empty, not an error
    HIST.append(root, HIST.row_from_step(_Step(3, 3.9, 3.4), method="m03r1",
                                         kernel="matern52"))
    HIST.append(root, HIST.row_from_step(_Step(4, 3.4, 3.0), method="m04r1",
                                         kernel="matern52"))
    rows, cols = HIST.read(root)
    assert len(rows) == 2 and cols[:2] == ["recorded_at", "run_order"]
    assert rows[1]["mean_sd_after"] == 3.0 and rows[1]["method"] == "m04r1"
    assert rows[1]["d_mean_sd"] == pytest.approx(0.4)
    assert HIST.latest(root)["method"] == "m04r1"


def test_the_history_widens_rather_than_dropping_a_column(tmp_path):
    """Switching run-order modelling on mid-campaign adds a lengthscale. The
    file must gain the column, not silently discard the number."""
    root = str(tmp_path)
    HIST.append(root, HIST.row_from_step(_Step(3, 3.9, 3.4), method="m03r1"))
    five = {"start_phi": .55, "end_phi": .73, "duration_min": .62, "T": .19,
            "run_order": 0.18}
    HIST.append(root, HIST.row_from_step(_Step(4, 3.4, 3.0, ls=five),
                                         method="m04r1", model_dim=5))
    rows, cols = HIST.read(root)
    assert "ls_run_order" in cols and len(rows) == 2
    assert rows[0]["ls_run_order"] is None      # the earlier fit never had one
    assert rows[1]["ls_run_order"] == pytest.approx(0.18)


def test_a_damaged_history_is_read_as_far_as_it_parses(tmp_path):
    """It is a derived file. A bad line must not take the diagnostics down."""
    root = str(tmp_path)
    HIST.append(root, HIST.row_from_step(_Step(3, 3.9, 3.4), method="m03r1"))
    with open(HIST.path_for(root), "a") as fh:
        fh.write("garbage,row\n")
    t = HIST.trend(root)
    assert t["n"] == 1                          # the unusable row carries no point
    assert HIST.ard_table(root)                 # and the readable one still reads


def test_the_trend_says_which_way_it_is_going(tmp_path):
    root = str(tmp_path)
    for i, (b, a) in enumerate([(3.9, 3.4), (3.4, 3.0), (3.0, 2.7)], start=3):
        HIST.append(root, HIST.row_from_step(_Step(i, b, a), method=f"m{i:02d}r1"))
    assert "campaign is learning" in HIST.trend(root)["reading"]

    root2 = str(tmp_path / "rising")
    for i, (b, a) in enumerate([(2.0, 2.2), (2.2, 2.6)], start=3):
        HIST.append(root2, HIST.row_from_step(_Step(i, b, a), method=f"m{i:02d}r1"))
    assert "RISEN" in HIST.trend(root2)["reading"]
    assert "no fit has been recorded" in HIST.trend(str(tmp_path / "empty"))["reading"].lower()


def test_the_ard_reading_works_on_a_stored_lengthscale(tmp_path):
    """The reading has to be available from the history, not only from a live
    fit - a diagnostic that needs a refit is one nobody looks at."""
    from gradient_bench.core import optimiser as O
    assert "floor" in O.lengthscale_reading("T", 0.02)
    assert "inert" in O.lengthscale_reading("T", 40.0)
    assert "quickly" in O.lengthscale_reading("T", 0.1)
    assert O.lengthscale_reading("T", 0.8) == "active"
    assert "drift" in O.lengthscale_reading("run_order", 0.1)
    assert "unreadable" in O.lengthscale_reading("T", float("nan"))

    root = str(tmp_path)
    HIST.append(root, HIST.row_from_step(_Step(3, 3.9, 3.4)))
    table = {a["name"]: a for a in HIST.ard_table(root)}
    assert table["T"]["value"] == pytest.approx(0.189)
    assert "quickly" in table["T"]["reading"]


def test_a_failed_fit_never_costs_a_recorded_run(tmp_path, monkeypatch):
    """The workbook write is the thing that must not be lost. torch is absent
    in this environment, so this is the real failure path, not a simulated one."""
    from gradient_bench import loop as L
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch)
    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 9.0, 1)
    for i, crf in enumerate([7.0, 8.0, 9.0], start=1):
        _add_run(c, f"m{i:02d}r1", SH.SOURCE_DESIGN, 1, crf, i + 1)

    written = [{"method": "m03r1", "source": SH.SOURCE_DESIGN}]
    out = L._record_uncertainty(c, written)
    assert out is not None and ("failed" in out or "reading" in out)
    if "failed" in out:
        assert "the workbook is intact" in out["note"]
    assert c.n_injections() == 4                 # the runs are still there

    # a reference-only record has nothing to teach the model and is skipped
    assert L._record_uncertainty(
        c, [{"method": "ref02", "source": SH.SOURCE_REFERENCE}]) is None


def test_fewer_than_three_design_runs_is_reported_not_attempted(tmp_path, monkeypatch):
    from gradient_bench import loop as L
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch)
    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 9.0, 1)
    _add_run(c, "m01r1", SH.SOURCE_DESIGN, 1, 7.0, 2)
    out = L._record_uncertainty(c, [{"method": "m01r1",
                                     "source": SH.SOURCE_DESIGN}])
    assert "skipped" in out and "three" in out["skipped"]


# ── fixes from the phase-5 audit ────────────────────────────────────────────
def test_the_chart_and_the_gate_use_the_same_drift_limits(tmp_path, monkeypatch):
    """The limits were written to Config, printed on the Method page and drawn
    on the control chart while the verdict still used the module defaults - so
    a campaign banded at 5.0 could HALT at 3.0."""
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch)
    cfg = c.config()
    cfg["drift_watch_delta"] = 2.5
    cfg["drift_halt_delta"] = 5.0
    cfg["drift_monotone_k"] = 4
    SH.write_config(c.workbook, cfg)

    lim = c.drift_limits()
    assert (lim["watch"], lim["halt"], lim["monotone_k"]) == (2.5, 5.0, 4)
    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 10.0, 1)
    _add_run(c, "ref02", SH.SOURCE_REFERENCE, 1, 6.0, 20)   # -4.0 from anchor
    # under the module defaults this is a HALT; under this campaign's own
    # limits it is only a WATCH, and the campaign's limits are what count
    assert c.verdict().verdict == "WATCH"


def test_the_reported_best_follows_the_drift_toggle(tmp_path, monkeypatch):
    """Otherwise switching drift adjustment on changes a caption and nothing
    else. The raw value still travels beside it."""
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch)
    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 12.0, 1)
    _add_run(c, "m01r1", SH.SOURCE_DESIGN, 1, 11.0, 2)
    _add_run(c, "ref02", SH.SOURCE_REFERENCE, 1, 6.0, 20)
    _add_run(c, "m02r1", SH.SOURCE_DESIGN, 1, 10.0, 21)

    s = c.summary()
    assert s["best_crf"] == pytest.approx(11.0) and s["best_method"] == "m01r1"
    assert s["best_crf_raw"] == pytest.approx(11.0) and s["phantom"]

    c.set_toggle("use_drift_adjusted", True)
    s = c.summary()
    assert s["best_method"] == "m02r1" and s["best_crf"] > 11.0
    assert s["best_crf_raw"] == pytest.approx(11.0)   # raw still available
    assert s["drift_adjusted"] is True
    # and the model is still fitted on raw scores
    _, y, _, _ = c.training_set()
    assert sorted(y.tolist()) == [10.0, 11.0]


def test_rescore_survives_a_blank_peak_count(tmp_path, monkeypatch):
    """`int(nan or -1)` raises, because NaN is truthy - and it would take the
    whole preview down, not just one row. A real zero must stay zero."""
    from gradient_bench import rescore as RS
    assert RS._int_or(float("nan"), -1) == -1
    assert RS._int_or(None, -1) == -1
    assert RS._int_or(0, -1) == 0            # not -1
    assert RS._int_or("7", -1) == 7


def test_a_reference_row_cannot_reach_the_health_channels(tmp_path, monkeypatch):
    """HANDOFF claims there is no way to hand design rows to `health` by
    accident. That gate has to be in the store, not in the API."""
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch)
    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 9.0, 1)
    _add_run(c, "m01r1", SH.SOURCE_DESIGN, 1, 7.0, 2)
    rows = c.reference_rows()
    assert [r["method"] for r in rows] == ["ref01"]
    assert all(r["source"] == SH.SOURCE_REFERENCE for r in rows)


def test_the_history_never_appends_under_a_mismatched_header(tmp_path):
    """Appending under a header that does not cover the row writes values into
    the wrong columns - silently, and unrecoverably once more rows follow."""
    root = str(tmp_path)
    # a header-only file, as a truncated first write would leave
    with open(HIST.path_for(root), "w") as fh:
        fh.write(",".join(HIST.BASE_COLUMNS) + "\n")
    HIST.append(root, HIST.row_from_step(_Step(3, 3.9, 3.4), method="m03r1"))
    rows, cols = HIST.read(root)
    assert len(rows) == 1
    assert "ls_T" in cols and rows[0]["ls_T"] == pytest.approx(0.189)
    assert rows[0]["mean_sd_after"] == 3.4     # not shifted into another column


def test_lengthscale_columns_follow_model_input_order(tmp_path):
    root = str(tmp_path)
    HIST.append(root, HIST.row_from_step(_Step(3, 3.9, 3.4)))
    _, cols = HIST.read(root)
    ls = [c for c in cols if c.startswith("ls_")]
    assert ls == ["ls_start_phi", "ls_end_phi", "ls_duration_min", "ls_T"]


def test_a_falling_reference_is_the_verdicts_own_judgement():
    v = D.drift_verdict([D.ReferenceRun(1, 10.0), D.ReferenceRun(11, 8.0)])
    assert v.falling
    assert not D.drift_verdict([D.ReferenceRun(1, 10.0),
                                D.ReferenceRun(11, 10.4)]).falling
    assert not D.drift_verdict([]).falling      # nothing measured, not falling


# ── phase 6: results and reporting ──────────────────────────────────────────
def test_methods_group_replicates_and_report_their_own_spread(tmp_path, monkeypatch):
    """A two-replicate method is one decision and one score. The within-method
    sd is the only direct read on noise the campaign has."""
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch)
    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 9.0, 1)
    _add_run(c, "m01r1", SH.SOURCE_DESIGN, 1, 7.0, 2)
    _add_run(c, "m01r2", SH.SOURCE_DESIGN, 2, 9.0, 3)
    _add_run(c, "m02r1", SH.SOURCE_DESIGN, 1, 11.0, 4)

    ms = {m["method"]: m for m in c.methods()}
    assert set(ms) == {"m01", "m02"}                  # references are excluded
    assert ms["m01"]["n"] == 2 and ms["m01"]["crf_mean"] == pytest.approx(8.0)
    assert ms["m01"]["crf_sd"] == pytest.approx(np.std([7.0, 9.0], ddof=1))
    assert ms["m02"]["n"] == 1 and ms["m02"]["crf_sd"] is None   # not zero
    assert ms["m01"]["runs"] == ["m01r1", "m01r2"]


def test_a_tied_pair_is_reported_as_tied_not_as_zero_noise(tmp_path, monkeypatch):
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch)
    _add_run(c, "m01r1", SH.SOURCE_DESIGN, 1, 13.0, 1)
    _add_run(c, "m01r2", SH.SOURCE_DESIGN, 2, 13.0, 2)
    m = c.methods()[0]
    assert m["tied"] and m["crf_sd"] == pytest.approx(0.0)


@pytest.mark.skipif(not HAVE_TRACES, reason="trace corpus not available")
def test_a_stored_trace_reproduces_its_own_recorded_score(tmp_path, monkeypatch):
    """The chromatogram beside a score must be the one that PRODUCED it -
    including per-run tuning - or the picture and the number disagree."""
    from gradient_bench import rescore as RS
    from gradient_bench.store import sheet as SH
    c = _campaign_with_traces(tmp_path, monkeypatch, [
        ("method9", SH.SOURCE_DESIGN, {"snr": 2.5}, None),     # tuned by hand
        ("method2", SH.SOURCE_DESIGN, None, [[6.74, 13.85]])]) # hump drawn

    for name, want in (("m01r1", 13.0), ("m02r1", 2.206)):
        res, row, exact = RS.analyse_as_recorded(c, name)
        assert exact                       # both rows carry a config stamp
        assert res.crf == pytest.approx(float(row["CRF"]), abs=5e-3)
        assert res.crf == pytest.approx(want, abs=5e-3)

    # changing the campaign settings afterwards must NOT change what is shown
    c_cfg = c.config()
    c_cfg["picker_config"] = json.dumps({"snr": 9.0}, sort_keys=True)
    SH.write_config(c.workbook, c_cfg)
    res, row, exact = RS.analyse_as_recorded(c, "m01r1")
    assert exact and res.crf == pytest.approx(13.0, abs=5e-3)

    with pytest.raises(ValueError, match="not a run"):
        RS.analyse_as_recorded(c, "m99r1")


@pytest.mark.skipif(not HAVE_TRACES, reason="trace corpus not available")
def test_an_unstamped_row_is_flagged_not_silently_replayed(tmp_path, monkeypatch):
    """Without a stamp the row can only be shown under the campaign's CURRENT
    settings, which may not reproduce the recorded score. Saying so is the
    difference between a diagnostic and a picture that contradicts the number
    printed beside it."""
    from gradient_bench import rescore as RS
    from gradient_bench.store import sheet as SH
    c = _campaign_with_traces(tmp_path, monkeypatch, [
        ("method9", SH.SOURCE_DESIGN, {"snr": 2.5}, None)])

    frames = SH.read_all(c.workbook)
    frames["Data"].loc[0, "picker_config"] = ""        # an older workbook
    SH._atomic_write(c.workbook, frames)
    cfg = c.config()
    cfg["picker_config"] = json.dumps({"snr": 15.0}, sort_keys=True)
    SH.write_config(c.workbook, cfg)

    res, row, exact = RS.analyse_as_recorded(c, "m01r1")
    assert exact is False
    assert res.crf != pytest.approx(float(row["CRF"]), abs=5e-3)


@pytest.mark.skipif(not HAVE_TRACES, reason="trace corpus not available")
def test_the_export_carries_its_own_provenance(tmp_path, monkeypatch):
    """A number without a chain of custody is what this exists to prevent."""
    import pandas as pd
    from gradient_bench import export as EXP
    from gradient_bench.store import sheet as SH
    c = _campaign_with_traces(tmp_path, monkeypatch, [
        ("method1", SH.SOURCE_REFERENCE, None, None),
        ("method4", SH.SOURCE_DESIGN, None, None)], budget=12)
    c.extend_budget(6, "two candidates still tied")
    c.edit_info({"column": "XBridge C18 2.1x50"}, reason="full part number")

    xl_path = EXP.workbook(c)
    assert os.path.exists(xl_path)
    sheets = pd.ExcelFile(xl_path).sheet_names
    for needed in ("Data", "Campaign", "Config", "Methods", "Budget history",
                   "Metadata history", "Export"):
        assert needed in sheets, needed
    exp = pd.read_excel(xl_path, sheet_name="Export").set_index("field")["value"]
    # One reference and one design method: the budget spends the METHOD only,
    # so runs_used is 1 against a budget of 18 methods (HANDOFF section 24).
    assert exp["run_budget"] == 18 and exp["runs_used"] == 1
    assert exp["kernel"] == "matern52"
    bh = pd.read_excel(xl_path, sheet_name="Budget history")
    assert set(bh["event"]) == {"created", "extended"}

    # the original record is never the thing being edited
    before = os.path.getmtime(c.workbook)
    EXP.workbook(c, os.path.join(str(tmp_path), "again.xlsx"))
    assert os.path.getmtime(c.workbook) == before
    assert len(pd.read_excel(c.workbook, sheet_name="Data")) == 2


@pytest.mark.skipif(not HAVE_TRACES, reason="trace corpus not available")
def test_the_pdf_report_is_a_real_pdf(tmp_path, monkeypatch):
    from gradient_bench import export as EXP
    from gradient_bench.store import sheet as SH
    c = _campaign_with_traces(tmp_path, monkeypatch, [
        ("method1", SH.SOURCE_REFERENCE, None, None),
        ("method4", SH.SOURCE_DESIGN, None, None),
        ("method12", SH.SOURCE_DESIGN, None, None)])
    p = EXP.pdf(c)
    raw = open(p, "rb").read()
    assert raw[:5] == b"%PDF-"
    assert raw.count(b"/Type /Page") - raw.count(b"/Type /Pages") == 5
    assert os.path.getsize(p) > 10_000


def test_an_empty_campaign_still_exports(tmp_path, monkeypatch):
    """A campaign with no runs must not be the thing that makes export raise."""
    from gradient_bench import export as EXP
    c = _campaign(tmp_path, monkeypatch)
    assert os.path.exists(EXP.workbook(c))
    p = EXP.pdf(c)
    assert open(p, "rb").read(5) == b"%PDF-"


def test_a_drawn_hump_is_not_recorded_as_a_picker_tune():
    """Two different acts: a tune moves a threshold under every run's reading,
    a drawn region annotates one chromatogram. The record has to say which."""
    from gradient_bench.core import picker as P
    _, _, dev = P.resolve_config({}, {"hump_override": [(1.0, 2.0)]})
    assert dev is False
    _, _, dev = P.resolve_config({}, {"snr": 2.5})
    assert dev is True
    _, _, dev = P.resolve_config({}, {"snr": 2.5,
                                      "hump_override": [(1.0, 2.0)]})
    assert dev is True
    # and a reference still refuses BOTH, because either is a per-run change
    with pytest.raises(ValueError, match="REFERENCE run"):
        P.resolve_config({}, {"hump_override": [(1.0, 2.0)]}, is_reference=True)


@pytest.mark.skipif(not HAVE_TRACES, reason="trace corpus not available")
def test_the_two_flags_are_recorded_separately(tmp_path, monkeypatch):
    from gradient_bench.store import sheet as SH
    c = _campaign_with_traces(tmp_path, monkeypatch, [
        ("method9", SH.SOURCE_DESIGN, {"snr": 2.5}, None),      # tuned only
        ("method2", SH.SOURCE_DESIGN, None, [[6.74, 13.85]])])  # drawn only
    df = c.data().set_index("method")
    assert SH.truthy(df.loc["m01r1", "picker_deviates"])
    assert not SH.has_text(df.loc["m01r1", "manual_humps"])
    assert not SH.truthy(df.loc["m02r1", "picker_deviates"])
    assert SH.has_text(df.loc["m02r1", "manual_humps"])
    ms = {m["method"]: m for m in c.methods()}
    assert ms["m01"]["deviates"] and not ms["m01"]["manual_humps"]
    assert ms["m02"]["manual_humps"] and not ms["m02"]["deviates"]


def test_the_sheets_flag_columns_have_one_reading():
    from gradient_bench.store import sheet as SH
    for yes in (True, np.True_, "True", "true", 1, 1.0, "1", "yes"):
        assert SH.truthy(yes), yes
    for no in (False, "False", "", " ", 0, 0.0, None, float("nan"), "nan"):
        assert not SH.truthy(no), no
    assert SH.has_text("6.74-13.85") and not SH.has_text(float("nan"))
    assert not SH.has_text("") and not SH.has_text("nan")


def test_the_dominated_noise_floor_reads_the_same_live_and_from_history():
    """floor_z >= 1 must mean the same thing whether it has just been computed
    or was read back off a history row months later - that is why the
    threshold is a pure function and not only a property of a live fit."""
    from gradient_bench.core import optimiser as O
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    sd = float(np.std(y, ddof=1))

    live = O.noise_floor_std(y, 2 * sd)         # noise twice the spread
    assert live.dominated
    assert O.floor_is_dominated(live.floor_z)   # same verdict off the number
    assert "cannot tell these methods apart" in live.reading

    ok_fit = O.noise_floor_std(y, sd / 4)
    assert not ok_fit.dominated and not O.floor_is_dominated(ok_fit.floor_z)
    assert "well above the noise" in ok_fit.reading

    # exactly 1.0 is dominated, and a missing value is never a false alarm
    assert O.floor_is_dominated(1.0) and not O.floor_is_dominated(0.999)
    assert not O.floor_is_dominated(None) and not O.floor_is_dominated("")
    assert O.floor_reading(None) == "not recorded"


def test_injections_and_design_runs_are_different_numbers(tmp_path, monkeypatch):
    """The two counts must never be published under one name again.

    A campaign with one instrument check and four design runs is a campaign of
    5 injections and 4 runs. The budget spends 5; the analyst has 4. Both are
    right, and the bug was calling both of them `n_runs` - the status bar said
    4 while the metadata banner said 5 on the very same campaign.
    """
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch)
    _add_run(c, "ref01", SH.SOURCE_REFERENCE, 1, 7.0, 1)
    _add_run(c, "m01r1", SH.SOURCE_DESIGN, 1, 5.0, 5)
    _add_run(c, "m01r2", SH.SOURCE_DESIGN, 2, 5.0, 5)
    _add_run(c, "m02r1", SH.SOURCE_DESIGN, 3, 3.0, 3)
    _add_run(c, "m02r2", SH.SOURCE_DESIGN, 4, 3.0, 3)

    assert c.n_injections() == 5
    assert c.n_design_runs() == 4
    assert c.n_methods() == 2                          # m01 and m02
    # The budget spends METHODS (HANDOFF section 24); injections ride along as
    # the bench-time figure and gate nothing.
    assert c.budget().used == c.n_methods()
    assert c.budget().injections == c.n_injections()
    assert c.summary()["n_runs"] == c.n_design_runs()   # the bar shows runs

    # and the metadata log agrees with the bar, while still keeping the
    # injection count so the record knows the bench time under the old value
    c.edit_info({"sample": "something else"}, reason="testing the counts")
    entry = c.metadata_history()[-1]
    assert entry["runs_recorded"] == 4
    assert entry["injections_recorded"] == 5


# ── phase 7: usable by someone who has never done this ──────────────────────
def test_the_bundled_corpus_is_where_the_package_says_it_is():
    """`testdata.py` is what verify_install and replay_corpus locate the traces
    with. It resolves relative to the package, not the working directory, so a
    copied folder still finds its own chromatograms."""
    from gradient_bench import testdata
    d = testdata.traces_dir()
    assert os.path.isabs(d)
    assert d.endswith(os.path.join("testdata", "traces"))
    # available() is a floor, not a count: a folder holding one stray file is a
    # broken install, and it should say so rather than report a small corpus.
    assert testdata.available() is (
        os.path.isdir(d)
        and len([f for f in os.listdir(d) if f.lower().endswith(".txt")]) >= 8)



def test_every_glossary_key_used_in_the_app_exists():
    """A <Term k="..."> naming a key that is not defined renders a tooltip with
    nothing in it - the failure is silent and only the reader notices."""
    import re
    root = os.path.dirname(PKG)
    src = ""
    # version 2 splits the interface into tabs/ and components/; every .jsx
    # under src/ is scanned so a term used in any of them is checked
    for dirpath, _, files in os.walk(os.path.join(root, "frontend", "src")):
        for name in files:
            if name.endswith(".jsx"):
                src += open(os.path.join(dirpath, name), encoding="utf-8").read()
    used = set(re.findall(r'<Term\s+k="([^"]+)"', src))
    gl = open(os.path.join(root, "frontend", "src", "glossary.js"),
              encoding="utf-8").read()
    defined = set(re.findall(r"^\s{2}'?([A-Za-z_][\w \-]*)'?:", gl, re.M))
    missing = sorted(t for t in used if t not in defined)
    assert not missing, f"used in the UI but not defined: {missing}"
    assert len(used) >= 25, f"only {len(used)} terms wired - the rule is dense"


def test_python_side_terms_name_real_glossary_entries():
    """Labels rendered from Python can carry a glossary key too. A typo there
    is the same silent failure, one layer further away."""
    import re
    from gradient_bench.core import picker as P
    root = os.path.dirname(PKG)
    gl = open(os.path.join(root, "frontend", "src", "glossary.js"),
              encoding="utf-8").read()
    defined = set(re.findall(r"^\s{2}'?([A-Za-z_][\w \-]*)'?:", gl, re.M))
    keys = [c.get("term") for c in P.PRIMARY_CONTROLS + P.advanced_controls()
            if c.get("term")]
    assert keys, "no picker control carries a glossary key"
    assert all(k in defined for k in keys), [k for k in keys if k not in defined]


def test_the_install_check_reports_rather_than_raising():
    """It runs on a machine that may be broken in any way, so it must survive
    every check failing and still print something actionable."""
    from gradient_bench.scripts import verify_install as V
    ran = [c for c in V.CHECKS if c.name in
           ("python", "core libraries", "the objective", "design space")]
    for c in ran:
        assert c.run() in (V.PASS, V.FAIL, V.WARN)
        assert c.detail and c.why and c.fix
    assert [c for c in ran if c.status == V.PASS]

    broken = V.Check("deliberately broken", "why", "the fix",
                     lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert broken.run() == V.FAIL and "boom" in broken.detail

    # every check states what it is for and what to do about it
    for c in V.CHECKS:
        assert len(c.why) > 10 and len(c.fix) > 10, c.name


# ── the optimiser itself ────────────────────────────────────────────────────
# Everything else in this file covers the picker, the store, the objective, the
# drift monitor, the budget and the exports. The MODEL is the subsystem that
# spends instrument time, and until these it had no test at all - installing
# torch would not have changed that. Guarded, because the stack is a large
# download and the rest of the suite must run without it.
def test_the_model_fits_and_proposes_a_runnable_method():
    pytest.importorskip("botorch")
    from gradient_bench.core import optimiser as O

    U = S.initial_sample(8, seed=3)
    # a response that genuinely depends on temperature, so the fit has
    # something to find rather than noise to chase
    y = np.array([5.0 + 3.0 * (u[3] - 25.0) / 35.0 for u in U])

    fit = O.fit_gp(U, y, sigma_crf=1.0, kernel="matern52")
    assert fit.n == 8 and fit.d == 4 and fit.kernel == "matern52"
    assert np.isfinite(fit.noise_sd_crf) and fit.noise_sd_crf > 0
    ls = {n: v for n, v, _ in fit.lengthscale_table()}
    assert set(ls) == set(S.P4_NAMES)
    assert all(np.isfinite(v) and v > 0 for v in ls.values())

    mu, sd = O.posterior(fit, U)
    assert mu.shape == sd.shape == (8,)
    assert np.all(np.isfinite(mu)) and np.all(sd >= 0)
    # the posterior at a training point should sit near what was measured
    assert np.abs(mu - y).max() < 5.0

    sug = O.suggest_next(U, y, sigma_crf=1.0, seed=1, kernel="matern52")
    assert S.check4(sug.u)[0], f"proposed an unrunnable method: {sug.u}"
    assert np.isfinite(sug.acq_value) and np.isfinite(sug.posterior_mean)
    assert sug.posterior_sd >= 0 and sug.phase == "optimise"


def test_the_measured_noise_floor_reaches_the_fitted_model():
    """The floor is the campaign's own replicate spread converted into the
    model's units. If it did not reach the likelihood, the model would explain
    replicate disagreement by inventing a wiggly surface instead."""
    pytest.importorskip("botorch")
    from gradient_bench.core import optimiser as O

    U = S.initial_sample(10, seed=5)
    y = np.array([6.0 + 2.0 * (u[1] - 0.1) for u in U])
    tight = O.fit_gp(U, y, sigma_crf=0.05, kernel="matern52")
    loose = O.fit_gp(U, y, sigma_crf=4.0, kernel="matern52")
    assert loose.noise.floor_z > tight.noise.floor_z
    assert loose.noise_sd_crf > tight.noise_sd_crf
    assert loose.noise.dominated and not tight.noise.dominated


def test_uncertainty_step_reports_what_a_run_bought():
    pytest.importorskip("botorch")
    from gradient_bench.core import uncertainty as UNC

    U = S.initial_sample(6, seed=7)
    y = np.array([4.0 + 5.0 * (u[3] - 25.0) / 35.0 for u in U])
    step = UNC.uncertainty_step(U, y, sigma_crf=1.0, kernel="matern52")
    assert step is not None and step.n_runs == 6
    assert np.isfinite(step.mean_sd_before) and np.isfinite(step.mean_sd_after)
    assert set(step.lengthscales) == set(S.P4_NAMES)
    assert step.reading()
    # fewer than three runs has nothing to compare against
    assert UNC.uncertainty_step(U[:2], y[:2], sigma_crf=1.0) is None


def test_recording_gates_on_feasibility(tmp_path, monkeypatch):
    """`space.py` says check4 is the ONLY gate and nothing is recorded without
    it. That held only transitively - `suggest` checked before persisting - so
    anything reaching state.json another way wrote unrunnable parameters
    straight into the workbook."""
    from gradient_bench import loop as L
    from gradient_bench.store import state as ST
    c = _campaign(tmp_path, monkeypatch)
    # a hand-edited pending file, which is exactly the route that bypassed it
    ST.save(c.state_path, ST.PendingRun(
        kind="design", u=[0.30, 0.10, 20.0, 40.0],     # end below start
        phase="cold_start", reason="hand-edited", n_replicates=1))
    with pytest.raises(ValueError, match="infeasible"):
        L.record(c)
    assert c.n_injections() == 0                        # nothing was written


# ── version 2: the surrogate and the acquisition are chosen, then locked ────
from gradient_bench.core import optimiser as O                # noqa: E402
def test_every_surrogate_option_builds_the_kernel_it_claims():
    """The option list is what the analyst chooses from, so every entry must
    build, and must build the smoothness it advertises."""
    pytest.importorskip("botorch")
    for opt in O.SURROGATE_OPTIONS:
        d = O.describe_kernel(opt["key"], 4)
        assert d["available"], (opt["key"], d.get("error"))
    assert O.describe_kernel("matern52", 4)["nu"] == 2.5
    assert O.describe_kernel("matern32", 4)["nu"] == 1.5
    assert O.describe_kernel("rbf", 4)["base"] == "RBFKernel"
    assert sum(1 for o in O.SURROGATE_OPTIONS if o["recommended"]) == 1
    assert set(o["key"] for o in O.SURROGATE_OPTIONS) == set(O.KERNELS)


def test_matern32_keeps_botorch_priors():
    """Choosing a rougher kernel must not cost the regularisation - the whole
    reason the module is built through BoTorch's constructor."""
    pytest.importorskip("botorch")
    a = O.make_covar_module(4, "matern52")
    b = O.make_covar_module(4, "matern32")
    assert type(a.lengthscale_prior).__name__ == type(b.lengthscale_prior).__name__
    assert float(a.lengthscale_prior.loc) == float(b.lengthscale_prior.loc)
    assert float(a.lengthscale_prior.scale) == float(b.lengthscale_prior.scale)
    assert (float(a.raw_lengthscale_constraint.lower_bound)
            == float(b.raw_lengthscale_constraint.lower_bound))
    assert b.nu == 1.5 and b.ard_num_dims == 4


def test_every_acquisition_proposes_a_runnable_method():
    pytest.importorskip("botorch")
    U = S.initial_sample(8, seed=11)
    y = np.array([3, 5, 4, 6, 2, 7, 5, 3], float)
    for opt in O.ACQUISITION_OPTIONS:
        sug = O.suggest_next(U, y, sigma_crf=1.0, seed=1, kernel="matern52",
                             acquisition=opt["key"], num_restarts=2,
                             raw_samples=32)
        assert S.check4(sug.u)[0], opt["key"]
        assert sug.acquisition == opt["key"]
        assert np.isfinite(sug.acq_value)
    assert set(o["key"] for o in O.ACQUISITION_OPTIONS) == set(O.ACQUISITIONS)


def test_acquisition_parameters_are_validated():
    assert O.resolve_acq_params("ucb", {"ucb_beta": 3}) == {"ucb_beta": 3.0}
    assert O.resolve_acq_params("ucb", None) == {"ucb_beta": 2.0}
    assert O.resolve_acq_params("qlognei", {"ucb_beta": 3}) == {}
    with pytest.raises(ValueError):
        O.resolve_acq_params("ucb", {"ucb_beta": 99})
    with pytest.raises(ValueError):
        O.resolve_acq_params("nope", {})
    assert O.describe_acquisition("qlognei") == \
        "qLogNoisyExpectedImprovement (prune_baseline=True)"
    assert O.describe_acquisition("ucb", {"ucb_beta": 1.5}) == \
        "qUpperConfidenceBound (ucb_beta=1.5)"


def test_a_campaign_records_and_locks_its_optimiser_choice(tmp_path, monkeypatch):
    c = _campaign(tmp_path, monkeypatch, kernel="matern32", acquisition="ucb",
                  acq_params={"ucb_beta": 3.5})
    assert c.kernel() == "matern32"
    assert c.acquisition() == "ucb"
    assert c.acq_params() == {"ucb_beta": 3.5}
    cfg = c.config()
    assert cfg["acquisition"] == "ucb"
    assert cfg["acquisition_note"] == "qUpperConfidenceBound (ucb_beta=3.5)"
    assert cfg["app_version"] == "2.0.0"
    s = c.summary()["optimiser"]
    assert s["acquisition_short"] == "UCB" and s["kernel_label"].startswith("Mat")
    # the choice travels with duplicated settings
    from gradient_bench.store import campaign as C
    d = C.duplicate_settings(c, str(tmp_path), {"name": "dup"})
    assert (d.kernel(), d.acquisition(), d.acq_params()) == \
        ("matern32", "ucb", {"ucb_beta": 3.5})
    with pytest.raises(ValueError):
        _campaign(tmp_path, monkeypatch, acquisition="thompson")


def test_a_version_one_campaign_reads_as_qlognei(tmp_path, monkeypatch):
    """Version 1 wrote a descriptive string, not a key, into `acquisition`.
    Every such campaign was proposed with qLogNEI."""
    from gradient_bench.store import sheet as SH
    c = _campaign(tmp_path, monkeypatch)
    cfg = c.config()
    cfg["acquisition"] = "qLogNoisyExpectedImprovement (prune_baseline=True)"
    cfg.pop("acq_params", None)
    SH.write_config(c.workbook, cfg)
    assert c.acquisition() == "qlognei"
    assert c.acq_params() == {}
    assert c.optimiser_summary()["acquisition_short"] == "qLogNEI"
