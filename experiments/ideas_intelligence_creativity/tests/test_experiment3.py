"""SPEC.md section 4: experiment3.py -- search arms over number systems (contracts 1 to 7)."""
import dataclasses
import multiprocessing
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.tests._iic_helpers import (  # noqa: E402
    PKG, Lazy, dual_kind, load, patch_budget_counter, spec_armresult_fields, spec_float, spec_literal, spec_section, true_floor,
)

REAL_ARMS = spec_literal("REAL_ARMS", 4)
CONTROLS = spec_literal("CONTROLS", 4)
L2_ARMS = spec_literal("L2_ARMS", 4)
ALL_ARMS = REAL_ARMS + CONTROLS + L2_ARMS
FLOOR_ARMS = REAL_ARMS + ("split_two_reals",)           # contract 3
COEFFS = spec_literal("COEFFS", 3)
X_TRUE, M_TRUE = true_floor(COEFFS)
DUAL_TOL = spec_float(r"\|delta\| ≤ (1e-\d+)·max", 3)
_box = re.search(r"alpha ∈ \[([−\-\d.]+), ([−\-\d.]+)\], beta ∈ \[([−\-\d.]+), ([−\-\d.]+)\]", spec_section(4))
ALPHA_BOX = (float(_box.group(1).replace("−", "-")), float(_box.group(2).replace("−", "-")))
BETA_BOX = (float(_box.group(3).replace("−", "-")), float(_box.group(4).replace("−", "-")))
START_DELTA = spec_float(r"starts from algebras with delta ≥ (\d+(?:\.\d+)?)", 4)
RECORD_KEYS = [k.strip() for k in re.search(r"keys: ([\w, ]+)", spec_section(4)).group(1).split(",")]
SMALL = {arm: (3001 if arm in L2_ARMS else 1237) for arm in ALL_ARMS}   # odd sizes: batches get truncated


def ex3():
    return load("experiment3")


def run(arm, max_evals, seed):
    return ex3().run_arm(arm, max_evals, seed)


def adoption_index(trace):
    below = np.flatnonzero(np.asarray(trace) < M_TRUE * (1 - 1e-9))
    return int(below[0]) if below.size else None


# ----------------------------------------------------------------------------- big Level-2 runs

def _worker(arm, max_evals, seed):
    import importlib
    import numpy as _np
    mod = importlib.import_module(PKG + ".experiment3")
    r = mod.run_arm(arm, max_evals, seed)
    return {"arm": r.arm, "trace": _np.asarray(r.trace, dtype=float), "n_evals": r.n_evals,
            "records": [dict(x) for x in r.records], "adoption_eval": r.adoption_eval,
            "first_complex_proposal_eval": r.first_complex_proposal_eval}


def _big_runs():
    load("experiment3")                                   # fail fast, as a test failure, if missing
    jobs = [("l2_evo", 200_000, s) for s in range(10)]
    jobs += [(arm, 20_000, s) for arm in L2_ARMS if arm != "l2_evo" for s in range(3)]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(12, max(2, (os.cpu_count() or 4) - 2)), mp_context=ctx) as pool:
        futs = {job: pool.submit(_worker, *job) for job in jobs}
        return {job: f.result() for job, f in futs.items()}


@pytest.fixture(scope="module")
def big():
    return Lazy(_big_runs)


# ----------------------------------------------------------------------------- shape

def test_arm_tuples_are_exported():
    m = ex3()
    assert tuple(m.REAL_ARMS) == REAL_ARMS
    assert tuple(m.CONTROLS) == CONTROLS
    assert tuple(m.L2_ARMS) == L2_ARMS


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_arm_result_fields(arm):
    m = ex3()
    r = m.run_arm(arm, SMALL[arm], 0)
    assert dataclasses.is_dataclass(r)
    names = {f.name for f in dataclasses.fields(r)}
    assert set(spec_armresult_fields()) <= names
    assert r.arm == arm
    assert isinstance(r.trace, np.ndarray)
    assert len(r.trace) == r.n_evals == SMALL[arm]
    assert np.all(np.isfinite(r.trace)) and np.all(r.trace >= 0)
    if arm in L2_ARMS:
        assert isinstance(r.records, list) and len(r.records) >= 1
    else:
        assert r.records == []
        assert r.adoption_eval is None and r.first_complex_proposal_eval is None


# ----------------------------------------------------------------------------- contract 1: budget

@pytest.mark.parametrize("arm", ALL_ARMS)
def test_contract1_every_evaluation_goes_through_budget(arm, monkeypatch):
    sc = load("search_core")
    m = ex3()
    count = patch_budget_counter(monkeypatch, sc.Budget)
    for B in (SMALL[arm], 1):
        count["n"], count["created"], count["instances"] = 0, 0, set()
        r = m.run_arm(arm, B, 3)
        assert len(r.trace) == r.n_evals == B
        assert count["n"] == B, f"{arm}: {count['n']} points went through search_core.Budget, expected {B}"
        # spec 9.2: one Budget per arm run; sub-limits are plain counters, never a second Budget
        assert count["created"] == 1, f"{arm}: {count['created']} Budgets were created in one run"
        assert len(count["instances"]) == 1


# ----------------------------------------------------------------------------- contracts 2, 3

@pytest.mark.parametrize("arm", ALL_ARMS)
def test_contract2_traces_never_increase(arm):
    for seed in (0, 1):
        t = np.asarray(run(arm, SMALL[arm], seed).trace)
        assert np.all(np.diff(t) <= 0), arm


@pytest.mark.parametrize("arm", FLOOR_ARMS)
def test_contract3_real_floor_holds_whatever_the_seed_or_budget(arm):
    for seed in range(5):
        for B in (1, 2, 64, 2500):
            t = np.asarray(run(arm, B, seed).trace)
            assert t.min() >= M_TRUE * (1 - 1e-12), (arm, seed, B, t.min() / M_TRUE)


@pytest.mark.parametrize("arm", [a for a in FLOOR_ARMS if a != "hc_fixed"])
def test_contract3_is_not_idle_the_arms_do_reach_the_floor(arm):
    """Negative control: the floor contract is only meaningful if the arms actually get to it."""
    finals = [float(run(arm, 20_000, s).trace[-1]) / M_TRUE for s in range(3)]
    assert min(finals) <= 1.01, (arm, finals)


@pytest.mark.parametrize("arm", ["hc_fixed", "hc_restart", "es_restart"])
def test_trace_starts_at_x0(arm):
    r = run(arm, 300, 4)
    x0 = np.asarray(r.x0, dtype=float).ravel()
    assert x0.size == 1, f"{arm}: a Level-1 start is one real number, got {r.x0!r}"
    assert r.trace[0] == pytest.approx(abs(np.polyval(COEFFS, x0[0])), rel=1e-12)


# ----------------------------------------------------------------------------- contracts 4 to 6 (Level 2)

def test_contract5_c_oracle_finds_a_root():
    hits = [float(run("c_oracle", 2000, s).trace[-1]) < 1e-6 * M_TRUE for s in range(10)]
    assert sum(hits) >= 9, hits


def test_contract6_level2_finds_c(big):
    runs = big.get()
    adopted = [runs[("l2_evo", 200_000, s)]["adoption_eval"] is not None for s in range(10)]
    assert sum(adopted) >= 8, adopted


def test_level2_runs_obey_budget_and_monotonicity(big):
    for (arm, B, seed), r in big.get().items():
        assert r["arm"] == arm
        assert len(r["trace"]) == r["n_evals"] == B, (arm, seed)
        assert np.all(np.diff(r["trace"]) <= 0), (arm, seed)


def test_adoption_eval_is_the_first_index_below_the_floor(big):
    for (arm, B, seed), r in big.get().items():
        assert r["adoption_eval"] == adoption_index(r["trace"]), (arm, seed)


def test_contract4_level2_escapes_the_floor_only_through_c(big):
    below = 0
    for (arm, B, seed), r in big.get().items():
        for rec in r["records"]:
            if rec["best_rho"] < M_TRUE * (1 - 1e-9):
                below += 1
                assert rec["delta"] < 0, (arm, seed, rec)
        if r["adoption_eval"] is not None:
            assert r["first_complex_proposal_eval"] is not None, (arm, seed)
            assert r["first_complex_proposal_eval"] <= r["adoption_eval"], (arm, seed)
            assert any(rec["best_rho"] < M_TRUE * (1 - 1e-9) for rec in r["records"]), (arm, seed)
    assert below > 0, "no record ever went below the floor: contract 4 was not exercised"


def test_level2_records_schema_and_box(big):
    for (arm, B, seed), r in big.get().items():
        recs = r["records"]
        assert recs, (arm, seed)
        for rec in recs:
            assert set(RECORD_KEYS) <= set(rec), (arm, seed, sorted(rec))
            a, b = float(rec["alpha"]), float(rec["beta"])
            assert ALPHA_BOX[0] <= a <= ALPHA_BOX[1] and BETA_BOX[0] <= b <= BETA_BOX[1], (arm, rec)
            assert rec["delta"] == pytest.approx(b * b + 4 * a, rel=1e-9, abs=1e-12)
            assert rec["kind"] == dual_kind(a, b, DUAL_TOL), rec
            assert int(rec["evals_used"]) == rec["evals_used"] and 1 <= rec["evals_used"] <= B
            assert int(rec["gen"]) == rec["gen"] and rec["gen"] >= 0
            assert rec["best_rho"] >= float(r["trace"][-1]) * (1 - 1e-12)   # no record beats the run's best
        if any(rec["delta"] < 0 for rec in recs):
            assert r["first_complex_proposal_eval"] is not None, (arm, seed)
        if r["first_complex_proposal_eval"] is not None:
            assert 0 <= r["first_complex_proposal_eval"] <= B


def test_level2_starts_away_from_c(big):
    """Spec 9.4: gen is 0-based and every gen-0 algebra has delta >= 1."""
    for (arm, B, seed), r in big.get().items():
        gens = [rec["gen"] for rec in r["records"]]
        assert min(gens) == 0, (arm, seed, min(gens))
        for rec in r["records"]:
            if rec["gen"] == 0:
                assert rec["delta"] >= START_DELTA, (arm, seed, rec)


def test_level2_evals_used_accounts_for_every_evaluation(big):
    """Spec 9.4: evals_used is what that algebra used in that generation (>= 1), so over all records
    it adds up to the run's evaluations (spec 9.2: they all come from the one Budget)."""
    for (arm, B, seed), r in big.get().items():
        assert sum(int(rec["evals_used"]) for rec in r["records"]) == B, (arm, seed)


def test_unknown_arm_raises_value_error():
    m = ex3()
    for bad in ("no_such_arm", "", "L2_EVO", "hc-fixed"):
        with pytest.raises(ValueError):
            m.run_arm(bad, 100, 0)


# ----------------------------------------------------------------------------- contract 7

@pytest.mark.parametrize("arm", ALL_ARMS)
def test_contract7_determinism(arm):
    m = ex3()
    B = SMALL[arm]
    np.random.seed(1)
    random.seed(1)
    a = m.run_arm(arm, B, 0)
    np.random.seed(2)
    random.seed(2)
    b = m.run_arm(arm, B, 0)
    c = m.run_arm(arm, B, 1)
    np.testing.assert_array_equal(a.trace, b.trace)
    assert a.records == b.records
    assert a.adoption_eval == b.adoption_eval
    assert a.first_complex_proposal_eval == b.first_complex_proposal_eval
    assert not np.array_equal(a.trace, c.trace), f"{arm}: seeds 0 and 1 gave the same trace"
