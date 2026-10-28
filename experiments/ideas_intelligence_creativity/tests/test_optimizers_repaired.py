"""SPEC.md section 6 and 9.8/9.9, intelligence.py and creativity.py: start point, budget, meta-selection.

Spec 9.8 names the attributes: optimizer.pool (list of records), optimizer.active_idx (int),
optimizer.active_strategy (== pool[active_idx].strategy), optimizer.history.positions and
optimizer.history.best_fitness. They are used here exactly, with no fallbacks.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.tests._iic_helpers import load, patch_budget_counter  # noqa: E402


def mods():
    return (load("landscape"), load("strategies"), load("intelligence"), load("creativity"))


def landscape(seed=21):
    return load("landscape").FitnessLandscape(n_peaks=8, seed=seed)


def instrument_landscape(land):
    """Record every value the landscape hands out, in order (batch internals counted once)."""
    values, state = [], {"inside": False}
    orig_eval, orig_batch = land.evaluate, land.evaluate_batch

    def evaluate(x):
        v = orig_eval(x)
        if not state["inside"]:
            values.append(float(v))
        return v

    def evaluate_batch(xs):
        state["inside"] = True
        try:
            vs = orig_batch(xs)
        finally:
            state["inside"] = False
        values.extend(float(v) for v in np.asarray(vs, dtype=float).ravel())
        return vs

    land.evaluate, land.evaluate_batch = evaluate, evaluate_batch
    return values


def make(kind, land, seed=5, x0=None, selection="ucb", meta_interval=5, mutator=None, replace_every=3):
    _, st, intel, crea = mods()
    strat = st.SearchStrategy(step_size=0.3, n_candidates=10, method="gaussian")
    if kind == "fixed":
        return intel.FixedStrategyOptimizer(land, strat, seed=seed, x0=x0)
    mutator = mutator or st.StrategyMutator(mutation_rate=0.3, seed=seed)
    return crea.MetaStrategyOptimizer(land, strat, mutator, meta_interval=meta_interval, strategy_pool_size=5,
                                      seed=seed, x0=x0, replace_every=replace_every, selection=selection)


# ----------------------------------------------------------------------------- start point

@pytest.mark.parametrize("kind", ["fixed", "meta"])
def test_given_x0_is_the_first_evaluation(kind):
    land = landscape()
    x0 = np.array([1.234, -2.5])
    opt = make(kind, land, x0=x0)
    opt.run(max_evals=40)
    np.testing.assert_array_equal(np.asarray(opt.history.positions[0], dtype=float), x0)
    f0 = float(land.evaluate(x0))
    assert float(opt.history.best_fitness[0]) == pytest.approx(f0, rel=1e-12)
    assert float(np.asarray(opt.trace)[0]) == pytest.approx(f0, rel=1e-12)   # the start point is evaluation 1


@pytest.mark.parametrize("kind", ["fixed", "meta"])
def test_without_x0_the_optimiser_draws_its_own_start(kind):
    starts = {}
    for seed in (1, 1, 2, 3):
        opt = make(kind, landscape(), seed=seed)
        opt.run(max_evals=15)
        p = np.asarray(opt.history.positions[0], dtype=float)
        assert np.all(p >= -5.0) and np.all(p <= 5.0)
        starts.setdefault(seed, []).append(p)
    np.testing.assert_array_equal(starts[1][0], starts[1][1])
    assert not np.array_equal(starts[1][0], starts[2][0]) or not np.array_equal(starts[1][0], starts[3][0])


# ----------------------------------------------------------------------------- budget

@pytest.mark.parametrize("kind", ["fixed", "meta"])
@pytest.mark.parametrize("B", [1, 2, 7, 11, 21, 333, 1000])
def test_max_evals_is_exact(kind, B, monkeypatch):
    sc = load("search_core")
    counter = patch_budget_counter(monkeypatch, sc.Budget)
    land = landscape()
    values = instrument_landscape(land)
    opt = make(kind, land, x0=np.array([0.5, 0.5]))
    opt.run(max_evals=B)
    trace = opt.trace
    assert isinstance(trace, np.ndarray)
    assert len(trace) == B
    assert len(values) == B, f"the landscape was evaluated at {len(values)} points, expected {B}"
    assert counter["n"] == B, f"{counter['n']} points went through search_core.Budget, expected {B}"
    np.testing.assert_array_equal(trace, np.maximum.accumulate(values))   # best so far after each evaluation


@pytest.mark.parametrize("kind", ["fixed", "meta"])
def test_same_seed_same_trace_different_seed_differs(kind):
    a = make(kind, landscape(), seed=8)
    b = make(kind, landscape(), seed=8)
    c = make(kind, landscape(), seed=9)
    for o in (a, b, c):
        o.run(max_evals=500)
    np.testing.assert_array_equal(a.trace, b.trace)
    assert not np.array_equal(a.trace, c.trace)
    assert np.all(np.diff(a.trace) >= 0)


# ----------------------------------------------------------------------------- meta-selection helpers

def active_record(opt):
    """Spec 9.8: active_idx is an int and active_strategy is pool[active_idx].strategy."""
    pool, idx = opt.pool, opt.active_idx
    assert isinstance(pool, list)
    assert isinstance(idx, (int, np.integer)) and not isinstance(idx, bool), type(idx)
    assert 0 <= idx < len(pool)
    assert opt.active_strategy is pool[idx].strategy
    return pool[idx]


def record_of(pool, strat):
    for r in pool:
        if r.strategy is strat:
            return r
    for r in pool:
        try:
            if r.strategy == strat:
                return r
        except Exception:  # noqa: BLE001
            pass
    return None


def instrument_mutator(mutator, holder, violations):
    holder.setdefault("calls", 0)
    for name in ("mutate", "crossover"):
        orig = getattr(mutator, name)

        def wrapped(*args, _orig=orig, _name=name, **kw):
            holder["calls"] += 1
            opt = holder.get("opt")
            for s in args:
                if not hasattr(s, "method"):
                    continue
                if opt is None:
                    violations.append(f"{_name} called before any strategy had run")
                    continue
                rec = record_of(opt.pool, s)
                if rec is None or int(rec.evals) <= 0:
                    violations.append(f"{_name} called with a strategy whose pool record has "
                                      f"{'no record' if rec is None else f'evals={rec.evals}'}")
            return _orig(*args, **kw)

        setattr(mutator, name, wrapped)


def check_records(pool):
    for r in pool:
        assert int(r.evals) == r.evals and r.evals >= 0
        assert isinstance(float(r.improvement), float)
        if r.evals == 0:
            assert r.score is None
        else:
            assert r.score == pytest.approx(r.improvement / r.evals, rel=1e-12, abs=1e-15)


def checked_meta_step(opt, selection, stats, holder=None):
    """Wrap opt.meta_step so every call (automatic or manual) is checked against spec 6 and 9.9."""
    orig = opt.meta_step

    def meta_step(*a, **kw):
        before = [(r.strategy, int(r.evals), r.score) for r in opt.pool]
        active_before = opt.active_strategy
        calls_before = holder["calls"] if holder else 0
        replace = kw.get("replace", a[0] if a else None)
        out = orig(*a, **kw)
        after = opt.pool
        arec = active_record(opt)
        check_records(after)
        untried = [r for r in after if r.evals == 0]
        if selection == "ucb":
            if untried:
                assert arec.evals == 0, "an untried strategy exists but a tried one is active"
            elif len({int(r.evals) for r in after}) == 1:
                stats["equal_evals_branch"] = stats.get("equal_evals_branch", 0) + 1
                assert arec.score == pytest.approx(max(r.score for r in after), rel=1e-12)
        elif untried and arec.evals > 0:
            stats["random_skipped_untried"] = stats.get("random_skipped_untried", 0) + 1
        dropped = [b for b in before if not any(b[0] is r.strategy for r in after)]
        added = [r for r in after if not any(r.strategy is b[0] for b in before)]
        assert len(dropped) <= 1 and len(added) <= 1, (len(dropped), len(added))
        for strat, evals, score in dropped:
            assert evals > 0, "a never-run strategy was dropped"
            assert strat is not active_before, "the active strategy was dropped"
            assert strat is not opt.active_strategy
            candidates = [b[2] for b in before if b[1] > 0 and b[0] is not active_before]
            assert score == pytest.approx(min(candidates)), "the dropped record is not the worst one with data"
        for r in added:
            assert r.evals == 0, "a new strategy must start without data"
        if dropped:
            assert added, "a record was dropped without a replacement being added"
        if replace is False:
            assert not dropped and not added, "meta_step(replace=False) changed the pool"
            if holder is not None:
                assert holder["calls"] == calls_before, "meta_step(replace=False) called the mutator"
        if replace is True:
            assert len(added) == 1, "meta_step(replace=True) did not add a new strategy"
        stats["checked"] = stats.get("checked", 0) + 1
        stats.setdefault("changed", []).append(bool(dropped or added))
        return out

    opt.meta_step = meta_step


def build_meta(seed, selection, violations, meta_interval=5, replace_every=3, holder=None):
    st = load("strategies")
    holder = {} if holder is None else holder
    mutator = st.StrategyMutator(mutation_rate=0.5, seed=seed)
    instrument_mutator(mutator, holder, violations)
    opt = make("meta", landscape(seed=30 + seed), seed=seed, selection=selection,
               meta_interval=meta_interval, mutator=mutator, replace_every=replace_every)
    holder["opt"] = opt
    return opt


# ----------------------------------------------------------------------------- spec 9.8 names

@pytest.mark.parametrize("kind", ["fixed", "meta"])
def test_history_has_positions_and_best_fitness(kind):
    opt = make(kind, landscape())
    opt.run(max_evals=60)
    assert len(opt.history.positions) == len(opt.history.best_fitness) >= 1
    assert np.all(np.diff(np.asarray(opt.history.best_fitness, dtype=float)) >= 0)


def test_pool_active_idx_and_records():
    opt = build_meta(0, "ucb", [])
    opt.run(max_evals=400)
    active_record(opt)
    assert len(opt.pool) >= 2
    check_records(opt.pool)
    assert any(r.evals > 0 for r in opt.pool)


# ----------------------------------------------------------------------------- spec 9.9 selection

def _prepared(seed, selection, holder, violations):
    opt = build_meta(seed, selection, violations, holder=holder)
    opt.run(max_evals=1500)
    assert len(opt.pool) >= 2, f"need a pool of at least two records to test selection, got {len(opt.pool)}"
    return opt


def _set_equal_evals(opt, best_idx, untried_idx=None):
    for i, r in enumerate(opt.pool):
        r.evals = 40
        r.improvement = 1000.0 if i == best_idx else 0.001 * (i + 1)
    if untried_idx is not None:
        opt.pool[untried_idx].evals = 0
        opt.pool[untried_idx].improvement = 0.0


def _not_active(opt, k):
    n = len(opt.pool)
    return [i for i in range(n) if i != opt.active_idx][k % (n - 1)]


@pytest.mark.parametrize("seed", range(6))
def test_selection_without_replacement_exploits_the_best_score(seed):
    """Equal evals > 0, distinct improvements, best record NOT active: meta_step(replace=False)
    must make the highest-.score record active (the UCB branch with equal counts)."""
    holder, violations, stats = {}, [], {}
    opt = _prepared(seed, "ucb", holder, violations)
    best = _not_active(opt, seed)
    _set_equal_evals(opt, best)
    strategies = [r.strategy for r in opt.pool]
    checked_meta_step(opt, "ucb", stats, holder)
    opt.meta_step(replace=False)
    assert [r.strategy for r in opt.pool] == strategies
    assert opt.active_strategy is strategies[best]
    assert active_record(opt).score == max(r.score for r in opt.pool)
    assert violations == [], violations


@pytest.mark.parametrize("seed", range(6))
def test_selection_without_replacement_runs_an_untried_strategy_first(seed):
    holder, violations, stats = {}, [], {}
    opt = _prepared(seed, "ucb", holder, violations)
    untried = _not_active(opt, seed + 1)
    _set_equal_evals(opt, opt.active_idx, untried_idx=untried)   # even the best-scoring record must yield
    strategies = [r.strategy for r in opt.pool]
    checked_meta_step(opt, "ucb", stats, holder)
    opt.meta_step(replace=False)
    assert opt.active_strategy is strategies[untried]
    assert active_record(opt).evals == 0
    assert violations == [], violations


def test_random_selection_breaks_both_rules():
    """Negative control: selection='random' must sometimes skip the best score and the untried record."""
    skipped_best = skipped_untried = 0
    for seed in range(30):
        holder, violations = {}, []
        opt = _prepared(seed, "random", holder, violations)
        best = _not_active(opt, seed)
        _set_equal_evals(opt, best)
        strategies = [r.strategy for r in opt.pool]
        opt.meta_step(replace=False)
        assert [r.strategy for r in opt.pool] == strategies          # same pool dynamics: no replacement
        skipped_best += opt.active_strategy is not strategies[best]
        untried = _not_active(opt, seed + 1)
        _set_equal_evals(opt, opt.active_idx, untried_idx=untried)
        opt.meta_step(replace=False)
        skipped_untried += opt.active_strategy is not strategies[untried]
        assert violations == [], violations
    assert skipped_best >= 1 and skipped_untried >= 1, (skipped_best, skipped_untried)


# ----------------------------------------------------------------------------- spec 9.9 replacement

@pytest.mark.parametrize("replace_every", [3, 2])
def test_replacement_happens_every_replace_every_th_meta_step(replace_every):
    holder, violations, stats = {}, [], {}
    opt = build_meta(1, "ucb", violations, meta_interval=10_000, replace_every=replace_every, holder=holder)
    opt.run(max_evals=1)                         # no automatic meta-step yet
    checked_meta_step(opt, "ucb", stats, holder)
    for _ in range(6):
        opt.meta_step()
    expected = [(i + 1) % replace_every == 0 for i in range(6)]
    assert stats["changed"] == expected, (stats["changed"], expected)
    assert violations == [], violations


def test_replace_true_adds_one_new_untried_strategy_and_drops_only_the_worst_with_data():
    for seed in range(4):
        holder, violations, stats = {}, [], {}
        opt = _prepared(seed, "ucb", holder, violations)
        checked_meta_step(opt, "ucb", stats, holder)
        for _ in range(4):
            opt.meta_step(replace=True)
        assert stats["checked"] == 4 and all(stats["changed"])
        assert violations == [], violations


@pytest.mark.parametrize("seed", range(10))
def test_meta_step_rules_hold_through_runs(seed):
    violations, stats, holder = [], {}, {}
    opt = build_meta(seed, "ucb", violations, holder=holder)
    checked_meta_step(opt, "ucb", stats, holder)   # also sees automatic meta-steps if run() calls meta_step()
    opt.run(max_evals=700 + 37 * seed)
    for replace in (None, True, False, None):
        opt.meta_step(replace=replace)
    assert stats["checked"] >= 4
    assert violations == [], violations


def test_meta_step_on_a_pool_where_nothing_new_has_run():
    for seed in range(4):
        violations, stats, holder = [], {}, {}
        opt = build_meta(seed, "ucb", violations, meta_interval=10_000, holder=holder)
        opt.run(max_evals=1)
        checked_meta_step(opt, "ucb", stats, holder)
        for _ in range(12):
            opt.meta_step()
        assert stats["checked"] == 12
        assert violations == [], violations
