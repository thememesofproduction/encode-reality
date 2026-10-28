"""Non-stationary credit: MetaStrategyOptimizer(discount=...) (coordinator decision, 2026-10-26).

The discount multiplies every record's evals and improvement at the start of each meta-step. A record
that earned a lot early and has stopped improving must lose the active slot to a record that is improving
now, but only when the discount is below 1. With discount 1.0 (the class default, SPEC section 6) it keeps
the slot. Records that have never run stay exactly at zero.

The scenario is driven by hand. Each epoch the active record runs one meta_interval of evaluations (10 steps
of 20 candidates, so 200 evaluations). Record A has early credit and stops improving. Record B improves at a
steady rate while it is active.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.creativity import MetaStrategyOptimizer, StrategyRecord  # noqa: E402
from experiments.ideas_intelligence_creativity.landscape import FitnessLandscape  # noqa: E402
from experiments.ideas_intelligence_creativity.strategies import SearchStrategy, StrategyMutator  # noqa: E402

EPOCH = 10 * 20          # meta_interval steps, each of n_candidates evaluations
HORIZON = 8              # meta-steps watched
STALL_GAIN = 0.0         # improvement per evaluation for record A while it is active
STEADY_GAIN = 0.3        # improvement per evaluation for record B while it is active


def build(discount):
    land = FitnessLandscape(n_peaks=5, seed=3)
    strategy = SearchStrategy(step_size=0.2, n_candidates=20)
    return MetaStrategyOptimizer(land, strategy, StrategyMutator(seed=3), meta_interval=10,
                                 strategy_pool_size=5, seed=3, x0=np.zeros(2), replace_every=10**9,
                                 selection="ucb", discount=discount)


def scenario(discount):
    """Pool of A (early credit, now stalled) and B (steady, both with the same evaluations); A active."""
    opt = build(discount)
    opt.pool = [
        StrategyRecord(strategy=SearchStrategy(step_size=0.2, n_candidates=20), evals=1000, improvement=1000.0),
        StrategyRecord(strategy=SearchStrategy(step_size=0.3, n_candidates=20), evals=1000, improvement=300.0),
    ]
    opt.active_idx = 0
    return opt


def active_per_meta_step(discount):
    opt = scenario(discount)
    gain = [STALL_GAIN, STEADY_GAIN]
    seq = []
    for _ in range(HORIZON):
        record = opt.pool[opt.active_idx]
        record.evals += EPOCH
        record.improvement += gain[opt.active_idx] * EPOCH
        opt.meta_step(replace=False)
        seq.append(opt.active_idx)
    return seq


def test_discount_one_keeps_the_stalled_record_active():
    assert active_per_meta_step(1.0) == [0] * HORIZON


def test_discount_below_one_hands_the_slot_to_the_improving_record():
    seq = active_per_meta_step(0.8)
    assert 1 in seq, f"the improving record never became active: {seq}"
    assert seq.index(1) < HORIZON, seq


def test_discount_leaves_never_run_records_at_exactly_zero():
    opt = build(0.8)
    opt.pool[0].evals, opt.pool[0].improvement = 500, 50.0
    opt.pool.append(StrategyRecord(strategy=SearchStrategy(), evals=0, improvement=0.0))
    for _ in range(3):
        opt.meta_step(replace=False)
    untried = opt.pool[1]
    assert untried.evals == 0 and untried.improvement == 0.0
    assert untried.score is None


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.2])
def test_discount_outside_zero_to_one_is_refused(bad):
    with pytest.raises(ValueError):
        build(bad)
