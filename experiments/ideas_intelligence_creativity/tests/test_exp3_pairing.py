"""Experiment 3 runs are paired: with the same seed, every arm that starts on the real line starts at the same x0.

The README and the essay call the 30 runs per arm "paired". That is only true if the start does not
depend on the arm.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.ideas_intelligence_creativity.experiment3 import run_arm

REAL_LINE_STARTERS = ("hc_fixed", "hc_restart", "es_restart", "portfolio", "split_two_reals", "c_oracle")


def _start_on_real_line(arm, seed):
    x0 = run_arm(arm, 5, seed).x0
    a = np.asarray(x0, dtype=float).ravel()
    if a.size == 2:
        assert a[1] == 0.0, f"{arm} must start on the real line, got {x0!r}"
    return float(a[0])


@pytest.mark.parametrize("seed", range(5))
def test_arms_with_the_same_seed_share_their_start(seed):
    starts = {arm: _start_on_real_line(arm, seed) for arm in REAL_LINE_STARTERS}
    assert len(set(starts.values())) == 1, starts


def test_different_seeds_start_in_different_places():
    # Negative control: a constant start would pass the test above.
    assert len({_start_on_real_line("hc_fixed", s) for s in range(5)}) == 5
