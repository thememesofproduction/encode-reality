"""Tests for search strategies and strategy mutation."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.ideas_intelligence_creativity.strategies import (
    SearchStrategy, StrategyMutator, METHODS
)


class TestSearchStrategy:

    def test_gaussian_propose_shape(self):
        s = SearchStrategy(step_size=0.5, n_candidates=30, method="gaussian")
        rng = np.random.default_rng(42)
        candidates = s.propose(np.array([0.0, 0.0]), rng)
        assert candidates.shape == (30, 2)

    def test_uniform_propose_shape(self):
        s = SearchStrategy(step_size=0.5, n_candidates=20, method="uniform")
        rng = np.random.default_rng(42)
        candidates = s.propose(np.array([0.0, 0.0]), rng)
        assert candidates.shape == (20, 2)

    def test_restart_propose_shape(self):
        s = SearchStrategy(step_size=0.5, n_candidates=20, method="restart")
        rng = np.random.default_rng(42)
        candidates = s.propose(np.array([0.0, 0.0]), rng)
        assert candidates.shape == (20, 2)

    def test_momentum_propose_shape(self):
        s = SearchStrategy(step_size=0.5, n_candidates=20, method="momentum")
        rng = np.random.default_rng(42)
        vel = np.array([0.1, -0.2])
        candidates = s.propose(np.array([0.0, 0.0]), rng, velocity=vel)
        assert candidates.shape == (20, 2)

    def test_candidates_within_bounds(self):
        s = SearchStrategy(step_size=1.0, n_candidates=100, method="gaussian")
        rng = np.random.default_rng(42)
        candidates = s.propose(np.array([4.5, 4.5]), rng, bounds=(-5.0, 5.0))
        assert candidates.min() >= -5.0
        assert candidates.max() <= 5.0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            SearchStrategy(method="invalid")

    def test_label_format(self):
        s = SearchStrategy(step_size=0.3, n_candidates=20, method="gaussian")
        label = s.label()
        assert "gaussian" in label
        assert "0.30" in label


class TestStrategyMutator:

    def test_mutate_returns_new_strategy(self):
        m = StrategyMutator(mutation_rate=1.0, seed=42)
        s = SearchStrategy()
        mutated = m.mutate(s)
        assert isinstance(mutated, SearchStrategy)

    def test_mutate_does_not_modify_original(self):
        m = StrategyMutator(mutation_rate=1.0, seed=42)
        s = SearchStrategy(step_size=0.3, n_candidates=20, method="gaussian")
        _ = m.mutate(s)
        assert s.step_size == 0.3
        assert s.n_candidates == 20
        assert s.method == "gaussian"

    def test_high_mutation_rate_changes_strategy(self):
        """With mutation_rate=1.0, at least something should change."""
        m = StrategyMutator(mutation_rate=1.0, seed=42)
        s = SearchStrategy(step_size=0.3, n_candidates=20, method="gaussian")
        mutated = m.mutate(s)
        changed = (mutated.step_size != s.step_size or
                   mutated.n_candidates != s.n_candidates or
                   mutated.method != s.method or
                   mutated.momentum_decay != s.momentum_decay)
        assert changed

    def test_crossover_combines_parents(self):
        m = StrategyMutator(seed=42)
        s1 = SearchStrategy(step_size=0.1, n_candidates=10, method="gaussian")
        s2 = SearchStrategy(step_size=1.0, n_candidates=50, method="restart")
        child = m.crossover(s1, s2)
        assert child.step_size in (s1.step_size, s2.step_size)
        assert child.n_candidates in (s1.n_candidates, s2.n_candidates)
        assert child.method in (s1.method, s2.method)

    def test_random_strategy_valid(self):
        m = StrategyMutator(seed=42)
        s = m.random_strategy()
        assert s.method in METHODS
        assert 0.01 <= s.step_size <= 5.0
        assert 5 <= s.n_candidates <= 100

    def test_mutate_step_size_bounded(self):
        """Step size should stay within reasonable bounds after mutation."""
        m = StrategyMutator(mutation_rate=1.0, seed=42)
        s = SearchStrategy(step_size=0.01)
        for _ in range(100):
            s = m.mutate(s)
        assert 0.01 <= s.step_size <= 5.0
