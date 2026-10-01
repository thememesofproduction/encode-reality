"""Tests for fixed and meta-strategy optimizers."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.ideas_intelligence_creativity.landscape import FitnessLandscape
from experiments.ideas_intelligence_creativity.strategies import SearchStrategy, StrategyMutator
from experiments.ideas_intelligence_creativity.intelligence import FixedStrategyOptimizer
from experiments.ideas_intelligence_creativity.creativity import MetaStrategyOptimizer


class TestFixedStrategyOptimizer:

    def test_deterministic(self):
        """Same seed produces same results."""
        landscape = FitnessLandscape(n_peaks=5, seed=42)
        strategy = SearchStrategy(step_size=0.3, n_candidates=20)
        opt1 = FixedStrategyOptimizer(landscape, strategy, seed=10)
        opt2 = FixedStrategyOptimizer(landscape, strategy, seed=10)
        r1 = opt1.run(50)
        r2 = opt2.run(50)
        assert r1["best_fitness"] == r2["best_fitness"]

    def test_fitness_nondecreasing(self):
        """Best fitness should never decrease."""
        landscape = FitnessLandscape(n_peaks=10, seed=42)
        strategy = SearchStrategy(step_size=0.5, n_candidates=30)
        opt = FixedStrategyOptimizer(landscape, strategy, seed=42)
        result = opt.run(100)
        fitnesses = result["best_fitness"]
        for i in range(1, len(fitnesses)):
            assert fitnesses[i] >= fitnesses[i - 1] - 1e-12

    def test_converges_on_unimodal(self):
        """On a simple landscape, should find the peak."""
        landscape = FitnessLandscape(n_peaks=1, seed=42)
        strategy = SearchStrategy(step_size=1.5, n_candidates=50, method="restart")
        opt = FixedStrategyOptimizer(landscape, strategy, seed=42)
        opt.run(500)
        _, global_val = landscape.get_global_optimum()
        # Should get within 50% of global optimum (generous threshold)
        assert opt.best_fitness > global_val * 0.5

    def test_history_length(self):
        landscape = FitnessLandscape(n_peaks=5, seed=42)
        strategy = SearchStrategy()
        opt = FixedStrategyOptimizer(landscape, strategy, seed=42)
        result = opt.run(100)
        # +1 for initial recording
        assert len(result["best_fitness"]) == 101

    def test_to_dict_keys(self):
        landscape = FitnessLandscape(n_peaks=5, seed=42)
        strategy = SearchStrategy()
        opt = FixedStrategyOptimizer(landscape, strategy, seed=42)
        result = opt.run(10)
        assert "best_fitness" in result
        assert "best_positions" in result
        assert "solution_diversity" in result


class TestMetaStrategyOptimizer:

    def _make_optimizer(self, seed=42, meta_interval=50, n_peaks=10):
        landscape = FitnessLandscape(n_peaks=n_peaks, seed=seed)
        strategy = SearchStrategy(step_size=0.3, n_candidates=20)
        mutator = StrategyMutator(mutation_rate=0.3, seed=seed)
        return MetaStrategyOptimizer(
            landscape, strategy, mutator,
            meta_interval=meta_interval,
            strategy_pool_size=5,
            seed=seed
        )

    def test_deterministic(self):
        opt1 = self._make_optimizer(seed=42)
        opt2 = self._make_optimizer(seed=42)
        r1 = opt1.run(200)
        r2 = opt2.run(200)
        assert r1["best_fitness"] == r2["best_fitness"]

    def test_fitness_nondecreasing(self):
        opt = self._make_optimizer()
        result = opt.run(200)
        fitnesses = result["best_fitness"]
        for i in range(1, len(fitnesses)):
            assert fitnesses[i] >= fitnesses[i - 1] - 1e-12

    def test_meta_events_occur(self):
        """Meta-steps should fire at the expected intervals."""
        opt = self._make_optimizer(meta_interval=50)
        result = opt.run(200)
        assert len(result["meta_events"]) == 4  # at steps 50, 100, 150, 200

    def test_strategy_diversity_tracked(self):
        opt = self._make_optimizer()
        result = opt.run(100)
        assert len(result["strategy_diversity"]) == 101

    def test_strategy_labels_tracked(self):
        opt = self._make_optimizer()
        result = opt.run(50)
        assert len(result["strategy_labels"]) == 51
        assert all(isinstance(s, str) for s in result["strategy_labels"])

    def test_meta_event_has_required_fields(self):
        opt = self._make_optimizer(meta_interval=50)
        result = opt.run(50)
        assert len(result["meta_events"]) >= 1
        event = result["meta_events"][0]
        assert "step" in event
        assert "dropped" in event
        assert "action" in event
        assert "new" in event

    def test_to_dict_keys(self):
        opt = self._make_optimizer()
        result = opt.run(10)
        assert "best_fitness" in result
        assert "strategy_diversity" in result
        assert "meta_events" in result
