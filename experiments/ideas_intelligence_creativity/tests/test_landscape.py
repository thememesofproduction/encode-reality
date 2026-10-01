"""Tests for the fitness landscape module."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.ideas_intelligence_creativity.landscape import FitnessLandscape


class TestFitnessLandscape:
    """Tests for FitnessLandscape."""

    def test_deterministic_with_seed(self):
        """Same seed produces identical landscapes."""
        l1 = FitnessLandscape(n_peaks=10, seed=42)
        l2 = FitnessLandscape(n_peaks=10, seed=42)
        x = np.array([1.0, 2.0])
        assert l1.evaluate(x) == l2.evaluate(x)

    def test_different_seeds_differ(self):
        """Different seeds produce different landscapes."""
        l1 = FitnessLandscape(n_peaks=10, seed=42)
        l2 = FitnessLandscape(n_peaks=10, seed=99)
        x = np.array([1.0, 2.0])
        assert l1.evaluate(x) != l2.evaluate(x)

    def test_evaluate_returns_float(self):
        landscape = FitnessLandscape(seed=42)
        val = landscape.evaluate(np.array([0.0, 0.0]))
        assert isinstance(val, float)

    def test_evaluate_nonnegative(self):
        """Gaussian peaks are always non-negative."""
        landscape = FitnessLandscape(seed=42)
        for _ in range(100):
            x = np.random.uniform(-5, 5, size=2)
            assert landscape.evaluate(x) >= 0.0

    def test_batch_matches_single(self):
        """Batch evaluation matches individual evaluations."""
        landscape = FitnessLandscape(n_peaks=10, seed=42)
        xs = np.random.default_rng(42).uniform(-5, 5, size=(50, 2))
        batch = landscape.evaluate_batch(xs)
        for i, x in enumerate(xs):
            assert abs(batch[i] - landscape.evaluate(x)) < 1e-10

    def test_global_optimum_exists(self):
        landscape = FitnessLandscape(seed=42)
        pos, val = landscape.get_global_optimum()
        assert val > 0
        assert len(pos) == 2

    def test_global_optimum_is_best_peak(self):
        """Global optimum should be at least as good as any peak center."""
        landscape = FitnessLandscape(n_peaks=10, seed=42)
        _, opt_val = landscape.get_global_optimum()
        for peak in landscape.peaks:
            assert opt_val >= landscape.evaluate(peak.center) - 1e-10

    def test_grid_generation(self):
        landscape = FitnessLandscape(seed=42)
        X, Y, Z = landscape.get_grid(resolution=50)
        assert X.shape == (50, 50)
        assert Y.shape == (50, 50)
        assert Z.shape == (50, 50)
        assert Z.min() >= 0

    def test_peak_count(self):
        """n_peaks + 1 global peak are created."""
        landscape = FitnessLandscape(n_peaks=10, seed=42)
        assert len(landscape.peaks) == 11  # 10 + 1 global
