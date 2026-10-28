"""Tests for local hidden-variable models and CHSH computation."""

import numpy as np
import pytest
from itertools import product

from experiments.bells_inequality.local_hv import (
    chsh_s,
    chsh_single_lambda,
    enumerate_deterministic,
    random_discrete_lhv,
    random_continuous_lhv,
    adversarial_optimize,
)
from experiments.bells_inequality.config import CLASSICAL_BOUND


class TestCHSHFormula:
    """Test the CHSH quantity computation."""

    def test_chsh_s_basic(self):
        """S = E11 - E12 + E21 + E22."""
        assert chsh_s(1.0, -1.0, 1.0, 1.0) == 1.0 - (-1.0) + 1.0 + 1.0
        assert chsh_s(0.5, 0.5, 0.5, 0.5) == 0.5 - 0.5 + 0.5 + 0.5

    def test_chsh_single_lambda_always_pm2(self):
        """For any deterministic assignment, X = +/-2."""
        for A0, A1, B0, B1 in product([-1, 1], repeat=4):
            X = chsh_single_lambda(A0, A1, B0, B1)
            assert X in (-2, 2), f"X={X} for A0={A0},A1={A1},B0={B0},B1={B1}"


class TestDeterministicStrategies:
    """Test exhaustive enumeration of deterministic strategies."""

    def test_enumerate_count(self):
        """There are exactly 2^4 = 16 deterministic strategies."""
        S = enumerate_deterministic()
        assert len(S) == 16

    def test_enumerate_all_pm2(self):
        """All deterministic strategies give |S| = 2."""
        S = enumerate_deterministic()
        np.testing.assert_array_equal(np.abs(S), 2.0)


class TestRandomDiscrete:
    """Test random discrete LHV models."""

    def test_bound_holds(self):
        """All random discrete strategies satisfy |S| <= 2."""
        rng = np.random.default_rng(42)
        S = random_discrete_lhv(500, 1000, rng)
        assert np.all(np.abs(S) <= CLASSICAL_BOUND + 1e-10)

    def test_reproducible(self):
        """Same seed produces same results."""
        S1 = random_discrete_lhv(10, 100, np.random.default_rng(99))
        S2 = random_discrete_lhv(10, 100, np.random.default_rng(99))
        np.testing.assert_array_equal(S1, S2)


class TestRandomContinuous:
    """Test random continuous LHV models."""

    def test_bound_holds(self):
        """All random continuous strategies satisfy |S| <= 2."""
        rng = np.random.default_rng(42)
        S = random_continuous_lhv(500, 1000, rng)
        assert np.all(np.abs(S) <= CLASSICAL_BOUND + 1e-10)


class TestAdversarial:
    """Test adversarial optimization of LHV models."""

    def test_cannot_exceed_bound(self):
        """Adversarial optimizer cannot exceed |S| = 2."""
        rng = np.random.default_rng(42)
        best_S = adversarial_optimize(20, rng)
        assert np.all(best_S <= CLASSICAL_BOUND + 1e-6), \
            f"Adversarial exceeded bound: max |S| = {np.max(best_S)}"

    def test_approaches_bound(self):
        """Adversarial optimizer should find strategies near |S| = 2."""
        rng = np.random.default_rng(42)
        best_S = adversarial_optimize(20, rng)
        assert np.max(best_S) > 1.9, \
            f"Adversarial too far from bound: max |S| = {np.max(best_S)}"
