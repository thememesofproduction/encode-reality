"""Tests for quantum singlet state measurements and CHSH violation."""

import numpy as np
import pytest

from experiments.bells_inequality.quantum import (
    singlet_correlation,
    quantum_chsh_s,
    singlet_joint_probabilities,
    simulate_measurements,
    estimate_correlation,
)
from experiments.bells_inequality.config import (
    OPTIMAL_ANGLES, CLASSICAL_BOUND, TSIRELSON_BOUND,
)


class TestSingletCorrelation:
    """Test the exact correlation function E(a,b) = -cos(a-b)."""

    def test_same_angle(self):
        """E(a,a) = -cos(0) = -1 (perfect anti-correlation)."""
        assert singlet_correlation(0, 0) == pytest.approx(-1.0)
        assert singlet_correlation(1.0, 1.0) == pytest.approx(-1.0)

    def test_opposite_angle(self):
        """E(a, a+pi) = -cos(pi) = 1 (perfect correlation)."""
        assert singlet_correlation(0, np.pi) == pytest.approx(1.0)

    def test_orthogonal(self):
        """E(a, a+pi/2) = -cos(pi/2) = 0 (no correlation)."""
        assert singlet_correlation(0, np.pi / 2) == pytest.approx(0.0)

    def test_symmetry(self):
        """E(a,b) = E(b,a) since cos is even."""
        a, b = 0.3, 1.7
        assert singlet_correlation(a, b) == pytest.approx(
            singlet_correlation(b, a))


class TestQuantumCHSH:
    """Test CHSH computation for the singlet state."""

    def test_optimal_angles_violate(self):
        """Optimal angles produce |S| = 2*sqrt(2)."""
        S = quantum_chsh_s(**OPTIMAL_ANGLES)
        assert abs(S) == pytest.approx(TSIRELSON_BOUND, rel=1e-10)

    def test_exceeds_classical_bound(self):
        """Quantum S exceeds classical bound of 2."""
        S = quantum_chsh_s(**OPTIMAL_ANGLES)
        assert abs(S) > CLASSICAL_BOUND

    def test_zero_for_aligned_settings(self):
        """All same angle: S = 0 (correlations cancel)."""
        S = quantum_chsh_s(0, 0, 0, 0)
        # E(0,0) = -1 for all pairs
        # S = -1 - (-1) + (-1) + (-1) = -2
        assert S == pytest.approx(-2.0)


class TestJointProbabilities:
    """Test Born rule joint probabilities for the singlet state."""

    def test_normalization(self):
        """Probabilities sum to 1."""
        probs = singlet_joint_probabilities(0.3, 1.2)
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_same_angle_anticorrelation(self):
        """Same angle: only (+1,-1) and (-1,+1) outcomes."""
        probs = singlet_joint_probabilities(0, 0)
        assert probs[(+1, +1)] == pytest.approx(0.0, abs=1e-15)
        assert probs[(-1, -1)] == pytest.approx(0.0, abs=1e-15)
        assert probs[(+1, -1)] == pytest.approx(0.5)
        assert probs[(-1, +1)] == pytest.approx(0.5)

    def test_opposite_angle_correlation(self):
        """Opposite angles: only (+1,+1) and (-1,-1) outcomes."""
        probs = singlet_joint_probabilities(0, np.pi)
        assert probs[(+1, -1)] == pytest.approx(0.0, abs=1e-15)
        assert probs[(-1, +1)] == pytest.approx(0.0, abs=1e-15)
        assert probs[(+1, +1)] == pytest.approx(0.5)
        assert probs[(-1, -1)] == pytest.approx(0.5)


class TestSimulatedMeasurements:
    """Test Monte Carlo simulation of singlet measurements."""

    def test_correlation_converges(self):
        """Simulated E(a,b) should converge to -cos(a-b)."""
        rng = np.random.default_rng(42)
        theta_a, theta_b = 0, np.pi / 4
        A, B = simulate_measurements(theta_a, theta_b, 100_000, rng)
        E_sim = estimate_correlation(A, B)
        E_theory = -np.cos(theta_a - theta_b)
        assert E_sim == pytest.approx(E_theory, abs=0.02)

    def test_outcomes_are_pm1(self):
        """All measurement outcomes are +/-1."""
        rng = np.random.default_rng(42)
        A, B = simulate_measurements(0, 1.0, 1000, rng)
        assert set(np.unique(A)) == {-1, 1}
        assert set(np.unique(B)) == {-1, 1}

    def test_marginals_uniform(self):
        """Each individual outcome is uniformly +/-1 (singlet property)."""
        rng = np.random.default_rng(42)
        A, B = simulate_measurements(0, np.pi / 3, 100_000, rng)
        assert np.mean(A) == pytest.approx(0, abs=0.02)
        assert np.mean(B) == pytest.approx(0, abs=0.02)

    def test_simulated_chsh_violation(self):
        """Simulated CHSH S at optimal angles should exceed 2."""
        rng = np.random.default_rng(42)
        angles = OPTIMAL_ANGLES
        n = 50_000

        E = {}
        for a_key, b_key in [("a1", "b1"), ("a1", "b2"),
                              ("a2", "b1"), ("a2", "b2")]:
            A, B = simulate_measurements(angles[a_key], angles[b_key], n, rng)
            E[(a_key, b_key)] = estimate_correlation(A, B)

        S = (E[("a1", "b1")] - E[("a1", "b2")]
             + E[("a2", "b1")] + E[("a2", "b2")])
        assert abs(S) > CLASSICAL_BOUND, f"|S| = {abs(S)} did not exceed 2"
