"""Tests for the SIRS ODE solver."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from experiments.models_define_world.sirs_ode import solve_sirs, SIRSResult


class TestSIRSResult:
    def test_to_dict_keys(self):
        result = solve_sirs(beta=0.3, gamma=0.1, omega=0.01, N=1000, I0=10, t_max=100)
        d = result.to_dict()
        assert set(d.keys()) == {"t", "S", "I", "R"}

    def test_to_dict_serializable(self):
        result = solve_sirs(beta=0.3, gamma=0.1, omega=0.01, N=1000, I0=10, t_max=100)
        d = result.to_dict()
        for v in d.values():
            assert isinstance(v, list)


class TestSolveSIRS:
    def test_conservation(self):
        """S + I + R = N at all times."""
        result = solve_sirs(beta=0.3, gamma=0.1, omega=0.01, N=1000, I0=10, t_max=200)
        total = result.S + result.I + result.R
        np.testing.assert_allclose(total, 1000.0, atol=0.1)

    def test_deterministic(self):
        """Same params produce identical results."""
        r1 = solve_sirs(beta=0.3, gamma=0.1, omega=0.01, N=1000, I0=10, t_max=100)
        r2 = solve_sirs(beta=0.3, gamma=0.1, omega=0.01, N=1000, I0=10, t_max=100)
        np.testing.assert_array_equal(r1.I, r2.I)

    def test_initial_conditions(self):
        """First time point matches initial conditions."""
        result = solve_sirs(beta=0.3, gamma=0.1, omega=0.01, N=1000, I0=10, t_max=100)
        assert result.S[0] == pytest.approx(990.0, abs=0.1)
        assert result.I[0] == pytest.approx(10.0, abs=0.1)
        assert result.R[0] == pytest.approx(0.0, abs=0.1)

    def test_no_infection_stable(self):
        """With I0=0, no dynamics occur."""
        result = solve_sirs(beta=0.3, gamma=0.1, omega=0.01, N=1000, I0=0, t_max=100)
        np.testing.assert_allclose(result.I, 0.0, atol=1e-6)

    def test_output_length(self):
        """Result arrays have expected length."""
        result = solve_sirs(beta=0.3, gamma=0.1, omega=0.01, N=1000, I0=10,
                            t_max=100, n_points=500)
        assert len(result.t) == 500
        assert len(result.S) == 500
        assert len(result.I) == 500
        assert len(result.R) == 500

    def test_r0_below_one_dies_out(self):
        """When R0 = beta/gamma < 1, infection dies out."""
        result = solve_sirs(beta=0.05, gamma=0.1, omega=0.01, N=1000, I0=10, t_max=500)
        assert result.I[-1] < 1.0

    def test_epidemic_peak(self):
        """With R0 > 1, infection should peak above initial."""
        result = solve_sirs(beta=0.3, gamma=0.1, omega=0.01, N=1000, I0=10, t_max=200)
        assert np.max(result.I) > 10.0


import pytest
