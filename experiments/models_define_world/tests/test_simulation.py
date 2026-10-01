"""Tests for the ABM simulation engine."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from experiments.models_define_world.abm_core import Population, S, I, R
from experiments.models_define_world.simulation import run_abm, SimulationHistory
from experiments.models_define_world.networks import generate_network


class TestSimulationHistory:
    def test_to_dict_keys(self):
        pop = Population(N=50, I0=5, beta=0.3, gamma=0.1, omega=0.01, seed=42)
        rng = np.random.default_rng(42)
        hist = run_abm(pop, n_steps=10, rng=rng, contact_rate=5, dt=0.1)
        d = hist.to_dict()
        assert set(d.keys()) == {"S", "I", "R"}

    def test_history_shape(self):
        pop = Population(N=50, I0=5, beta=0.3, gamma=0.1, omega=0.01, seed=42)
        rng = np.random.default_rng(42)
        hist = run_abm(pop, n_steps=100, rng=rng, contact_rate=5, dt=0.1)
        assert len(hist.S_counts) == 100
        assert len(hist.I_counts) == 100
        assert len(hist.R_counts) == 100


class TestRunABM:
    def test_conservation(self):
        """S + I + R = N at every timestep."""
        N = 100
        pop = Population(N=N, I0=10, beta=0.3, gamma=0.1, omega=0.01, seed=42)
        rng = np.random.default_rng(42)
        hist = run_abm(pop, n_steps=200, rng=rng, contact_rate=10, dt=0.1)
        total = hist.S_counts + hist.I_counts + hist.R_counts
        np.testing.assert_array_equal(total, N)

    def test_deterministic_with_seed(self):
        """Same seed produces identical trajectory."""
        def run_once(seed):
            pop = Population(N=100, I0=10, beta=0.3, gamma=0.1, omega=0.01, seed=seed)
            rng = np.random.default_rng(seed)
            return run_abm(pop, n_steps=50, rng=rng, contact_rate=10, dt=0.1)
        h1 = run_once(42)
        h2 = run_once(42)
        np.testing.assert_array_equal(h1.I_counts, h2.I_counts)

    def test_no_infection_stable(self):
        """With I0=0, no infections occur."""
        pop = Population(N=100, I0=0, beta=0.3, gamma=0.1, omega=0.01, seed=42)
        rng = np.random.default_rng(42)
        hist = run_abm(pop, n_steps=50, rng=rng, contact_rate=10, dt=0.1)
        np.testing.assert_array_equal(hist.I_counts, 0)

    def test_with_network(self):
        """Simulation runs on a network without error."""
        G = generate_network("barabasi_albert", n_nodes=100, seed=42, m=3)
        pop = Population(N=100, I0=10, beta=0.3, gamma=0.1, omega=0.01, seed=42)
        rng = np.random.default_rng(42)
        hist = run_abm(pop, n_steps=50, rng=rng, network=G, dt=0.1)
        total = hist.S_counts + hist.I_counts + hist.R_counts
        np.testing.assert_array_equal(total, 100)

    def test_recovery_approximates_ode(self):
        """ABM ensemble mean peak I within 15% of ODE peak I (homogeneous mixing)."""
        from experiments.models_define_world.sirs_ode import solve_sirs

        N = 1000
        beta, gamma, omega = 0.3, 0.1, 0.01
        dt = 300 / 3000
        ode = solve_sirs(beta=beta, gamma=gamma, omega=omega, N=N, I0=10, t_max=300,
                         n_points=3000)
        ode_peak = np.max(ode.I)

        peaks = []
        for seed in range(10):
            pop = Population(N=N, I0=10, beta=beta, gamma=gamma, omega=omega, seed=seed)
            rng = np.random.default_rng(seed)
            hist = run_abm(pop, n_steps=3000, rng=rng, contact_rate=10, dt=dt)
            peaks.append(np.max(hist.I_counts))

        abm_mean_peak = np.mean(peaks)
        assert abs(abm_mean_peak - ode_peak) / ode_peak < 0.15, \
            f"ABM mean peak {abm_mean_peak:.1f} too far from ODE peak {ode_peak:.1f}"

    def test_dt_affects_dynamics(self):
        """Different dt values produce different peak timing."""
        N = 500
        beta, gamma, omega = 0.3, 0.1, 0.01
        n_steps = 1000

        def run_with_dt(dt_val, seed=42):
            pop = Population(N=N, I0=10, beta=beta, gamma=gamma, omega=omega, seed=seed)
            rng = np.random.default_rng(seed)
            hist = run_abm(pop, n_steps=n_steps, rng=rng, contact_rate=10, dt=dt_val)
            return np.argmax(hist.I_counts)

        peak_small_dt = run_with_dt(0.05)
        peak_large_dt = run_with_dt(0.5)
        assert peak_small_dt != peak_large_dt, \
            "Different dt values should produce different peak timing"
