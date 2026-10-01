"""Tests for ABM core: Agent and Population."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from experiments.models_define_world.abm_core import Agent, Population, S, I, R


class TestAgent:
    def test_default_compliance(self):
        agent = Agent(state=S, beta=0.3, gamma=0.1, omega=0.01)
        assert agent.compliance == 0.0

    def test_state_constants(self):
        assert S == 0
        assert I == 1
        assert R == 2


class TestPopulation:
    def test_initial_counts(self):
        """N-I0 susceptible, I0 infected, 0 recovered."""
        pop = Population(N=100, I0=5, beta=0.3, gamma=0.1, omega=0.01, seed=42)
        counts = pop.get_counts()
        assert counts[S] == 95
        assert counts[I] == 5
        assert counts[R] == 0

    def test_total_population(self):
        pop = Population(N=200, I0=10, beta=0.3, gamma=0.1, omega=0.01, seed=42)
        counts = pop.get_counts()
        assert sum(counts.values()) == 200

    def test_deterministic_with_seed(self):
        pop1 = Population(N=100, I0=5, beta=0.3, gamma=0.1, omega=0.01, seed=42,
                          heterogeneity=True, beta_std=0.1)
        pop2 = Population(N=100, I0=5, beta=0.3, gamma=0.1, omega=0.01, seed=42,
                          heterogeneity=True, beta_std=0.1)
        betas1 = [a.beta for a in pop1.agents]
        betas2 = [a.beta for a in pop2.agents]
        assert betas1 == betas2

    def test_homogeneous_identical_beta(self):
        pop = Population(N=50, I0=5, beta=0.3, gamma=0.1, omega=0.01, seed=42,
                         heterogeneity=False)
        betas = {a.beta for a in pop.agents}
        assert len(betas) == 1
        assert betas.pop() == pytest.approx(0.3)

    def test_heterogeneous_varied_beta(self):
        pop = Population(N=200, I0=5, beta=0.3, gamma=0.1, omega=0.01, seed=42,
                         heterogeneity=True, beta_std=0.1)
        betas = [a.beta for a in pop.agents]
        assert np.std(betas) > 0.01

    def test_heterogeneous_beta_clipped(self):
        pop = Population(N=200, I0=5, beta=0.3, gamma=0.1, omega=0.01, seed=42,
                         heterogeneity=True, beta_std=0.5)
        for a in pop.agents:
            assert 0.01 <= a.beta <= 1.0

    def test_reset(self):
        pop = Population(N=100, I0=5, beta=0.3, gamma=0.1, omega=0.01, seed=42)
        # Manually change some states
        pop.agents[0].state = R
        pop.reset()
        counts = pop.get_counts()
        assert counts[S] == 95
        assert counts[I] == 5
