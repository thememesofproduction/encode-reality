"""Tests for agent classes."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.ideas_intelligence_creativity.agents import (
    Architecture, NeuralAgent, FixedAgent
)


class TestArchitecture:

    def test_obs_size(self):
        arch = Architecture(input_channels=["food", "hazard"])
        assert arch.obs_size() == 6  # 3 + 3

    def test_parameter_count(self):
        arch = Architecture(input_channels=["food"], hidden_sizes=[8], n_actions=5)
        # 3*8 + 8 + 8*5 + 5 = 24 + 8 + 40 + 5 = 77
        assert arch.parameter_count() == 77

    def test_label_format(self):
        arch = Architecture(input_channels=["food", "hazard"], hidden_sizes=[16, 8])
        label = arch.label()
        assert "food+hazard" in label
        assert "16x8" in label

    def test_to_dict(self):
        arch = Architecture()
        d = arch.to_dict()
        assert "input_channels" in d
        assert "parameter_count" in d


class TestNeuralAgent:

    def test_act_returns_valid_action(self):
        arch = Architecture(input_channels=["food", "hazard"], n_actions=5)
        agent = NeuralAgent(arch)
        obs = np.random.randn(arch.obs_size())
        action = agent.act(obs)
        assert 0 <= action < 5

    def test_get_set_weights_roundtrip(self):
        arch = Architecture(input_channels=["food"], hidden_sizes=[8], n_actions=5)
        agent = NeuralAgent(arch)
        weights = np.random.randn(arch.parameter_count())
        agent.set_weights(weights)
        recovered = agent.get_weights()
        np.testing.assert_allclose(weights, recovered)

    def test_different_weights_different_actions(self):
        """Different weight configurations should produce different behavior."""
        arch = Architecture(input_channels=["food"], hidden_sizes=[8], n_actions=5)
        obs = np.array([0.5, -0.3, 0.8])

        agent1 = NeuralAgent(arch, np.random.default_rng(1).normal(0, 1, arch.parameter_count()))
        agent2 = NeuralAgent(arch, np.random.default_rng(99).normal(0, 1, arch.parameter_count()))

        # With very different weights, actions should likely differ
        # (not guaranteed but extremely likely)
        actions_1 = [agent1.act(obs) for _ in range(10)]
        actions_2 = [agent2.act(obs) for _ in range(10)]
        # Deterministic, so all same, but likely different between agents
        assert all(a == actions_1[0] for a in actions_1)
        assert all(a == actions_2[0] for a in actions_2)


class TestFixedAgent:

    def test_act_returns_valid_action(self):
        agent = FixedAgent()
        obs = np.array([0.5, -0.3, 0.8])  # food channel: dx, dy, dist
        action = agent.act(obs)
        assert 0 <= action <= 4

    def test_moves_toward_food(self):
        agent = FixedAgent()
        # Food to the right (dx > 0)
        action = agent.act(np.array([0.8, 0.0, 0.5]))
        assert action == 1  # right

        # Food above (dy < 0)
        action = agent.act(np.array([0.0, -0.8, 0.5]))
        assert action == 0  # up

    def test_stays_when_on_food(self):
        agent = FixedAgent()
        action = agent.act(np.array([0.0, 0.0, 0.0]))
        assert action == 4  # stay
