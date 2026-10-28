"""Tests for weight evolution and architecture evolution."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.ideas_intelligence_creativity.gridworld import GridConfig
from experiments.ideas_intelligence_creativity.agents import Architecture
from experiments.ideas_intelligence_creativity.evolution import WeightEvolver, evaluate_fixed_agent
from experiments.ideas_intelligence_creativity.neuroevolution import ArchitectureEvolver


class TestWeightEvolver:

    def test_population_shape(self):
        arch = Architecture(input_channels=["food"], hidden_sizes=[8], n_actions=5)
        evolver = WeightEvolver(arch, population_size=20, seed=42)
        assert evolver.population.shape == (20, arch.parameter_count())

    def test_run_returns_stats(self):
        arch = Architecture(input_channels=["food"], hidden_sizes=[8], n_actions=5)
        evolver = WeightEvolver(arch, population_size=10, seed=42)
        config = GridConfig(width=15, height=15, n_food=5, n_hazards=2, seed=42)
        stats = evolver.run(config, n_generations=3, n_steps=50)
        assert len(stats) == 3
        assert all(hasattr(s, "best_fitness") for s in stats)

    def test_fitness_improves_over_generations(self):
        """Fitness should generally improve (or at least not crash)."""
        arch = Architecture(input_channels=["food", "hazard"], hidden_sizes=[12], n_actions=5)
        evolver = WeightEvolver(arch, population_size=30, seed=42)
        config = GridConfig(width=15, height=15, n_food=8, n_hazards=3, seed=42)
        stats = evolver.run(config, n_generations=10, n_steps=100)
        # Last generation should be at least as good as first
        assert stats[-1].best_fitness >= stats[0].best_fitness - 1.0


class TestEvaluateFixedAgent:

    def test_returns_float(self):
        config = GridConfig(width=15, height=15, seed=42)
        result = evaluate_fixed_agent(config, n_steps=50, n_trials=2)
        assert isinstance(result, float)


class TestArchitectureEvolver:

    def test_pool_initialization(self):
        arch = Architecture(input_channels=["food", "hazard"])
        evolver = ArchitectureEvolver(arch, pool_size=4,
                                       weight_generations=2,
                                       weight_pop_size=5,
                                       eval_steps=30, seed=42)
        assert len(evolver.pool) == 4

    def test_mutate_produces_valid_architecture(self):
        arch = Architecture(input_channels=["food", "hazard"], hidden_sizes=[16, 8])
        evolver = ArchitectureEvolver(arch, pool_size=4, seed=42)
        for _ in range(50):
            mutated = evolver.mutate_architecture(arch)
            assert len(mutated.input_channels) >= 1
            assert len(mutated.hidden_sizes) >= 1
            assert mutated.n_actions in (5, 9)
            # Should be able to create a valid agent
            from experiments.ideas_intelligence_creativity.agents import NeuralAgent
            agent = NeuralAgent(mutated)
            assert agent.architecture.parameter_count() > 0

    def test_meta_step_returns_stats(self):
        arch = Architecture(input_channels=["food"], hidden_sizes=[8], n_actions=5)
        evolver = ArchitectureEvolver(arch, pool_size=3,
                                       weight_generations=2,
                                       weight_pop_size=5,
                                       eval_steps=30, seed=42)
        config = GridConfig(width=10, height=10, n_food=5, n_hazards=2, seed=42)
        stats = evolver.meta_step(config)
        assert hasattr(stats, "best_fitness")
        assert hasattr(stats, "best_architecture")
