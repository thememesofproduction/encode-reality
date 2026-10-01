"""
Fixed-strategy optimizer (Level 1 — Intelligence).

Searches for solutions within a fixed space using a single, unchanging
search strategy. This is the computational analogue of intelligence:
generating ideas within a given generator.
"""

import numpy as np
from dataclasses import dataclass, field
from .landscape import FitnessLandscape
from .strategies import SearchStrategy


@dataclass
class OptimizationHistory:
    """Tracks the full history of an optimization run."""
    best_fitness: list[float] = field(default_factory=list)
    best_positions: list[np.ndarray] = field(default_factory=list)
    all_positions: list[np.ndarray] = field(default_factory=list)
    solution_diversity: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_fitness": self.best_fitness,
            "best_positions": [p.tolist() for p in self.best_positions],
            "solution_diversity": self.solution_diversity,
        }


class FixedStrategyOptimizer:
    """
    Level 1 — Intelligence: search within a fixed strategy.

    This optimizer uses a single, unchanging SearchStrategy to explore
    the fitness landscape. It can find good solutions within the space
    defined by its strategy, but cannot escape structural limitations
    of that strategy (e.g., a small step size trapping it in a local basin).
    """

    def __init__(self, landscape: FitnessLandscape, strategy: SearchStrategy,
                 seed: int = 42):
        self.landscape = landscape
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

        # Initialize at a random position
        lo, hi = landscape.bounds
        self.position = self.rng.uniform(lo, hi, size=landscape.dims)
        self.best_fitness = landscape.evaluate(self.position)
        self.best_position = self.position.copy()
        self.velocity = np.zeros(landscape.dims)

        self.history = OptimizationHistory()
        self._record()

    def step(self) -> float:
        """Execute one optimization step. Returns current best fitness."""
        candidates = self.strategy.propose(
            self.position, self.rng,
            velocity=self.velocity,
            bounds=self.landscape.bounds
        )
        fitnesses = self.landscape.evaluate_batch(candidates)
        best_idx = np.argmax(fitnesses)

        if fitnesses[best_idx] > self.best_fitness:
            old_pos = self.position.copy()
            self.position = candidates[best_idx]
            self.best_fitness = fitnesses[best_idx]
            self.best_position = self.position.copy()
            self.velocity = self.position - old_pos
        else:
            self.velocity *= 0.5  # decay velocity on no improvement

        # Track diversity as std of candidate positions
        diversity = float(np.mean(np.std(candidates, axis=0)))

        self.history.best_fitness.append(self.best_fitness)
        self.history.best_positions.append(self.best_position.copy())
        self.history.all_positions.append(self.position.copy())
        self.history.solution_diversity.append(diversity)

        return self.best_fitness

    def _record(self):
        """Record initial state."""
        self.history.best_fitness.append(self.best_fitness)
        self.history.best_positions.append(self.best_position.copy())
        self.history.all_positions.append(self.position.copy())
        self.history.solution_diversity.append(0.0)

    def run(self, n_steps: int) -> dict:
        """Run the optimizer for n_steps and return history."""
        for _ in range(n_steps):
            self.step()
        return self.history.to_dict()
