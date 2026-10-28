"""
Fixed-strategy optimizer (Level 1 — Intelligence).

Searches for solutions within a fixed space using a single, unchanging
search strategy. This is the computational analogue of intelligence:
generating ideas within a given generator.

Every evaluation goes through one ``search_core.Budget`` (SPEC section 2 and 6). The start point is
evaluation 1, and the first ``run`` call evaluates and records it, so the history always begins at the
start the optimiser was given.
"""

import numpy as np
from dataclasses import dataclass, field
from .landscape import FitnessLandscape
from .search_core import Budget
from .strategies import SearchStrategy


@dataclass
class OptimizationHistory:
    """One entry per recorded point: the start, then each step."""
    positions: list[np.ndarray] = field(default_factory=list)
    best_fitness: list[float] = field(default_factory=list)
    solution_diversity: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_fitness": list(self.best_fitness),
            "best_positions": [p.tolist() for p in self.positions],
            "solution_diversity": list(self.solution_diversity),
        }


class FixedStrategyOptimizer:
    """
    Level 1 — Intelligence: search within a fixed strategy.

    This optimizer uses a single, unchanging SearchStrategy to explore
    the fitness landscape. It can find good solutions within the space
    defined by its strategy, but cannot escape structural limitations
    of that strategy (e.g., a small step size trapping it in a local basin).

    ``seed`` may be an int or a numpy Generator. With ``x0`` None, the start is drawn from the stream.
    """

    def __init__(self, landscape: FitnessLandscape, strategy: SearchStrategy,
                 seed: int = 42, x0=None):
        self.landscape = landscape
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

        lo, hi = landscape.bounds
        start = self.rng.uniform(lo, hi, size=landscape.dims) if x0 is None else x0
        self.position = np.array(start, dtype=float)
        self.best_fitness = -np.inf
        self.best_position = self.position.copy()
        self.velocity = np.zeros(landscape.dims)

        self.budget: Budget | None = None
        self.history = OptimizationHistory()

    @property
    def trace(self) -> np.ndarray:
        """Best fitness so far after each evaluation; empty before ``run``."""
        return np.empty(0) if self.budget is None else self.budget.trace

    def step(self) -> float:
        """Execute one optimization step. Returns current best fitness."""
        candidates = self.strategy.propose(
            self.position, self.rng,
            velocity=self.velocity,
            bounds=self.landscape.bounds
        )
        values = self.budget.batch(candidates)
        k = len(values)
        best_idx = int(np.argmax(values))

        if values[best_idx] > self.best_fitness:
            old_pos = self.position.copy()
            self.position = np.array(candidates[best_idx], dtype=float)
            self.best_fitness = float(values[best_idx])
            self.best_position = self.position.copy()
            self.velocity = self.position - old_pos
        else:
            self.velocity *= 0.5  # decay velocity on no improvement

        # Track diversity as std of the candidates that were evaluated
        diversity = float(np.mean(np.std(candidates[:k], axis=0)))
        self._record(diversity)
        return self.best_fitness

    def _start(self):
        """Evaluate and record the start point: evaluation 1."""
        self.best_fitness = self.budget(self.position)
        self.best_position = self.position.copy()
        self._record(0.0)

    def _record(self, diversity: float):
        self.history.positions.append(self.position.copy())
        self.history.best_fitness.append(self.best_fitness)
        self.history.solution_diversity.append(diversity)

    def run(self, n_steps: int | None = None, max_evals: int | None = None) -> dict:
        """Run until ``max_evals`` evaluations are spent (the last batch is truncated), or for ``n_steps``.

        With only ``n_steps``, the budget is ``1 + n_steps * n_candidates``, so the run is exactly
        ``n_steps`` steps. Each optimiser runs once.
        """
        if self.budget is not None:
            raise RuntimeError("run() has already been called on this optimiser")
        if max_evals is None:
            if n_steps is None:
                raise ValueError("give n_steps or max_evals")
            max_evals = 1 + n_steps * self.strategy.n_candidates
        if max_evals < 1:
            raise ValueError(f"max_evals must be at least 1, not {max_evals}")
        self.budget = Budget(self.landscape.evaluate, max_evals, mode="max")
        self._start()
        steps = 0
        while self.budget.remaining > 0 and (n_steps is None or steps < n_steps):
            self.step()
            steps += 1
        return self.history.to_dict()
