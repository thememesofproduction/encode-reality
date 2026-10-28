"""
Search strategies and strategy mutation for Experiment 1.

A SearchStrategy encapsulates how candidate solutions are proposed.
The StrategyMutator modifies strategies, enabling meta-level search.
"""

import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy


METHODS = ("gaussian", "uniform", "restart", "momentum")
ACTIVATIONS = ("relu", "tanh", "sigmoid")  # for future use


@dataclass
class SearchStrategy:
    """
    A parameterized search strategy for exploring the fitness landscape.

    Attributes:
        step_size: Scale of perturbation when proposing candidates.
        n_candidates: Number of candidates to propose per step.
        method: How candidates are generated.
        momentum_decay: Decay factor for momentum method.
    """
    step_size: float = 0.3
    n_candidates: int = 20
    method: str = "gaussian"
    momentum_decay: float = 0.9

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"Unknown method: {self.method}. Must be one of {METHODS}")

    def propose(self, current: np.ndarray, rng: np.random.Generator,
                velocity: np.ndarray | None = None,
                bounds: tuple[float, float] = (-5.0, 5.0)) -> np.ndarray:
        """
        Propose candidate solutions around the current position.

        Returns:
            Array of shape (n_candidates, dims) with proposed positions.
        """
        dims = len(current)
        lo, hi = bounds

        if self.method == "gaussian":
            perturbations = rng.normal(0, self.step_size, size=(self.n_candidates, dims))
            candidates = current + perturbations

        elif self.method == "uniform":
            perturbations = rng.uniform(-self.step_size * 2, self.step_size * 2,
                                        size=(self.n_candidates, dims))
            candidates = current + perturbations

        elif self.method == "restart":
            # Half near current, half random restarts
            n_local = self.n_candidates // 2
            n_restart = self.n_candidates - n_local
            local = current + rng.normal(0, self.step_size, size=(n_local, dims))
            restarts = rng.uniform(lo, hi, size=(n_restart, dims))
            candidates = np.vstack([local, restarts])

        elif self.method == "momentum":
            if velocity is None:
                velocity = np.zeros(dims)
            # Bias proposals in the direction of momentum
            biased_center = current + velocity * self.momentum_decay
            perturbations = rng.normal(0, self.step_size, size=(self.n_candidates, dims))
            candidates = biased_center + perturbations

        # Clip to bounds
        candidates = np.clip(candidates, lo, hi)
        return candidates

    def label(self) -> str:
        """Short human-readable label."""
        return f"{self.method}(s={self.step_size:.2f}, n={self.n_candidates})"


class StrategyMutator:
    """
    Mutates search strategies to create new ones.

    This is the mechanism that enables creativity: rather than searching
    within a fixed strategy, the mutator modifies the strategy itself.
    """

    def __init__(self, mutation_rate: float = 0.3, seed: int = 42):
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)

    def mutate(self, strategy: SearchStrategy) -> SearchStrategy:
        """Create a mutated copy of the strategy."""
        new = deepcopy(strategy)

        if self.rng.random() < self.mutation_rate:
            # Mutate step size (log-scale)
            log_step = np.log(new.step_size) + self.rng.normal(0, 0.5)
            new.step_size = float(np.clip(np.exp(log_step), 0.01, 5.0))

        if self.rng.random() < self.mutation_rate:
            # Mutate candidate count
            new.n_candidates = int(np.clip(
                new.n_candidates + self.rng.integers(-10, 11), 5, 100
            ))

        if self.rng.random() < self.mutation_rate:
            # Switch method entirely
            new.method = self.rng.choice(METHODS)

        if self.rng.random() < self.mutation_rate:
            new.momentum_decay = float(np.clip(
                new.momentum_decay + self.rng.normal(0, 0.1), 0.5, 0.99
            ))

        return new

    def crossover(self, s1: SearchStrategy, s2: SearchStrategy) -> SearchStrategy:
        """Create a child strategy by combining two parents."""
        return SearchStrategy(
            step_size=s1.step_size if self.rng.random() < 0.5 else s2.step_size,
            n_candidates=s1.n_candidates if self.rng.random() < 0.5 else s2.n_candidates,
            method=s1.method if self.rng.random() < 0.5 else s2.method,
            momentum_decay=s1.momentum_decay if self.rng.random() < 0.5 else s2.momentum_decay,
        )

    def random_strategy(self) -> SearchStrategy:
        """Generate a completely novel random strategy."""
        return SearchStrategy(
            step_size=float(np.exp(self.rng.uniform(np.log(0.01), np.log(5.0)))),
            n_candidates=int(self.rng.integers(5, 100)),
            method=self.rng.choice(METHODS),
            momentum_decay=float(self.rng.uniform(0.5, 0.99)),
        )
