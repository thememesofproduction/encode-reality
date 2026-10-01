"""
Meta-strategy optimizer (Level 2 — Creativity).

Maintains a pool of search strategies and periodically mutates them
based on performance. This is the computational analogue of creativity:
a generator that operates over generators, producing new ways of
generating solutions.
"""

import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy
from .landscape import FitnessLandscape
from .strategies import SearchStrategy, StrategyMutator


@dataclass
class StrategyRecord:
    """Tracks a strategy and its recent performance."""
    strategy: SearchStrategy
    recent_improvements: list[float] = field(default_factory=list)
    total_steps: int = 0

    @property
    def score(self) -> float:
        if not self.recent_improvements:
            return 0.0
        return float(np.mean(self.recent_improvements))


@dataclass
class CreativityHistory:
    """Tracks the full history including strategy-level changes."""
    best_fitness: list[float] = field(default_factory=list)
    best_positions: list[np.ndarray] = field(default_factory=list)
    all_positions: list[np.ndarray] = field(default_factory=list)
    solution_diversity: list[float] = field(default_factory=list)
    strategy_labels: list[str] = field(default_factory=list)
    strategy_diversity: list[float] = field(default_factory=list)
    meta_events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_fitness": self.best_fitness,
            "best_positions": [p.tolist() for p in self.best_positions],
            "solution_diversity": self.solution_diversity,
            "strategy_labels": self.strategy_labels,
            "strategy_diversity": self.strategy_diversity,
            "meta_events": self.meta_events,
        }


class MetaStrategyOptimizer:
    """
    Level 2 — Creativity: search over search strategies.

    Maintains a pool of SearchStrategy instances. At the base level, it
    uses the active strategy to search the fitness landscape (just like
    FixedStrategyOptimizer). But periodically, it performs a meta-step:
    evaluating strategies by their recent improvement rate, dropping
    underperformers, and generating new strategies via mutation/crossover.

    This is a generator of generators. It does not just produce solutions —
    it produces new ways of producing solutions.
    """

    def __init__(self, landscape: FitnessLandscape,
                 initial_strategy: SearchStrategy,
                 mutator: StrategyMutator,
                 meta_interval: int = 50,
                 strategy_pool_size: int = 5,
                 improvement_window: int = 25,
                 novel_strategy_prob: float = 0.2,
                 seed: int = 42):
        self.landscape = landscape
        self.mutator = mutator
        self.meta_interval = meta_interval
        self.pool_size = strategy_pool_size
        self.improvement_window = improvement_window
        self.novel_strategy_prob = novel_strategy_prob
        self.rng = np.random.default_rng(seed)

        # Initialize position
        lo, hi = landscape.bounds
        self.position = self.rng.uniform(lo, hi, size=landscape.dims)
        self.best_fitness = landscape.evaluate(self.position)
        self.best_position = self.position.copy()
        self.velocity = np.zeros(landscape.dims)

        # Initialize strategy pool
        self.pool: list[StrategyRecord] = [
            StrategyRecord(strategy=deepcopy(initial_strategy))
        ]
        for _ in range(strategy_pool_size - 1):
            self.pool.append(StrategyRecord(strategy=mutator.mutate(initial_strategy)))

        self.active_idx = 0
        self.step_count = 0

        self.history = CreativityHistory()
        self._record_initial()

    @property
    def active_strategy(self) -> SearchStrategy:
        return self.pool[self.active_idx].strategy

    def _record_initial(self):
        self.history.best_fitness.append(self.best_fitness)
        self.history.best_positions.append(self.best_position.copy())
        self.history.all_positions.append(self.position.copy())
        self.history.solution_diversity.append(0.0)
        self.history.strategy_labels.append(self.active_strategy.label())
        self.history.strategy_diversity.append(self._strategy_entropy())

    def _strategy_entropy(self) -> float:
        """Shannon entropy over the method distribution in the pool."""
        methods = [r.strategy.method for r in self.pool]
        unique, counts = np.unique(methods, return_counts=True)
        probs = counts / counts.sum()
        entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
        return entropy

    def step(self) -> float:
        """Execute one search step with the active strategy."""
        record = self.pool[self.active_idx]

        candidates = record.strategy.propose(
            self.position, self.rng,
            velocity=self.velocity,
            bounds=self.landscape.bounds
        )
        fitnesses = self.landscape.evaluate_batch(candidates)
        best_idx = np.argmax(fitnesses)

        improvement = 0.0
        if fitnesses[best_idx] > self.best_fitness:
            old_pos = self.position.copy()
            improvement = fitnesses[best_idx] - self.best_fitness
            self.position = candidates[best_idx]
            self.best_fitness = fitnesses[best_idx]
            self.best_position = self.position.copy()
            self.velocity = self.position - old_pos
        else:
            self.velocity *= 0.5

        # Track strategy performance
        record.recent_improvements.append(improvement)
        if len(record.recent_improvements) > self.improvement_window:
            record.recent_improvements.pop(0)
        record.total_steps += 1

        # Track diversity
        diversity = float(np.mean(np.std(candidates, axis=0)))

        self.history.best_fitness.append(self.best_fitness)
        self.history.best_positions.append(self.best_position.copy())
        self.history.all_positions.append(self.position.copy())
        self.history.solution_diversity.append(diversity)
        self.history.strategy_labels.append(self.active_strategy.label())
        self.history.strategy_diversity.append(self._strategy_entropy())

        self.step_count += 1

        # Trigger meta-step periodically
        if self.step_count % self.meta_interval == 0:
            self.meta_step()

        return self.best_fitness

    def meta_step(self):
        """
        Evaluate and evolve the strategy pool.

        This is the creative act: modifying the generator itself based on
        how well different generators have been performing.
        """
        # Score all strategies
        scores = [r.score for r in self.pool]

        # Find worst performer
        worst_idx = int(np.argmin(scores))
        best_idx = int(np.argmax(scores))

        # Record the meta-event
        event = {
            "step": self.step_count,
            "dropped": self.pool[worst_idx].strategy.label(),
            "best": self.pool[best_idx].strategy.label(),
            "scores": scores.copy(),
        }

        # Replace worst with a new strategy
        if self.rng.random() < self.novel_strategy_prob:
            # Completely novel strategy
            new_strategy = self.mutator.random_strategy()
            event["action"] = "novel"
        elif len(self.pool) >= 2:
            # Crossover + mutation of top performers
            sorted_indices = np.argsort(scores)[::-1]
            parent1 = self.pool[sorted_indices[0]].strategy
            parent2 = self.pool[sorted_indices[min(1, len(sorted_indices) - 1)]].strategy
            child = self.mutator.crossover(parent1, parent2)
            new_strategy = self.mutator.mutate(child)
            event["action"] = "crossover+mutate"
        else:
            new_strategy = self.mutator.mutate(self.pool[best_idx].strategy)
            event["action"] = "mutate"

        event["new"] = new_strategy.label()
        self.pool[worst_idx] = StrategyRecord(strategy=new_strategy)

        # Switch active strategy to the current best
        self.active_idx = best_idx

        self.history.meta_events.append(event)

    def run(self, n_steps: int) -> dict:
        """Run the optimizer for n_steps and return history."""
        for _ in range(n_steps):
            self.step()
        return self.history.to_dict()
