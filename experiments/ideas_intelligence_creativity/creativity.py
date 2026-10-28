"""
Meta-strategy optimizer (Level 2 — Creativity).

Maintains a pool of search strategies and periodically mutates them
based on performance. This is the computational analogue of creativity:
a generator that operates over generators, producing new ways of
generating solutions.

The meta-level follows SPEC.md section 6 and clarifications 9.8 to 9.12. Each pool record counts the
evaluations its strategy has spent and the improvement they made, so its score is improvement per
evaluation. Every evaluation goes through one ``search_core.Budget``.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy
from .landscape import FitnessLandscape
from .search_core import Budget
from .strategies import SearchStrategy, StrategyMutator

SELECTIONS = ("ucb", "random")
UCB_C = 0.5                 # the exploration constant c in s + c * sqrt(ln N / n), in normalised units
MAX_CANDIDATES = 100        # StrategyMutator keeps n_candidates within [5, 100]


@dataclass
class StrategyRecord:
    """A pool entry: a strategy and the evaluations and improvement it has produced."""
    strategy: SearchStrategy
    evals: int = 0
    improvement: float = 0.0

    @property
    def score(self) -> float | None:
        """Improvement per evaluation, or None when the strategy has never run."""
        return None if self.evals == 0 else self.improvement / self.evals


@dataclass
class CreativityHistory:
    """Tracks the full history including strategy-level changes."""
    positions: list[np.ndarray] = field(default_factory=list)
    best_fitness: list[float] = field(default_factory=list)
    solution_diversity: list[float] = field(default_factory=list)
    strategy_labels: list[str] = field(default_factory=list)
    strategy_diversity: list[float] = field(default_factory=list)
    meta_events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_fitness": list(self.best_fitness),
            "best_positions": [p.tolist() for p in self.positions],
            "solution_diversity": list(self.solution_diversity),
            "strategy_labels": list(self.strategy_labels),
            "strategy_diversity": list(self.strategy_diversity),
            "meta_events": list(self.meta_events),
        }


class MetaStrategyOptimizer:
    """
    Level 2 — Creativity: search over search strategies.

    The pool starts with the initial strategy alone and grows. The active strategy searches the
    landscape, as in FixedStrategyOptimizer. Every ``meta_interval`` steps a meta-step runs:

    (a) replacement, on every ``replace_every``-th meta-step: once the pool is full, the worst record
        with data that is not active is dropped. Then one new strategy is added (novel, crossover or
        mutant), with no data. Parents of a crossover or mutant always have data;
    (b) selection, always: an untried record becomes active; otherwise the best UCB record does.
        With ``selection="random"``, the active record is chosen uniformly instead.

    Before each meta-step, every record's evals and improvement are multiplied by ``discount``. The
    default 1.0 is no discount (SPEC section 6). A discount below 1 lets recent credit outweigh old credit.
    A record that has never run keeps exactly zero, so it stays untried.

    The UCB selection works in normalised units. The score is divided by the best score in the pool,
    and the count is in epochs of ``meta_interval * n_candidates`` evaluations of the initial strategy.

    ``improvement_window`` is kept for the signature of SPEC section 6. The score is the record's whole
    improvement per evaluation, so the window is not used.
    ``seed`` may be an int or a numpy Generator. With ``x0`` None, the start is drawn from the stream.
    """

    def __init__(self, landscape: FitnessLandscape,
                 initial_strategy: SearchStrategy,
                 mutator: StrategyMutator,
                 meta_interval: int = 50,
                 strategy_pool_size: int = 5,
                 improvement_window: int = 25,
                 novel_strategy_prob: float = 0.2,
                 seed: int = 42,
                 x0=None,
                 replace_every: int = 3,
                 selection: str = "ucb",
                 discount: float = 1.0):
        if selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}, not {selection!r}")
        if strategy_pool_size < 1:
            raise ValueError(f"strategy_pool_size must be at least 1, not {strategy_pool_size}")
        if replace_every < 1:
            raise ValueError(f"replace_every must be at least 1, not {replace_every}")
        if not 0.0 < discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1], not {discount}")
        self.landscape = landscape
        self.mutator = mutator
        self.meta_interval = meta_interval
        self.pool_size = strategy_pool_size
        self.improvement_window = improvement_window
        self.novel_strategy_prob = novel_strategy_prob
        self.replace_every = replace_every
        self.selection = selection
        self.discount = discount
        self.epoch_evals = max(1, meta_interval * initial_strategy.n_candidates)
        self.rng = np.random.default_rng(seed)

        lo, hi = landscape.bounds
        start = self.rng.uniform(lo, hi, size=landscape.dims) if x0 is None else x0
        self.position = np.array(start, dtype=float)
        self.best_fitness = -np.inf
        self.best_position = self.position.copy()
        self.velocity = np.zeros(landscape.dims)

        self.pool: list[StrategyRecord] = [StrategyRecord(strategy=deepcopy(initial_strategy))]
        self.active_idx = 0
        self.step_count = 0
        self.meta_step_count = 0

        self.budget: Budget | None = None
        self.history = CreativityHistory()

    @property
    def active_strategy(self) -> SearchStrategy:
        return self.pool[self.active_idx].strategy

    @property
    def trace(self) -> np.ndarray:
        """Best fitness so far after each evaluation; empty before ``run``."""
        return np.empty(0) if self.budget is None else self.budget.trace

    def _start(self):
        """Evaluate and record the start point: evaluation 1."""
        self.best_fitness = self.budget(self.position)
        self.best_position = self.position.copy()
        self._record(0.0)

    def _record(self, diversity: float):
        self.history.positions.append(self.position.copy())
        self.history.best_fitness.append(self.best_fitness)
        self.history.solution_diversity.append(diversity)
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
        values = self.budget.batch(candidates)
        k = len(values)
        best_idx = int(np.argmax(values))

        improvement = 0.0
        if values[best_idx] > self.best_fitness:
            old_pos = self.position.copy()
            improvement = float(values[best_idx]) - self.best_fitness
            self.position = np.array(candidates[best_idx], dtype=float)
            self.best_fitness = float(values[best_idx])
            self.best_position = self.position.copy()
            self.velocity = self.position - old_pos
        else:
            self.velocity *= 0.5

        record.evals += k
        record.improvement += improvement

        diversity = float(np.mean(np.std(candidates[:k], axis=0)))
        self._record(diversity)

        self.step_count += 1

        # Trigger meta-step periodically
        if self.step_count % self.meta_interval == 0:
            self.meta_step()

        return self.best_fitness

    def meta_step(self, replace: bool | None = None):
        """
        One meta-step: optional replacement (a), then selection (b). See the class docstring.

        ``replace=None`` replaces on every ``replace_every``-th meta-step; True or False forces it.
        Every call counts towards ``replace_every``.
        """
        self.meta_step_count += 1
        if self.discount != 1.0:
            for r in self.pool:
                r.evals *= self.discount           # 0 stays 0: a never-run record is still untried
                r.improvement *= self.discount
        if replace is None:
            replace = self.meta_step_count % self.replace_every == 0

        event = {"step": self.step_count, "dropped": None, "action": "none", "new": None}
        if replace:
            event.update(self._replace())
        self._select()
        event["active"] = self.active_strategy.label()
        self.history.meta_events.append(event)

    def _replace(self) -> dict:
        """(a) Drop the worst record with data that is not active, if the pool is full; add one new strategy.

        When the pool is full and no record may be dropped, nothing changes and an empty dict is returned.
        """
        slot = None
        if len(self.pool) >= self.pool_size:
            droppable = [i for i, r in enumerate(self.pool) if r.evals > 0 and i != self.active_idx]
            if not droppable:
                return {}
            slot = min(droppable, key=lambda i: (self.pool[i].score, i))

        strategy, action = self._new_strategy()
        record = StrategyRecord(strategy=strategy)
        if slot is None:
            self.pool.append(record)
            dropped = None
        else:
            dropped = self.pool[slot].strategy.label()
            self.pool[slot] = record
        return {"dropped": dropped, "action": action, "new": strategy.label()}

    def _new_strategy(self) -> tuple[SearchStrategy, str]:
        """A novel strategy, or a crossover or mutant of records with data. Only tried records are parents."""
        tried = [r for r in self.pool if r.evals > 0]
        if not tried or self.rng.random() < self.novel_strategy_prob:
            return self.mutator.random_strategy(), "novel"
        ranked = sorted(tried, key=lambda r: r.score, reverse=True)
        if len(ranked) >= 2 and self.rng.random() < 0.5:
            return self.mutator.crossover(ranked[0].strategy, ranked[1].strategy), "crossover"
        return self.mutator.mutate(ranked[0].strategy), "mutate"

    def _select(self):
        """(b) Untried first; otherwise the best UCB record; or a uniform choice with selection='random'."""
        if self.selection == "random":
            self.active_idx = int(self.rng.integers(len(self.pool)))
            return
        untried = [i for i, r in enumerate(self.pool) if r.evals == 0]
        if untried:
            self.active_idx = untried[0]
            return
        best = max(r.score for r in self.pool)
        epochs = [r.evals / self.epoch_evals for r in self.pool]
        log_total = math.log(max(sum(epochs), 1.0))

        def ucb_key(i: int):
            r = self.pool[i]
            normalised = r.score / best if best > 0 else 0.0
            bonus = UCB_C * math.sqrt(log_total / epochs[i])
            return (normalised + bonus, r.score, -i)  # equal evals: equal bonus, so the highest score wins

        self.active_idx = max(range(len(self.pool)), key=ucb_key)

    def run(self, n_steps: int | None = None, max_evals: int | None = None) -> dict:
        """Run until ``max_evals`` evaluations are spent (the last batch is truncated), or for ``n_steps``.

        With only ``n_steps``, the budget is ``1 + n_steps * max(MAX_CANDIDATES, n_candidates)``, which is
        enough for any strategy the pool can hold. Each optimiser runs once.
        """
        if self.budget is not None:
            raise RuntimeError("run() has already been called on this optimiser")
        if max_evals is None:
            if n_steps is None:
                raise ValueError("give n_steps or max_evals")
            max_evals = 1 + n_steps * max(MAX_CANDIDATES, self.pool[0].strategy.n_candidates)
        if max_evals < 1:
            raise ValueError(f"max_evals must be at least 1, not {max_evals}")
        self.budget = Budget(self.landscape.evaluate, max_evals, mode="max")
        self._start()
        steps = 0
        while self.budget.remaining > 0 and (n_steps is None or steps < n_steps):
            self.step()
            steps += 1
        return self.history.to_dict()
