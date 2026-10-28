"""SPEC.md section 2: an evaluation budget every arm must go through.

A ``Budget`` wraps a point function and counts every point it evaluates. An arm run with
``max_evals = B`` evaluates exactly B points, and the best-so-far trace is kept as it goes.
"""

import numpy as np


class BudgetExhausted(Exception):
    """Raised when a point is asked of a Budget with no evaluations left."""


class Budget:
    """Evaluates points with ``f`` until ``max_evals`` evaluations have been spent.

    ``mode`` is ``"min"`` (lower is better) or ``"max"`` (higher is better). ``best`` and ``best_x``
    are ``inf``/``-inf`` and ``None`` until the first evaluation.
    """

    def __init__(self, f, max_evals: int, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', not {mode!r}")
        if max_evals < 0:
            raise ValueError(f"max_evals must be non-negative, not {max_evals}")
        self._f = f
        self.max_evals = int(max_evals)
        self.mode = mode
        self.used = 0
        self._trace = np.empty(self.max_evals, dtype=float)
        self.best = np.inf if mode == "min" else -np.inf
        self.best_x = None

    @property
    def remaining(self) -> int:
        return self.max_evals - self.used

    @property
    def trace(self) -> np.ndarray:
        """Best-so-far after each evaluation (a view); its length is ``used``."""
        return self._trace[:self.used]

    def __call__(self, x) -> float:
        """Evaluate one point. Raises BudgetExhausted, without calling f, when no evaluations are left."""
        if self.used >= self.max_evals:
            raise BudgetExhausted(f"all {self.max_evals} evaluations are spent")
        value = float(self._f(x))
        self._record(x, value)
        return value

    def batch(self, xs) -> np.ndarray:
        """Evaluate the points of ``xs`` in order, up to the remaining budget, and return their values."""
        if self.remaining == 0:
            raise BudgetExhausted(f"all {self.max_evals} evaluations are spent")
        n = min(len(xs), self.remaining)
        values = np.empty(n, dtype=float)
        for i in range(n):
            values[i] = self(xs[i])
        return values

    def _record(self, x, value: float) -> None:
        better = value < self.best if self.mode == "min" else value > self.best
        if self.used == 0 or better:
            self.best = value
            self.best_x = np.array(x, dtype=float)      # a copy: the caller may reuse its buffer
        self._trace[self.used] = self.best
        self.used += 1
