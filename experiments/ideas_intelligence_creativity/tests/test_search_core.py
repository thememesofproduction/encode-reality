"""SPEC.md section 2: search_core.py -- an evaluation budget every arm must go through."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.tests._iic_helpers import load  # noqa: E402


class Recorder:
    """A point function that records every point it is called with."""

    def __init__(self, fn=None):
        self.fn = fn or (lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2)))
        self.points = []

    def __call__(self, x):
        self.points.append(np.array(x, dtype=float, copy=True))
        return self.fn(x)


def first_coord(x):
    return float(np.asarray(x, dtype=float).ravel()[0])


def test_budget_exhausted_is_an_exception():
    sc = load("search_core")
    assert issubclass(sc.BudgetExhausted, Exception)


def test_call_evaluates_one_point_and_counts_it():
    sc = load("search_core")
    f = Recorder()
    b = sc.Budget(f, 5)
    assert b.max_evals == 5 and b.used == 0 and b.remaining == 5
    pts = [np.array([1.0, 2.0]), np.array([0.5, -1.0]), np.array([3.0, 0.0])]
    for i, x in enumerate(pts, start=1):
        v = b(x)
        assert isinstance(v, float)
        assert v == pytest.approx(float(np.sum(x ** 2)), rel=1e-15)
        assert b.used == i and b.remaining == 5 - i
    assert len(f.points) == 3
    for got, want in zip(f.points, pts):
        np.testing.assert_array_equal(got, want)


def test_call_raises_when_exhausted_and_does_not_call_f():
    sc = load("search_core")
    f = Recorder()
    b = sc.Budget(f, 4)
    for x in range(4):
        b(np.array([float(x)]))
    assert b.used == 4 and b.remaining == 0
    for _ in range(3):
        with pytest.raises(sc.BudgetExhausted):
            b(np.array([9.0]))
    assert len(f.points) == 4          # f was not called for the refused points
    assert b.used == 4 and len(b.trace) == 4


@pytest.mark.parametrize("make_xs", [
    lambda n: np.arange(n, dtype=float).reshape(n, 1) + 0.25,       # 2-D array of 1-D points
    lambda n: [np.array([float(i), -float(i)]) for i in range(n)],   # list of points
    lambda n: np.linspace(-3.0, 3.0, n),                             # 1-D array of scalar points
], ids=["2d-array", "list", "scalars"])
def test_batch_evaluates_in_order_and_truncates_to_remaining(make_xs):
    sc = load("search_core")
    f = Recorder(first_coord)
    b = sc.Budget(f, 10)
    b(make_xs(1)[0])
    b(make_xs(2)[1])
    b(make_xs(3)[2])
    xs = make_xs(12)
    vals = b.batch(xs)
    assert isinstance(vals, np.ndarray)
    assert len(vals) == 7                                            # min(12, remaining=7)
    np.testing.assert_array_equal(vals, [first_coord(x) for x in list(xs)[:7]])
    assert b.used == 10 and b.remaining == 0
    assert len(f.points) == 10
    for got, want in zip(f.points[3:], list(xs)[:7]):
        np.testing.assert_array_equal(got.ravel(), np.asarray(want, dtype=float).ravel())
    with pytest.raises(sc.BudgetExhausted):
        b.batch(xs)
    with pytest.raises(sc.BudgetExhausted):
        b(xs[0])
    assert len(f.points) == 10


def test_batch_does_not_raise_while_budget_remains():
    sc = load("search_core")
    b = sc.Budget(Recorder(), 3)
    assert len(b.batch(np.zeros((0, 2)))) == 0          # nothing asked, nothing evaluated, no raise
    assert b.used == 0
    out = b.batch(np.ones((50, 2)))                      # far more than remaining: truncate, no raise
    assert len(out) == 3 and b.remaining == 0


@pytest.mark.parametrize("mode", ["min", "max"])
def test_trace_is_the_running_best_by_mode(mode):
    sc = load("search_core")
    rng = np.random.default_rng(11 if mode == "min" else 12)
    acc = np.minimum.accumulate if mode == "min" else np.maximum.accumulate
    for trial in range(40):
        n = int(rng.integers(1, 120))
        values = rng.normal(size=n) * 10 ** rng.uniform(-3, 3)
        pts = values.reshape(n, 1)
        b = sc.Budget(first_coord, n, mode=mode)
        i = 0
        while i < n:                                     # mix single calls and batches
            if rng.random() < 0.5:
                b(pts[i])
                i += 1
            else:
                k = int(rng.integers(1, 9))
                got = b.batch(pts[i:i + k])
                i += len(got)
        trace = np.asarray(b.trace)
        assert isinstance(b.trace, np.ndarray)
        assert len(trace) == b.used == n
        assert trace[0] == values[0]
        np.testing.assert_array_equal(trace, acc(values))
        diffs = np.diff(trace)
        assert np.all(diffs <= 0) if mode == "min" else np.all(diffs >= 0)
        j = int(np.argmin(values) if mode == "min" else np.argmax(values))
        assert b.best == values[j]
        np.testing.assert_array_equal(np.asarray(b.best_x, dtype=float).ravel(), pts[j])


def test_default_mode_is_min():
    sc = load("search_core")
    b = sc.Budget(first_coord, 3)
    b.batch(np.array([[3.0], [1.0], [2.0]]))
    np.testing.assert_array_equal(b.trace, [3.0, 1.0, 1.0])
    assert b.best == 1.0


def test_budgets_are_independent():
    sc = load("search_core")
    b1, b2 = sc.Budget(first_coord, 5), sc.Budget(first_coord, 7)
    b1.batch(np.array([[1.0], [2.0], [0.5]]))
    assert b2.used == 0 and b2.remaining == 7 and len(b2.trace) == 0
    b2(np.array([4.0]))
    assert b1.used == 3 and b1.remaining == 2
    np.testing.assert_array_equal(b1.trace, [1.0, 1.0, 0.5])
    np.testing.assert_array_equal(b2.trace, [4.0])
