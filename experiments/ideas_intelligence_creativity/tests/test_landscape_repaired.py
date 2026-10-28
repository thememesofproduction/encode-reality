"""SPEC.md section 6, landscape.py: global_peak conditions, from_peaks and a true ceiling()."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.tests._iic_helpers import load, spec_exp1_conditions  # noqa: E402

CONDITIONS = spec_exp1_conditions()          # none, random, corner (from the runner's --conditions)
EXTRA_HEIGHT = 8.0                           # "height 8, width 0.3"
# The spec allows 1e-9 against the grid and the peak centres. Against an independent local maximiser
# it states no tolerance; 1e-7 still rejects a grid-only ceiling (off by ~1e-4 where peaks overlap).
TRUE_MAX_TOL = 1e-7


def L():
    return load("landscape").FitnessLandscape


def centres(land):
    return [np.asarray(p.center, dtype=float) for p in land.peaks]


def grid_max(land, lo, hi, n=401):
    g = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(g, g)
    vals = np.asarray(land.evaluate_batch(np.column_stack([X.ravel(), Y.ravel()])), dtype=float)
    i = int(np.argmax(vals))
    return float(vals[i]), np.array([X.ravel()[i], Y.ravel()[i]]), vals, X.ravel(), Y.ravel()


def local_max(land, starts, lo, hi):
    """Independent multi-start bounded maximisation of land.evaluate (never exceeds the true max)."""
    best = -np.inf
    dims = len(starts[0])
    for s in starts:
        res = minimize(lambda x: -float(land.evaluate(np.asarray(x))), np.asarray(s, dtype=float),
                       method="L-BFGS-B", bounds=[(lo, hi)] * dims, options={"ftol": 1e-15, "gtol": 1e-12})
        best = max(best, -float(res.fun), float(land.evaluate(np.clip(res.x, lo, hi))))
    return best


def check_ceiling(land, lo, hi):
    x, f = land.ceiling()
    x = np.asarray(x, dtype=float)
    assert np.all(x >= lo) and np.all(x <= hi), x
    assert abs(float(land.evaluate(x)) - f) <= 1e-9 * max(1.0, abs(f)), "ceiling value is not the value at its x"
    for c in centres(land):
        if np.all(c >= lo) and np.all(c <= hi):
            assert f >= float(land.evaluate(c)) - 1e-9
    return x, float(f)


def test_condition_names_are_the_global_peak_values():
    assert CONDITIONS == ["none", "random", "corner"]
    L()(n_peaks=3, seed=1, global_peak="none")


def test_default_is_none_with_fifteen_peaks():
    land = L()()
    assert len(land.peaks) == 15
    assert len(L()(n_peaks=10, seed=42).peaks) == 10


@pytest.mark.parametrize("n", [1, 4, 10, 15])
def test_peak_counts_per_condition(n):
    for seed in (0, 3):
        assert len(L()(n_peaks=n, seed=seed, global_peak="none").peaks) == n
        assert len(L()(n_peaks=n, seed=seed, global_peak="random").peaks) == n + 1
        assert len(L()(n_peaks=n, seed=seed, global_peak="corner").peaks) == n + 1


@pytest.mark.parametrize("bad", ["", "None", "NONE", "Random", "global", "center", None, 1])
def test_other_global_peak_values_raise(bad):
    with pytest.raises(ValueError):
        L()(n_peaks=5, seed=0, global_peak=bad)


def _tall_centres(land):
    return [c for c in centres(land) if float(land.evaluate(c)) >= EXTRA_HEIGHT - 1e-9]


def test_corner_peak_is_the_same_place_for_every_seed():
    lands = [L()(n_peaks=10, seed=s, global_peak="corner") for s in range(6)]
    common = [c for c in centres(lands[0])
              if all(any(np.allclose(c, d, atol=1e-12) for d in centres(o)) for o in lands[1:])]
    assert len(common) == 1, common
    for land in lands:
        assert float(land.evaluate(common[0])) >= EXTRA_HEIGHT - 1e-9
        _, f = check_ceiling(land, -5.0, 5.0)
        assert f >= EXTRA_HEIGHT - 1e-9


def test_random_peak_moves_with_the_seed():
    lands = [L()(n_peaks=10, seed=s, global_peak="random") for s in range(6)]
    for land in lands:
        assert _tall_centres(land), "no peak centre reaches the extra peak's height 8"
    common = [c for c in centres(lands[0])
              if all(any(np.allclose(c, d, atol=1e-12) for d in centres(o)) for o in lands[1:])]
    assert common == [], f"a peak sits at the same place for every seed: {common}"
    tall = [tuple(np.round(_tall_centres(land)[0], 9)) for land in lands]
    assert len(set(tall)) > 1


def test_same_seed_same_landscape_for_each_condition():
    for cond in CONDITIONS:
        a, b = L()(n_peaks=8, seed=9, global_peak=cond), L()(n_peaks=8, seed=9, global_peak=cond)
        pts = np.random.default_rng(0).uniform(-5, 5, size=(200, 2))
        np.testing.assert_array_equal(a.evaluate_batch(pts), b.evaluate_batch(pts))


# ----------------------------------------------------------------------------- from_peaks

def test_from_peaks_single_peak():
    FL = L()
    land = FL.from_peaks([[1.0, -2.0]], [3.0], [0.7])
    assert len(land.peaks) == 1
    np.testing.assert_allclose(centres(land)[0], [1.0, -2.0])
    assert float(land.evaluate(np.array([1.0, -2.0]))) == pytest.approx(3.0, rel=1e-12)
    d = np.array([0.6, 0.8])
    vals = [float(land.evaluate(np.array([1.0, -2.0]) + t * d)) for t in (0.0, 0.1, 0.5, 1.0, 2.0)]
    assert all(v1 > v2 for v1, v2 in zip(vals, vals[1:]))
    x, f = check_ceiling(land, -5.0, 5.0)
    np.testing.assert_allclose(x, [1.0, -2.0], atol=1e-4)
    assert f == pytest.approx(3.0, abs=1e-9)


def test_from_peaks_heights_and_widths_are_used():
    FL = L()
    c = [[0.0, 0.0], [2.0, 1.0], [-3.0, 2.5]]
    a = FL.from_peaks(c, [1.0, 2.0, 0.5], [0.5, 0.8, 1.2])
    b = FL.from_peaks(c, [2.5, 5.0, 1.25], [0.5, 0.8, 1.2])
    pts = np.random.default_rng(1).uniform(-5, 5, size=(300, 2))
    np.testing.assert_allclose(b.evaluate_batch(pts), 2.5 * np.asarray(a.evaluate_batch(pts)), rtol=1e-12, atol=1e-300)
    narrow = FL.from_peaks([[0.0, 0.0]], [1.0], [0.3])
    wide = FL.from_peaks([[0.0, 0.0]], [1.0], [1.0])
    p = np.array([1.0, 0.0])
    assert float(narrow.evaluate(p)) < float(wide.evaluate(p))


@pytest.mark.parametrize("centres_,heights,widths", [
    ([[0.0, 0.0], [0.5, 0.0]], [1.0, 1.0], [0.5, 0.5]),                    # symmetric overlap
    ([[0.0, 0.0], [0.45, 0.1]], [1.0, 0.8], [0.4, 0.6]),                   # asymmetric overlap
    ([[-1.0, 1.0], [-0.6, 1.3], [-0.9, 1.6]], [0.7, 0.9, 0.6], [0.5, 0.45, 0.55]),
], ids=["symmetric", "asymmetric", "three"])
def test_ceiling_finds_the_maximum_where_peaks_overlap(centres_, heights, widths):
    land = L().from_peaks(centres_, heights, widths)
    x, f = check_ceiling(land, -5.0, 5.0)
    top_centre = max(float(land.evaluate(np.asarray(c))) for c in centres_)
    assert f > top_centre + 1e-6, "the true maximum lies between the overlapping peaks, above both centres"
    starts = [np.mean(centres_, axis=0)] + [np.asarray(c) for c in centres_]
    assert f >= local_max(land, starts, -5.0, 5.0) - TRUE_MAX_TOL


def test_ceiling_respects_custom_bounds():
    land = L().from_peaks([[1.2, -1.5], [-1.7, 1.9]], [1.0, 1.5], [0.6, 0.4], bounds=(-2.0, 2.0))
    x, f = check_ceiling(land, -2.0, 2.0)
    gm = grid_max(land, -2.0, 2.0)[0]
    assert f >= gm - 1e-9
    land2 = L()(n_peaks=6, bounds=(0.0, 3.0), seed=4)
    x2, f2 = check_ceiling(land2, 0.0, 3.0)
    assert f2 >= grid_max(land2, 0.0, 3.0)[0] - 1e-9


@pytest.mark.parametrize("cond,seed", list(itertools.product(CONDITIONS, [0, 1, 2])))
def test_ceiling_is_the_true_maximum_of_random_landscapes(cond, seed):
    land = L()(n_peaks=15, seed=seed, global_peak=cond)
    x, f = check_ceiling(land, -5.0, 5.0)
    gm, gx, vals, X, Y = grid_max(land, -5.0, 5.0)
    assert f >= gm - 1e-9
    order = np.argsort(vals)[::-1][:400]
    starts, seen = [], []
    for i in order:
        p = np.array([X[i], Y[i]])
        if all(np.linalg.norm(p - q) > 0.3 for q in seen):
            seen.append(p)
            starts.append(p)
        if len(starts) == 10:
            break
    assert f >= local_max(land, starts + centres(land), -5.0, 5.0) - TRUE_MAX_TOL
    gx_, gf = land.get_global_optimum()
    np.testing.assert_allclose(np.asarray(gx_, dtype=float), x)
    assert gf == f


def test_ceiling_in_three_dimensions():
    land = L()(n_peaks=6, dims=3, seed=5)
    x, f = check_ceiling(land, -5.0, 5.0)
    assert np.asarray(x).shape == (3,)
    assert f >= local_max(land, centres(land), -5.0, 5.0) - TRUE_MAX_TOL
