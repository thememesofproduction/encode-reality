"""
Rugged fitness landscape for Experiment 1 (SPEC.md section 6).

A surface made of Gaussian peaks. ``global_peak`` adds one extra tall, narrow peak: ``"none"`` adds
nothing, ``"random"`` puts it at a random place, and ``"corner"`` puts it at the old hand-placed corner.
``ceiling()`` is the landscape's true maximum over the bounds, found numerically, so a search's best
value can be read as a fraction of it.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

GLOBAL_PEAK_KINDS = ("none", "random", "corner")
EXTRA_PEAK_HEIGHT = 8.0
EXTRA_PEAK_WIDTH = 0.3
CEILING_GRID_AXIS = 401          # per-axis grid for dims == 2 (SPEC section 6)
CEILING_STARTS = 50              # the top grid cells, used as extra starts for the local maximiser
CEILING_CHUNK = 20_000           # grid points evaluated per call, to bound memory


@dataclass
class Peak:
    """A single Gaussian peak in the landscape."""
    center: np.ndarray
    height: float
    width: float


class FitnessLandscape:
    """
    A rugged landscape: a sum of Gaussian peaks over a box.

    Randomness comes from ``rng`` when given, otherwise from ``seed``. A Generator passed as ``rng``
    is used directly, so callers can hand in their own stream (SPEC section 1).
    """

    def __init__(self, n_peaks: int = 15, dims: int = 2,
                 bounds: tuple[float, float] = (-5.0, 5.0),
                 seed: int = 42, global_peak: str = "none", rng=None):
        if global_peak not in GLOBAL_PEAK_KINDS:
            raise ValueError(f"global_peak must be one of {GLOBAL_PEAK_KINDS}, not {global_peak!r}")
        self.n_peaks = n_peaks
        self.dims = dims
        self.bounds = bounds
        self.global_peak = global_peak
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        self.peaks: list[Peak] = []
        self._generate_peaks()
        self._refresh()
        self._ceiling = None

    def _generate_peaks(self):
        lo, hi = self.bounds
        for _ in range(self.n_peaks):
            center = self.rng.uniform(lo, hi, size=self.dims)
            height = self.rng.uniform(1.0, 4.0)
            width = self.rng.uniform(0.2, 0.8)  # narrow peaks = more local traps
            self.peaks.append(Peak(center=center, height=height, width=width))

        if self.global_peak == "random":
            center = self.rng.uniform(lo, hi, size=self.dims)
            self.peaks.append(Peak(center=center, height=EXTRA_PEAK_HEIGHT, width=EXTRA_PEAK_WIDTH))
        elif self.global_peak == "corner":
            corner = np.full(self.dims, hi * 0.85)
            self.peaks.append(Peak(center=corner, height=EXTRA_PEAK_HEIGHT, width=EXTRA_PEAK_WIDTH))

    @classmethod
    def from_peaks(cls, centers, heights, widths, bounds: tuple[float, float] = (-5.0, 5.0)):
        """A landscape with exactly these peaks: ``centers`` (n, dims), ``heights`` (n,), ``widths`` (n,)."""
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 2 or len(centers) == 0:
            raise ValueError("centers must be a non-empty list of points")
        heights = np.asarray(heights, dtype=float)
        widths = np.asarray(widths, dtype=float)
        if heights.shape != (len(centers),) or widths.shape != (len(centers),):
            raise ValueError("heights and widths must have one entry per centre")
        if np.any(widths <= 0):
            raise ValueError("widths must be positive")
        land = cls(n_peaks=0, dims=centers.shape[1], bounds=bounds, seed=0)
        land.n_peaks = len(centers)
        land.peaks = [Peak(center=c.copy(), height=float(h), width=float(w))
                      for c, h, w in zip(centers, heights, widths)]
        land._refresh()
        return land

    def _refresh(self):
        """Cache the peaks as arrays for evaluation. Call again after changing ``peaks``."""
        n = len(self.peaks)
        self._centers = np.array([p.center for p in self.peaks], dtype=float).reshape(n, self.dims)
        self._heights = np.array([p.height for p in self.peaks], dtype=float)
        self._inv_2w2 = np.array([1.0 / (2.0 * p.width ** 2) for p in self.peaks], dtype=float)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate fitness at position x."""
        x = np.asarray(x, dtype=float)
        d2 = np.sum((x - self._centers) ** 2, axis=1)
        return float(self._heights @ np.exp(-d2 * self._inv_2w2))

    def evaluate_batch(self, xs: np.ndarray) -> np.ndarray:
        """Evaluate fitness for multiple positions at once."""
        xs = np.asarray(xs, dtype=float)
        d2 = np.sum((xs[:, None, :] - self._centers[None, :, :]) ** 2, axis=2)
        return np.exp(-d2 * self._inv_2w2) @ self._heights

    def _negative_value_and_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """(-f(x), -grad f(x)): the objective the bounded minimiser descends."""
        diff = x - self._centers
        g = self._heights * np.exp(-np.sum(diff ** 2, axis=1) * self._inv_2w2)
        value = float(g.sum())
        grad = (g[:, None] * diff * (2.0 * self._inv_2w2)[:, None]).sum(axis=0)
        return -value, grad

    def ceiling(self) -> tuple[np.ndarray, float]:
        """(x, f): the landscape's true maximum over the bounds, computed numerically.

        The candidates are every peak centre inside the bounds, the top grid cells, and the result of a
        bounded L-BFGS-B climb from each of those starts. The value is the landscape's own value at x,
        so the ceiling is never below any centre or grid value.
        """
        if self._ceiling is None:
            self._ceiling = self._find_ceiling()
        x, f = self._ceiling
        return x.copy(), f

    def get_global_optimum(self) -> tuple[np.ndarray, float]:
        """Return (position, fitness) of the global optimum. Same as ``ceiling()``."""
        return self.ceiling()

    def _find_ceiling(self) -> tuple[np.ndarray, float]:
        lo, hi = self.bounds
        inside = np.all((self._centers >= lo) & (self._centers <= hi), axis=1)
        centres = self._centers[inside]

        grid = self._grid_points()
        grid_values = np.concatenate([self.evaluate_batch(grid[i:i + CEILING_CHUNK])
                                      for i in range(0, len(grid), CEILING_CHUNK)])
        top = np.argsort(grid_values)[::-1][:CEILING_STARTS]

        best_x, best_f = grid[top[0]].copy(), float(grid_values[top[0]])
        for c in centres:
            f = self.evaluate(c)
            if f > best_f:
                best_x, best_f = c.copy(), f

        for s in list(centres) + list(grid[top]):
            res = minimize(self._negative_value_and_gradient, np.asarray(s, dtype=float), jac=True,
                           method="L-BFGS-B", bounds=[(lo, hi)] * self.dims,
                           options={"ftol": 1e-15, "gtol": 1e-12})
            x = np.clip(res.x, lo, hi)
            f = self.evaluate(x)
            if f > best_f:
                best_x, best_f = x, f
        return np.asarray(best_x, dtype=float), float(best_f)

    def _grid_points(self) -> np.ndarray:
        """A grid over the bounds: CEILING_GRID_AXIS points per axis for 2-D, about the same size otherwise."""
        lo, hi = self.bounds
        if self.dims == 2:
            n = CEILING_GRID_AXIS
        else:
            n = max(2, int((CEILING_GRID_AXIS ** 2) ** (1.0 / self.dims)))
        axis = np.linspace(lo, hi, n)
        mesh = np.meshgrid(*([axis] * self.dims), indexing="ij")
        return np.column_stack([m.ravel() for m in mesh])

    def get_grid(self, resolution: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return meshgrid (X, Y, Z) for surface plotting. Only works for dims=2."""
        if self.dims != 2:
            raise ValueError("Grid visualization only supported for 2D landscapes")
        lo, hi = self.bounds
        x = np.linspace(lo, hi, resolution)
        y = np.linspace(lo, hi, resolution)
        X, Y = np.meshgrid(x, y)
        positions = np.column_stack([X.ravel(), Y.ravel()])
        Z = self.evaluate_batch(positions).reshape(X.shape)
        return X, Y, Z
