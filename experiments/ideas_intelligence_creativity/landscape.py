"""
Rugged fitness landscape for Experiment 1.

A 2D surface generated as a sum of Gaussian peaks with varying heights,
widths, and positions. Multiple local optima with one global optimum.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Peak:
    """A single Gaussian peak in the landscape."""
    center: np.ndarray
    height: float
    width: float


class FitnessLandscape:
    """
    A rugged 2D fitness landscape with multiple local optima.

    The landscape is a sum of Gaussian peaks, creating a surface with
    many basins of attraction. This makes it ideal for demonstrating
    the difference between fixed-strategy search (gets trapped) and
    meta-strategy search (escapes via strategy mutation).
    """

    def __init__(self, n_peaks: int = 15, dims: int = 2,
                 bounds: tuple[float, float] = (-5.0, 5.0),
                 seed: int = 42):
        self.n_peaks = n_peaks
        self.dims = dims
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)

        self.peaks: list[Peak] = []
        self._generate_peaks()
        self._global_optimum = self._find_global_optimum()

    def _generate_peaks(self):
        lo, hi = self.bounds
        for _ in range(self.n_peaks):
            center = self.rng.uniform(lo, hi, size=self.dims)
            height = self.rng.uniform(1.0, 4.0)
            width = self.rng.uniform(0.2, 0.8)  # narrow peaks = more local traps
            self.peaks.append(Peak(center=center, height=height, width=width))

        # Global peak: high but very narrow and distant from cluster center
        # Placed near a corner so small-step optimizers are unlikely to reach it
        corner = np.array([hi * 0.85] * self.dims)
        self.peaks.append(Peak(center=corner, height=8.0, width=0.3))

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate fitness at position x."""
        x = np.asarray(x, dtype=float)
        total = 0.0
        for peak in self.peaks:
            dist_sq = np.sum((x - peak.center) ** 2)
            total += peak.height * np.exp(-dist_sq / (2 * peak.width ** 2))
        return total

    def evaluate_batch(self, xs: np.ndarray) -> np.ndarray:
        """Evaluate fitness for multiple positions at once."""
        xs = np.asarray(xs, dtype=float)
        results = np.zeros(len(xs))
        for peak in self.peaks:
            diffs = xs - peak.center
            dist_sq = np.sum(diffs ** 2, axis=1)
            results += peak.height * np.exp(-dist_sq / (2 * peak.width ** 2))
        return results

    def _find_global_optimum(self) -> tuple[np.ndarray, float]:
        """Find the global optimum by evaluating at all peak centers."""
        best_pos = self.peaks[0].center
        best_val = self.evaluate(best_pos)
        for peak in self.peaks:
            val = self.evaluate(peak.center)
            if val > best_val:
                best_val = val
                best_pos = peak.center
        return best_pos, best_val

    def get_global_optimum(self) -> tuple[np.ndarray, float]:
        """Return (position, fitness) of the global optimum."""
        return self._global_optimum

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
