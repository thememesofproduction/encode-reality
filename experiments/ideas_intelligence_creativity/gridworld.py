"""
Grid world environment for Experiment 2.

A 2D world with food (positive reward) and hazards (negative reward).
The environment can be "shifted" to represent qualitatively new problem classes,
demonstrating that Level 1 agents (fixed architecture) fail while Level 2 agents
(evolvable architecture) adapt.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class GridConfig:
    """Configuration for a grid world environment."""
    width: int = 30
    height: int = 30
    n_food: int = 15
    n_hazards: int = 8
    food_reward: float = 1.0
    hazard_penalty: float = -1.5
    step_cost: float = -0.01
    food_regrow_rate: float = 0.02
    seed: int = 42


class GridWorld:
    """
    A 2D grid world with food and hazards.

    Agents navigate the grid, consuming food (positive reward) and
    taking damage from hazards (negative reward). The environment
    can be shifted to create qualitatively new challenges.

    Observation channels (selectable per agent):
        "food"     : distance/direction to nearest food
        "hazard"   : distance/direction to nearest hazard
        "density"  : local food/hazard density
        "memory"   : trace of recently visited cells
        "gradient" : food gradient direction
    """

    CHANNELS = ("food", "hazard", "density", "memory", "gradient")

    def __init__(self, config: GridConfig | None = None):
        self.config = config or GridConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.width = self.config.width
        self.height = self.config.height

        # State grids
        self.food = np.zeros((self.height, self.width), dtype=float)
        self.hazard = np.zeros((self.height, self.width), dtype=float)
        self.visit_count = np.zeros((self.height, self.width), dtype=float)

        self._place_items()
        self.step_count = 0
        self.phase = "A"

    def _place_items(self):
        """Randomly place food and hazards."""
        self.food[:] = 0
        self.hazard[:] = 0

        for _ in range(self.config.n_food):
            r, c = self.rng.integers(0, self.height), self.rng.integers(0, self.width)
            self.food[r, c] = 1.0

        for _ in range(self.config.n_hazards):
            r, c = self.rng.integers(0, self.height), self.rng.integers(0, self.width)
            self.hazard[r, c] = 1.0

    def reset(self):
        """Reset environment state for a new episode."""
        self._place_items()
        self.visit_count[:] = 0
        self.step_count = 0

    def get_observation(self, position: tuple[int, int],
                        channels: list[str],
                        sensory_range: int = 5) -> np.ndarray:
        """
        Get observation vector for an agent at the given position.

        Each channel contributes a fixed-size vector. The total observation
        is the concatenation of all requested channels.
        """
        r, c = position
        obs_parts = []

        for ch in channels:
            if ch == "food":
                obs_parts.append(self._sense_nearest(r, c, self.food, sensory_range))
            elif ch == "hazard":
                obs_parts.append(self._sense_nearest(r, c, self.hazard, sensory_range))
            elif ch == "density":
                obs_parts.append(self._sense_density(r, c, sensory_range))
            elif ch == "memory":
                obs_parts.append(self._sense_memory(r, c, sensory_range))
            elif ch == "gradient":
                obs_parts.append(self._sense_gradient(r, c, self.food, sensory_range))
            else:
                raise ValueError(f"Unknown channel: {ch}")

        return np.concatenate(obs_parts)

    def _sense_nearest(self, r: int, c: int, grid: np.ndarray,
                       sense_range: int) -> np.ndarray:
        """Return [dx, dy, distance] to nearest item within range."""
        r_lo = max(0, r - sense_range)
        r_hi = min(self.height, r + sense_range + 1)
        c_lo = max(0, c - sense_range)
        c_hi = min(self.width, c + sense_range + 1)

        patch = grid[r_lo:r_hi, c_lo:c_hi]
        positions = np.argwhere(patch > 0)

        if len(positions) == 0:
            return np.array([0.0, 0.0, 1.0])  # nothing found, max distance

        # Convert to global coords
        positions[:, 0] += r_lo
        positions[:, 1] += c_lo

        diffs = positions - np.array([r, c])
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        nearest_idx = np.argmin(dists)

        dx = diffs[nearest_idx, 1] / (sense_range + 1e-8)
        dy = diffs[nearest_idx, 0] / (sense_range + 1e-8)
        dist = dists[nearest_idx] / (sense_range * np.sqrt(2) + 1e-8)

        return np.array([dx, dy, dist])

    def _sense_density(self, r: int, c: int, sense_range: int) -> np.ndarray:
        """Return [food_density, hazard_density] in local area."""
        r_lo = max(0, r - sense_range)
        r_hi = min(self.height, r + sense_range + 1)
        c_lo = max(0, c - sense_range)
        c_hi = min(self.width, c + sense_range + 1)

        area = (r_hi - r_lo) * (c_hi - c_lo)
        food_d = self.food[r_lo:r_hi, c_lo:c_hi].sum() / max(area, 1)
        hazard_d = self.hazard[r_lo:r_hi, c_lo:c_hi].sum() / max(area, 1)

        return np.array([food_d, hazard_d])

    def _sense_memory(self, r: int, c: int, sense_range: int) -> np.ndarray:
        """Return [visit_density, novelty_score] — how explored is the local area."""
        r_lo = max(0, r - sense_range)
        r_hi = min(self.height, r + sense_range + 1)
        c_lo = max(0, c - sense_range)
        c_hi = min(self.width, c + sense_range + 1)

        patch = self.visit_count[r_lo:r_hi, c_lo:c_hi]
        visit_density = patch.mean() / max(patch.max(), 1)
        novelty = 1.0 - visit_density

        return np.array([visit_density, novelty])

    def _sense_gradient(self, r: int, c: int, grid: np.ndarray,
                        sense_range: int) -> np.ndarray:
        """Return [grad_x, grad_y] — direction of increasing food concentration."""
        r_lo = max(0, r - sense_range)
        r_hi = min(self.height, r + sense_range + 1)
        c_lo = max(0, c - sense_range)
        c_hi = min(self.width, c + sense_range + 1)

        patch = grid[r_lo:r_hi, c_lo:c_hi]
        if patch.sum() < 1e-8:
            return np.array([0.0, 0.0])

        rows, cols = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
        center_r = r - r_lo
        center_c = c - c_lo

        weights = patch.ravel()
        total = weights.sum()
        if total < 1e-8:
            return np.array([0.0, 0.0])

        grad_r = (weights * (rows.ravel() - center_r)).sum() / total
        grad_c = (weights * (cols.ravel() - center_c)).sum() / total

        norm = np.sqrt(grad_r ** 2 + grad_c ** 2) + 1e-8
        return np.array([grad_c / norm, grad_r / norm])

    def apply_action(self, position: tuple[int, int],
                     action: int) -> tuple[tuple[int, int], float]:
        """
        Apply an action and return (new_position, reward).

        Actions: 0=up, 1=right, 2=down, 3=left, 4=stay
        Extended actions: 5=up-right, 6=down-right, 7=down-left, 8=up-left
        """
        r, c = position
        deltas = {
            0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (0, 0),
            5: (-1, 1), 6: (1, 1), 7: (1, -1), 8: (-1, -1),
        }
        dr, dc = deltas.get(action, (0, 0))
        nr = max(0, min(self.height - 1, r + dr))
        nc = max(0, min(self.width - 1, c + dc))

        reward = self.config.step_cost

        # Consume food
        if self.food[nr, nc] > 0:
            reward += self.config.food_reward
            self.food[nr, nc] = 0

        # Take hazard damage
        if self.hazard[nr, nc] > 0:
            reward += self.config.hazard_penalty

        self.visit_count[nr, nc] += 1
        self.step_count += 1

        # Food regrowth
        if self.rng.random() < self.config.food_regrow_rate:
            rr = self.rng.integers(0, self.height)
            rc = self.rng.integers(0, self.width)
            if self.food[rr, rc] == 0 and self.hazard[rr, rc] == 0:
                self.food[rr, rc] = 1.0

        return (nr, nc), reward

    def shift_environment(self, phase: str = "B"):
        """
        Shift the environment to create a qualitatively new challenge.

        Phase B: Redistributes food/hazards (same quantities, new locations).
        Phase C: Adds moving hazards and sparse clustered food (requires
                 memory and gradient sensing to solve effectively).
        """
        self.phase = phase
        self.visit_count[:] = 0

        if phase == "B":
            # Same quantities but totally different layout
            self._place_items()

        elif phase == "C":
            # Harder: fewer food items clustered together, more hazards
            self.food[:] = 0
            self.hazard[:] = 0

            # Clustered food in 2-3 patches
            n_clusters = self.rng.integers(2, 4)
            for _ in range(n_clusters):
                cr = self.rng.integers(5, self.height - 5)
                cc = self.rng.integers(5, self.width - 5)
                for _ in range(self.config.n_food // n_clusters):
                    fr = cr + self.rng.integers(-3, 4)
                    fc = cc + self.rng.integers(-3, 4)
                    fr = max(0, min(self.height - 1, fr))
                    fc = max(0, min(self.width - 1, fc))
                    self.food[fr, fc] = 1.0

            # More hazards, scattered
            for _ in range(self.config.n_hazards * 2):
                r = self.rng.integers(0, self.height)
                c = self.rng.integers(0, self.width)
                self.hazard[r, c] = 1.0

    def obs_size_for_channels(self, channels: list[str]) -> int:
        """Return the total observation vector size for given channels."""
        sizes = {"food": 3, "hazard": 3, "density": 2, "memory": 2, "gradient": 2}
        return sum(sizes[ch] for ch in channels)
