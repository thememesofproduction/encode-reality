"""Tests for the grid world environment."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.ideas_intelligence_creativity.gridworld import GridWorld, GridConfig


class TestGridWorld:

    def test_initialization(self):
        config = GridConfig(width=20, height=20, n_food=10, n_hazards=5, seed=42)
        world = GridWorld(config)
        assert world.food.sum() > 0
        assert world.hazard.sum() > 0

    def test_deterministic(self):
        c1 = GridConfig(seed=42)
        c2 = GridConfig(seed=42)
        w1, w2 = GridWorld(c1), GridWorld(c2)
        assert np.array_equal(w1.food, w2.food)
        assert np.array_equal(w1.hazard, w2.hazard)

    def test_observation_channels(self):
        world = GridWorld(GridConfig(seed=42))
        pos = (15, 15)
        for channels in [["food"], ["hazard"], ["food", "hazard"],
                         ["food", "hazard", "density", "memory", "gradient"]]:
            obs = world.get_observation(pos, channels)
            expected_size = world.obs_size_for_channels(channels)
            assert obs.shape == (expected_size,), f"Failed for {channels}"

    def test_action_stays_in_bounds(self):
        world = GridWorld(GridConfig(width=10, height=10, seed=42))
        # Corner cases
        for pos in [(0, 0), (0, 9), (9, 0), (9, 9)]:
            for action in range(9):
                new_pos, _ = world.apply_action(pos, action)
                assert 0 <= new_pos[0] < 10
                assert 0 <= new_pos[1] < 10

    def test_food_consumption(self):
        config = GridConfig(width=10, height=10, n_food=1, n_hazards=0,
                            food_regrow_rate=0, seed=42)
        world = GridWorld(config)
        # Find the food
        food_pos = tuple(np.argwhere(world.food > 0)[0])
        # Place agent on food
        _, reward = world.apply_action(food_pos, 4)  # stay
        assert reward > 0  # got food reward
        assert world.food[food_pos[0], food_pos[1]] == 0  # food consumed

    def test_hazard_penalty(self):
        config = GridConfig(width=10, height=10, n_food=0, n_hazards=1,
                            food_regrow_rate=0, seed=42)
        world = GridWorld(config)
        hazard_pos = tuple(np.argwhere(world.hazard > 0)[0])
        _, reward = world.apply_action(hazard_pos, 4)  # stay on hazard
        assert reward < 0

    def test_environment_shift_b(self):
        world = GridWorld(GridConfig(seed=42))
        food_before = world.food.copy()
        world.shift_environment("B")
        assert world.phase == "B"
        # Food should be redistributed (very unlikely to be identical)
        assert not np.array_equal(food_before, world.food)

    def test_environment_shift_c(self):
        world = GridWorld(GridConfig(seed=42))
        world.shift_environment("C")
        assert world.phase == "C"
        # Phase C has more hazards
        assert world.hazard.sum() > 0

    def test_obs_size_for_channels(self):
        world = GridWorld(GridConfig(seed=42))
        assert world.obs_size_for_channels(["food"]) == 3
        assert world.obs_size_for_channels(["food", "hazard"]) == 6
        assert world.obs_size_for_channels(["food", "hazard", "density"]) == 8
