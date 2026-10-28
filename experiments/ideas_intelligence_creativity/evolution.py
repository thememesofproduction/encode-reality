"""
Evolutionary weight optimization (Level 1 — Intelligence).

Given a fixed Architecture, evolves the weights of a population of
NeuralAgents to maximize fitness in the grid world. This is a generator
that produces solutions within a fixed representational space.
"""

import numpy as np
from dataclasses import dataclass, field
from .gridworld import GridWorld, GridConfig
from .agents import Architecture, NeuralAgent, FixedAgent


@dataclass
class GenerationStats:
    """Statistics for one generation of evolution."""
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    best_weights: np.ndarray | None = None


class WeightEvolver:
    """
    Level 1 — Intelligence: evolve weights within a fixed architecture.

    Uses a simple evolutionary strategy (ES):
    1. Maintain a population of weight vectors
    2. Evaluate each by running the agent in the grid world
    3. Select top performers, reproduce with mutation

    The architecture (what channels, how many layers, what actions)
    is FROZEN. This system can only find better solutions within
    the space defined by the architecture.
    """

    def __init__(self, architecture: Architecture,
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 elite_fraction: float = 0.2,
                 seed: int = 42):
        self.architecture = architecture
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_count = max(2, int(population_size * elite_fraction))
        self.rng = np.random.default_rng(seed)

        n_params = architecture.parameter_count()
        self.population = self.rng.normal(0, 0.5, size=(population_size, n_params))

    def evaluate_agent(self, weights: np.ndarray, world: GridWorld,
                       n_steps: int = 200) -> float:
        """Evaluate a single agent's fitness in the world."""
        agent = NeuralAgent(self.architecture, weights)
        position = (world.height // 2, world.width // 2)
        total_reward = 0.0

        for _ in range(n_steps):
            obs = world.get_observation(
                position, self.architecture.input_channels
            )
            action = agent.act(obs)
            position, reward = world.apply_action(position, action)
            total_reward += reward

        return total_reward

    def evaluate_population(self, world_config: GridConfig,
                            n_steps: int = 200) -> np.ndarray:
        """Evaluate all agents. Creates a fresh world for each to ensure fairness."""
        fitnesses = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            world = GridWorld(world_config)
            fitnesses[i] = self.evaluate_agent(
                self.population[i], world, n_steps
            )
        return fitnesses

    def select_and_reproduce(self, fitnesses: np.ndarray):
        """Select elites and fill population via mutation."""
        sorted_idx = np.argsort(fitnesses)[::-1]
        elites = self.population[sorted_idx[:self.elite_count]].copy()

        new_pop = [elites[i] for i in range(self.elite_count)]

        while len(new_pop) < self.pop_size:
            parent_idx = self.rng.integers(0, self.elite_count)
            parent = elites[parent_idx]
            child = parent + self.rng.normal(0, self.mutation_rate, size=parent.shape)
            new_pop.append(child)

        self.population = np.array(new_pop)

    def run_generation(self, world_config: GridConfig,
                       n_steps: int = 200) -> GenerationStats:
        """Run one generation: evaluate → select → reproduce."""
        fitnesses = self.evaluate_population(world_config, n_steps)
        best_idx = int(np.argmax(fitnesses))

        stats = GenerationStats(
            generation=0,
            best_fitness=float(fitnesses[best_idx]),
            mean_fitness=float(fitnesses.mean()),
            std_fitness=float(fitnesses.std()),
            best_weights=self.population[best_idx].copy(),
        )

        self.select_and_reproduce(fitnesses)
        return stats

    def run(self, world_config: GridConfig, n_generations: int,
            n_steps: int = 200) -> list[GenerationStats]:
        """Run the full evolutionary optimization."""
        history = []
        for gen in range(n_generations):
            stats = self.run_generation(world_config, n_steps)
            stats.generation = gen
            history.append(stats)
        return history

    def get_best_agent(self, world_config: GridConfig,
                       n_steps: int = 200) -> NeuralAgent:
        """Return the best agent from current population."""
        fitnesses = self.evaluate_population(world_config, n_steps)
        best_idx = int(np.argmax(fitnesses))
        return NeuralAgent(self.architecture, self.population[best_idx])


def evaluate_fixed_agent(world_config: GridConfig,
                         n_steps: int = 200,
                         n_trials: int = 10) -> float:
    """Evaluate the Level 0 fixed agent over multiple trials."""
    agent = FixedAgent()
    total = 0.0
    for trial in range(n_trials):
        config = GridConfig(
            width=world_config.width, height=world_config.height,
            n_food=world_config.n_food, n_hazards=world_config.n_hazards,
            seed=world_config.seed + trial,
        )
        world = GridWorld(config)
        position = (world.height // 2, world.width // 2)
        trial_reward = 0.0
        for _ in range(n_steps):
            obs = world.get_observation(position, agent.architecture.input_channels)
            action = agent.act(obs)
            position, reward = world.apply_action(position, action)
            trial_reward += reward
        total += trial_reward
    return total / n_trials
