"""
Architecture evolution (Level 2 — Creativity).

Evolves the Architecture itself — the representational space in which
agents operate. This is a generator of generators: it produces new
architectures, each of which defines a new space of possible agents.
"""

import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy
from .gridworld import GridWorld, GridConfig
from .agents import Architecture, NeuralAgent
from .evolution import WeightEvolver


ALL_CHANNELS = ["food", "hazard", "density", "memory", "gradient"]
ACTIVATIONS = ["tanh", "relu", "sigmoid"]


@dataclass
class ArchitectureRecord:
    """An architecture and its evaluated fitness."""
    architecture: Architecture
    fitness: float = 0.0
    generation_born: int = 0
    parent_label: str = ""


@dataclass
class MetaStats:
    """Statistics for one meta-generation of architecture evolution."""
    meta_step: int
    best_fitness: float
    mean_fitness: float
    best_architecture: dict
    pool_labels: list[str]
    capability_events: list[dict] = field(default_factory=list)


class ArchitectureEvolver:
    """
    Level 2 — Creativity: evolve the architecture (the generator itself).

    Maintains a pool of Architectures. For each architecture, runs a
    Level 1 WeightEvolver to find the best weights, then scores the
    architecture by the best achievable fitness. Poorly performing
    architectures are replaced by mutations of successful ones.

    The key creative operations:
    - Add/remove sensory channels (new ways of perceiving)
    - Add/remove hidden layers (new processing depth)
    - Change layer sizes (new processing capacity)
    - Change activation functions (new computational primitives)
    - Change action count (new behavioral repertoire)
    """

    def __init__(self, initial_architecture: Architecture,
                 pool_size: int = 8,
                 weight_generations: int = 15,
                 weight_pop_size: int = 30,
                 eval_steps: int = 150,
                 seed: int = 42):
        self.pool_size = pool_size
        self.weight_generations = weight_generations
        self.weight_pop_size = weight_pop_size
        self.eval_steps = eval_steps
        self.rng = np.random.default_rng(seed)
        self.seed_counter = seed + 1000

        # Initialize pool with the base architecture + mutations
        self.pool: list[ArchitectureRecord] = [
            ArchitectureRecord(architecture=deepcopy(initial_architecture))
        ]
        for i in range(pool_size - 1):
            mutated = self.mutate_architecture(deepcopy(initial_architecture))
            self.pool.append(ArchitectureRecord(
                architecture=mutated, parent_label=initial_architecture.label()
            ))

        self.meta_step_count = 0
        self.history: list[MetaStats] = []
        self.capability_log: list[dict] = []

    def _next_seed(self) -> int:
        self.seed_counter += 1
        return self.seed_counter

    def mutate_architecture(self, arch: Architecture) -> Architecture:
        """
        Mutate an architecture — the creative act.

        Each mutation changes the *kind of thing* the agent can be,
        not just how well it performs within its current kind.
        """
        new = deepcopy(arch)
        mutation_type = self.rng.choice([
            "add_channel", "remove_channel",
            "add_layer", "remove_layer",
            "resize_layer", "change_activation",
            "change_actions",
        ])

        if mutation_type == "add_channel":
            available = [ch for ch in ALL_CHANNELS if ch not in new.input_channels]
            if available:
                new.input_channels.append(self.rng.choice(available))

        elif mutation_type == "remove_channel":
            if len(new.input_channels) > 1:
                idx = int(self.rng.integers(0, len(new.input_channels)))
                new.input_channels.pop(idx)

        elif mutation_type == "add_layer":
            if len(new.hidden_sizes) < 4:
                size = int(self.rng.choice([8, 12, 16, 24, 32]))
                pos = int(self.rng.integers(0, len(new.hidden_sizes) + 1))
                new.hidden_sizes.insert(pos, size)

        elif mutation_type == "remove_layer":
            if len(new.hidden_sizes) > 1:
                idx = int(self.rng.integers(0, len(new.hidden_sizes)))
                new.hidden_sizes.pop(idx)

        elif mutation_type == "resize_layer":
            if new.hidden_sizes:
                idx = int(self.rng.integers(0, len(new.hidden_sizes)))
                delta = int(self.rng.integers(-8, 9))
                new.hidden_sizes[idx] = max(4, new.hidden_sizes[idx] + delta)

        elif mutation_type == "change_activation":
            new.activation = self.rng.choice(ACTIVATIONS)

        elif mutation_type == "change_actions":
            new.n_actions = 9 if new.n_actions == 5 else 5

        return new

    def evaluate_architecture(self, arch: Architecture,
                              world_config: GridConfig) -> float:
        """
        Evaluate an architecture by running Level 1 weight evolution on it.

        This is the key hierarchy: to evaluate a generator, we must run it.
        The architecture's score is the best fitness achievable by any agent
        with that architecture after weight optimization.
        """
        evolver = WeightEvolver(
            architecture=arch,
            population_size=self.weight_pop_size,
            mutation_rate=0.1,
            seed=self._next_seed(),
        )
        stats = evolver.run(world_config, self.weight_generations, self.eval_steps)

        if stats:
            return max(s.best_fitness for s in stats)
        return 0.0

    def _detect_capability_changes(self, old_best: Architecture,
                                    new_best: Architecture, step: int):
        """Log when new capabilities emerge."""
        old_ch = set(old_best.input_channels)
        new_ch = set(new_best.input_channels)

        for ch in new_ch - old_ch:
            event = {
                "step": step,
                "capability": f"{ch} sensing",
                "description": f"Added {ch} channel to sensory repertoire",
            }
            self.capability_log.append(event)

        if new_best.n_actions > old_best.n_actions:
            event = {
                "step": step,
                "capability": "8-directional movement",
                "description": "Expanded action space to include diagonal movement",
            }
            self.capability_log.append(event)

        if len(new_best.hidden_sizes) > len(old_best.hidden_sizes):
            event = {
                "step": step,
                "capability": "deeper processing",
                "description": f"Added hidden layer (now {len(new_best.hidden_sizes)} layers)",
            }
            self.capability_log.append(event)

    def meta_step(self, world_config: GridConfig) -> MetaStats:
        """
        One step of architecture evolution:
        1. Evaluate all architectures (each via full Level 1 optimization)
        2. Drop worst, replace with mutation of best
        """
        # Evaluate all
        for record in self.pool:
            record.fitness = self.evaluate_architecture(
                record.architecture, world_config
            )

        fitnesses = [r.fitness for r in self.pool]
        best_idx = int(np.argmax(fitnesses))
        worst_idx = int(np.argmin(fitnesses))

        old_best = deepcopy(self.pool[best_idx].architecture)

        # Replace worst with mutation of best
        if best_idx != worst_idx:
            parent = self.pool[best_idx].architecture
            child = self.mutate_architecture(parent)
            self.pool[worst_idx] = ArchitectureRecord(
                architecture=child,
                generation_born=self.meta_step_count,
                parent_label=parent.label(),
            )

        new_best = self.pool[best_idx].architecture
        self._detect_capability_changes(old_best, new_best, self.meta_step_count)

        stats = MetaStats(
            meta_step=self.meta_step_count,
            best_fitness=float(fitnesses[best_idx]),
            mean_fitness=float(np.mean(fitnesses)),
            best_architecture=new_best.to_dict(),
            pool_labels=[r.architecture.label() for r in self.pool],
        )

        self.meta_step_count += 1
        self.history.append(stats)
        return stats

    def run(self, world_config: GridConfig, n_meta_steps: int,
            environment_shifts: dict[int, str] | None = None) -> dict:
        """
        Run the full architecture evolution experiment.

        Args:
            world_config: Base world configuration.
            n_meta_steps: Number of meta-generations.
            environment_shifts: Dict mapping meta-step -> phase ("B", "C").
        """
        shifts = environment_shifts or {}

        for step in range(n_meta_steps):
            if step in shifts:
                phase = shifts[step]
                print(f"  [meta-step {step}] Environment shift -> Phase {phase}")
                # Update config seed to create a different layout
                world_config = GridConfig(
                    width=world_config.width,
                    height=world_config.height,
                    n_food=world_config.n_food,
                    n_hazards=world_config.n_hazards,
                    seed=world_config.seed + step * 100,
                )

            stats = self.meta_step(world_config)
            print(f"  [meta-step {step}] best={stats.best_fitness:.2f} "
                  f"mean={stats.mean_fitness:.2f} "
                  f"arch={self.pool[int(np.argmax([r.fitness for r in self.pool]))].architecture.label()}")

        return {
            "history": [
                {
                    "meta_step": s.meta_step,
                    "best_fitness": s.best_fitness,
                    "mean_fitness": s.mean_fitness,
                    "best_architecture": s.best_architecture,
                    "pool_labels": s.pool_labels,
                }
                for s in self.history
            ],
            "capability_events": self.capability_log,
        }
