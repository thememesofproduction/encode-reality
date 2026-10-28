#!/usr/bin/env python3
"""
Experiment 2: The Generator Hierarchy (Agent-Based)

Demonstrates three levels of adaptation in a grid world:
  Level 0: Fixed agent (a single idea)
  Level 1: Weight evolution within a fixed architecture (intelligence)
  Level 2: Architecture evolution — evolving the generator itself (creativity)

Usage:
    python run_experiment_2.py
    python run_experiment_2.py --meta-steps 20 --seed 42
"""

import sys
from pathlib import Path
import argparse
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.ideas_intelligence_creativity.gridworld import GridConfig
from experiments.ideas_intelligence_creativity.agents import Architecture
from experiments.ideas_intelligence_creativity.evolution import WeightEvolver, evaluate_fixed_agent
from experiments.ideas_intelligence_creativity.neuroevolution import ArchitectureEvolver
from experiments.ideas_intelligence_creativity.plotting import (
    plot_hierarchy_fitness, plot_capability_timeline
)


def run_experiment(meta_steps: int = 20,
                   weight_generations: int = 20,
                   seed: int = 42,
                   output_dir: str | None = None):
    """Run the three-level hierarchy comparison."""
    print("=" * 60)
    print("EXPERIMENT 2: The Generator Hierarchy")
    print("Agent-Based Grid World")
    print("=" * 60)

    world_config = GridConfig(
        width=25, height=25, n_food=12, n_hazards=6, seed=seed
    )

    # Environment shift schedule
    shift_step_b = meta_steps // 3
    shift_step_c = 2 * meta_steps // 3
    shifts = {shift_step_b: "B", shift_step_c: "C"}

    # --- Level 0: Fixed Agent ---
    print("\n--- Level 0: Fixed Agent (Idea) ---")
    t0 = time.time()
    level0_fitness = evaluate_fixed_agent(world_config, n_steps=150, n_trials=5)
    print(f"Fitness: {level0_fitness:.2f} ({time.time() - t0:.1f}s)")

    # Evaluate across all phases
    level0_history = []
    for step in range(meta_steps):
        cfg = world_config
        if step >= shift_step_c:
            cfg = GridConfig(
                width=world_config.width, height=world_config.height,
                n_food=world_config.n_food, n_hazards=world_config.n_hazards,
                seed=world_config.seed + step * 100,
            )
        elif step >= shift_step_b:
            cfg = GridConfig(
                width=world_config.width, height=world_config.height,
                n_food=world_config.n_food, n_hazards=world_config.n_hazards,
                seed=world_config.seed + step * 100,
            )
        fitness = evaluate_fixed_agent(cfg, n_steps=150, n_trials=3)
        level0_history.append(fitness)

    # --- Level 1: Weight Evolution (fixed architecture) ---
    print(f"\n--- Level 1: Intelligence (weight evolution, {weight_generations} gen/step) ---")
    base_arch = Architecture(
        input_channels=["food", "hazard"],
        hidden_sizes=[16, 8],
        n_actions=5,
    )
    print(f"Architecture: {base_arch.label()} ({base_arch.parameter_count()} params)")

    level1_history = []
    evolver = WeightEvolver(
        base_arch, population_size=40,
        mutation_rate=0.1, seed=seed + 10
    )
    t0 = time.time()
    for step in range(meta_steps):
        cfg = world_config
        if step == shift_step_b:
            print(f"  [step {step}] Environment shift -> Phase B")
            cfg = GridConfig(
                width=world_config.width, height=world_config.height,
                n_food=world_config.n_food, n_hazards=world_config.n_hazards,
                seed=world_config.seed + step * 100,
            )
        elif step == shift_step_c:
            print(f"  [step {step}] Environment shift -> Phase C")
            cfg = GridConfig(
                width=world_config.width, height=world_config.height,
                n_food=world_config.n_food, n_hazards=world_config.n_hazards,
                seed=world_config.seed + step * 100,
            )

        stats = evolver.run(cfg, n_generations=weight_generations, n_steps=150)
        best = max(s.best_fitness for s in stats) if stats else 0
        level1_history.append(best)
        if step % 5 == 0:
            print(f"  [step {step}] best={best:.2f}")

    print(f"Level 1 complete ({time.time() - t0:.1f}s)")

    # --- Level 2: Architecture Evolution ---
    print(f"\n--- Level 2: Creativity (architecture evolution, {meta_steps} meta-steps) ---")
    t0 = time.time()
    arch_evolver = ArchitectureEvolver(
        initial_architecture=base_arch,
        pool_size=6,
        weight_generations=weight_generations,
        weight_pop_size=30,
        eval_steps=150,
        seed=seed + 20,
    )
    level2_result = arch_evolver.run(world_config, meta_steps, shifts)
    level2_history = [h["best_fitness"] for h in level2_result["history"]]
    print(f"Level 2 complete ({time.time() - t0:.1f}s)")

    # --- Save Results ---
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "results")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "params": {
            "meta_steps": meta_steps,
            "weight_generations": weight_generations,
            "seed": seed,
            "shifts": {str(k): v for k, v in shifts.items()},
        },
        "level0": {"best_fitness": level0_history},
        "level1": {"best_fitness": level1_history},
        "level2": level2_result,
    }
    results_path = out / "experiment_2_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Generate Plots ---
    print("\nGenerating plots...")
    img_dir = Path(__file__).parent.parent.parent / "static" / "images" / "ideas_intelligence_creativity"
    img_dir.mkdir(parents=True, exist_ok=True)

    p = img_dir / "hierarchy_fitness.png"
    plot_hierarchy_fitness(
        {"best_fitness": level0_history},
        {"best_fitness": level1_history},
        {"best_fitness": level2_history},
        shift_points=[shift_step_b, shift_step_c],
        output_path=p,
    )
    print(f"  Saved: {p}")

    if level2_result["capability_events"]:
        p = img_dir / "capability_timeline.png"
        plot_capability_timeline(level2_result["capability_events"], output_path=p)
        print(f"  Saved: {p}")

    print("\n" + "=" * 60)
    print("Experiment 2 complete.")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: The Generator Hierarchy"
    )
    parser.add_argument("--meta-steps", type=int, default=20,
                        help="Number of meta-evolution steps")
    parser.add_argument("--weight-generations", type=int, default=20,
                        help="Weight evolution generations per meta-step")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    run_experiment(
        meta_steps=args.meta_steps,
        weight_generations=args.weight_generations,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
