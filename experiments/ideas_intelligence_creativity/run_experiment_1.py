#!/usr/bin/env python3
"""
Experiment 1: Intelligence vs Creativity on a Fitness Landscape

Demonstrates the difference between a fixed-strategy optimizer (Level 1)
and a meta-strategy optimizer (Level 2) on a rugged fitness landscape.

Usage:
    python run_experiment_1.py
    python run_experiment_1.py --n-steps 2000 --seed 42
    python run_experiment_1.py --output-dir ./results
"""

import sys
from pathlib import Path
import argparse
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.ideas_intelligence_creativity.landscape import FitnessLandscape
from experiments.ideas_intelligence_creativity.strategies import SearchStrategy, StrategyMutator
from experiments.ideas_intelligence_creativity.intelligence import FixedStrategyOptimizer
from experiments.ideas_intelligence_creativity.creativity import MetaStrategyOptimizer
from experiments.ideas_intelligence_creativity.plotting import save_all_experiment1_plots
from experiments.ideas_intelligence_creativity.config import (
    LANDSCAPE_DEFAULTS, FIXED_STRATEGY_DEFAULTS,
    META_OPTIMIZER_DEFAULTS, EXPERIMENT_1_DEFAULTS,
)


def run_experiment(n_steps: int = 1000, seed: int = 42,
                   n_peaks: int = 15, output_dir: str | None = None):
    """Run the Intelligence vs Creativity comparison experiment."""
    print("=" * 60)
    print("EXPERIMENT 1: Intelligence vs Creativity")
    print("Fitness Landscape Optimization")
    print("=" * 60)

    # Setup
    import numpy as np

    landscape = FitnessLandscape(n_peaks=n_peaks, seed=seed)
    opt_pos, opt_val = landscape.get_global_optimum()
    print(f"\nLandscape: {n_peaks} peaks, global optimum = {opt_val:.3f}")

    strategy = SearchStrategy(**FIXED_STRATEGY_DEFAULTS)
    mutator = StrategyMutator(mutation_rate=0.3, seed=seed + 1)

    # Both optimizers start from the same position (center of space)
    start_pos = np.zeros(landscape.dims)

    # Run fixed-strategy optimizer (Intelligence)
    print(f"\n--- Level 1: Intelligence (fixed strategy) ---")
    print(f"Strategy: {strategy.label()}")
    t0 = time.time()
    fixed_opt = FixedStrategyOptimizer(landscape, strategy, seed=seed)
    fixed_opt.position = start_pos.copy()
    fixed_opt.best_fitness = landscape.evaluate(start_pos)
    fixed_opt.best_position = start_pos.copy()
    fixed_result = fixed_opt.run(n_steps)
    t_fixed = time.time() - t0
    print(f"Best fitness: {fixed_result['best_fitness'][-1]:.4f} ({t_fixed:.2f}s)")
    print(f"Reached {fixed_result['best_fitness'][-1] / opt_val * 100:.1f}% of global optimum")

    # Run meta-strategy optimizer (Creativity)
    print(f"\n--- Level 2: Creativity (meta-strategy) ---")
    t0 = time.time()
    meta_opt = MetaStrategyOptimizer(
        landscape, strategy, mutator,
        meta_interval=META_OPTIMIZER_DEFAULTS["meta_interval"],
        strategy_pool_size=META_OPTIMIZER_DEFAULTS["strategy_pool_size"],
        improvement_window=META_OPTIMIZER_DEFAULTS["improvement_window"],
        novel_strategy_prob=META_OPTIMIZER_DEFAULTS["novel_strategy_prob"],
        seed=seed + 2,
    )
    meta_opt.position = start_pos.copy()
    meta_opt.best_fitness = landscape.evaluate(start_pos)
    meta_opt.best_position = start_pos.copy()
    meta_result = meta_opt.run(n_steps)
    t_meta = time.time() - t0
    print(f"Best fitness: {meta_result['best_fitness'][-1]:.4f} ({t_meta:.2f}s)")
    print(f"Reached {meta_result['best_fitness'][-1] / opt_val * 100:.1f}% of global optimum")
    print(f"Meta-events: {len(meta_result['meta_events'])}")

    # Save results
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "results")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "params": {
            "n_steps": n_steps, "seed": seed, "n_peaks": n_peaks,
            "strategy": FIXED_STRATEGY_DEFAULTS,
            "meta": META_OPTIMIZER_DEFAULTS,
        },
        "global_optimum": {"position": opt_pos.tolist(), "fitness": opt_val},
        "fixed": fixed_result,
        "meta": meta_result,
    }
    results_path = out / "experiment_1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    print("\nGenerating plots...")
    img_dir = Path(__file__).parent.parent.parent / "static" / "images" / "ideas_intelligence_creativity"
    plots = save_all_experiment1_plots(fixed_result, meta_result, landscape, img_dir)
    for p in plots:
        print(f"  Saved: {p}")

    print("\n" + "=" * 60)
    print("Experiment 1 complete.")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: Intelligence vs Creativity on a Fitness Landscape"
    )
    parser.add_argument("--n-steps", type=int, default=EXPERIMENT_1_DEFAULTS["n_steps"],
                        help="Number of optimization steps")
    parser.add_argument("--seed", type=int, default=EXPERIMENT_1_DEFAULTS["seed"],
                        help="Random seed")
    parser.add_argument("--n-peaks", type=int, default=LANDSCAPE_DEFAULTS["n_peaks"],
                        help="Number of landscape peaks")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    run_experiment(
        n_steps=args.n_steps,
        seed=args.seed,
        n_peaks=args.n_peaks,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
