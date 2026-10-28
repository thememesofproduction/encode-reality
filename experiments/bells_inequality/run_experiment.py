"""CLI runner for the Bell's inequality experiments."""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.bells_inequality.config import (
    EXHAUSTION_DEFAULTS, QUANTUM_DEFAULTS, DEFAULT_SEED,
    CLASSICAL_BOUND, TSIRELSON_BOUND,
)
from experiments.bells_inequality.local_hv import run_exhaustion_experiment
from experiments.bells_inequality.quantum import run_quantum_experiment
from experiments.bells_inequality.plotting import save_all_plots
from experiments.bells_inequality.cache import save_cache, load_cache


def run_experiment(seed=DEFAULT_SEED, experiment="all", output_dir="./results",
                   plot_only=False):
    """Run experiments and generate outputs."""
    print("=" * 60)
    print("  Bell's Inequality — Experiments")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    img_dir = Path(__file__).parent.parent.parent / "static" / "images" / "bells_inequality"
    plot_dir = Path(__file__).parent.parent.parent / "static" / "plots" / "bells_inequality"

    if plot_only:
        print("  --plot-only: loading from cache, skipping simulation")
        cached = load_cache(output_dir)
        exhaustion_data = cached["exhaustion"]
        quantum_data = cached["quantum"]
    else:
        print(f"  seed={seed}")
        exhaustion_data = None
        quantum_data = None

        if experiment in ("all", "exhaustion"):
            exhaustion_data = run_exhaustion_experiment(seed=seed)

            # Verify bound
            all_S = np.concatenate([
                np.abs(exhaustion_data["deterministic"]),
                np.abs(exhaustion_data["random_discrete"]),
                np.abs(exhaustion_data["random_continuous"]),
                exhaustion_data["adversarial"],
            ])
            max_S = np.max(all_S)
            print(f"\n  VERIFICATION: max |S| across all LHV = {max_S:.6f}")
            assert max_S <= CLASSICAL_BOUND + 1e-6, \
                f"Classical bound violated: |S| = {max_S}"

        if experiment in ("all", "quantum"):
            quantum_data = run_quantum_experiment(seed=seed)

            # Verify violation
            print(f"\n  VERIFICATION: |S_theory| = {abs(quantum_data['theoretical_S']):.6f}")
            assert abs(quantum_data["theoretical_S"]) > CLASSICAL_BOUND, \
                "Quantum S does not violate classical bound"

        # Save cache
        save_cache(output_dir, exhaustion_data, quantum_data)

    # Save summary JSON
    results = {"params": {"seed": seed}}
    if exhaustion_data:
        results["experiment_1_exhaustion"] = {
            "n_deterministic": len(exhaustion_data["deterministic"]),
            "n_random_discrete": len(exhaustion_data["random_discrete"]),
            "n_random_continuous": len(exhaustion_data["random_continuous"]),
            "n_adversarial": len(exhaustion_data["adversarial"]),
            "max_abs_S_discrete": float(np.max(np.abs(exhaustion_data["random_discrete"]))),
            "max_abs_S_continuous": float(np.max(np.abs(exhaustion_data["random_continuous"]))),
            "max_abs_S_adversarial": float(np.max(exhaustion_data["adversarial"])),
        }
    if quantum_data:
        results["experiment_2_quantum"] = {
            "theoretical_S": quantum_data["theoretical_S"],
            "simulated_S": quantum_data["simulated_S"],
            "tsirelson_bound": TSIRELSON_BOUND,
            "classical_bound": CLASSICAL_BOUND,
        }

    results_file = output_path / "experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    # Generate plots
    saved = save_all_plots(exhaustion_data, quantum_data, img_dir, plot_dir)
    for p in saved:
        print(f"  Plot saved: {p}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Bell's Inequality experiments")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--experiment",
                        choices=["all", "exhaustion", "quantum", "interactive"],
                        default="all")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulation; load cache and regenerate plots.")
    args = parser.parse_args()

    run_experiment(
        seed=args.seed,
        experiment=args.experiment,
        output_dir=args.output_dir,
        plot_only=args.plot_only,
    )


if __name__ == "__main__":
    main()
