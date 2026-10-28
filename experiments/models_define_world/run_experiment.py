"""CLI runner for the Models Define the World experiments."""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.models_define_world.config import (
    SIRS_DEFAULTS, ABM_DEFAULTS, RECOVERY_DEFAULTS,
    NETWORK_DEFAULTS, SWEEP_DEFAULTS, STRUCTURED_DEFAULTS, DEFAULT_SEED,
)
from experiments.models_define_world.sirs_ode import solve_sirs
from experiments.models_define_world.abm_core import Population
from experiments.models_define_world.networks import generate_network
from experiments.models_define_world.simulation import run_abm
from experiments.models_define_world.layered_network import generate_layered_network
from experiments.models_define_world.interventions import (
    random_vaccination, targeted_vaccination, layer_contact_reduction,
)
from experiments.models_define_world.plotting import save_all_plots, save_structured_plots
from experiments.models_define_world.cache import save_cache, load_cache


def run_recovery(seed, n_ensemble, N, n_steps, beta, gamma, omega, I0, dt):
    """Experiment 1: ABM with homogeneous mixing recovers SIRS ODE."""
    print("\n--- Experiment 1: Recovery ---")
    ode = solve_sirs(beta=beta, gamma=gamma, omega=omega, N=N, I0=I0,
                     t_max=SIRS_DEFAULTS["t_max"], n_points=n_steps)
    print(f"  ODE peak I: {np.max(ode.I):.1f}")

    ensemble = []
    for i in range(n_ensemble):
        run_seed = seed + i
        pop = Population(N=N, I0=I0, beta=beta, gamma=gamma, omega=omega, seed=run_seed)
        rng = np.random.default_rng(run_seed)
        hist = run_abm(pop, n_steps=n_steps, rng=rng,
                       contact_rate=ABM_DEFAULTS["contact_rate"], dt=dt)
        ensemble.append(hist)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{n_ensemble} runs")

    peaks = [np.max(h.I_counts) for h in ensemble]
    print(f"  ABM mean peak I: {np.mean(peaks):.1f} +/- {np.std(peaks):.1f}")
    return ode, ensemble


def run_divergence(seed, n_ensemble, N, n_steps, beta, gamma, omega, I0, dt):
    """Experiment 2: Network structure breaks ODE equivalence."""
    print("\n--- Experiment 2: Divergence ---")
    G = generate_network(
        NETWORK_DEFAULTS["graph_type"], n_nodes=N, seed=seed,
        m=NETWORK_DEFAULTS["ba_m"],
    )
    print(f"  Network: {NETWORK_DEFAULTS['graph_type']} (N={N}, m={NETWORK_DEFAULTS['ba_m']})")

    runs = []
    for i in range(n_ensemble):
        run_seed = seed + 1000 + i
        pop = Population(N=N, I0=I0, beta=beta, gamma=gamma, omega=omega,
                         seed=run_seed, heterogeneity=NETWORK_DEFAULTS["heterogeneity"],
                         beta_std=NETWORK_DEFAULTS["beta_std"])
        rng = np.random.default_rng(run_seed)
        hist = run_abm(pop, n_steps=n_steps, rng=rng, network=G, dt=dt)
        runs.append(hist)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{n_ensemble} runs")

    peaks = [np.max(h.I_counts) for h in runs]
    print(f"  Network ABM mean peak I: {np.mean(peaks):.1f} +/- {np.std(peaks):.1f}")
    return runs


def run_irreducibility(seed, n_ensemble, N, n_steps, beta, gamma, omega, I0, dt):
    """Experiment 3: Compliance sweep reveals threshold effects."""
    print("\n--- Experiment 3: Irreducibility ---")
    sweep_values = SWEEP_DEFAULTS["sweep_values"]

    G = generate_network(
        NETWORK_DEFAULTS["graph_type"], n_nodes=N, seed=seed,
        m=NETWORK_DEFAULTS["ba_m"],
    )

    peak_means = []
    peak_stds = []

    for compliance in sweep_values:
        peaks = []
        for i in range(n_ensemble):
            run_seed = seed + 2000 + i
            pop = Population(N=N, I0=I0, beta=beta, gamma=gamma, omega=omega,
                             seed=run_seed)
            # Set compliance on all agents
            for agent in pop.agents:
                agent.compliance = compliance

            rng = np.random.default_rng(run_seed)
            hist = run_abm(pop, n_steps=n_steps, rng=rng, network=G, dt=dt)
            peaks.append(np.max(hist.I_counts) / N)

        peak_means.append(float(np.mean(peaks)))
        peak_stds.append(float(np.std(peaks)))
        print(f"  compliance={compliance:.1f}: peak={peak_means[-1]:.3f} +/- {peak_stds[-1]:.3f}")

    return sweep_values, {"peak_mean": peak_means, "peak_std": peak_stds}


def run_structured(seed, n_ensemble, N, n_steps, beta, gamma, omega, I0, dt):
    """Experiment 4: Layered contact networks & intervention ranking."""
    print("\n--- Experiment 4: Structured Networks & Interventions ---")

    vax_frac = STRUCTURED_DEFAULTS["vaccination_fraction"]
    layer_red = STRUCTURED_DEFAULTS["layer_reduction"]

    G = generate_layered_network(N, seed=seed)
    print(f"  Layered network: N={N}, edges={G.number_of_edges()}")

    conditions = {
        "no_intervention": {},
        "random_vax": {"vaccination": "random", "fraction": vax_frac},
        "targeted_vax": {"vaccination": "targeted", "fraction": vax_frac},
        "close_workplace": {"layer_reduction": ("workplace", layer_red)},
        "close_community": {"layer_reduction": ("community", layer_red)},
    }

    # Also run homogeneous ABM with same interventions for comparison
    all_results = {}

    for cond_name, cond_params in conditions.items():
        print(f"  Condition: {cond_name}")
        peaks = []
        for i in range(n_ensemble):
            run_seed = seed + 4000 + i
            pop = Population(N=N, I0=I0, beta=beta, gamma=gamma, omega=omega,
                             seed=run_seed)
            rng = np.random.default_rng(run_seed)

            network = G.copy()

            # Apply interventions
            if "vaccination" in cond_params:
                if cond_params["vaccination"] == "random":
                    random_vaccination(pop, cond_params["fraction"], rng)
                elif cond_params["vaccination"] == "targeted":
                    targeted_vaccination(pop, network, cond_params["fraction"], rng)

            if "layer_reduction" in cond_params:
                layer, reduction = cond_params["layer_reduction"]
                network = layer_contact_reduction(network, layer, reduction, rng)

            hist = run_abm(pop, n_steps=n_steps, rng=rng, network=network, dt=dt)
            peaks.append(np.max(hist.I_counts) / N)

        all_results[cond_name] = {
            "peak_mean": float(np.mean(peaks)),
            "peak_std": float(np.std(peaks)),
            "peaks": [float(p) for p in peaks],
        }
        print(f"    peak={all_results[cond_name]['peak_mean']:.3f} "
              f"+/- {all_results[cond_name]['peak_std']:.3f}")

    # Run homogeneous ABM with same interventions for comparison
    homogeneous_results = {}
    for cond_name in ["no_intervention", "random_vax", "targeted_vax"]:
        cond_params = conditions[cond_name]
        print(f"  Homogeneous - {cond_name}")
        peaks = []
        for i in range(n_ensemble):
            run_seed = seed + 5000 + i
            pop = Population(N=N, I0=I0, beta=beta, gamma=gamma, omega=omega,
                             seed=run_seed)
            rng = np.random.default_rng(run_seed)

            if "vaccination" in cond_params:
                if cond_params["vaccination"] == "random":
                    random_vaccination(pop, cond_params["fraction"], rng)
                elif cond_params["vaccination"] == "targeted":
                    # For homogeneous, targeted = random (no network structure)
                    random_vaccination(pop, cond_params["fraction"], rng)

            hist = run_abm(pop, n_steps=n_steps, rng=rng,
                           contact_rate=ABM_DEFAULTS["contact_rate"], dt=dt)
            peaks.append(np.max(hist.I_counts) / N)

        homogeneous_results[cond_name] = {
            "peak_mean": float(np.mean(peaks)),
            "peak_std": float(np.std(peaks)),
            "peaks": [float(p) for p in peaks],
        }
        print(f"    peak={homogeneous_results[cond_name]['peak_mean']:.3f} "
              f"+/- {homogeneous_results[cond_name]['peak_std']:.3f}")

    # Also collect trajectory data for the comparison plot
    # ODE baseline
    ode = solve_sirs(beta=beta, gamma=gamma, omega=omega, N=N, I0=I0,
                     t_max=SIRS_DEFAULTS["t_max"], n_points=n_steps)

    # Homogeneous ABM ensemble (no intervention)
    homogeneous_ensemble = []
    for i in range(n_ensemble):
        run_seed = seed + 5000 + i
        pop = Population(N=N, I0=I0, beta=beta, gamma=gamma, omega=omega, seed=run_seed)
        rng = np.random.default_rng(run_seed)
        hist = run_abm(pop, n_steps=n_steps, rng=rng,
                       contact_rate=ABM_DEFAULTS["contact_rate"], dt=dt)
        homogeneous_ensemble.append(hist)

    # Layered ABM ensemble (no intervention)
    layered_ensemble = []
    for i in range(n_ensemble):
        run_seed = seed + 4000 + i
        pop = Population(N=N, I0=I0, beta=beta, gamma=gamma, omega=omega, seed=run_seed)
        rng = np.random.default_rng(run_seed)
        hist = run_abm(pop, n_steps=n_steps, rng=rng, network=G, dt=dt)
        layered_ensemble.append(hist)

    return {
        "ode": ode,
        "homogeneous_ensemble": homogeneous_ensemble,
        "layered_ensemble": layered_ensemble,
        "layered_results": all_results,
        "homogeneous_results": homogeneous_results,
    }


def run_experiment(seed=DEFAULT_SEED, n_ensemble=20, n_steps=3000,
                   N=1000, experiment="all", output_dir="./results",
                   plot_only=False):
    """Run all experiments and generate outputs."""
    beta = SIRS_DEFAULTS["beta"]
    gamma = SIRS_DEFAULTS["gamma"]
    omega = SIRS_DEFAULTS["omega"]
    I0 = SIRS_DEFAULTS["I0"]

    dt = SIRS_DEFAULTS["t_max"] / n_steps

    print("=" * 60)
    print("  Models Define the World -- Experiments")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if plot_only:
        print("  --plot-only: loading from cache, skipping simulation")
        cached = load_cache(output_dir)
        ode = cached["ode"]
        recovery_ensemble = cached["recovery_ensemble"]
        network_runs = cached["network_runs"]
        sweep_values = cached["sweep_values"]
        sweep_results = cached["sweep_results"]
        structured_data = cached["structured_data"]
        n_steps = cached["n_steps"] if n_steps == 3000 else n_steps
        dt = cached["dt"] if dt == SIRS_DEFAULTS["t_max"] / 3000 else dt
        n_ensemble = cached["n_ensemble"] if n_ensemble == 20 else n_ensemble
    else:
        print(f"  seed={seed}, N={N}, n_steps={n_steps}, n_ensemble={n_ensemble}, dt={dt}")
        print(f"  beta={beta}, gamma={gamma}, omega={omega}, I0={I0}")

        ode = solve_sirs(beta=beta, gamma=gamma, omega=omega, N=N, I0=I0,
                         t_max=SIRS_DEFAULTS["t_max"], n_points=n_steps)
        recovery_ensemble = None
        network_runs = None
        sweep_values = None
        sweep_results = None
        structured_data = None

        if experiment in ("all", "recovery"):
            ode, recovery_ensemble = run_recovery(seed, n_ensemble, N, n_steps,
                                                  beta, gamma, omega, I0, dt)

        if experiment in ("all", "divergence"):
            network_runs = run_divergence(seed, n_ensemble, N, n_steps,
                                          beta, gamma, omega, I0, dt)

        if experiment in ("all", "irreducibility"):
            sweep_values, sweep_results = run_irreducibility(
                seed, n_ensemble, N, n_steps, beta, gamma, omega, I0, dt)

        if experiment in ("all", "structured"):
            structured_data = run_structured(seed, n_ensemble, N, n_steps,
                                             beta, gamma, omega, I0, dt)

        # Save compressed cache for future --plot-only runs
        save_cache(output_dir, ode, recovery_ensemble, network_runs,
                   sweep_values, sweep_results, structured_data,
                   n_steps=n_steps, dt=dt, n_ensemble=n_ensemble)

    # Save summary JSON
    results = {
        "params": {
            "seed": seed, "N": N, "n_steps": n_steps, "n_ensemble": n_ensemble,
            "beta": beta, "gamma": gamma, "omega": omega, "I0": I0,
        },
    }
    if recovery_ensemble:
        results["experiment_1_recovery"] = {
            "ode": ode.to_dict(),
            "abm_peak_mean": float(np.mean([np.max(h.I_counts) for h in recovery_ensemble])),
            "abm_peak_std": float(np.std([np.max(h.I_counts) for h in recovery_ensemble])),
        }
    if sweep_results:
        results["experiment_3_irreducibility"] = {
            "sweep_values": sweep_values,
            "peak_mean": sweep_results["peak_mean"],
            "peak_std": sweep_results["peak_std"],
        }
    if structured_data:
        results["experiment_4_structured"] = {
            "layered_results": structured_data["layered_results"],
            "homogeneous_results": structured_data["homogeneous_results"],
        }

    results_file = output_path / "experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    # Generate plots
    img_dir = Path(__file__).parent.parent.parent / "static" / "images" / "models_define_world"

    if recovery_ensemble and network_runs and sweep_results:
        saved = save_all_plots(ode, recovery_ensemble, network_runs,
                               sweep_values, sweep_results, img_dir,
                               n_steps=n_steps, dt=dt, n_ensemble=n_ensemble)
        for p in saved:
            print(f"  Plot saved: {p}")

    if structured_data:
        saved = save_structured_plots(structured_data, img_dir,
                                       n_steps=n_steps, dt=dt,
                                       n_ensemble=n_ensemble)
        for p in saved:
            print(f"  Plot saved: {p}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Models Define the World experiments")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-ensemble", type=int, default=RECOVERY_DEFAULTS["n_ensemble"])
    parser.add_argument("--n-steps", type=int, default=ABM_DEFAULTS["n_steps"])
    parser.add_argument("--population", type=int, default=ABM_DEFAULTS["N"])
    parser.add_argument("--experiment", choices=["all", "recovery", "divergence", "irreducibility", "structured"],
                        default="all")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulation; load cache and regenerate plots only.")
    args = parser.parse_args()

    run_experiment(
        seed=args.seed,
        n_ensemble=args.n_ensemble,
        n_steps=args.n_steps,
        N=args.population,
        experiment=args.experiment,
        output_dir=args.output_dir,
        plot_only=args.plot_only,
    )


if __name__ == "__main__":
    main()
