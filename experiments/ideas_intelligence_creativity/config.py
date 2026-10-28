"""
Shared configuration: default parameters, color schemes, random seeds.
"""

# Reproducibility
DEFAULT_SEED = 42

# --- Experiment 1: Fitness Landscape ---

LANDSCAPE_DEFAULTS = {
    "n_peaks": 15,
    "dims": 2,
    "bounds": (-5.0, 5.0),
    "seed": DEFAULT_SEED,
    "global_peak": "none",
}

FIXED_STRATEGY_DEFAULTS = {
    "step_size": 0.2,
    "n_candidates": 20,
    "method": "gaussian",
}

META_OPTIMIZER_DEFAULTS = {
    "meta_interval": 10,         # steps: 200 evaluations at 20 candidates, so a meta-step every 200 evaluations
    "strategy_pool_size": 5,
    "improvement_window": 25,
    "novel_strategy_prob": 0.2,
    "replace_every": 2,          # a new strategy every 400 evaluations; the pool fills by about 1600
    "mutation_rate": 0.3,
    "discount": 0.95,            # per meta-step: recent credit outweighs old credit (non-stationary).
                                 # Frozen from a pilot on held-out root seed 1000 (10 landscapes x 3 starts per
                                 # condition). Settings (discount, UCB_C) = (0.8, 0.5), (0.95, 0.5), (0.95, 0.2) and
                                 # (1.0, 0.5) were all within noise: meta_ucb succeeded in 0.88-0.91 of runs.
}

EXPERIMENT_1_DEFAULTS = {
    "profile": "full",
    "root_seed": DEFAULT_SEED,
    "workers": 1,
}

# Profiles set the sizes of a run (SPEC section 6). Command-line flags override them.
EXPERIMENT_1_PROFILES = {
    "full": {"n_landscapes": 40, "n_starts": 5, "max_evals": 20_000},
    "quick": {"n_landscapes": 3, "n_starts": 2, "max_evals": 2_000},
}

# Canonical orders: a condition's and an arm's index is its seeding key, whatever subset is run.
EXPERIMENT_1_CONDITIONS = ("none", "random", "corner")
EXPERIMENT_1_ARMS = ("hc_fixed", "restart_fixed", "swap_random", "meta_ucb")
EXPERIMENT_1_SUCCESS = 0.99         # a run succeeds when best >= 0.99 * ceiling
EXPERIMENT_1_BOOTSTRAP = 2000       # resamples of the 95% cluster bootstrap over landscapes

# --- Experiment 2: Generator Hierarchy ---

GRIDWORLD_DEFAULTS = {
    "width": 50,
    "height": 50,
    "n_food": 20,
    "n_hazards": 10,
    "seed": DEFAULT_SEED,
}

EVOLUTION_DEFAULTS = {
    "population_size": 50,
    "mutation_rate": 0.1,
    "n_generations": 100,
    "eval_steps": 200,
}

NEUROEVOLUTION_DEFAULTS = {
    "pool_size": 10,
    "weight_evolver_generations": 20,
    "n_meta_steps": 30,
    "environment_shifts_at": [100, 200],
}

# --- Plotting ---

COLORS = {
    "intelligence": "#6699CC",   # steel blue
    "creativity": "#CC6666",     # muted red
    "level_0": "#999999",        # gray
    "level_1": "#6699CC",        # steel blue
    "level_2": "#CC6666",        # muted red
    "landscape": "viridis",      # colormap
    "background": "#1e1e1e",     # dark background
}

PLOT_DEFAULTS = {
    "figsize": (10, 6),
    "dpi": 150,
    "style": "dark_background",
}
