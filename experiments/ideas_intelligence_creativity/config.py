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
}

FIXED_STRATEGY_DEFAULTS = {
    "step_size": 0.2,
    "n_candidates": 20,
    "method": "gaussian",
}

META_OPTIMIZER_DEFAULTS = {
    "meta_interval": 50,
    "strategy_pool_size": 5,
    "improvement_window": 25,
    "novel_strategy_prob": 0.2,
}

EXPERIMENT_1_DEFAULTS = {
    "n_steps": 1000,
    "seed": DEFAULT_SEED,
}

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
