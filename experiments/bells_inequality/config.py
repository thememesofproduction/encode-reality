"""Shared defaults, color schemes, and plot settings for Bell's inequality experiments."""

import numpy as np

DEFAULT_SEED = 42

# --- CHSH angle defaults (radians) ---
# Optimal angles for maximal quantum violation of CHSH
OPTIMAL_ANGLES = {
    "a1": 0.0,
    "a2": np.pi / 2,
    "b1": np.pi / 4,
    "b2": 3 * np.pi / 4,
}

# --- Experiment defaults ---
EXHAUSTION_DEFAULTS = {
    "n_random_discrete": 2000,
    "n_random_continuous": 2000,
    "n_adversarial": 100,
    "n_lambda": 1000,       # samples of hidden variable per strategy
    "seed": DEFAULT_SEED,
}

QUANTUM_DEFAULTS = {
    "n_trials": 100_000,
    "seed": DEFAULT_SEED,
}

# --- Theoretical constants ---
CLASSICAL_BOUND = 2.0
TSIRELSON_BOUND = 2 * np.sqrt(2)  # ≈ 2.828

# --- Plotting ---
COLORS = {
    "classical": "#6699CC",
    "quantum": "#CC6666",
    "bound_classical": "#FFFFFF",
    "bound_tsirelson": "#66CC99",
    "deterministic": "#CC9966",
    "random_discrete": "#9966CC",
    "random_continuous": "#66CCCC",
    "adversarial": "#CC6666",
    "violation": "#CC6666",
    "background": "#1e1e1e",
}

PLOT_DEFAULTS = {
    "figsize": (10, 6),
    "dpi": 150,
    "style": "dark_background",
}
