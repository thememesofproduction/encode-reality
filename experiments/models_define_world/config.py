"""Shared defaults, color schemes, and plot settings."""

DEFAULT_SEED = 42

# --- SIRS ODE Defaults ---
SIRS_DEFAULTS = {
    "beta": 0.3,
    "gamma": 0.1,
    "omega": 0.01,
    "N": 1000,
    "I0": 10,
    "t_max": 300,
}

# --- ABM Defaults ---
ABM_DEFAULTS = {
    "N": 1000,
    "contact_rate": 10,
    "beta": 0.3,
    "gamma": 0.1,
    "omega": 0.01,
    "I0": 10,
    "n_steps": 3000,
    "dt": 0.1,
    "seed": DEFAULT_SEED,
}

# --- Experiment 1: Recovery ---
RECOVERY_DEFAULTS = {
    "n_ensemble": 20,
    "seed": DEFAULT_SEED,
}

# --- Experiment 2: Divergence (Network) ---
NETWORK_DEFAULTS = {
    "graph_type": "barabasi_albert",
    "n_nodes": 1000,
    "ba_m": 3,
    "ws_k": 10,
    "ws_p": 0.1,
    "heterogeneity": True,
    "beta_std": 0.15,
    "n_ensemble": 10,
    "seed": DEFAULT_SEED,
}

# --- Experiment 3: Irreducibility (Parameter Sweep) ---
SWEEP_DEFAULTS = {
    "sweep_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "n_ensemble": 10,
    "seed": DEFAULT_SEED,
}

# --- Experiment 4: Structured Networks & Interventions ---
STRUCTURED_DEFAULTS = {
    "N": 1000,
    "n_ensemble": 10,
    "I0": 10,
    "n_steps": 3000,
    "vaccination_fraction": 0.2,
    "layer_reduction": 0.5,
    "seed": DEFAULT_SEED,
}

# --- Plotting ---
COLORS = {
    "susceptible": "#6699CC",
    "infected": "#CC6666",
    "recovered": "#66CC99",
    "ode": "#FFFFFF",
    "network_abm": "#CC6666",
    "sweep_line": "#CC9966",
    "background": "#1e1e1e",
}

PLOT_DEFAULTS = {
    "figsize": (10, 6),
    "dpi": 150,
    "style": "dark_background",
}
