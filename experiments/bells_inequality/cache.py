"""Save and load experiment results to/from a compressed numpy cache.

Usage
-----
Save after a run::

    from experiments.bells_inequality.cache import save_cache
    save_cache(output_dir, exhaustion_data, quantum_data)

Load for re-plotting::

    from experiments.bells_inequality.cache import load_cache
    data = load_cache(output_dir)
"""

import numpy as np
from pathlib import Path

CACHE_FILE = "experiment_cache.npz"


def save_cache(output_dir, exhaustion_data=None, quantum_data=None):
    """Compress all experiment outputs to a single .npz file."""
    arrays = {}

    if exhaustion_data is not None:
        arrays["exh_deterministic"] = exhaustion_data["deterministic"]
        arrays["exh_random_discrete"] = exhaustion_data["random_discrete"]
        arrays["exh_random_continuous"] = exhaustion_data["random_continuous"]
        arrays["exh_adversarial"] = exhaustion_data["adversarial"]

    if quantum_data is not None:
        arrays["qm_theoretical_S"] = np.array(quantum_data["theoretical_S"])
        arrays["qm_simulated_S"] = np.array(quantum_data["simulated_S"])
        arrays["qm_convergence_n"] = quantum_data["convergence_n"]
        arrays["qm_convergence_S"] = quantum_data["convergence_S"]
        # Store correlations as parallel arrays
        pairs = list(quantum_data["correlations"].keys())
        arrays["qm_corr_keys"] = np.array([f"{a}_{b}" for a, b in pairs])
        arrays["qm_corr_vals"] = np.array([quantum_data["correlations"][p]
                                            for p in pairs])

    path = Path(output_dir) / CACHE_FILE
    np.savez_compressed(path, **arrays)
    print(f"  Cache saved: {path}")
    return path


def load_cache(output_dir):
    """Load experiment outputs from a compressed .npz cache."""
    path = Path(output_dir) / CACHE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No cache found at {path}. Run without --plot-only first.")

    d = np.load(path, allow_pickle=False)
    result = {"exhaustion": None, "quantum": None}

    if "exh_deterministic" in d:
        result["exhaustion"] = {
            "deterministic": d["exh_deterministic"],
            "random_discrete": d["exh_random_discrete"],
            "random_continuous": d["exh_random_continuous"],
            "adversarial": d["exh_adversarial"],
        }

    if "qm_theoretical_S" in d:
        keys = [tuple(k.split("_")) for k in d["qm_corr_keys"]]
        vals = d["qm_corr_vals"]
        result["quantum"] = {
            "theoretical_S": float(d["qm_theoretical_S"]),
            "simulated_S": float(d["qm_simulated_S"]),
            "convergence_n": d["qm_convergence_n"],
            "convergence_S": d["qm_convergence_S"],
            "correlations": dict(zip(keys, vals)),
        }

    print(f"  Cache loaded: {path}")
    return result
