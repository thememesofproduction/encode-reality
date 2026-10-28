"""Save and load experiment results to/from a compressed numpy cache.

Trajectories are stored as stacked 2-D arrays (n_runs × n_steps), which
compress well and round-trip exactly.  String condition names are stored as
numpy unicode arrays so the file needs no pickling.

Usage
-----
Save after a run::

    from experiments.models_define_world.cache import save_cache
    save_cache(output_dir, ode, recovery_ensemble, network_runs,
               sweep_values, sweep_results, structured_data,
               n_steps=n_steps, dt=dt, n_ensemble=n_ensemble)

Load for re-plotting::

    from experiments.models_define_world.cache import load_cache
    data = load_cache(output_dir)
    # data keys: ode, recovery_ensemble, network_runs,
    #            sweep_values, sweep_results, structured_data,
    #            n_steps, dt, n_ensemble
"""

import numpy as np
from pathlib import Path

from experiments.models_define_world.sirs_ode import SIRSResult
from experiments.models_define_world.simulation import SimulationHistory

CACHE_FILE = "experiment_cache.npz"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _stack_ensemble(ensemble):
    """Return (S, I, R) as (n_runs × n_steps) arrays."""
    S = np.stack([h.S_counts for h in ensemble])
    I = np.stack([h.I_counts for h in ensemble])
    R = np.stack([h.R_counts for h in ensemble])
    return S, I, R


def _unstack_ensemble(S, I, R):
    """Reconstruct a list of SimulationHistory from stacked arrays."""
    return [
        SimulationHistory(S_counts=S[i], I_counts=I[i], R_counts=R[i])
        for i in range(S.shape[0])
    ]


def _encode_results(prefix, results_dict, arrays):
    """Encode {condition: {peak_mean, peak_std}} into parallel numpy arrays."""
    conds = list(results_dict.keys())
    arrays[f"{prefix}_conds"] = np.array(conds)          # unicode, no pickle
    arrays[f"{prefix}_peak_mean"] = np.array(
        [results_dict[c]["peak_mean"] for c in conds]
    )
    arrays[f"{prefix}_peak_std"] = np.array(
        [results_dict[c]["peak_std"] for c in conds]
    )


def _decode_results(prefix, d):
    """Reconstruct {condition: {peak_mean, peak_std}} from parallel arrays."""
    conds = list(d[f"{prefix}_conds"])
    means = d[f"{prefix}_peak_mean"]
    stds = d[f"{prefix}_peak_std"]
    return {c: {"peak_mean": float(m), "peak_std": float(s)}
            for c, m, s in zip(conds, means, stds)}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def save_cache(output_dir, ode, recovery_ensemble, network_runs,
               sweep_values, sweep_results, structured_data,
               n_steps=None, dt=None, n_ensemble=None):
    """Compress all experiment outputs to a single .npz file.

    Parameters
    ----------
    output_dir : str or Path
    ode : SIRSResult
    recovery_ensemble : list[SimulationHistory] or None
    network_runs : list[SimulationHistory] or None
    sweep_values : list[float] or None
    sweep_results : dict or None   keys: peak_mean, peak_std
    structured_data : dict or None  keys: ode, homogeneous_ensemble,
                                         layered_ensemble, layered_results,
                                         homogeneous_results
    n_steps, dt, n_ensemble : scalars stored as metadata
    """
    arrays = {}

    # metadata
    arrays["meta_n_steps"] = np.array(n_steps if n_steps is not None else -1)
    arrays["meta_dt"] = np.array(dt if dt is not None else float("nan"))
    arrays["meta_n_ensemble"] = np.array(n_ensemble if n_ensemble is not None else -1)

    # ODE baseline
    if ode is not None:
        arrays.update({"ode_t": ode.t, "ode_S": ode.S, "ode_I": ode.I, "ode_R": ode.R})

    # Experiment 1 — recovery ensemble
    if recovery_ensemble:
        S, I, R = _stack_ensemble(recovery_ensemble)
        arrays.update({"rec_S": S, "rec_I": I, "rec_R": R})

    # Experiment 2 — network divergence runs
    if network_runs:
        S, I, R = _stack_ensemble(network_runs)
        arrays.update({"net_S": S, "net_I": I, "net_R": R})

    # Experiment 3 — compliance sweep
    if sweep_values is not None and sweep_results is not None:
        arrays["sweep_values"] = np.array(sweep_values)
        arrays["sweep_peak_mean"] = np.array(sweep_results["peak_mean"])
        arrays["sweep_peak_std"] = np.array(sweep_results["peak_std"])

    # Experiment 4 — structured networks
    if structured_data:
        str_ode = structured_data.get("ode")
        if str_ode is not None:
            arrays.update({
                "str_ode_t": str_ode.t, "str_ode_S": str_ode.S,
                "str_ode_I": str_ode.I, "str_ode_R": str_ode.R,
            })

        hom_ens = structured_data.get("homogeneous_ensemble", [])
        if hom_ens:
            S, I, R = _stack_ensemble(hom_ens)
            arrays.update({"str_hom_S": S, "str_hom_I": I, "str_hom_R": R})

        lay_ens = structured_data.get("layered_ensemble", [])
        if lay_ens:
            S, I, R = _stack_ensemble(lay_ens)
            arrays.update({"str_lay_S": S, "str_lay_I": I, "str_lay_R": R})

        lay_res = structured_data.get("layered_results")
        if lay_res:
            _encode_results("str_lay", lay_res, arrays)

        hom_res = structured_data.get("homogeneous_results")
        if hom_res:
            _encode_results("str_hom_res", hom_res, arrays)

    path = Path(output_dir) / CACHE_FILE
    np.savez_compressed(path, **arrays)
    print(f"  Cache saved: {path}")
    return path


def load_cache(output_dir):
    """Load experiment outputs from a compressed .npz cache.

    Returns
    -------
    dict with keys:
        ode, recovery_ensemble, network_runs,
        sweep_values, sweep_results, structured_data,
        n_steps, dt, n_ensemble
    """
    path = Path(output_dir) / CACHE_FILE
    if not path.exists():
        raise FileNotFoundError(f"No cache found at {path}. Run without --plot-only first.")

    d = np.load(path, allow_pickle=False)

    result = {
        "n_steps": int(d["meta_n_steps"]),
        "dt": float(d["meta_dt"]),
        "n_ensemble": int(d["meta_n_ensemble"]),
        "ode": None,
        "recovery_ensemble": None,
        "network_runs": None,
        "sweep_values": None,
        "sweep_results": None,
        "structured_data": None,
    }

    if "ode_t" in d:
        result["ode"] = SIRSResult(t=d["ode_t"], S=d["ode_S"], I=d["ode_I"], R=d["ode_R"])

    if "rec_S" in d:
        result["recovery_ensemble"] = _unstack_ensemble(d["rec_S"], d["rec_I"], d["rec_R"])

    if "net_S" in d:
        result["network_runs"] = _unstack_ensemble(d["net_S"], d["net_I"], d["net_R"])

    if "sweep_values" in d:
        result["sweep_values"] = d["sweep_values"].tolist()
        result["sweep_results"] = {
            "peak_mean": d["sweep_peak_mean"].tolist(),
            "peak_std": d["sweep_peak_std"].tolist(),
        }

    if "str_ode_t" in d:
        str_ode = SIRSResult(
            t=d["str_ode_t"], S=d["str_ode_S"], I=d["str_ode_I"], R=d["str_ode_R"]
        )
        hom_ens = (_unstack_ensemble(d["str_hom_S"], d["str_hom_I"], d["str_hom_R"])
                   if "str_hom_S" in d else [])
        lay_ens = (_unstack_ensemble(d["str_lay_S"], d["str_lay_I"], d["str_lay_R"])
                   if "str_lay_S" in d else [])
        lay_res = _decode_results("str_lay", d) if "str_lay_conds" in d else {}
        hom_res = _decode_results("str_hom_res", d) if "str_hom_res_conds" in d else {}

        result["structured_data"] = {
            "ode": str_ode,
            "homogeneous_ensemble": hom_ens,
            "layered_ensemble": lay_ens,
            "layered_results": lay_res,
            "homogeneous_results": hom_res,
        }

    print(f"  Cache loaded: {path}")
    return result
