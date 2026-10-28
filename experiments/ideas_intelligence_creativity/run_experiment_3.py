#!/usr/bin/env python3
"""
Experiment 3: searching over number systems (SPEC.md sections 4 and 5).

Level 0 is a point, Level 1 searches the real line, and Level 2 changes the space. Every arm runs over
several seeds, then the algebra map, the real-line and complex-plane grids are computed. The results
go to number_systems_results.json, the three interactive figures and the two instruments are written beside it.

Usage:
    python run_experiment_3.py                              # full profile, one worker
    python run_experiment_3.py --profile quick --workers 4
    python run_experiment_3.py --profile quick --publish-plots-dir <a directory inside the repository>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from experiments.ideas_intelligence_creativity import experiment3 as ex3  # noqa: E402
from experiments.ideas_intelligence_creativity.cli_common import (  # noqa: E402
    check_publish_dir, parallel_map, prepare_output_dir, publish_plots, write_json_strict,
)
from experiments.ideas_intelligence_creativity.number_systems import (  # noqa: E402
    COEFFS, Algebra, complex_roots, p_real, real_critical_points,
)
from experiments.ideas_intelligence_creativity.plots_number_systems import write_number_system_plots  # noqa: E402
from experiments.ideas_intelligence_creativity.instruments import write_instruments  # noqa: E402

SCHEMA = "iic.number_systems/1"
JSON_NAME = "number_systems_results.json"

PROFILES = {
    "full": {"n_seeds": 30, "max_evals": 1_000_000, "alpha_n": 61, "beta_n": 81, "map_evals": 500},
    "quick": {"n_seeds": 3, "max_evals": 20_000, "alpha_n": 16, "beta_n": 21, "map_evals": 200},
}
ALPHA_RANGE = ex3.L2_ALPHA                    # the algebra box; the map covers all of it
BETA_RANGE = ex3.L2_BETA
MAP_CLIP = (-12.0, 0.5)                       # log10(best / m) in the algebra map
REAL_LINE = (-3.0, 4.0, 401)                  # |p| on the real line for the anchor figure
COMPLEX_RE = (-3.0, 4.0, 141)                 # log10 |p| over the complex plane
COMPLEX_IM = (-2.0, 2.0, 81)
GRID_CHECK = (-6.0, 6.0, 2_000_001)           # the dense grid that checks real_floor
FLOOR_SLACK = 1e-6                            # a floor success: best <= m (1 + FLOOR_SLACK)
ROOT_TOL = 1e-6                               # a root success: best < ROOT_TOL * m
FLOOR_ARMS = ex3.REAL_ARMS + ("split_two_reals",)
CONSTANTS = ("HC_SIGMA", "HC_RESTART_AFTER", "ES_SIGMA0", "ES_UP", "ES_DOWN", "ES_SIGMA_MIN", "ALG_BOUND",
             "PORTFOLIO_EPOCH", "PORTFOLIO_UCB_C", "L2_INIT", "L2_MU", "L2_LAMBDA", "L2_UNIFORM_PROB",
             "L2_SIGMA", "L2_CHILD_EVALS", "L2_TOPUP_EVALS")


def _arm_task(task):
    """One arm and seed, reduced to what the JSON keeps: checkpoints, adoption and the Level-2 records."""
    arm, max_evals, seed, root_seed = task
    m = ex3.M_FLOOR
    r = ex3.run_arm(arm, max_evals, seed, root_seed)
    out = {"arm": arm, "seed": seed,
           "best": [float(r.trace[c - 1]) for c in ex3.checkpoints(max_evals)]}
    if arm not in ex3.L2_ARMS:
        return out
    out["adoption_eval"] = r.adoption_eval
    out["first_complex_proposal_eval"] = r.first_complex_proposal_eval
    out["adoption_value_over_m"] = (float(r.trace[r.adoption_eval]) / m
                                    if r.adoption_eval is not None else None)
    out["adoption_algebra"] = list(r.adoption_algebra) if r.adoption_algebra is not None else None
    best = min(r.records, key=lambda rec: rec["best_rho"])
    out["final"] = {"alpha": best["alpha"], "beta": best["beta"], "delta": best["delta"],
                    "kind": best["kind"], "best_over_m": best["best_rho"] / m}
    out["below_floor_nonnegative_delta"] = sum(
        1 for rec in r.records
        if rec["kind"] != "complex" and rec["best_rho"] < m * (1.0 - ex3.ADOPTION_TOL))
    out["proposals"] = [[rec["alpha"], rec["beta"], rec["best_rho"] / m, rec["gen"]]
                        for rec in r.records if rec["proposal"]]
    out["elite_path"] = r.elite_path
    return out


def _map_task(task):
    alpha, beta, evals, cell, root_seed = task
    return ex3.algebra_cell_ratio(alpha, beta, evals, cell, root_seed)


def _summary(arm, best, m):
    """Per checkpoint: median, quartiles of best/m, and the fraction of seeds that succeed."""
    ratio = np.asarray(best, dtype=float) / m
    if arm in FLOOR_ARMS:
        criterion, ok = "floor", ratio <= 1.0 + FLOOR_SLACK
    else:
        criterion, ok = "root", ratio < ROOT_TOL
    return {"criterion": criterion,
            "median": np.median(ratio, axis=0).tolist(),
            "q1": np.percentile(ratio, 25, axis=0).tolist(),
            "q3": np.percentile(ratio, 75, axis=0).tolist(),
            "success": ok.mean(axis=0).tolist()}


def _level2_summary(runs):
    adopted = [o["adoption_eval"] for o in runs if o["adoption_eval"] is not None]
    firsts = [o["first_complex_proposal_eval"] for o in runs if o["first_complex_proposal_eval"] is not None]
    return {"adoption_rate": len(adopted) / len(runs),
            "median_adoption_eval": float(np.median(adopted)) if adopted else None,
            "median_first_complex_proposal_eval": float(np.median(firsts)) if firsts else None}


def _exemplar(l2_runs, m):
    """The l2_evo seed with the median adoption_eval among the adopting seeds (the lower median for an
    even count); ties go to the lowest seed. Its elite path and proposals go in the JSON."""
    adopting = sorted((o["adoption_eval"], o["seed"]) for o in l2_runs if o["adoption_eval"] is not None)
    if not adopting:
        return None
    adoption_eval, seed = adopting[(len(adopting) - 1) // 2]
    run = next(o for o in l2_runs if o["seed"] == seed)
    return {"arm": "l2_evo", "seed": seed,
            "rule": "median adoption_eval among adopting seeds (lower median); ties to the lowest seed",
            "adoption_eval": adoption_eval,
            "adoption_value_over_m": run["adoption_value_over_m"],
            "adoption_algebra": run["adoption_algebra"],
            "adoption_kind": Algebra(*run["adoption_algebra"]).kind,
            "elite_path": run["elite_path"],
            "proposals": run["proposals"]}


def _real_line_block(m):
    xs = np.linspace(REAL_LINE[0], REAL_LINE[1], REAL_LINE[2])
    minima = []
    h = 1e-4
    for x in real_critical_points():
        v = abs(float(p_real(x)))
        if v < abs(float(p_real(x - h))) and v < abs(float(p_real(x + h))):
            minima.append({"x": x, "abs_p": v})
    return {"x": xs.tolist(), "abs_p": np.round(np.abs(p_real(xs)), 6).tolist(), "minima": minima,
            "m": m}


def _complex_plane_block():
    re = np.linspace(COMPLEX_RE[0], COMPLEX_RE[1], COMPLEX_RE[2])
    im = np.linspace(COMPLEX_IM[0], COMPLEX_IM[1], COMPLEX_IM[2])
    X, Y = np.meshgrid(re, im)                 # rows follow im, columns follow re
    z = np.polyval(COEFFS, X + 1j * Y)
    log_abs = np.log10(np.maximum(np.abs(z), 1e-300))
    return {"re": re.tolist(), "im": im.tolist(), "log10_abs_p": np.round(log_abs, 4).tolist()}


def _grid_check(m):
    lo, hi, n = GRID_CHECK
    grid = np.linspace(lo, hi, n)
    vals = np.abs(p_real(grid))
    k = int(np.argmin(vals))
    return {"interval": [lo, hi], "points": int(n), "min_abs_p": float(vals[k]), "x": float(grid[k]),
            "relative_excess": float((vals[k] - m) / m)}


def build_results(profile: str, root_seed: int = ex3.DEFAULT_ROOT_SEED, workers: int = 1,
                  n_seeds: int | None = None) -> dict:
    """Every number in number_systems_results.json (SPEC.md section 5)."""
    prof = dict(PROFILES[profile])
    if n_seeds is not None:
        prof["n_seeds"] = int(n_seeds)
    n, max_evals = prof["n_seeds"], prof["max_evals"]
    x_star, m = ex3.X_STAR, ex3.M_FLOOR
    cps = ex3.checkpoints(max_evals)

    tasks = [(arm, max_evals, seed, root_seed) for arm in ex3.ALL_ARMS for seed in range(n)]
    results = parallel_map(_arm_task, tasks, workers)
    by_arm = {arm: [] for arm in ex3.ALL_ARMS}
    for out in results:
        by_arm[out["arm"]].append(out)
    for arm in by_arm:
        by_arm[arm].sort(key=lambda o: o["seed"])

    alphas = np.linspace(ALPHA_RANGE[0], ALPHA_RANGE[1], prof["alpha_n"])
    betas = np.linspace(BETA_RANGE[0], BETA_RANGE[1], prof["beta_n"])
    cells = [(float(a), float(b)) for a in alphas for b in betas]      # row-major: alpha, then beta
    map_tasks = [(a, b, prof["map_evals"], k, root_seed) for k, (a, b) in enumerate(cells)]
    ratios = np.asarray(parallel_map(_map_task, map_tasks, workers), dtype=float)
    log_ratio = np.clip(np.log10(np.maximum(ratios, 1e-300)), MAP_CLIP[0], MAP_CLIP[1])
    log_ratio = log_ratio.reshape(prof["alpha_n"], prof["beta_n"])

    arms = {}
    summary = {}
    for arm in ex3.ALL_ARMS:
        best = [o["best"] for o in by_arm[arm]]
        kind = "level2" if arm in ex3.L2_ARMS else ("real" if arm in ex3.REAL_ARMS else "control")
        arms[arm] = {"kind": kind, "best": best}
        summary[arm] = _summary(arm, best, m)
        if arm in ex3.L2_ARMS:
            summary[arm].update(_level2_summary(by_arm[arm]))

    level2 = {}
    for arm in ex3.L2_ARMS:
        level2[arm] = {"seeds": [
            {"seed": o["seed"], "adoption_eval": o["adoption_eval"],
             "first_complex_proposal_eval": o["first_complex_proposal_eval"],
             "final_best": o["final"]} for o in by_arm[arm]]}
    level2["exemplar"] = _exemplar(by_arm["l2_evo"], m)

    below = sum(o["below_floor_nonnegative_delta"] for arm in ex3.L2_ARMS for o in by_arm[arm])
    min_real = min(float(np.min(arms[arm]["best"])) for arm in FLOOR_ARMS) / m

    return {
        "schema": SCHEMA,
        "params": {
            "profile": profile, "n_seeds": n, "max_evals": max_evals, "root_seed": root_seed,
            "checkpoints_per_decade": ex3.CHECKPOINTS_PER_DECADE,
            "algebra_map": {"alpha_points": prof["alpha_n"], "beta_points": prof["beta_n"],
                            "evals_per_cell": prof["map_evals"]},
            "constants": {name: getattr(ex3, name) for name in CONSTANTS},
        },
        "polynomial": {
            "coeffs": list(COEFFS),
            "roots": [[float(z.real), float(z.imag)] for z in complex_roots()],
            "floor": {
                "x_star": x_star, "m": m,
                "method": "real critical points of p (numpy roots of p', Newton-polished); smallest |p| among them",
                "grid_check": _grid_check(m),
            },
        },
        "checkpoints": cps,
        "arms": arms,
        "level2": level2,
        "algebra_map": {"alpha": alphas.tolist(), "beta": betas.tolist(),
                        "clip": list(MAP_CLIP),
                        "log10_ratio": np.round(log_ratio, 4).tolist()},
        "real_line": _real_line_block(m),
        "complex_plane": _complex_plane_block(),
        "summary": summary,
        "invariants": {"min_real_best_over_m": min_real, "below_floor_from_nonnegative_delta": int(below)},
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Experiment 3: searching over number systems.")
    p.add_argument("--output-dir", default=None, help="default: experiments/ideas_intelligence_creativity/results")
    p.add_argument("--profile", choices=sorted(PROFILES), default="full")
    p.add_argument("--n-seeds", type=int, default=None, help="seeds per arm (default: the profile's)")
    p.add_argument("--root-seed", type=int, default=ex3.DEFAULT_ROOT_SEED,
                   help="shifts every random stream (default 0)")
    p.add_argument("--workers", type=int, default=1, help="processes (the output does not depend on it)")
    p.add_argument("--no-plots", action="store_true", help="write only the JSON")
    p.add_argument("--publish-plots-dir", default=None,
                   help="also copy the plot HTML here; must resolve inside the repository")
    args = p.parse_args(argv)
    if args.n_seeds is not None and args.n_seeds < 1:
        p.error("--n-seeds must be at least 1")
    if args.workers < 1:
        p.error("--workers must be at least 1")
    if args.root_seed < 0:
        p.error("--root-seed must be non-negative")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    publish = check_publish_dir(args.publish_plots_dir)     # refused before anything is created or computed
    if publish is not None and args.no_plots:
        print("error: --publish-plots-dir needs the plots; drop --no-plots", file=sys.stderr)
        return 2
    out_dir = prepare_output_dir(args.output_dir)
    doc = build_results(args.profile, args.root_seed, args.workers, args.n_seeds)
    write_json_strict(doc, out_dir / JSON_NAME)
    if not args.no_plots:
        plots_dir = out_dir / "plots"
        write_number_system_plots(doc, plots_dir)
        write_instruments(doc, plots_dir)
        if publish is not None:
            publish_plots(plots_dir, publish)
    return 0


if __name__ == "__main__":
    sys.exit(main())
