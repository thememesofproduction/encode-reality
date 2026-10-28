#!/usr/bin/env python3
"""
Experiment 1 (repaired): Intelligence vs Creativity on a peaks landscape (SPEC.md section 6).

Four arms search the same landscape from the same start point: a fixed Gaussian hill-climber
(hc_fixed), a fixed restart strategy (restart_fixed), the meta-level with random strategy selection
(swap_random) and the meta-level with UCB selection (meta_ucb). Every arm spends exactly --max-evals
evaluations, through one search_core.Budget.

Outputs: DIR/landscape_results.json, and unless --no-plots, DIR/plots/*.html. No matplotlib, no PNGs.

Usage:
    python run_experiment_1.py                          # full profile: 40 landscapes x 5 starts
    python run_experiment_1.py --profile quick
    python run_experiment_1.py --workers 8 --publish-plots-dir <a directory inside the repository>
"""

import sys
from pathlib import Path
import argparse
import dataclasses

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy.stats import binomtest

from experiments.ideas_intelligence_creativity.cli_common import (
    check_publish_dir, parallel_map, prepare_output_dir, publish_plots, write_json_strict,
)
from experiments.ideas_intelligence_creativity.config import (
    EXPERIMENT_1_ARMS, EXPERIMENT_1_BOOTSTRAP, EXPERIMENT_1_CONDITIONS, EXPERIMENT_1_DEFAULTS,
    EXPERIMENT_1_PROFILES, EXPERIMENT_1_SUCCESS, FIXED_STRATEGY_DEFAULTS, LANDSCAPE_DEFAULTS,
    META_OPTIMIZER_DEFAULTS,
)
from experiments.ideas_intelligence_creativity.creativity import UCB_C, MetaStrategyOptimizer
from experiments.ideas_intelligence_creativity.intelligence import FixedStrategyOptimizer
from experiments.ideas_intelligence_creativity.landscape import CEILING_GRID_AXIS, FitnessLandscape
from experiments.ideas_intelligence_creativity.plots_landscape import write_landscape_plots
from experiments.ideas_intelligence_creativity.seeding import (
    EXP1, ROLE_LANDSCAPE, ROLE_META, ROLE_SEARCH, ROLE_START, rng_for,
)
from experiments.ideas_intelligence_creativity.strategies import SearchStrategy, StrategyMutator

SCHEMA = "iic.landscape/1"
JSON_NAME = "landscape_results.json"
CHECKPOINTS_PER_DECADE = 10
CONFIDENCE = 0.95
START_CLEARANCE = 1e-6          # a start point must be farther than this from every peak centre
META_SELECTION = {"swap_random": "random", "meta_ucb": "ucb"}


def checkpoints_for(max_evals: int) -> list[int]:
    """About 10 per decade, log-spaced from 1 to max_evals: unique integers that include max_evals."""
    n = max(2, round(CHECKPOINTS_PER_DECADE * np.log10(max_evals)) + 1)
    points = np.unique(np.round(np.logspace(0.0, np.log10(max_evals), n)).astype(int))
    points = np.unique(np.append(points[points < max_evals], max_evals))
    return [int(p) for p in points]


def draw_start(rng, land: FitnessLandscape, dims: int, lo: float, hi: float) -> np.ndarray:
    """A uniform start in the bounds, redrawn while it sits on a peak centre (SPEC section 6)."""
    centres = np.array([p.center for p in land.peaks], dtype=float)
    while True:
        x = rng.uniform(lo, hi, size=dims)
        if np.min(np.linalg.norm(centres - x, axis=1)) > START_CLEARANCE:
            return x


def make_arm(arm: str, land: FitnessLandscape, x0: np.ndarray, search_rng, meta_rng):
    """One arm on the shared landscape and start. The fixed arms use the same step size as the meta-level."""
    base = SearchStrategy(**FIXED_STRATEGY_DEFAULTS)
    if arm == "hc_fixed":
        return FixedStrategyOptimizer(land, base, seed=search_rng, x0=x0)
    if arm == "restart_fixed":
        return FixedStrategyOptimizer(land, dataclasses.replace(base, method="restart"), seed=search_rng, x0=x0)
    if arm not in META_SELECTION:
        raise ValueError(f"unknown arm {arm!r}")
    mutator = StrategyMutator(mutation_rate=META_OPTIMIZER_DEFAULTS["mutation_rate"], seed=meta_rng)
    return MetaStrategyOptimizer(
        land, base, mutator,
        meta_interval=META_OPTIMIZER_DEFAULTS["meta_interval"],
        strategy_pool_size=META_OPTIMIZER_DEFAULTS["strategy_pool_size"],
        improvement_window=META_OPTIMIZER_DEFAULTS["improvement_window"],
        novel_strategy_prob=META_OPTIMIZER_DEFAULTS["novel_strategy_prob"],
        seed=search_rng, x0=x0,
        replace_every=META_OPTIMIZER_DEFAULTS["replace_every"],
        selection=META_SELECTION[arm],
        discount=META_OPTIMIZER_DEFAULTS["discount"],
    )


def run_landscape(task: tuple) -> dict:
    """Every arm on every start of one landscape (one condition, one landscape index).

    The result depends only on the task, so it is the same whichever worker runs it. Each start is
    shared by the four arms, and each run's best-so-far is taken at the checkpoints, divided by the ceiling.
    """
    cond_id, land_idx, cfg = task
    root, dims = cfg["root_seed"], cfg["dims"]
    lo, hi = cfg["bounds"]
    land = FitnessLandscape(n_peaks=cfg["n_peaks"], dims=dims, bounds=(lo, hi),
                            global_peak=EXPERIMENT_1_CONDITIONS[cond_id],
                            rng=rng_for(root, EXP1, ROLE_LANDSCAPE, cond_id, land_idx))
    _, ceiling = land.ceiling()
    checkpoints = np.asarray(cfg["checkpoints"]) - 1
    runs = {arm: [] for arm in EXPERIMENT_1_ARMS}
    for start_idx in range(cfg["n_starts"]):
        x0 = draw_start(rng_for(root, EXP1, ROLE_START, cond_id, land_idx, start_idx), land, dims, lo, hi)
        for arm_id, arm in enumerate(EXPERIMENT_1_ARMS):
            search_rng = rng_for(root, EXP1, ROLE_SEARCH, arm_id, cond_id, land_idx, start_idx)
            meta_rng = rng_for(root, EXP1, ROLE_META, arm_id, cond_id, land_idx, start_idx)
            optimiser = make_arm(arm, land, x0, search_rng, meta_rng)
            optimiser.run(max_evals=cfg["max_evals"])
            best = np.asarray(optimiser.trace, dtype=float)[checkpoints]
            runs[arm].append((best / ceiling).tolist())
    return {"ceiling": float(ceiling), "runs": runs}


def mcnemar_p(arm_ok: np.ndarray, ucb_ok: np.ndarray) -> float:
    """Exact two-sided McNemar test on the paired runs. 1.0 when no run is discordant."""
    arm_only = int(np.sum(arm_ok & ~ucb_ok))
    ucb_only = int(np.sum(~arm_ok & ucb_ok))
    n = arm_only + ucb_only
    if n == 0:
        return 1.0
    return float(binomtest(min(arm_only, ucb_only), n, 0.5, alternative="two-sided").pvalue)


def summarise(cond_id: int, block: dict, n_landscapes: int, n_starts: int, root_seed: int) -> dict:
    """Final success rate per arm, its 95% cluster bootstrap CI over landscapes, and McNemar p against meta_ucb."""
    ok = {arm: np.asarray([run[-1] for run in block[arm]["normalised_best"]]) >= EXPERIMENT_1_SUCCESS
          for arm in EXPERIMENT_1_ARMS}
    summary = {}
    for arm_id, arm in enumerate(EXPERIMENT_1_ARMS):
        per_landscape = ok[arm].reshape(n_landscapes, n_starts).mean(axis=1)
        rng = rng_for(root_seed, EXP1, ROLE_META, cond_id, arm_id)
        picks = rng.integers(0, n_landscapes, size=(EXPERIMENT_1_BOOTSTRAP, n_landscapes))
        boot = per_landscape[picks].mean(axis=1)
        tail = 100 * (1 - CONFIDENCE) / 2
        lo, hi = np.percentile(boot, [tail, 100 - tail])
        rate = float(ok[arm].mean())
        summary[arm] = {"success_rate": rate, "ci": [float(min(lo, rate)), float(max(hi, rate))]}
    summary["mcnemar_p"] = {arm: mcnemar_p(ok[arm], ok["meta_ucb"])
                            for arm in EXPERIMENT_1_ARMS if arm != "meta_ucb"}
    return summary


def build_results(cfg: dict, cond_ids: list[int], per_task: list[dict]) -> dict:
    """The JSON document (SPEC section 6 and clarification 9.10). Runs are landscape-major, start-minor."""
    n_l = cfg["n_landscapes"]
    conditions, summary = {}, {}
    for k, cond_id in enumerate(cond_ids):
        chunk = per_task[k * n_l:(k + 1) * n_l]
        cond = EXPERIMENT_1_CONDITIONS[cond_id]
        block = {"ceilings": [t["ceiling"] for t in chunk]}
        for arm in EXPERIMENT_1_ARMS:
            block[arm] = {"normalised_best": [run for t in chunk for run in t["runs"][arm]]}
        conditions[cond] = block
        summary[cond] = summarise(cond_id, block, n_l, cfg["n_starts"], cfg["root_seed"])
    params = {
        "profile": cfg["profile"],
        "root_seed": cfg["root_seed"],
        "n_landscapes": n_l,
        "n_starts": cfg["n_starts"],
        "max_evals": cfg["max_evals"],
        "conditions": [EXPERIMENT_1_CONDITIONS[c] for c in cond_ids],
        "arms": list(EXPERIMENT_1_ARMS),
        "landscape": {"n_peaks": cfg["n_peaks"], "dims": cfg["dims"], "bounds": list(cfg["bounds"]),
                      "ceiling_grid_axis": CEILING_GRID_AXIS},
        "hc_fixed_strategy": dict(FIXED_STRATEGY_DEFAULTS),
        "meta": dict(META_OPTIMIZER_DEFAULTS, ucb_c=UCB_C,
                     epoch_evals=META_OPTIMIZER_DEFAULTS["meta_interval"] * FIXED_STRATEGY_DEFAULTS["n_candidates"]),
        "meta_selection": dict(META_SELECTION),
        "success_fraction": EXPERIMENT_1_SUCCESS,
        "bootstrap_resamples": EXPERIMENT_1_BOOTSTRAP,
        "ci_level": CONFIDENCE,
    }
    return {"schema": SCHEMA, "params": params, "checkpoints": cfg["checkpoints"],
            "conditions": conditions, "summary": summary}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Experiment 1 (repaired): intelligence vs creativity on a peaks landscape")
    parser.add_argument("--output-dir", default=None,
                        help="where to write the results (default: <this directory>/results)")
    parser.add_argument("--profile", choices=sorted(EXPERIMENT_1_PROFILES), default=EXPERIMENT_1_DEFAULTS["profile"],
                        help="sizes of the run (default: full)")
    parser.add_argument("--n-landscapes", type=int, default=None, help="landscapes per condition")
    parser.add_argument("--n-starts", type=int, default=None, help="start points per landscape")
    parser.add_argument("--max-evals", type=int, default=None, help="evaluations per run")
    parser.add_argument("--root-seed", type=int, default=EXPERIMENT_1_DEFAULTS["root_seed"], help="root seed")
    parser.add_argument("--workers", type=int, default=EXPERIMENT_1_DEFAULTS["workers"],
                        help="processes (the output does not depend on this)")
    parser.add_argument("--conditions", default=",".join(EXPERIMENT_1_CONDITIONS),
                        help="comma-separated subset of: " + ",".join(EXPERIMENT_1_CONDITIONS))
    parser.add_argument("--no-plots", action="store_true", help="write the JSON only")
    parser.add_argument("--publish-plots-dir", default=None,
                        help="also copy the plot pages here; must resolve inside the repository")
    args = parser.parse_args(argv)

    sizes = EXPERIMENT_1_PROFILES[args.profile]
    for name in ("n_landscapes", "n_starts", "max_evals"):
        value = getattr(args, name)
        if value is None:
            setattr(args, name, sizes[name])
        elif value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1, not {value}")
    if args.workers < 1:
        parser.error(f"--workers must be at least 1, not {args.workers}")
    requested = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if not requested or any(c not in EXPERIMENT_1_CONDITIONS for c in requested):
        parser.error("--conditions must be a comma-separated subset of " + ",".join(EXPERIMENT_1_CONDITIONS))
    args.cond_ids = [i for i, c in enumerate(EXPERIMENT_1_CONDITIONS) if c in requested]
    if args.publish_plots_dir is not None and args.no_plots:
        parser.error("--publish-plots-dir copies the plots, so it cannot be combined with --no-plots")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Refused before any directory is created or anything is computed (SPEC section 9.7).
    publish_dir = check_publish_dir(args.publish_plots_dir)
    out_dir = prepare_output_dir(args.output_dir)

    cfg = {
        "profile": args.profile,
        "root_seed": args.root_seed,
        "n_landscapes": args.n_landscapes,
        "n_starts": args.n_starts,
        "max_evals": args.max_evals,
        "checkpoints": checkpoints_for(args.max_evals),
        "n_peaks": LANDSCAPE_DEFAULTS["n_peaks"],
        "dims": LANDSCAPE_DEFAULTS["dims"],
        "bounds": LANDSCAPE_DEFAULTS["bounds"],
    }
    tasks = [(cond_id, land_idx, cfg) for cond_id in args.cond_ids for land_idx in range(args.n_landscapes)]
    per_task = parallel_map(run_landscape, tasks, args.workers)
    results = build_results(cfg, args.cond_ids, per_task)

    json_path = out_dir / JSON_NAME
    write_json_strict(results, json_path)
    print(f"wrote {json_path}")
    if not args.no_plots:
        plots_dir = out_dir / "plots"
        for path in write_landscape_plots(results, plots_dir):
            print(f"wrote {path}")
        if publish_dir is not None:
            for path in publish_plots(plots_dir, publish_dir):
                print(f"published {path}")


if __name__ == "__main__":
    main()
