"""SPEC.md 10.3: the data for the two instruments, and the pages that carry it.

Every number comes from the recorded results document or from the constants the experiment uses, so a
changed result changes the page. The algebra map codes follow plots_number_systems' rule for the figure.
"""

import math
from pathlib import Path

from experiments.ideas_intelligence_creativity import experiment3 as ex3
from experiments.ideas_intelligence_creativity.figkit.build import build_instrument
from experiments.ideas_intelligence_creativity.number_systems import DUAL_TOL
from experiments.ideas_intelligence_creativity.plots_number_systems import FLOOR_TOL, ROOT_TOL

LINE_PAGE = "instrument_line.html"
ALGEBRAS_PAGE = "instrument_algebras.html"
VIEW = {"re": [-3.0, 4.0], "im": [-2.0, 2.0]}
WALKERS = 128
LINE_SEED = 20261026                          # the searchers' first positions: the same on every visit
ALGEBRAS_SEED = 20261027
EXEMPLAR_KEYS = ("proposals", "elite_path", "adoption_algebra", "adoption_eval")

LINE_TITLE = "The answer is off the line"
LINE_ARIA = "Searchers on the real line and in the plane, with the lowest point on the line and the four roots"
LINE_CAPTION = (
    "Live: the searchers above run the experiment's own step rule in your browser. On the line they pile up "
    "at the lowest point, and once a second direction is allowed they flow to the four roots. "
    "Recorded: the essay's numbers come from 30 recorded runs of the same search."
)
ALGEBRAS_TITLE = "Choose a number system"
ALGEBRAS_ARIA = "Map of number systems, with a probe to choose one and a live search inside it"
ALGEBRAS_CAPTION = (
    "Live: the search inside the number system you choose runs here, in your browser, with the experiment's "
    "own rule, and it restarts as you move the probe. Recorded: the background map and the replayed run come "
    "from the 30 recorded runs behind the essay."
)


def cell_code(log10_ratio: float) -> int:
    """0 stayed at the floor, 1 below the floor with no root, 2 a root: plots_number_systems' rule."""
    if log10_ratio < math.log10(ROOT_TOL):
        return 2
    if log10_ratio < -FLOOR_TOL:
        return 1
    return 0


def line_data(results: dict) -> dict:
    poly = results["polynomial"]
    return {
        "coeffs": list(poly["coeffs"]),
        "m": poly["floor"]["m"],
        "x_star": poly["floor"]["x_star"],
        "minima": [[mm["x"], mm["abs_p"]] for mm in results["real_line"]["minima"]],
        "roots": [list(r) for r in poly["roots"]],
        "sigma": ex3.HC_SIGMA,
        "view": {"re": list(VIEW["re"]), "im": list(VIEW["im"])},
        "start": [max(ex3.START_LO, VIEW["re"][0]), min(ex3.START_HI, VIEW["re"][1])],
        "walkers": WALKERS,
        "seed": LINE_SEED,
    }


def guess_schedule(results: dict) -> dict:
    """The guesses (evaluations of p) that Level 2 spends, as experiment3's loop spends them: the initial population,
    then per generation its children and the elites' top-ups. ``cap`` is the run's budget."""
    return {
        "first": ex3.L2_INIT * ex3.L2_CHILD_EVALS,
        "per_round": ex3.L2_LAMBDA * ex3.L2_CHILD_EVALS + ex3.L2_MU * ex3.L2_TOPUP_EVALS,
        "cap": int(results["params"]["max_evals"]),
    }


def algebras_data(results: dict) -> dict:
    amap = results["algebra_map"]
    consts = results["params"]["constants"]
    exemplar = results["level2"]["exemplar"]
    return {
        "alpha": list(amap["alpha"]),
        "beta": list(amap["beta"]),
        "cat": [[cell_code(v) for v in row] for row in amap["log10_ratio"]],
        "coeffs": list(results["polynomial"]["coeffs"]),
        "m": results["polynomial"]["floor"]["m"],
        "box": {"alpha": list(ex3.L2_ALPHA), "beta": list(ex3.L2_BETA)},
        "dual_tol": DUAL_TOL,
        "es": {
            "ES_SIGMA0": consts["ES_SIGMA0"],
            "ES_UP": consts["ES_UP"],
            "ES_DOWN": consts["ES_DOWN"],
            "ES_SIGMA_MIN": consts["ES_SIGMA_MIN"],
            "child_evals": ex3.L2_CHILD_EVALS,
        },
        "start": [ex3.START_LO, ex3.START_HI],
        "bound": ex3.ALG_BOUND,
        "exemplar": None if exemplar is None else {k: exemplar[k] for k in EXEMPLAR_KEYS},
        "schedule": guess_schedule(results),
        "seed": ALGEBRAS_SEED,
    }


def write_instruments(results: dict, plots_dir) -> list:
    """Write both instrument pages into plots_dir, from the results document alone."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return [
        build_instrument("line", line_data(results), plots_dir / LINE_PAGE,
                         title=LINE_TITLE, aria_label=LINE_ARIA, caption=LINE_CAPTION),
        build_instrument("algebras", algebras_data(results), plots_dir / ALGEBRAS_PAGE,
                         title=ALGEBRAS_TITLE, aria_label=ALGEBRAS_ARIA, caption=ALGEBRAS_CAPTION),
    ]
