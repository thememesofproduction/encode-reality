"""
Interactive figures for Experiment 1 (SPEC.md sections 6 and 7).

``write_landscape_plots(results, plots_dir)`` writes two self-contained pages from the dict that
run_experiment_1.py writes as landscape_results.json:

- ``landscape_anytime.html``: the best value so far divided by the highest point, against evaluations
  (the median line for each arm);
- ``landscape_success.html``: the share of runs within 1% of the highest point, against evaluations.

Every arm is blue, since each searches inside a fixed space; the arms differ by dash style, and each line
ends in a direct label. The highest point of each landscape is a neutral dashed line. A dropdown switches
between the conditions, and the end labels switch with the lines they belong to.
"""

import math
from pathlib import Path

import numpy as np

from .config import EXPERIMENT_1_ARMS, EXPERIMENT_1_CONDITIONS, EXPERIMENT_1_SUCCESS
from .plots_interactive import BLUE, FG, GRID, MUTED, PLOT_BG, base_layout, note, write_page

ARM_LABELS = {                    # the names the essay uses, in hover text and on the lines
    "hc_fixed": "fixed small step",
    "restart_fixed": "restarts",
    "swap_random": "switches at random",
    "meta_ucb": "learns which strategy",
}
ARM_END = {                       # the direct label at the end of each line
    "hc_fixed": "fixed small step",
    "restart_fixed": "restarts",
    "swap_random": "switches at random",
    "meta_ucb": "learns which strategy",
}
ARM_DASH = {"hc_fixed": "solid", "restart_fixed": "dash", "swap_random": "dot", "meta_ucb": "dashdot"}
ARM_WIDTH = {"hc_fixed": 1.6, "restart_fixed": 2.5, "swap_random": 2.5, "meta_ucb": 2.5}
CONDITION_LABELS = {
    "none": "random peaks",
    "random": "+ one tall random peak",
    "corner": "+ one peak hidden in a corner (the original setup)",
}
ANYTIME_TITLE = "Better search reaches the ceiling sooner, never above it"
ANYTIME_LAYOUT_TITLE = "Better search reaches the ceiling<br>sooner, never above it"
SUCCESS_TITLE = "How often each strategy finds the highest point"
SUCCESS_LAYOUT_TITLE = "How often each strategy<br>finds the highest point"
TOP_MARGIN = 124     # room for the two-line title and, under it, the dropdown
RIGHT_MARGIN = 118   # room for the widest direct label ("fixed restarts 100%") at the end of each line
LABEL_STEP = 14      # end labels are kept this many pixels apart
PLOT_PX = 330.0     # about how many pixels one unit of the value axis takes on these pages
TWO_LINE_STEP = 26   # the same for the two-line end labels of landscape_success (a name, then a percentage)
CEILING_LABEL = "highest point of the<br>landscape (computed)"


def _conditions(results: dict) -> list[str]:
    return [c for c in EXPERIMENT_1_CONDITIONS if c in results["conditions"]]


def _arm_array(results: dict, cond: str, arm: str) -> np.ndarray:
    """Runs x checkpoints array of normalised best values."""
    return np.asarray(results["conditions"][cond][arm]["normalised_best"], dtype=float)


def _stack_shifts(finals: list[float], step: int, top: float) -> list[float]:
    """Pixel shifts that keep the end labels at least ``step`` pixels apart, top to bottom, and inside the plot
    (its top is the axis value ``top``): a label whose line ends too near the one above it is pushed down below
    it. A shift is a plotly yshift (positive moves a label up)."""
    order = sorted(range(len(finals)), key=lambda i: -finals[i])
    shifts = [0.0] * len(finals)
    prev = -top * PLOT_PX - step / 2                # the first label may sit half a label below the top
    for i in order:
        y = -finals[i] * PLOT_PX                  # pixels below the axis origin, for a value on the axis
        placed = max(y, prev + step)
        shifts[i] = y - placed
        prev = placed
    return shifts


def _end_labels(x_end: float, finals: list[float], texts: list[str], step: int, top: float) -> list[dict]:
    """One direct label per arm, at the end of its line (annotations, so the dropdown can move them)."""
    shifts = _stack_shifts(finals, step, top)
    return [note(text, x_end, value, ax=0, ay=0, log_x=True, xanchor="left", yanchor="middle", yshift=shift,
                 xshift=8, font={"size": 11, "color": FG})
            for text, value, shift in zip(texts, finals, shifts)]


def _dropdown(conds: list[str], groups: list[str], annotations: dict) -> list[dict]:
    """One button per condition; each shows the traces of that condition only, and its end labels."""
    buttons = [{"label": CONDITION_LABELS[c], "method": "update",
                "args": [{"visible": [g == c for g in groups]}, {"annotations": annotations[c]}]} for c in conds]
    return [{"type": "dropdown", "direction": "down", "x": 0, "xanchor": "left", "y": 1.0, "yanchor": "bottom",
             "showactive": True, "active": 0, "buttons": buttons, "bgcolor": PLOT_BG, "bordercolor": GRID,
             "font": {"color": FG, "size": 12}}]


def _style_axes(layout: dict, x: list[int], x_title: str, y_title: str, y_range: list[float]):
    layout["xaxis"].update(type="log", range=[0.0, math.log10(x[-1])], dtick=2, exponentformat="power",
                           title={"text": x_title, "font": {"color": MUTED}})
    layout["yaxis"].update(title={"text": y_title, "font": {"color": MUTED}}, range=y_range)


def _anytime_figure(results: dict, conds: list[str]) -> dict:
    x = list(results["checkpoints"])
    traces, groups, labels = [], [], {}
    for cond in conds:
        finals, texts = [], []
        for arm in EXPERIMENT_1_ARMS:
            nb = _arm_array(results, cond, arm)
            q25, median, q75 = np.percentile(nb, [25, 50, 75], axis=0)
            finals.append(float(median[-1]))
            texts.append(ARM_END[arm])
            traces.append({"type": "scatter", "mode": "lines", "x": x, "y": median.tolist(),
                           "name": ARM_LABELS[arm],
                           "customdata": [[a, b] for a, b in zip(q25.tolist(), q75.tolist())],
                           "line": {"color": BLUE, "width": ARM_WIDTH[arm], "dash": ARM_DASH[arm]},
                           "hovertemplate": (f"{ARM_LABELS[arm]}<br>evaluations: %{{x:,.0f}}"
                                             "<br>median best ÷ highest: %{y:.3f}"
                                             "<br>quartiles: %{customdata[0]:.3f} to %{customdata[1]:.3f}"
                                             "<extra></extra>")})
            groups.append(cond)
        labels[cond] = _end_labels(x[-1], finals, texts, LABEL_STEP, 1.14)
        traces.append({"type": "scatter", "mode": "lines+text", "x": [x[0], x[-1]], "y": [1.0, 1.0],
                       "line": {"color": MUTED, "width": 1.6, "dash": "dash"}, "hoverinfo": "skip",
                       "showlegend": False, "name": "highest point (computed)",
                       "text": ["", CEILING_LABEL], "textposition": "top left",
                       "textfont": {"color": MUTED, "size": 11}, "cliponaxis": False})
        groups.append(cond)

    layout = base_layout(title=ANYTIME_LAYOUT_TITLE, height=520, showlegend=False,
                         margin={"l": 46, "r": RIGHT_MARGIN, "t": TOP_MARGIN, "b": 60})
    _style_axes(layout, x, "evaluations (log scale)", "best ÷ highest point", [0, 1.14])
    layout["updatemenus"] = _dropdown(conds, groups, labels)
    layout["annotations"] = labels[conds[0]]
    visible = [g == conds[0] for g in groups]
    for trace, show in zip(traces, visible):
        trace["visible"] = show
    return {"data": traces, "layout": layout}


def _success_figure(results: dict, conds: list[str]) -> dict:
    x = list(results["checkpoints"])
    traces, groups, labels = [], [], {}
    for cond in conds:
        shares = [np.mean(_arm_array(results, cond, arm) >= EXPERIMENT_1_SUCCESS, axis=0)
                  for arm in EXPERIMENT_1_ARMS]
        finals = [float(share[-1]) for share in shares]
        texts = [f"{ARM_END[arm]}<br>{round(100 * final)}%" for arm, final in zip(EXPERIMENT_1_ARMS, finals)]
        labels[cond] = _end_labels(x[-1], finals, texts, TWO_LINE_STEP, 1.02)
        for arm, share in zip(EXPERIMENT_1_ARMS, shares):
            traces.append({"type": "scatter", "mode": "lines", "x": x, "y": share.tolist(),
                           "name": ARM_LABELS[arm],
                           "line": {"color": BLUE, "width": ARM_WIDTH[arm], "dash": ARM_DASH[arm]},
                           "hovertemplate": (f"{ARM_LABELS[arm]}<br>evaluations: %{{x:,.0f}}"
                                             "<br>runs within 1%: %{y:.0%}<extra></extra>")})
            groups.append(cond)

    layout = base_layout(title=SUCCESS_LAYOUT_TITLE, height=520, showlegend=False,
                         margin={"l": 46, "r": RIGHT_MARGIN, "t": TOP_MARGIN, "b": 60})
    _style_axes(layout, x, "guesses checked (log scale)", "share of runs within 1% of the highest point", [0, 1.02])
    layout["yaxis"].update(tickformat=".0%")
    layout["updatemenus"] = _dropdown(conds, groups, labels)
    layout["annotations"] = labels[conds[0]]
    visible = [g == conds[0] for g in groups]
    for trace, show in zip(traces, visible):
        trace["visible"] = show
    return {"data": traces, "layout": layout}


def write_landscape_plots(results: dict, plots_dir) -> list[Path]:
    """Write landscape_anytime.html and landscape_success.html into ``plots_dir`` (created if missing)."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    conds = _conditions(results)
    n_runs = len(results["conditions"][conds[0]][EXPERIMENT_1_ARMS[0]]["normalised_best"])
    n_starts = results["params"]["n_starts"]
    n_landscapes = results["params"]["n_landscapes"]
    runs = (f"{n_runs} runs for each kind of landscape: {n_landscapes} landscapes, each searched from "
            f"{n_starts} starting points")
    return [
        write_page(_anytime_figure(results, conds), plots_dir / "landscape_anytime.html",
                   title=ANYTIME_TITLE,
                   caption=(f"Median over {runs}; hover for the quartiles. A value of 1 is the highest point "
                            "of the landscape.")),
        write_page(_success_figure(results, conds), plots_dir / "landscape_success.html",
                   title=SUCCESS_TITLE,
                   caption=(f"{runs}. A run succeeds when it gets within 1% of the highest point; each line is "
                            "the share of runs that have succeeded by then.")),
    ]
