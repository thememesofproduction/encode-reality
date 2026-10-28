"""SPEC.md section 7: the three interactive figures of Experiment 3.

Every figure is built from the results document (number_systems_results.json), so the pages can be
rebuilt from the JSON alone. Colours follow the spec's roles: BLUE for a search inside a fixed space (the
dash styles tell the strategies apart), ORANGE for a change of the space (Level 2, the roots, the region
better than the real line), and neutral dashed lines for every limit. Cell and region fills are flat
colours, not colourscales, so they read the same on both site themes. Annotation lines are kept to 28
characters or fewer, with <br> between them, and each label sits on a backing so it reads over the fills.
The pages must read at 360 px, so the labels stay inside the plot area. Figure B's twelve legend names stack
one to a row, which is why that figure is taller than the others.
"""

import math
from pathlib import Path

import numpy as np

from experiments.ideas_intelligence_creativity.plots_interactive import (
    BG, BLUE, FG, GRID, MUTED, ORANGE, PLOT_BG, base_layout, note, rgba, write_page,
)

BACKING = rgba(PLOT_BG, 0.85)            # annotation backing: keeps labels readable over fills and contours
CELL_FLOOR = "rgba(160,160,160,0.16)"    # a cell that stayed at the floor: low contrast on both themes
CELL_BELOW = rgba(ORANGE, 0.45)          # below the floor, but no root
CELL_ROOT = ORANGE                       # a root found (below 10^-6 times the floor)
ROOT_TOL = 1e-6                          # a root: best |p| below 1e-6 times the floor
FLOOR_TOL = 1e-9                         # ratios within this of 1 are at the floor (rounding noise)

# Real arms: BLUE, one dash per strategy.
REAL_STYLE = {
    "hc_fixed": ("dot", "hill, fixed σ"),
    "hc_restart": ("dash", "hill, restarts"),
    "es_restart": ("solid", "ES, restarts"),
    "portfolio": ("dashdot", "portfolio"),
}
# Controls and the other Level-2 variants start hidden; button 3 of figure B's guided tour shows them. The
# comparisons that are not Level 2 are grey, each with its own dash (the floor line has the dash), so they are
# not read as blue searches or as orange Level-2 variants.
CONTROL_STYLE = {
    "c_oracle": (MUTED, "dot", "ℂ by hand"),
    "split_two_reals": (MUTED, "dashdot", "two reals"),
    "flat_joint": (MUTED, "longdashdot", "flat joint"),
    "l2_mutation_only": (ORANGE, "dash", "L2: mutation"),
    "l2_uniform_only": (ORANGE, "dot", "L2: uniform"),
    "l2_no_selection": (ORANGE, "longdash", "L2: no select"),
}
L2_MAIN = "l2_evo"
L2_MAIN_LABEL = "Level 2: ES over algebras"
# The legend names of SPEC 10.4, declared once here: figure B's legend and trace names are read from this table.
# Plain words, no jargon; the hover keeps the older short labels above.
LEGEND_NAME = {
    "floor": "best possible on the line",
    "hc_fixed": "hill-climber, fixed step",
    "hc_restart": "hill-climber, restarts",
    "es_restart": "evolution strategy, restarts",
    "portfolio": "portfolio of strategies",
    L2_MAIN: "Level 2: searches number systems",
    "c_oracle": "complex numbers given by hand",
    "split_two_reals": "two real numbers",
    "flat_joint": "one search over everything",
    "l2_mutation_only": "Level 2: small changes only",
    "l2_uniform_only": "Level 2: random proposals only",
    "l2_no_selection": "Level 2: no selection",
}


def _axis(**kw) -> dict:
    """An axis in the house style (the colours of plots_interactive.base_layout), with overrides."""
    axis = {"color": FG, "gridcolor": GRID, "zerolinecolor": GRID, "linecolor": GRID,
            "tickfont": {"color": MUTED}}
    axis.update(kw)
    return axis


def _label(text, x, y, color=FG, **kw):
    """An annotation on a backing, so it reads over the fills and contours. ``color`` sets its text colour."""
    kw.setdefault("bgcolor", BACKING)
    kw.setdefault("borderpad", 2)
    return note(text, x, y, color=color, font={"size": 11, "color": color}, **kw)


def _category_scale() -> list:
    """A discrete three-colour scale for cell categories 0, 1 and 2 (the z values 0, 1 and 2 with zmin 0, zmax 2)."""
    return [[0.0, CELL_FLOOR], [1 / 3, CELL_FLOOR], [1 / 3, CELL_BELOW], [2 / 3, CELL_BELOW],
            [2 / 3, CELL_ROOT], [1.0, CELL_ROOT]]


def _pos(x: float) -> float:
    """A value for a log axis: never zero or negative."""
    return max(float(x), 1e-14)


def _legend(y: float) -> dict:
    """A horizontal legend under the plot, wrapping by width (plotly.js 2.35 supports wrapping). The symbol is
    narrow, so a phone-width plot fits the default legend in about three rows."""
    return {"orientation": "h", "x": 0.5, "xanchor": "center", "y": y, "yanchor": "top",
            "bgcolor": "rgba(0,0,0,0)", "font": {"color": FG, "size": 11}, "itemwidth": 20}


def _leaves_floor(cps: list, med: list) -> tuple:
    """The last checkpoint at which a median is still on the floor: where the curve leaves it, (evaluations, 1)."""
    on_floor = [i for i, v in enumerate(med) if abs(v - 1.0) < 1e-6]
    i = max(on_floor) if on_floor else 0
    return cps[i], 1.0


# ----------------------------------------------------------------------------- complex_plane

def _complex_plane(results: dict, path: Path) -> Path:
    poly = results["polynomial"]
    m = float(poly["floor"]["m"])
    x_star = float(poly["floor"]["x_star"])
    rl = results["real_line"]
    cp = results["complex_plane"]
    roots = [complex(float(r), float(i)) for r, i in poly["roots"]]
    dips = sorted(rl["minima"], key=lambda d: d["abs_p"])
    other = dips[1] if len(dips) > 1 else None

    z = np.clip(np.asarray(cp["log10_abs_p"], dtype=float), -2.0, 3.0)
    re_grid, im_grid = cp["re"], cp["im"]
    x_lo, x_hi = rl["x"][0], rl["x"][-1]
    strip_lo, strip_hi = math.log10(6.0), math.log10(600.0)   # the top strip's log10 |p| range
    strip_dom = (0.78, 1.0)                                    # the top strip's domain in paper coordinates

    traces = [
        # the top strip: |p(x)| along the real line, and the floor
        {"type": "scatter", "mode": "lines", "name": "|p(x)| on ℝ", "x": rl["x"],
         "y": [_pos(v) for v in rl["abs_p"]], "xaxis": "x2", "yaxis": "y2", "showlegend": False,
         "line": {"color": BLUE, "width": 2.5},
         "hovertemplate": "x = %{x:.3f}<br>|p| = %{y:.4g}<extra></extra>"},
        {"type": "scatter", "mode": "lines", "name": f"floor m = {m:.2f}", "x": [x_lo, x_hi], "y": [m, m],
         "xaxis": "x2", "yaxis": "y2", "showlegend": False, "hoverinfo": "skip",
         "line": {"color": MUTED, "width": 1.5, "dash": "dash"}},
        # the main panel: neutral contour lines of log10 |p|, then the region better than the real line
        {"type": "contour", "x": re_grid, "y": im_grid, "z": z.tolist(), "showscale": False,
         "showlegend": False, "hoverinfo": "skip",
         "contours": {"coloring": "none", "start": -1.5, "end": 2.5, "size": 0.5},
         "line": {"color": MUTED, "width": 0.8}},
        {"type": "contour", "x": re_grid, "y": im_grid, "z": z.tolist(), "showscale": False,
         "showlegend": False, "hoverinfo": "skip",
         "contours": {"type": "constraint", "operation": ">", "value": math.log10(m), "coloring": "fill"},
         "fillcolor": rgba(ORANGE, 0.5), "line": {"width": 0}},
        {"type": "scatter", "mode": "lines", "name": "ℝ, the only line searched", "x": [-3.0, 4.0],
         "y": [0.0, 0.0], "showlegend": False, "hoverinfo": "skip", "line": {"color": BLUE, "width": 5}},
        {"type": "scatter", "mode": "markers", "name": "best on ℝ", "x": [x_star], "y": [0.0],
         "showlegend": False, "hovertemplate": "best on ℝ<br>x = %{x:.4f}<br>|p| = %{customdata:.4g}<extra></extra>",
         "customdata": [m],
         "marker": {"symbol": "diamond", "size": 11, "color": BLUE, "line": {"color": BG, "width": 1.5}}},
        {"type": "scatter", "mode": "markers", "name": "roots of p (|p| = 0)", "x": [r.real for r in roots],
         "y": [r.imag for r in roots], "showlegend": False,
         "hovertemplate": "root<br>%{x:.3f}%{y:+.3f} i<extra></extra>",
         "marker": {"symbol": "x", "size": 12, "color": ORANGE, "line": {"color": FG, "width": 1}}},
        # the strip's markers: the best on ℝ and the other dip
        {"type": "scatter", "mode": "markers", "name": "best on ℝ", "x": [x_star], "y": [m],
         "xaxis": "x2", "yaxis": "y2", "showlegend": False, "hoverinfo": "skip",
         "marker": {"symbol": "diamond", "size": 9, "color": BLUE, "line": {"color": BG, "width": 1}}},
    ]
    if other is not None:
        traces.append({"type": "scatter", "mode": "markers", "name": "the other dip", "x": [other["x"]],
                       "y": [other["abs_p"]], "xaxis": "x2", "yaxis": "y2", "showlegend": False,
                       "hovertemplate": "trap<br>x = %{x:.3f}<br>|p| = %{y:.4g}<extra></extra>",
                       "marker": {"symbol": "circle-open", "size": 9, "color": BLUE,
                                  "line": {"width": 2, "color": BLUE}}})

    annotations = [
        _label(f"floor m = {m:.2f}", x=x_hi, y=m, xref="x2", yref="y2", log_y=True, color=MUTED,
               ax=0, ay=0, xanchor="right", yanchor="bottom", yshift=3),
        _label("ℝ — the only line<br>intelligence searches", x=3.95, y=0.08, ax=0, ay=0,
               xanchor="right", yanchor="bottom", color=BLUE),
        _label(f"best on ℝ:<br>|p| = {m:.2f}", x=-1.25, y=-0.25, ax=0, ay=0, xanchor="right", yanchor="top",
               color=FG),
        _label("better than<br>anything on ℝ", x=-1.0, y=1.25, ax=0, ay=0, xanchor="center", yanchor="middle",
               color=ORANGE),
        _label("root −1 ± ½i", x=-1.0, y=-0.68, ax=0, ay=0, xanchor="center", yanchor="top"),
        _label("root 2 ± i", x=2.0, y=1.3, ax=0, ay=0, xanchor="center", yanchor="middle"),
        _label("the line only touches<br>the better region here", x=x_star, y=0.0, ax=90, ay=40,
               xanchor="center", yanchor="middle", color=FG),
    ]
    if other is not None:
        annotations.append(_label(f"trap: {other['abs_p']:.1f}", x=other["x"], y=other["abs_p"], xref="x2",
                                  yref="y2", log_y=True, ax=0, ay=0, xanchor="right", yanchor="bottom",
                                  xshift=-6, yshift=-8))

    layout = base_layout(
        title="The answer is off the line", height=520, showlegend=False,
        margin={"l": 46, "r": 12, "t": 44, "b": 52},
        xaxis=_axis(domain=[0.0, 1.0], range=[-3.0, 4.0], anchor="y", title={"text": "Re"}),
        yaxis=_axis(domain=[0.0, 0.72], range=[-2.0, 2.0], anchor="x", title={"text": "Im"}),
        xaxis2=_axis(domain=[0.0, 1.0], range=[-3.0, 4.0], anchor="y2", showticklabels=False),
        yaxis2=_axis(domain=[strip_dom[0], strip_dom[1]], type="log", range=[strip_lo, strip_hi],
                     anchor="x2", tickmode="array", tickvals=[10, 100], ticktext=["10", "100"],
                     title={"text": "|p|"}),
        annotations=annotations,
    )
    caption = "Level 0 is a point. Level 1 searches along ℝ. Level 2 changes the space so the plane is reachable."
    return write_page({"data": traces, "layout": layout}, path, "Experiment 3: the answer is off the line",
                      caption=caption)


# ----------------------------------------------------------------------------- residual_vs_evaluations

def _residual(results: dict, path: Path) -> Path:
    cps = [int(c) for c in results["checkpoints"]]
    summary = results["summary"]   # best |p| over the floor, so the floor is 1

    def arm(name, colour, dash, label, visible=True, band=False) -> list:
        s = summary[name]
        legend_name = LEGEND_NAME[name]
        med = [_pos(v) for v in s["median"]]
        q1 = [_pos(v) for v in s["q1"]]
        q3 = [_pos(v) for v in s["q3"]]
        out = []
        if band:
            out.append({"type": "scatter", "mode": "lines", "x": cps + cps[::-1], "y": q3 + q1[::-1],
                        "fill": "toself", "fillcolor": rgba(colour, 0.16), "line": {"width": 0},
                        "hoverinfo": "skip", "showlegend": False, "legendgroup": name, "visible": visible})
        out.append({"type": "scatter", "mode": "lines", "name": legend_name, "x": cps, "y": med,
                    "legendgroup": name, "visible": visible, "customdata": [[a, b] for a, b in zip(q1, q3)],
                    "line": {"color": colour, "width": 2.5, "dash": dash},
                    "hovertemplate": (f"{label}<br>evaluations: %{{x:,.0f}}<br>median best ÷ floor: %{{y:.3g}}"
                                      "<br>quartiles: %{customdata[0]:.3g} to %{customdata[1]:.3g}<extra></extra>")})
        return out

    traces = [
        {"type": "scatter", "mode": "lines", "name": LEGEND_NAME["floor"], "x": [cps[0], cps[-1]], "y": [1.0, 1.0],
         "line": {"color": MUTED, "width": 1.6, "dash": "dash"}, "hoverinfo": "skip"},
        {"type": "scatter", "mode": "lines", "x": [cps[0], cps[-1]], "y": [1e-6, 1e-6], "showlegend": False,
         "line": {"color": rgba(MUTED, 0.6), "width": 1, "dash": "dot"}, "hoverinfo": "skip"},
    ]
    owner = [None, None]   # the arm each trace belongs to: the floor and the threshold belong to none

    def add(name, *args, **kw):
        new = arm(name, *args, **kw)
        traces.extend(new)
        owner.extend([name] * len(new))

    for name, (dash, label) in REAL_STYLE.items():
        add(name, BLUE, dash, label)
    add(L2_MAIN, ORANGE, "solid", L2_MAIN_LABEL, band=True)
    for name, (colour, dash, label) in CONTROL_STYLE.items():
        if name in summary:
            add(name, colour, dash, label, visible=False)

    def shown(*arms) -> list:
        """Trace visibility for a tour state: the floor and the threshold always, and the traces of the arms named."""
        return [o is None or o in arms for o in owner]

    # The guided tour (SPEC 10.4). Each button sets the traces and the annotations of one state; button 2 is the
    # initial state, so the figure starts with exactly what button 2 shows.
    med_l2 = summary[L2_MAIN]["median"]
    x_cross, y_cross = _leaves_floor(cps, med_l2)
    line_only = shown(*REAL_STYLE)
    new_space = shown(*REAL_STYLE, L2_MAIN)
    everything = shown(*REAL_STYLE, L2_MAIN, *CONTROL_STYLE)

    floor_note = _label("the floor: the best any<br>search on the line can do", x=1e6, y=1.0, log_x=True,
                        log_y=True, ax=0, ay=0, xanchor="right", yanchor="bottom", yshift=4)
    threshold_note = _label("a solution:<br>a millionth of the<br>floor or less", x=1.0, y=1e-6, log_x=True,
                            log_y=True, ax=0, ay=0, xanchor="left", yanchor="top", yshift=-3, color=MUTED)
    every_search_note = _label("every search on the line<br>stops at the floor,<br>however long it runs", x=1e6,
                               y=1.0, log_x=True, log_y=True, ax=0, ay=0, xanchor="right", yanchor="top", yshift=-4)
    level2_note = _label("Level 2 finds a number<br>system with a square root<br>of −1, and drops through<br>the floor",
                         x=x_cross, y=y_cross, log_x=True, log_y=True, ax=-50, ay=70, xanchor="center", color=ORANGE)
    two_reals_note = _label("two real numbers (a control)<br>also stops at the floor", x=1e6, y=1.0, log_x=True,
                            log_y=True, ax=-10, ay=130, xanchor="right", color=MUTED)
    # The current step, top left of the plot area. The buttons show no active state (the paper theme cannot
    # re-skin Plotly's highlight), so this line says which step is showing.
    def step(text):
        return _label(text, x=0.01, y=0.99, xref="paper", yref="paper", ax=0, ay=0, xanchor="left", yanchor="top")

    step1 = step("step 1 of 3: searches on the line")
    step2 = step("step 2 of 3: a new space")
    step3 = step("step 3 of 3: comparison runs")
    line_notes = [floor_note, threshold_note, every_search_note, step1]
    space_notes = [floor_note, threshold_note, level2_note, step2]
    all_notes = [floor_note, threshold_note, level2_note, two_reals_note, step3]

    updatemenus = [{
        "type": "buttons", "direction": "right", "x": 0.0, "xanchor": "left", "y": 1.0, "yanchor": "bottom",
        "showactive": False, "active": 1, "bgcolor": MUTED, "bordercolor": GRID,
        "font": {"color": BG, "size": 11}, "pad": {"l": 4, "r": 4, "t": 2, "b": 2},
        "buttons": [
            {"label": "1 the line", "method": "update",
             "args": [{"visible": line_only}, {"annotations": line_notes}]},
            {"label": "2 a new space", "method": "update",
             "args": [{"visible": new_space}, {"annotations": space_notes}]},
            {"label": "3 comparisons", "method": "update",
             "args": [{"visible": everything}, {"annotations": all_notes}]},
        ],
    }]

    height = 760   # the twelve legend names, one to a row at 390 px, need this height to show without a scroll
    layout = base_layout(
        title="Searching harder vs changing the space", height=height,
        margin={"l": 80, "r": 16, "t": 92, "b": 150},
        xaxis=_axis(type="log", range=[0.0, math.log10(cps[-1])], dtick=2, exponentformat="power",
                    title={"text": "guesses checked (log scale)"}),
        yaxis=_axis(type="log", range=[-12.0, 2.0], dtick=3, exponentformat="power",
                    title={"text": "best so far, as a fraction of the floor<br>(log scale; 1 = floor; lower is better)"}),
        legend=_legend(-0.155),
        hovermode="closest", updatemenus=updatemenus, annotations=space_notes,
    )
    layout["title"]["font"]["size"] = 14
    n_runs = results["params"]["n_seeds"]
    caption = (f"The line for each search is the middle of {n_runs} runs; the shaded band holds the middle half. "
               "Every search on the line stops at the floor, however long it runs. Level 2 searches over number "
               "systems; once it picks one with a square root of −1, it finds a root.")
    return write_page({"data": traces, "layout": layout}, path, "Experiment 3: searching harder or changing the space",
                      caption=caption)


# ----------------------------------------------------------------------------- algebra_map

def _algebra_map(results: dict, path: Path) -> Path:
    am = results["algebra_map"]
    alphas = am["alpha"]
    betas = am["beta"]
    z = np.asarray(am["log10_ratio"], dtype=float)     # log10 of best |p| over the floor, per cell
    exemplar = results["level2"]["exemplar"]

    # 0: stayed at the floor, 1: below the floor, no root, 2: a root (below 1e-6 times the floor)
    cat = np.where(z < math.log10(ROOT_TOL), 2, np.where(z < -FLOOR_TOL, 1, 0))
    names = np.array(["stayed at the floor", "below the floor, no root", "root found"])
    traces = [{
        "type": "heatmap", "x": betas, "y": alphas, "z": cat.tolist(), "zmin": 0, "zmax": 2,
        "colorscale": _category_scale(), "showscale": False, "name": "cells", "showlegend": False,
        "text": names[cat].tolist(), "hovertemplate": "β %{x:.2f}, α %{y:.2f}<br>%{text}<extra></extra>",
    }]
    for name, colour in [("stayed at the floor", CELL_FLOOR), ("below the floor, no root", CELL_BELOW),
                         ("root found", CELL_ROOT)]:
        traces.append({"type": "scatter", "mode": "markers", "name": name, "x": [None], "y": [None],
                       "marker": {"symbol": "square", "size": 12, "color": colour}})

    beta_par = np.linspace(-2.0, 2.0, 101)
    traces.append({"type": "scatter", "mode": "lines", "name": "Δ = 0 (dual numbers)",
                   "x": beta_par.tolist(), "y": (-beta_par ** 2 / 4.0).tolist(), "showlegend": False,
                   "hoverinfo": "skip", "line": {"color": MUTED, "width": 2, "dash": "dash"}})
    traces.append({"type": "scatter", "mode": "markers", "name": "e² = −1 (ℂ)", "x": [0.0], "y": [-1.0],
                   "showlegend": False, "hovertemplate": "e² = −1, ℂ as usually written<extra></extra>",
                   "marker": {"symbol": "circle-open", "size": 13, "color": FG, "line": {"width": 1.5}}})

    # Labels sit inside the box at 390 px too, so they are centred or anchored to an edge, never run off it. The
    # arrows are in data units (axref/ayref), so they keep their place when the plot is narrower.
    annotations = [
        _label("Δ > 0: two real numbers<br>in disguise — no roots", x=0.0, y=3.6, ax=0, ay=0,
               xanchor="center", yanchor="middle"),
        _label("Δ = 0:<br>dual numbers", x=0.0, y=0.12, ax=0, ay=0, xanchor="center", yanchor="bottom", color=MUTED),
        # the cap is Δ < 0, so this label sits in the empty upper left and points in
        _label("Δ < 0: ℂ in<br>disguise — roots<br>exist", x=-1.2, y=-0.55, ax=-3.8, ay=1.3, axref="x", ayref="y",
               xanchor="left", yanchor="middle"),
        _label("e² = −1", x=-0.2, y=-1.0, ax=0, ay=0, xanchor="right", yanchor="bottom", yshift=3),
    ]

    if exemplar is not None:
        proposals = exemplar["proposals"]                     # [alpha, beta, best/floor, generation]
        below = [p for p in proposals if p[2] < 1.0 - FLOOR_TOL]
        at_floor = [p for p in proposals if p[2] >= 1.0 - FLOOR_TOL]
        traces.append({"type": "scatter", "mode": "markers", "name": "proposal, stayed at the floor",
                       "x": [p[1] for p in at_floor], "y": [p[0] for p in at_floor],
                       "marker": {"symbol": "circle", "size": 4, "color": MUTED},
                       "hovertemplate": "β %{x:.3f}, α %{y:.3f}<extra>proposal, at the floor</extra>"})
        traces.append({"type": "scatter", "mode": "markers", "name": "proposal, below the floor",
                       "x": [p[1] for p in below], "y": [p[0] for p in below],
                       "marker": {"symbol": "circle", "size": 5, "color": ORANGE, "line": {"color": BG, "width": 0.6}},
                       "hovertemplate": "β %{x:.3f}, α %{y:.3f}<extra>proposal, below the floor</extra>"})
        a, b = exemplar["adoption_algebra"]
        traces.append({"type": "scatter", "mode": "markers", "name": "first proposal inside", "x": [b], "y": [a],
                       "showlegend": False, "hoverinfo": "skip",
                       "marker": {"symbol": "star", "size": 15, "color": FG, "line": {"color": BG, "width": 1}}})
        # outside the cap, on the right: the callout's tail is at (3.9, 1.0) in data units
        annotations.append(_label(f"first proposal<br>inside: evaluation<br>{int(exemplar['adoption_eval']):,}",
                                  x=b, y=a, ax=3.9, ay=1.0, axref="x", ayref="y", xanchor="right",
                                  yanchor="middle"))

    layout = base_layout(
        title="Searching over number systems", height=620,
        margin={"l": 48, "r": 16, "t": 44, "b": 150},
        xaxis=_axis(title={"text": "β"}, range=[betas[0], betas[-1]]),
        yaxis=_axis(title={"text": "α"}, range=[alphas[0], alphas[-1]]),
        legend=_legend(-0.13),
        annotations=annotations,
    )
    caption = ("Each cell is one number system e² = α + βe, coloured by what a 500-evaluation search inside it "
               "achieved. Only the cap under the parabola reaches below the real floor; it covers 5.6% of the box. "
               "Dots: one Level-2 run's proposals.")
    return write_page({"data": traces, "layout": layout}, path, "Experiment 3: searching over number systems",
                      caption=caption)


# ----------------------------------------------------------------------------- the entry point

def write_number_system_plots(results: dict, plots_dir) -> list:
    """Write the three Experiment 3 pages into plots_dir, from the results document alone."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return [
        _complex_plane(results, plots_dir / "complex_plane.html"),
        _residual(results, plots_dir / "residual_vs_evaluations.html"),
        _algebra_map(results, plots_dir / "algebra_map.html"),
    ]
