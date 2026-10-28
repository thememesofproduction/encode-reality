"""SPEC.md 10.4: plain language in figure B (residual_vs_evaluations) and figure D (landscape_success).

The figures come from the runners' quick-profile output (the session fixtures in conftest.py, shared with
test_plots_interactive.py), so they are built by the current code, not read from results/plots.
Every label, title, jargon word and legend name is parsed from SPEC.md.

Visibility after a button is computed by applying the button's restyle/update/relayout args to the figure
the way Plotly does; `visible: "legendonly"` counts as hidden. Each button is applied from every reachable
state (the initial one and the state after each button), so a button that only works from the initial
state does not pass.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.tests._iic_helpers import (  # noqa: E402
    CLI_TIMEOUT, PY, REPO, EXP_DIR, extract_figures, list_files, load, require_ok, spec_literal,
)
from experiments.ideas_intelligence_creativity.tests._fk_helpers import (  # noqa: E402
    Page, ann_key, apply_button, axis_keys, initial_state, jargon_hits, legend_names, numeric,
    sets_annotations, shown, spec_b_x_title, spec_b_y_title_parts, spec_d_axis, spec_d_dropdown,
    spec_figure_files, spec_jargon, spec_legend_table, spec_log_phrase, spec_replaced_labels,
    spec_square_root_phrase, spec_tour_labels, strip_markup, text_of, visible_annotations, visible_texts,
)

FIG_B, FIG_D = spec_figure_files()
TABLE = spec_legend_table()
FLOOR_KEY = "floor line"
REAL_ARMS = spec_literal("REAL_ARMS", 4)
CONTROLS = spec_literal("CONTROLS", 4)
L2_ARMS = spec_literal("L2_ARMS", 4)
ALL_ARMS = REAL_ARMS + CONTROLS + L2_ARMS
TOUR = spec_tour_labels()
FIGURES = [("exp3", FIG_B), ("exp1", FIG_D)]


def figure(request, runner: str, name: str):
    cli = request.getfixturevalue("exp3_cli" if runner == "exp3" else "exp1_cli").get()
    require_ok(cli["r1"], f"{runner} quick run")
    path = cli["r1_dir"] / "plots" / name
    assert path.exists(), f"{name} not written; files: {list_files(cli['r1_dir'])}"
    html = path.read_text(encoding="utf-8")
    figs = extract_figures(html)
    assert figs, f"{name}: no figure JSON found in the page"
    return figs[0], Page(html)


def fig_b(request):
    return figure(request, "exp3", FIG_B)


def fig_d(request):
    return figure(request, "exp1", FIG_D)


# ----------------------------------------------------------------------------- the legend-name table

def test_legend_names_are_declared_once_in_plots_number_systems():
    assert set(TABLE) - {FLOOR_KEY} == set(ALL_ARMS), "SPEC 10.4's table does not cover SPEC 4's arms"
    mod = load("plots_number_systems")
    ln = getattr(mod, "LEGEND_NAME", None)
    assert ln is not None, "plots_number_systems.LEGEND_NAME is not defined"
    ln = dict(ln)
    for arm in ALL_ARMS:
        assert ln.get(arm) == TABLE[arm], f"LEGEND_NAME[{arm!r}] = {ln.get(arm)!r}, expected {TABLE[arm]!r}"
    assert TABLE[FLOOR_KEY] in ln.values(), "LEGEND_NAME has no entry for the floor line"
    assert sorted(ln.values()) == sorted(TABLE.values()), sorted(ln.values())


def test_figure_b_legend_names_are_the_table(request):
    fig, _ = fig_b(request)
    names = legend_names(fig)
    assert sorted(names) == sorted(TABLE.values()), \
        f"legend names {sorted(names)} differ from SPEC 10.4's table"


LEGEND_DRIVER = r'''
import importlib, os, runpy, sys
script, repo, marker = sys.argv[1], sys.argv[2], sys.argv[3]
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
sys.path.insert(1, repo)
patched = 0
for name in ("experiments.ideas_intelligence_creativity.plots_number_systems", "plots_number_systems"):
    try:
        mod = importlib.import_module(name)
    except Exception:
        continue
    table = getattr(mod, "LEGEND_NAME", None)
    if table is None:
        continue
    new = {k: v + marker for k, v in dict(table).items()}
    try:
        table.clear(); table.update(new)
    except Exception:
        setattr(mod, "LEGEND_NAME", new)
    patched += 1
if not patched:
    sys.stderr.write("plots_number_systems.LEGEND_NAME not found\n")
    sys.exit(3)
sys.argv = [script] + sys.argv[4:]
runpy.run_path(script, run_name="__main__")
'''


def test_figure_b_legend_names_follow_LEGEND_NAME(tmp_path):
    """Config perturbation: rename every entry of LEGEND_NAME and rerun the quick runner; the legend
    must follow, so the names are declared once and used, not repeated in the figure code."""
    marker = " [renamed]"
    driver = tmp_path / "legend_driver.py"
    driver.write_text(LEGEND_DRIVER, encoding="utf-8")
    out = tmp_path / "out"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    proc = subprocess.run(
        [PY, str(driver), str(EXP_DIR / "run_experiment_3.py"), str(REPO), marker, "--profile", "quick",
         "--workers", "1", "--n-seeds", "2", "--output-dir", str(out)],
        cwd=str(cwd), capture_output=True, text=True, timeout=CLI_TIMEOUT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert proc.returncode == 0, f"patched quick run failed (exit {proc.returncode}):\n{proc.stderr[-2500:]}"
    html = (out / "plots" / FIG_B).read_text(encoding="utf-8")
    figs = extract_figures(html)
    assert figs, "no figure in the patched run's figure B"
    names = legend_names(figs[0])
    assert sorted(names) == sorted(v + marker for v in TABLE.values()), \
        f"legend names did not follow LEGEND_NAME: {sorted(names)}"


# ----------------------------------------------------------------------------- jargon and log axes

@pytest.mark.parametrize("runner,name", FIGURES)
def test_visible_text_uses_no_jargon(request, runner, name):
    # negative control on the matcher: ES only as a whole, case-sensitive word; the rest anywhere, any case
    words = spec_jargon()
    assert jargon_hits("the ES arm", words) == ["ES"]
    assert jargon_hits("YES, best, rest, es", words) == []
    assert jargon_hits("Seeds and <b>Evaluations</b>", words) == ["evaluations", "seed"]
    fig, page = figure(request, runner, name)
    texts = visible_texts(fig, page)
    assert texts, f"{name}: no visible text found"
    hits = [(where, jargon_hits(t), t) for where, t in texts if jargon_hits(t)]
    assert hits == [], f"{name}: jargon in visible text (where, words, text): {hits[:8]}"


def _log_axes(fig):
    lay = fig["layout"]
    out = []
    for ax in axis_keys(lay):
        if (lay.get(ax) or {}).get("type") == "log":
            out.append((ax, text_of((lay.get(ax) or {}).get("title")) or ""))
    return out


def test_every_log_axis_title_says_log_scale(request):
    phrase = spec_log_phrase()
    found = []
    for runner, name in FIGURES:
        fig, _ = figure(request, runner, name)
        for ax, title in _log_axes(fig):
            found.append((name, ax, title))
    assert found, "no log-type axis in B or D (B's x-axis is titled as a log scale)"
    bad = [f for f in found if phrase not in strip_markup(f[2])]
    assert bad == [], f"log axes whose title lacks {phrase!r}: {bad}"


# ----------------------------------------------------------------------------- figure B: axes

def test_figure_b_axis_titles(request):
    fig, _ = fig_b(request)
    lay = fig["layout"]
    x = text_of((lay.get("xaxis") or {}).get("title"))
    y = text_of((lay.get("yaxis") or {}).get("title")) or ""
    assert x == spec_b_x_title(), f"x-axis title {x!r}"
    for part in spec_b_y_title_parts():
        assert part in strip_markup(y), f"y-axis title {y!r} lacks {part!r}"


# ----------------------------------------------------------------------------- figure B: the guided tour

def tour_buttons(fig):
    menus = fig["layout"].get("updatemenus") or []
    tours = [m for m in menus if m.get("type") == "buttons"]
    assert len(tours) == 1, f"expected one updatemenus entry of type 'buttons', found {len(tours)}"
    buttons = tours[0].get("buttons") or []
    labels = [b.get("label") for b in buttons]
    assert labels == TOUR, f"tour labels {labels}, expected {TOUR}"
    return buttons


def traces_by_arm(fig):
    """(main trace index per legend key, owner key per trace index or None)."""
    name_to_key = {v: k for k, v in TABLE.items()}
    main = {}
    for i, t in enumerate(fig["data"]):
        if t.get("showlegend", True) is not False and t.get("name") in name_to_key:
            key = name_to_key[t["name"]]
            assert key not in main, f"two legend entries named {t['name']!r}"
            main[key] = i
    missing = sorted(set(TABLE) - set(main))
    assert not missing, f"no legend trace for {missing}"
    groups = {}
    for key, i in main.items():
        g = fig["data"][i].get("legendgroup")
        if g:
            groups.setdefault(g, []).append(key)
    owner = {}
    for i, t in enumerate(fig["data"]):
        if t.get("name") in name_to_key:
            owner[i] = name_to_key[t["name"]]
        elif t.get("legendgroup") and len(groups.get(t["legendgroup"], [])) == 1:
            owner[i] = groups[t["legendgroup"]][0]
        else:
            owner[i] = None
    return main, owner


def states(fig, buttons):
    s0 = initial_state(fig)
    return [s0] + [apply_button(fig, s0, b) for b in buttons]


def test_tour_replaces_show_and_hide_controls(request):
    fig, _ = fig_b(request)
    tour_buttons(fig)
    old = {s.lower() for s in spec_replaced_labels()}
    labels = [str(b.get("label", "")).lower() for m in fig["layout"].get("updatemenus") or []
              for b in m.get("buttons") or []]
    assert not old & set(labels), f"old control buttons still present: {labels}"


def test_tour_button_1_shows_the_line_only(request):
    fig, _ = fig_b(request)
    b1 = tour_buttons(fig)[0]
    main, owner = traces_by_arm(fig)
    for prior in states(fig, tour_buttons(fig)):
        vis, _ = apply_button(fig, prior, b1)
        assert shown(vis[main[FLOOR_KEY]]), "button 1 hides the floor line"
        for arm in REAL_ARMS:
            assert shown(vis[main[arm]]), f"button 1 hides {arm}"
        leaked = [owner[i] for i, v in enumerate(vis) if shown(v) and owner[i] in L2_ARMS + CONTROLS]
        assert leaked == [], f"button 1 shows Level-2 arms or controls: {leaked}"


def test_tour_button_2_adds_l2_evo_and_its_band(request):
    fig, _ = fig_b(request)
    buttons = tour_buttons(fig)
    main, owner = traces_by_arm(fig)
    s0 = initial_state(fig)
    one = {i for i, v in enumerate(apply_button(fig, s0, buttons[0])[0]) if shown(v)}
    for prior in states(fig, buttons):
        two = {i for i, v in enumerate(apply_button(fig, prior, buttons[1])[0]) if shown(v)}
        assert one <= two, f"button 2 hides traces button 1 shows: {sorted(one - two)}"
        extra = two - one
        assert main["l2_evo"] in extra, "button 2 does not show l2_evo"
        foreign = sorted({owner[i] for i in extra if owner[i] not in (None, "l2_evo")})
        assert foreign == [], f"button 2 also shows {foreign}"
        assert len(extra - {main["l2_evo"]}) >= 1, "button 2 shows no band for l2_evo"


def test_tour_button_2_is_the_initial_state(request):
    fig, _ = fig_b(request)
    buttons = tour_buttons(fig)
    s0 = initial_state(fig)
    want_vis = [shown(v) for v in s0[0]]
    want_anns = sorted(ann_key(a) for a in visible_annotations(s0[1]))
    for prior in states(fig, buttons):
        vis, anns = apply_button(fig, prior, buttons[1])
        assert [shown(v) for v in vis] == want_vis, "button 2's trace visibility differs from the initial state"
        assert sorted(ann_key(a) for a in visible_annotations(anns)) == want_anns, \
            "button 2's annotations differ from the initial annotations"


def test_tour_button_3_shows_every_trace(request):
    fig, _ = fig_b(request)
    buttons = tour_buttons(fig)
    for prior in states(fig, buttons):
        vis, _ = apply_button(fig, prior, buttons[2])
        hidden = [fig["data"][i].get("name") for i, v in enumerate(vis) if not shown(v)]
        assert hidden == [], f"button 3 leaves traces hidden: {hidden}"


def _is_floor_annotation(text: str) -> bool:
    t = strip_markup(text).lower()
    return "floor" in t or TABLE[FLOOR_KEY].lower() in t


def test_tour_annotations(request):
    fig, _ = fig_b(request)
    buttons = tour_buttons(fig)
    phrase = spec_square_root_phrase()
    for label, b in zip(TOUR, buttons):
        assert sets_annotations(b), f"button {label!r} does not set the layout's annotations"
    per_button = []
    for k, b in enumerate(buttons):
        sets = []
        for prior in states(fig, buttons):
            _, anns = apply_button(fig, prior, b)
            texts = [strip_markup(str(a.get("text", ""))) for a in visible_annotations(anns)]
            assert any(_is_floor_annotation(t) for t in texts), \
                f"button {TOUR[k]!r} leaves no floor annotation: {texts}"
            sets.append(set(texts))
        per_button.append(sets[0])
    common = set.intersection(*per_button)
    assert any(_is_floor_annotation(t) for t in common), \
        f"no floor annotation common to all three buttons: {per_button}"
    assert any(phrase in t for t in per_button[1]), f"button 2's annotations lack {phrase!r}: {per_button[1]}"


# ----------------------------------------------------------------------------- figure D

def test_figure_d_y_axis_is_a_share(request):
    fig, _ = fig_d(request)
    fmt, rng = spec_d_axis()
    y = fig["layout"].get("yaxis") or {}
    assert y.get("tickformat") == fmt, f"yaxis.tickformat = {y.get('tickformat')!r}"
    got = y.get("range")
    assert isinstance(got, list) and len(got) == 2 and np.allclose(got, rng, rtol=0, atol=1e-12), \
        f"yaxis.range = {got!r}, expected {rng}"
    checked = 0
    for t in fig["data"]:
        ys = numeric(t.get("y"))
        if ys is None or ys.size == 0:
            continue
        checked += 1
        assert ys.min() >= rng[0] - 1e-12 and ys.max() <= rng[1] + 1e-12, \
            f"trace {t.get('name')!r} has y outside {rng} ({ys.min()}..{ys.max()}): not a share"
    assert checked > 0, "no trace with y values in figure D"


def test_figure_d_dropdown_labels(request):
    fig, _ = fig_d(request)
    want = spec_d_dropdown()
    menus = [m for m in fig["layout"].get("updatemenus") or [] if m.get("type", "dropdown") == "dropdown"]
    labels = [[b.get("label") for b in m.get("buttons") or []] for m in menus]
    assert want in labels, f"no dropdown with labels {want}; dropdowns: {labels}"
