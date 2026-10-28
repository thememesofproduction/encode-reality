"""SPEC.md section 7: plots_interactive.py -- the interactive figures the runners write (quick profile)."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.tests._iic_helpers import (  # noqa: E402
    extract_figures, list_files, load, require_ok, spec_colours, spec_plot_names, spec_plotly_url,
    spec_section, typed_arrays, walk_strings,
)

PLOTLY_URL = spec_plotly_url()
BG, FG = spec_colours()
# Section 7 governs the Plotly pages. The canvas instruments that section 5 also lists are section 10's,
# with their own tests (no Plotly, no script src at all), so they are not cases here.
INSTRUMENTS = set(re.findall(r"`(instrument_\w+\.html)`", spec_section(10)))
CASES = ([("exp3", n) for n in spec_plot_names(5) if n not in INSTRUMENTS]
         + [("exp1", n) for n in spec_plot_names(6)])
LATEX = re.compile(r"\\[A-Za-z]{2,}|^\s*\$.*\$\s*$", re.S)


def page(request, runner, name):
    cli = request.getfixturevalue("exp3_cli" if runner == "exp3" else "exp1_cli").get()
    require_ok(cli["r1"], f"{runner} quick run")
    path = cli["r1_dir"] / "plots" / name
    assert path.exists(), f"{name} not written; files: {list_files(cli['r1_dir'])}"
    html = path.read_text(encoding="utf-8")
    figs = extract_figures(html)
    assert figs, f"{name}: no figure JSON passed to Plotly.newPlot/react found in the page"
    return html, figs


def test_spec_parse_and_module_exists():
    assert PLOTLY_URL == "https://cdn.plot.ly/plotly-2.35.2.min.js" and (BG, FG) == ("#1e1e1e", "#e0e0e0")
    assert len(CASES) == 5
    assert len(INSTRUMENTS) == 2 and INSTRUMENTS <= set(spec_plot_names(5)), INSTRUMENTS
    load("plots_interactive")


@pytest.mark.parametrize("runner,name", CASES)
def test_plotly_comes_from_the_pinned_cdn_script_only(request, runner, name):
    html, _ = page(request, runner, name)
    assert "<html" in html.lower()
    srcs = re.findall(r"<script\b[^>]*\bsrc\s*=\s*[\"']([^\"']+)[\"']", html, re.I)
    assert PLOTLY_URL in srcs, srcs
    assert all(s == PLOTLY_URL for s in srcs), f"other external scripts: {srcs}"
    other = set(re.findall(r"cdn\.plot\.ly/plotly-[^\"'\s]+", html)) - {PLOTLY_URL.split("https://")[1]}
    assert not other, other
    assert "fetch(" not in html and "XMLHttpRequest" not in html      # self-contained: data is inline


@pytest.mark.parametrize("runner,name", CASES)
def test_dark_source_colours_are_set_explicitly(request, runner, name):
    html, figs = page(request, runner, name)
    assert BG in html.lower() and FG in html.lower()
    for fig in figs:
        lay = fig["layout"]
        assert str(lay.get("paper_bgcolor", "")).lower() == BG, lay.get("paper_bgcolor")
        assert str((lay.get("font") or {}).get("color", "")).lower() == FG, lay.get("font")


@pytest.mark.parametrize("runner,name", CASES)
def test_no_fixed_width(request, runner, name):
    _, figs = page(request, runner, name)
    for fig in figs:
        assert fig["layout"].get("width") is None, fig["layout"].get("width")
        tmpl = (fig["layout"].get("template") or {}).get("layout") or {}
        assert tmpl.get("width") is None


@pytest.mark.parametrize("runner,name", CASES)
def test_no_nan_anywhere(request, runner, name):
    html, figs = page(request, runner, name)
    assert not re.search(r"\bNaN\b", html)
    for fig in figs:
        for arr in typed_arrays(fig):
            assert arr is not None, "undecodable typed array"
            if arr.dtype.kind == "f":
                assert not np.isnan(arr).any(), "NaN inside a binary-encoded array"


@pytest.mark.parametrize("runner,name", CASES)
def test_text_is_unicode_not_latex(request, runner, name):
    _, figs = page(request, runner, name)
    for fig in figs:
        for s in walk_strings(fig):
            assert not LATEX.search(s), repr(s)


@pytest.mark.parametrize("runner,name", CASES)
def test_each_figure_carries_annotations(request, runner, name):
    _, figs = page(request, runner, name)
    for fig in figs:
        lay = fig["layout"]
        labelled_shapes = [s for s in (lay.get("shapes") or []) if (s.get("label") or {}).get("text")]
        text_traces = [t for t in fig["data"] if "text" in str(t.get("mode", ""))]
        assert lay.get("annotations") or labelled_shapes or text_traces, f"{name}: no annotations"


def test_experiment3_figures_annotate_the_floor_and_the_roots(request):
    texts = []
    for runner, name in CASES:
        if runner == "exp3":
            for fig in page(request, runner, name)[1]:
                texts += [s.lower() for s in walk_strings(fig)]
    assert any("floor" in t for t in texts), "no figure text mentions the real floor"
    assert any("root" in t for t in texts), "no figure text mentions where roots exist"
