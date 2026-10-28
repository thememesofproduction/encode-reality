"""SPEC.md section 7: interactive figures as self-contained Plotly HTML pages.

Styled like experiments/bells_inequality/plotting.py: a dark source background and light text, set
explicitly, so the site can re-skin the page to the reader's theme (coauthor's plot-embed re-maps the
neutrals and keeps the accents; colourscales are left alone). The figure JSON is written here, by
hand, as plain lists with rounded floats and no NaN, so a page is byte-stable and renders under
plotly.js 2.35.
"""

import html as html_lib
import importlib
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np

BG = "#1e1e1e"          # paper background
PLOT_BG = "#2a2a2a"     # plot area background
FG = "#e0e0e0"          # text
GRID = "#3a3a3a"
MUTED = "#aaaaaa"       # reference lines and limits (always dashed), secondary text

BLUE = "#3987E5"        # a search inside a fixed space: every search over ℝ, every Experiment 1 arm
ORANGE = "#D95926"      # a change of the space: Level 2, the roots, the region better than ℝ

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'

_LATEX = re.compile(r"\$|\\[A-Za-z]{2,}")

# Builders that live in their own modules. They import this module, so they are looked up lazily.
_OPTIONAL_BUILDERS = {
    "write_landscape_plots": "plots_landscape",
    "write_number_system_plots": "plots_number_systems",
}


def base_layout(title=None, height=500, **kw) -> dict:
    """A layout in the house style: dark paper, responsive width, phone-sized margins, a horizontal legend below.

    Keyword arguments replace top-level keys. A ``width`` is refused: figures are responsive.
    """
    if "width" in kw:
        raise ValueError("figures are responsive: no fixed width in the layout")
    axis = {"color": FG, "gridcolor": GRID, "zerolinecolor": GRID, "linecolor": GRID,
            "tickfont": {"color": MUTED}}
    layout = {
        "paper_bgcolor": BG,
        "plot_bgcolor": PLOT_BG,
        "font": {"color": FG, "size": 12, "family": FONT_FAMILY},
        "autosize": True,
        "height": height,
        "margin": {"l": 56, "r": 20, "t": 44 if title else 20, "b": 120},
        "showlegend": True,
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.22, "yanchor": "top",
                   "bgcolor": "rgba(0,0,0,0)", "font": {"color": FG}},
        "xaxis": dict(axis),
        "yaxis": dict(axis),
        "hoverlabel": {"bgcolor": PLOT_BG, "bordercolor": GRID, "font": {"color": FG}},
    }
    if title is not None:
        layout["title"] = {"text": title, "x": 0.02, "xanchor": "left", "font": {"color": FG, "size": 15}}
    layout.update(kw)
    return layout


def rgba(hex_colour: str, alpha: float) -> str:
    """A ``#rrggbb`` colour as an ``rgba()`` string with the given opacity."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def note(text, x, y, ax=0, ay=-30, color=FG, log_x=False, log_y=False, **kw) -> dict:
    """An annotation at (x, y). An arrow is drawn unless (ax, ay) is (0, 0); ``xshift``/``yshift`` (in pixels)
    move a label without an arrow. Text is Unicode, not LaTeX.

    On a log axis plotly.js reads an annotation's x or y as a log10 value: a y of 1e-5 is drawn at 1. Pass
    ``log_x=True`` or ``log_y=True`` to give the data value on that log axis and let this function take log10.
    Leave them False for paper coordinates (``xref="paper"``).
    """
    if _LATEX.search(text):
        raise ValueError(f"annotation text must be Unicode, not LaTeX: {text!r}")
    if log_x:
        x = math.log10(x)
    if log_y:
        y = math.log10(y)
    ann = {"x": x, "y": y, "text": text, "showarrow": bool(ax) or bool(ay), "ax": ax, "ay": ay,
           "font": {"color": color, "size": 12}, "arrowcolor": MUTED}
    ann.update(kw)
    return ann


def _plain(obj):
    """Plain JSON types: numpy to Python, floats to 6 significant digits, NaN and Inf to null."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        return _plain(obj.item())
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return float(f"{obj:.6g}") if math.isfinite(obj) else None
    raise TypeError(f"cannot put a {type(obj).__name__} into a figure page")


def _js(obj) -> str:
    text = json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    # Escaped JSON, so the data can never close the script element or break a JS string literal.
    return (text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _data_and_layout(fig):
    if isinstance(fig, dict):
        data, layout = fig.get("data", []), fig.get("layout", {})
    else:
        # A plotly Figure. Its to_plotly_json() would base64-encode the arrays, so take the traces one by one.
        data = [trace.to_plotly_json() for trace in fig.data]
        layout = fig.layout.to_plotly_json()
    # plotly.py's default template is not house style: every colour here is explicit.
    layout = {k: v for k, v in dict(layout).items() if k != "template"}
    if "width" in layout:
        raise ValueError("figure layout has a fixed width; figures are responsive")
    return list(data), layout


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="{cdn}"></script>
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; overflow: hidden; }}
  body {{ background: {bg}; color: {fg}; font-family: {font}; padding: 8px; }}
  #plot {{ width: 100%; }}
  .caption {{ margin: 6px 12px 0; font-size: 0.95em; line-height: 1.4; color: {muted}; }}
</style>
</head>
<body>
<div id="plot"></div>{caption}
<script>
var DATA = {data};
var LAYOUT = {layout};
Plotly.newPlot("plot", DATA, LAYOUT, {{"responsive": true, "displaylogo": false}});
</script>
</body>
</html>
"""


def write_page(fig, path, title, caption=None) -> Path:
    """Write ``fig`` (a plotly Figure, or a dict with ``data`` and ``layout``) as one self-contained HTML page."""
    data, layout = _data_and_layout(fig)
    caption_html = f'\n<p class="caption">{html_lib.escape(caption)}</p>' if caption else ""
    page = _PAGE.format(title=html_lib.escape(title), cdn=PLOTLY_CDN, bg=BG, fg=FG, font=FONT_FAMILY,
                        muted=MUTED, caption=caption_html, data=_js(data), layout=_js(layout))
    path = Path(path)
    path.write_text(page, encoding="utf-8")
    return path


def __getattr__(name):
    """Re-export the figure builders from plots_landscape and plots_number_systems, when those modules exist."""
    module_name = _OPTIONAL_BUILDERS.get(name)
    if module_name is not None:
        full_name = f"{__package__}.{module_name}"
        if importlib.util.find_spec(full_name) is not None:
            return getattr(importlib.import_module(full_name), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
