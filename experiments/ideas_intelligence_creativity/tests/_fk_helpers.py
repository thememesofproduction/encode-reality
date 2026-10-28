"""Helpers for the SPEC.md section 10 tests (instruments and plain-language figures).

Not a test module. Everything here is derived from SPEC.md section 10; nothing here knows how figkit,
instruments.py or the plot modules are implemented. Strings the spec declares (the jargon list, the
privacy-guard tokens, the legend-name table, the button and dropdown labels) are parsed out of SPEC.md,
so the spec stays the single source of truth.
"""
from __future__ import annotations

import html as htmllib
import json
import math
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest

from experiments.ideas_intelligence_creativity.tests._iic_helpers import (
    EXP_DIR, typed_arrays, spec_text,
)

FIGKIT_DIR = EXP_DIR / "figkit"
SYSTEMS_DIR = FIGKIT_DIR / "systems"
RESULTS_DIR = EXP_DIR / "results"


# --------------------------------------------------------------------------- SPEC section 10

def spec_sub(number: str) -> str:
    """Text of SPEC.md subsection `### <number> ...` up to the next ### or ## heading."""
    m = re.search(rf"^### {re.escape(number)} .*?(?=^### |^## |\Z)", spec_text(), re.S | re.M)
    assert m, f"SPEC.md has no subsection {number}"
    return m.group(0)


def spec_privacy_tokens() -> tuple[list[str], list[str]]:
    """(case-sensitive tokens, case-insensitive tokens) listed as literals in the privacy guard (10.1)."""
    sec = spec_sub("10.1")
    block = sec[sec.index("**The privacy guard.**"):sec.index("**Determinism.**")]
    sensitive, insensitive = [], []
    for line in re.findall(r"^\s*- (.*)$", block, re.M):
        toks = re.findall(r"`([^`]+)`", line)
        (insensitive if "case-insensitive" in line else sensitive).extend(toks)
    assert len(sensitive) + len(insensitive) == 3, (sensitive, insensitive)
    return sensitive, insensitive


def spec_allowed_stylesheets() -> list[str]:
    return re.findall(r"`(/assets/er/[\w.-]+\.css)`", spec_sub("10.1"))


def spec_embed_meta() -> tuple[str, str]:
    m = re.search(r'<meta name="([^"]+)" content="([^"]+)">', spec_sub("10.1"))
    assert m, "SPEC 10.1 declares no embed meta"
    return m.group(1), m.group(2)


def spec_jargon() -> list[str]:
    line = re.search(r"\*\*Jargon list\.\*\*[^:]*:(.*?)(?=\n\s*- |\n\s*\n)", spec_sub("10.4"), re.S).group(1)
    words = re.findall(r"`([^`]+)`", line)
    assert len(words) == 13 and "ES" in words, words
    return words


def spec_whole_word_case_sensitive() -> set:
    return set(re.findall(r"Match `([^`]+)` as a whole word and case-sensitively", spec_sub("10.4")))


def spec_legend_table() -> dict:
    """{'floor line' | arm: legend name} from the table in 10.4."""
    rows = re.findall(r"^\| ([^|`]+?) \| `([^`]+)` \|\s*$", spec_sub("10.4"), re.M)
    table = {k.strip(): v for k, v in rows}
    assert len(table) == 12 and "floor line" in table, table
    return table


def spec_tour_labels() -> list[str]:
    m = re.search(r"whose labels in order are\s+(.*?)\.\s", spec_sub("10.4"), re.S)
    labels = re.findall(r"`([^`]+)`", m.group(1))
    assert len(labels) == 3, labels
    return labels


def spec_replaced_labels() -> list[str]:
    m = re.search(r'It replaces "([^"]+)" and "([^"]+)"', spec_sub("10.4"))
    return [m.group(1), m.group(2)]


def spec_figure_files() -> tuple[str, str]:
    m = re.search(r"figures B \(`(\w+)`\) and D \(`(\w+)`\)", spec_sub("10.4"))
    return m.group(1) + ".html", m.group(2) + ".html"


def spec_b_x_title() -> str:
    return re.search(r"\*\*x-axis title:\*\* `([^`]+)`", spec_sub("10.4")).group(1)


def spec_b_y_title_parts() -> list[str]:
    line = re.search(r"\*\*y-axis title:\*\*(.*)", spec_sub("10.4")).group(1)
    parts = re.findall(r"`([^`]+)`", line)
    assert len(parts) == 2, parts
    return parts


def spec_log_phrase() -> str:
    return re.search(r"Every log-type axis title contains `([^`]+)`", spec_sub("10.4")).group(1)


def spec_d_axis() -> tuple[str, list]:
    sec = spec_sub("10.4")
    line = re.search(r"\*\*y-axis:\*\*(.*)", sec[sec.index("**Figure D:**"):]).group(1)
    fmt = re.search(r"`tickformat` `([^`]+)`", line).group(1)
    rng = [float(v) for v in re.search(r"range \[([^\]]+)\]", line).group(1).split(",")]
    return fmt, rng


def spec_d_dropdown() -> list[str]:
    line = re.search(r"\*\*dropdown labels:\*\*(.*)", spec_sub("10.4")).group(1)
    labels = re.findall(r"`([^`]+)`", line)
    assert len(labels) == 3, labels
    return labels


def spec_square_root_phrase() -> str:
    return re.search(r"annotation\s+containing\s+`([^`]+)`", spec_sub("10.4")).group(1)


# --------------------------------------------------------------------------- jargon

def strip_markup(s: str) -> str:
    return htmllib.unescape(re.sub(r"<[^>]*>", " ", s))


def jargon_hits(text: str, words=None) -> list[str]:
    """Words of the 10.4 jargon list found in `text`, with the 10.4 matching rules."""
    words = spec_jargon() if words is None else words
    exact = spec_whole_word_case_sensitive()
    plain = strip_markup(text)
    hits = []
    for w in words:
        if w in exact:
            if re.search(rf"(?<!\w){re.escape(w)}(?!\w)", plain):
                hits.append(w)
        elif w.lower() in plain.lower():
            hits.append(w)
    return hits


# --------------------------------------------------------------------------- node

def require_node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.fail("node is not on PATH: SPEC 10.1/10.2 checks need node; a missing node is a failure, "
                    "not a skip", pytrace=False)
    return node


def node_check(source: str, workdir: Path, name: str, module: bool = False) -> subprocess.CompletedProcess:
    node = require_node()
    path = Path(workdir) / (name + (".mjs" if module else ".js"))
    path.write_text(source, encoding="utf-8")
    return subprocess.run([node, "--check", str(path)], capture_output=True, text=True, timeout=60)


def run_node(script: str, args: list, workdir: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    node = require_node()
    path = Path(workdir) / "driver.js"
    path.write_text(script, encoding="utf-8")
    return subprocess.run([node, str(path), *map(str, args)], capture_output=True, text=True,
                          timeout=timeout, cwd=str(workdir))


# --------------------------------------------------------------------------- HTML pages

_SCRIPT_END = re.compile(r"</script(?=[\s/>])", re.I)


class _Collector(HTMLParser):
    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self._line_starts = [0]
        for m in re.finditer(r"\n", text):
            self._line_starts.append(m.end())
        self.elements = []          # dicts: tag, attrs, offset, in_head
        self.saw_head = False
        self.in_head = False
        self.captures = {}          # id(element) -> list[str]
        self._open = []             # stack of (tag, element) for capture elements
        self._skip = 0              # inside script/style/template
        self.body_text = []

    def _offset(self):
        line, col = self.getpos()
        return self._line_starts[line - 1] + col

    def handle_starttag(self, tag, attrs):
        if tag == "head":
            self.saw_head, self.in_head = True, True
        elif tag == "body":
            self.in_head = False
        el = {"tag": tag, "attrs": {k: (v if v is not None else "") for k, v in attrs},
              "offset": self._offset(), "in_head": self.in_head, "text": None}
        self.elements.append(el)
        if tag in ("script", "style", "template"):
            self._skip += 1
        if tag in ("title", "p", "figcaption"):
            el["text"] = []
            self._open.append((tag, el))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag in ("script", "style", "template"):
            self._skip -= 1
        if self._open and self._open[-1][1] is self.elements[-1]:
            self._open.pop()

    def handle_endtag(self, tag):
        if tag == "head":
            self.in_head = False
        if tag in ("script", "style", "template") and self._skip:
            self._skip -= 1
        for i in range(len(self._open) - 1, -1, -1):
            if self._open[i][0] == tag:
                del self._open[i:]
                break

    def handle_data(self, data):
        for _, el in self._open:
            el["text"].append(data)
        if not self._skip and not self.in_head:
            self.body_text.append(data)


class Page:
    """A parsed HTML page: elements in document order, inline scripts (browser end-tag rule)."""

    def __init__(self, text: str):
        self.html = text
        c = _Collector(text)
        c.feed(text)
        c.close()
        self.saw_head = c.saw_head
        self.elements = c.elements
        for el in self.elements:
            if el["text"] is not None:
                el["text"] = "".join(el["text"])
        self.body_text = " ".join(" ".join(c.body_text).split())
        self.scripts = []
        for el in self.elements:
            if el["tag"] != "script":
                continue
            start = self.html.index(">", el["offset"]) + 1
            m = _SCRIPT_END.search(self.html, start)
            el["content"] = self.html[start:m.start()] if m else self.html[start:]
            self.scripts.append(el)

    def find(self, tag=None, **attrs):
        out = []
        for el in self.elements:
            if tag is not None and el["tag"] != tag:
                continue
            if all(el["attrs"].get(k.replace("_", "-")) == v for k, v in attrs.items()):
                out.append(el)
        return out

    def stylesheets(self):
        return [el for el in self.elements
                if el["tag"] == "style"
                or (el["tag"] == "link" and "stylesheet" in el["attrs"].get("rel", "").lower().split())]

    def data_scripts(self):
        return [s for s in self.scripts
                if s["attrs"].get("type", "").strip().lower() == "application/json"
                and s["attrs"].get("data-fk") == "data"]

    def js_scripts(self):
        """Inline non-JSON scripts."""
        return [s for s in self.scripts
                if "json" not in s["attrs"].get("type", "").lower() and "src" not in s["attrs"]]


def read_page(path: Path) -> Page:
    return Page(Path(path).read_text(encoding="utf-8"))


def page_data(page: Page):
    blocks = page.data_scripts()
    assert len(blocks) == 1, f"expected exactly one data script, found {len(blocks)}"
    return json.loads(blocks[0]["content"])


def caption_text(page: Page):
    caps = page.find("p", data_fk="caption")
    assert len(caps) == 1, f"expected exactly one <p data-fk=\"caption\">, found {len(caps)}"
    return caps[0]["text"]


# --------------------------------------------------------------------------- plain data

def plain(obj):
    """Tuples and arrays to lists, numpy scalars to Python, so outputs compare like JSON."""
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [plain(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def same_json(a, b) -> bool:
    """Exact structural equality, with int, float and bool kept apart."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, float) and isinstance(b, float)):
            return False
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same_json(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def load_results(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- Plotly figures

def text_of(t):
    if isinstance(t, dict):
        t = t.get("text")
    return t if isinstance(t, str) else None


def axis_keys(layout: dict) -> list[str]:
    return sorted(k for k in layout if re.fullmatch(r"[xy]axis\d*", k))


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and not re.match(r"annotations(\[\d+\])?$", key) and not key.endswith("title"):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def button_parts(button: dict):
    """(data update, layout update, trace indices) of an updatemenu button, as Plotly reads them."""
    method = button.get("method", "restyle")
    args = list(button.get("args") or [])
    if method == "restyle":
        return (args[0] if args else {}) or {}, {}, args[1] if len(args) > 1 else None
    if method == "update":
        return ((args[0] if args else {}) or {}, (args[1] if len(args) > 1 else {}) or {},
                args[2] if len(args) > 2 else None)
    if method == "relayout":
        return {}, (args[0] if args else {}) or {}, None
    pytest.fail(f"button {button.get('label')!r} uses method {method!r}; the tests model restyle, "
                f"relayout and update", pytrace=False)


def initial_state(fig: dict):
    vis = [t.get("visible", True) for t in fig["data"]]
    anns = [dict(a) for a in (fig["layout"].get("annotations") or [])]
    return vis, anns


def apply_button(fig: dict, state, button: dict):
    """Apply one button's args to (trace visibility, annotations) as Plotly does."""
    vis, anns = list(state[0]), [dict(a) for a in state[1]]
    data_upd, layout_upd, idx = button_parts(button)
    n = len(fig["data"])
    if "visible" in data_upd:
        v = data_upd["visible"]
        targets = list(range(n)) if idx is None else ([idx] if isinstance(idx, int) else list(idx))
        for k, t in enumerate(targets):
            val = v[k % len(v)] if isinstance(v, list) else v
            vis[t] = True if val is None else val
    flat = _flatten(layout_upd)
    if "annotations" in flat:
        anns = [dict(a) for a in (flat["annotations"] or [])]
    for key in sorted(flat):
        m = re.fullmatch(r"annotations\[(\d+)\](?:\.(.+))?", key)
        if not m:
            continue
        i, prop = int(m.group(1)), m.group(2)
        while len(anns) <= i:
            anns.append({})
        if prop is None:
            anns[i] = dict(flat[key]) if flat[key] else {"visible": False}
        else:
            cur = anns[i]
            parts = prop.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = flat[key]
    return vis, anns


def shown(v) -> bool:
    """Trace `visible`: True shows it; False and "legendonly" hide it."""
    return v is True or v is None


def visible_annotations(anns) -> list:
    return [a for a in anns if a.get("visible", True) is not False]


def ann_key(a: dict):
    return (strip_markup(str(a.get("text", ""))).strip(), a.get("x"), a.get("y"))


def sets_annotations(button: dict) -> bool:
    _, layout_upd, _ = button_parts(button)
    return any(re.match(r"annotations", k) for k in _flatten(layout_upd))


def legend_names(fig: dict) -> list[str]:
    return [t.get("name") for t in fig["data"]
            if t.get("showlegend", True) is not False and isinstance(t.get("name"), str)]


def visible_texts(fig: dict, page: Page | None = None) -> list[tuple[str, str]]:
    """(where, text) for every piece of visible text (SPEC 10.4); hover text excluded."""
    lay = fig["layout"]
    out = []
    t = text_of(lay.get("title"))
    if t:
        out.append(("figure title", t))
    for ax in axis_keys(lay):
        t = text_of((lay.get(ax) or {}).get("title"))
        if t:
            out.append((f"{ax} title", t))
    t = text_of(((lay.get("legend") or {}).get("title")))
    if t:
        out.append(("legend title", t))
    for name in legend_names(fig):
        out.append(("legend name", name))
    for tr in fig["data"]:
        t = text_of(tr.get("legendgrouptitle"))
        if t:
            out.append(("legend group title", t))
    for a in lay.get("annotations") or []:
        if isinstance(a.get("text"), str):
            out.append(("annotation", a["text"]))
    for mi, menu in enumerate(lay.get("updatemenus") or []):
        for bi, b in enumerate(menu.get("buttons") or []):
            if isinstance(b.get("label"), str):
                out.append((f"menu {mi} label", b["label"]))
            data_upd, layout_upd, _ = button_parts(b)
            names = data_upd.get("name")
            for nm in (names if isinstance(names, list) else [names]):
                if isinstance(nm, str):
                    out.append((f"menu {mi} button {bi} legend name", nm))
            for key, val in _flatten(layout_upd).items():
                if key == "annotations":
                    for a in val or []:
                        if isinstance(a, dict) and isinstance(a.get("text"), str):
                            out.append((f"menu {mi} button {bi} annotation", a["text"]))
                elif re.fullmatch(r"annotations\[\d+\]", key) and isinstance(val, dict):
                    if isinstance(val.get("text"), str):
                        out.append((f"menu {mi} button {bi} annotation", val["text"]))
                elif re.fullmatch(r"annotations\[\d+\]\.text", key) and isinstance(val, str):
                    out.append((f"menu {mi} button {bi} annotation", val))
                elif re.search(r"(^|\.)title(\.text)?$", key):
                    t = text_of(val)
                    if t:
                        out.append((f"menu {mi} button {bi} {key}", t))
    for si, slider in enumerate(lay.get("sliders") or []):
        for step in slider.get("steps") or []:
            if isinstance(step.get("label"), str):
                out.append((f"slider {si} label", step["label"]))
    if page is not None and page.body_text:
        out.append(("page caption", page.body_text))
    return out


def numeric(values):
    """A trace coordinate (list or Plotly typed array) as a float array; None when absent."""
    if values is None:
        return None
    if isinstance(values, dict):
        arrs = list(typed_arrays(values))
        assert arrs and arrs[0] is not None, "undecodable typed array"
        return np.asarray(arrs[0], dtype=float)
    flat = []
    for v in np.asarray(values, dtype=object).ravel():
        if v is None:
            continue
        try:
            flat.append(float(v))
        except (TypeError, ValueError):
            pass
    return np.asarray(flat, dtype=float)
