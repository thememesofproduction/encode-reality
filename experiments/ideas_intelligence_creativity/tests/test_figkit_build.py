"""SPEC.md 10.1: figkit/build.py -- build_instrument writes one self-contained page per instrument.

Every string the spec declares (the embed meta, the allowed stylesheets, the privacy-guard tokens) is
parsed from SPEC.md. The system used for most builds is `numsys_math`: 10.2 puts that file at
figkit/systems/numsys_math.js, so by 10.1 it is a valid system name.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.tests._iic_helpers import (  # noqa: E402
    EXP_DIR, PY, REPO, load, tree_digest,
)
from experiments.ideas_intelligence_creativity.tests._fk_helpers import (  # noqa: E402
    FIGKIT_DIR, SYSTEMS_DIR, Page, caption_text, node_check, read_page, require_node, run_node, same_json,
    spec_allowed_stylesheets, spec_embed_meta, spec_privacy_tokens,
)

SYSTEM = "numsys_math"
ALLOWED_CSS = spec_allowed_stylesheets()
META_NAME, META_CONTENT = spec_embed_meta()

TITLE = 'Roots & "lines" <off the line>'
ARIA = 'Plane of re & im, "drag" a searcher'
CAPTION = 'Live: the quillwort searchers move here. Recorded: the map <b>bold</b> & more.'
CAPTION_MARKER = "quillwort"

HEX39 = ("0123456789abcdef" * 3)[:39]
CLEAN = {
    "coeffs": [4.0, -8.0, -7.0, 20.0, 25.0],
    "m": 9.770194456419759,
    "walkers": 128,
    "flags": [True, False, None],
    "nested": {"tiny": 1e-300, "big": 1.7976931348623157e308, "neg": -2.5, "zero": 0.0, "int": 0,
               "grid": [[0, 1, 2], [2, 1, 0]]},
    "label": "Searchers on the line: α, β, ρ ≥ 0 ✓",
    "closing": "</script><script>window.__pwned = 1</script>",
    "upper": "</SCRIPT >",
    "bare": "</",
    "</script>": "a key that would close the element",
    "line_sep": "a b c",
    "near_misses": ["srv", "home page", HEX39, "2026-10-26", "14:05", "5 @ 2"],
}


def build_fn():
    mod = load("figkit.build")
    fn = getattr(mod, "build_instrument", None)
    if not callable(fn):
        pytest.fail("figkit.build defines no build_instrument", pytrace=False)
    return fn


def build(out: Path, data=None, *, system=SYSTEM, title=TITLE, aria_label=ARIA, caption=CAPTION, **extra):
    kwargs = dict(title=title, aria_label=aria_label, **extra)
    if caption is not None:
        kwargs["caption"] = caption
    return build_fn()(system, CLEAN if data is None else data, out, **kwargs)


def built(tmp_path, name="page.html", **kw) -> Page:
    out = tmp_path / name
    build(out, **kw)
    assert out.is_file(), f"build_instrument did not write {out.name}"
    return read_page(out)


def discovered_systems() -> list:
    assert SYSTEMS_DIR.is_dir(), f"{SYSTEMS_DIR.relative_to(EXP_DIR)} does not exist"
    names = sorted(p.stem for p in SYSTEMS_DIR.glob("*.js"))
    assert SYSTEM in names, f"figkit/systems/{SYSTEM}.js (SPEC 10.2) not found; systems: {names}"
    return names


def html_outside_scripts(page: Page) -> str:
    out, pos = [], 0
    for s in page.scripts:
        start = page.html.index(">", s["offset"]) + 1
        out.append(page.html[pos:start])
        pos = start + len(s["content"])
    out.append(page.html[pos:])
    return "".join(out)


# ----------------------------------------------------------------------------- the page

def test_writes_one_file_at_out_and_returns_its_path(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = out_dir / "instrument.html"
    before = tree_digest(EXP_DIR)
    ret = build(out)
    after = tree_digest(EXP_DIR)
    assert isinstance(ret, Path), f"build_instrument returned {type(ret).__name__}, not a Path"
    assert ret.resolve() == out.resolve()
    assert sorted(p.name for p in out_dir.iterdir()) == ["instrument.html"], "more than one file written"
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    assert changed == [], f"building a page changed the experiment dir: {changed[:10]}"
    assert "<html" in out.read_text(encoding="utf-8").lower()


def test_no_src_and_no_http_anywhere(tmp_path):
    page = built(tmp_path)
    assert "src=" not in page.html.lower(), "the page contains src="
    with_src = [el["tag"] for el in page.elements if "src" in el["attrs"]]
    assert with_src == [], f"elements with a src attribute: {with_src}"
    urls = re.findall(r"https?://[^\s\"'<>)]*", page.html, re.I)
    assert urls == [], f"http(s) URLs in the page: {urls[:5]}"


def test_the_only_links_are_the_two_same_origin_stylesheets(tmp_path):
    assert len(ALLOWED_CSS) == 2, ALLOWED_CSS
    page = built(tmp_path)
    links = [el["attrs"].get("href") for el in page.elements if el["tag"] == "link"]
    assert all(h in ALLOWED_CSS for h in links), f"<link> hrefs: {links}"
    hrefs = [(el["tag"], el["attrs"]["href"]) for el in page.elements if "href" in el["attrs"]]
    bad = [(t, h) for t, h in hrefs if not (h in ALLOWED_CSS or h.startswith("#"))]
    assert bad == [], f"href to something other than the allowed stylesheets: {bad}"
    markup = html_outside_scripts(page)
    imports = re.findall(r"@import\s+(?:url\()?\s*['\"]?([^'\")\s;]+)", markup)
    assert all(i in ALLOWED_CSS for i in imports), f"@import targets: {imports}"
    css_urls = re.findall(r"url\(\s*['\"]?([^'\")\s]+)", markup)
    assert all(u.startswith(("data:", "#")) for u in css_urls), f"url() references: {css_urls}"


def test_embed_meta_exactly_once_and_in_head(tmp_path):
    page = built(tmp_path)
    assert page.saw_head, "no <head> element"
    metas = page.find("meta", name=META_NAME)
    assert len(metas) == 1, f'expected one <meta name="{META_NAME}">, found {len(metas)}'
    assert metas[0]["attrs"].get("content") == META_CONTENT, metas[0]["attrs"]
    assert metas[0]["in_head"], "the embed meta is not inside <head>"
    raw = re.findall(rf"<meta\b[^>]*{re.escape(META_NAME)}", page.html, re.I)
    assert len(raw) == 1, f"the embed meta appears {len(raw)} times in the file"


def test_title_is_the_argument_html_escaped(tmp_path):
    other = "Choose a number system"
    for name, title in (("a.html", TITLE), ("b.html", other)):
        page = built(tmp_path, name=name, title=title)
        titles = page.find("title")
        assert len(titles) == 1, f"expected one <title>, found {len(titles)}"
        assert titles[0]["in_head"], "<title> is not in <head>"
        assert titles[0]["text"] == title, (titles[0]["text"], title)
    assert "<off the line>" not in (tmp_path / "a.html").read_text(encoding="utf-8"), "title not HTML-escaped"


THEME_DRIVER = r"""
const vm = require('vm'); const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
function run(stored, mode) {
  const dataset = {};
  const documentElement = {
    dataset, style: {},
    setAttribute(n, v) { if (n === 'data-theme') dataset.theme = String(v); },
    getAttribute(n) { return n === 'data-theme' && dataset.theme !== undefined ? dataset.theme : null; },
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
  };
  const sandbox = { document: { documentElement }, console };
  const store = {};
  if (stored !== undefined) store['er-theme'] = stored;
  const ls = Object.assign({
    getItem(k) { return Object.prototype.hasOwnProperty.call(store, k) ? store[k] : null; },
    setItem() {}, removeItem() {},
  }, store);
  if (mode === 'throws') {
    Object.defineProperty(sandbox, 'localStorage', { get() { throw new Error('SecurityError: storage denied'); } });
  } else {
    sandbox.localStorage = ls;
  }
  sandbox.window = sandbox; sandbox.self = sandbox;
  sandbox.matchMedia = () => ({ matches: false, addEventListener() {}, addListener() {} });
  vm.createContext(sandbox);
  try { vm.runInContext(src, sandbox, { timeout: 2000 }); }
  catch (e) { return { threw: String(e) }; }
  return { theme: dataset.theme === undefined ? null : dataset.theme };
}
console.log(JSON.stringify({
  paper: run('paper'), ink: run('ink'), bogus: run('neon'), none: run(undefined), throws: run('ink', 'throws'),
}));
"""


def theme_behaviour(source: str, workdir: Path) -> dict:
    (workdir / "candidate.js").write_text(source, encoding="utf-8")
    proc = run_node(THEME_DRIVER, [workdir / "candidate.js"], workdir)
    assert proc.returncode == 0, f"theme driver failed: {proc.stderr[-1500:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_theme_boot_script_comes_before_any_stylesheet_and_sets_the_stored_theme(tmp_path):
    require_node()
    page = built(tmp_path)
    sheets = page.stylesheets()
    assert sheets, "no stylesheet at all: kit.css is to be inlined"
    first_sheet = min(s["offset"] for s in sheets)
    candidates = [s for s in page.js_scripts() if s["in_head"] and s["offset"] < first_sheet and s["content"].strip()]
    assert candidates, "no inline script in <head> before the first stylesheet"
    report = []
    for i, s in enumerate(candidates):
        work = tmp_path / f"cand{i}"
        work.mkdir()
        got = theme_behaviour(s["content"], work)
        ok = (got["paper"].get("theme") == "paper" and got["ink"].get("theme") == "ink"
              and "threw" not in got["bogus"] and got["bogus"].get("theme") != "neon"
              and "threw" not in got["none"] and "threw" not in got["throws"])
        if ok:
            return
        report.append(got)
    pytest.fail("no script before the first stylesheet sets dataset.theme from localStorage['er-theme'] "
                f"(paper/ink only, storage failure caught): {report}", pytrace=False)


# ----------------------------------------------------------------------------- data

ROUND_TRIP = {
    "clean": CLEAN,
    "empty": {},
    "only closers": {"a": "</", "b": ["</script>", "</style>", "</ScRiPt>"], "</": {"</x": "</"}},
}


@pytest.mark.parametrize("case", list(ROUND_TRIP))
def test_data_is_one_json_script_that_parses_back_exactly(tmp_path, case):
    data = ROUND_TRIP[case]
    page = built(tmp_path, data=data)
    blocks = page.data_scripts()
    assert len(blocks) == 1, f'expected one <script type="application/json" data-fk="data">, found {len(blocks)}'
    text = blocks[0]["content"]
    assert "</" not in text, "an unescaped '</' inside the data"
    got = json.loads(text)
    assert same_json(got, data), "the embedded data does not parse back to exactly `data`"


def test_data_can_never_close_its_script_early(tmp_path):
    hostile = {"s": "</script><script>window.__pwned = 1</script><p data-fk=\"caption\">fake</p>"}
    benign = {"s": "plain text of the same kind"}
    a = built(tmp_path, name="hostile.html", data=hostile)
    b = built(tmp_path, name="benign.html", data=benign)
    assert [e["tag"] for e in a.elements] == [e["tag"] for e in b.elements], \
        "the data changed the page's element structure"
    assert "__pwned" not in "".join(s["content"] for s in a.js_scripts())


NON_FINITE = {
    "nan at top level": lambda v: {"x": v},
    "inside a list": lambda v: {"x": [1.0, 2.0, v]},
    "nested dict": lambda v: {"x": {"y": {"z": v}}},
    "list of lists": lambda v: {"x": [[0.0], [1.0, v]]},
    "dict in list": lambda v: {"x": [{"y": v}]},
}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("where", list(NON_FINITE))
def test_non_finite_numbers_raise(tmp_path, where, value):
    shape = NON_FINITE[where]
    build(tmp_path / "control.html", data=shape(1.5))       # the same shape with a finite value builds
    with pytest.raises(ValueError):
        build(tmp_path / "bad.html", data=shape(value))


# ----------------------------------------------------------------------------- accessibility

def test_aria_label_is_on_the_stage_and_follows_the_argument(tmp_path):
    other = "Map of number systems"
    a = built(tmp_path, name="a.html", aria_label=ARIA)
    b = built(tmp_path, name="b.html", aria_label=other)
    assert a.find(None, aria_label=ARIA), "no element carries the given aria-label"
    assert not a.find(None, aria_label=other)
    assert b.find(None, aria_label=other), "the aria-label did not follow the argument"
    assert not b.find(None, aria_label=ARIA)


def test_caption_appears_exactly_once_html_escaped(tmp_path):
    page = built(tmp_path, caption=CAPTION)
    assert caption_text(page) == CAPTION
    assert page.html.count(CAPTION_MARKER) == 1, f"the caption appears {page.html.count(CAPTION_MARKER)} times"
    assert "<b>bold</b>" not in page.html, "the caption is not HTML-escaped"


def test_no_caption_element_with_text_when_caption_is_not_given(tmp_path):
    page = built(tmp_path, caption=None)
    texts = [(el["text"] or "").strip() for el in page.find("p", data_fk="caption")]
    assert all(t == "" for t in texts), texts


# ----------------------------------------------------------------------------- systems

def test_unknown_system_raises_and_every_known_system_builds(tmp_path):
    for name in discovered_systems():
        build(tmp_path / f"{name}.html", system=name)
        assert (tmp_path / f"{name}.html").is_file()
    for bad in ("no_such_system", "numsys_math_typo", ""):
        with pytest.raises(ValueError):
            build(tmp_path / "bad.html", system=bad)


def _code_lines(path: Path) -> list:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith(("//", "/*", "*", "*/")):
            continue
        lines.append(s)
    return lines


def test_kit_and_system_files_are_inlined(tmp_path):
    kit_js, kit_css = FIGKIT_DIR / "kit.js", FIGKIT_DIR / "kit.css"
    assert kit_js.is_file() and kit_css.is_file(), "figkit/kit.js and figkit/kit.css must exist"
    for name in discovered_systems():
        page = built(tmp_path, name=f"{name}.html", system=name)
        for src in (kit_js, kit_css, SYSTEMS_DIR / f"{name}.js"):
            lines = _code_lines(src)
            assert lines, f"{src.name} is empty"
            missing = [ln for ln in lines if ln not in page.html and ln.replace("</", "<\\/") not in page.html]
            assert missing == [], f"{src.name} not inlined in the {name} page; e.g. {missing[:3]}"


# ----------------------------------------------------------------------------- the privacy guard

def _privacy_cases() -> list:
    sensitive, insensitive = spec_privacy_tokens()
    cases = []
    for i, tok in enumerate(sensitive):
        cases.append((f"literal-{i}", f"see {tok}x for details"))
    for i, tok in enumerate(insensitive):
        cases.append((f"name-{i}-lower", f"by {tok.lower()} here"))
        cases.append((f"name-{i}-upper", f"by {tok.upper()} here"))
        cases.append((f"name-{i}-mixed", f"by {tok[0].upper() + tok[1:]} here"))
    cases.append(("email", "write to reader" + "@" + "example.org today"))
    cases.append(("iso-time-plus-offset", "at 2026-10-26T14:05:09" + "+02:00 local"))
    cases.append(("iso-time-minus-offset", "at 2026-10-26T14:05:09.250" + "-05:00 local"))
    cases.append(("hex-40", "commit " + ("0123456789abcdef" * 3)[:40] + " was used"))
    return cases


PRIVACY = _privacy_cases()
SITES = ("data", "title", "caption", "aria_label")


def _site_kwargs(site: str, text: str) -> dict:
    if site == "data":
        return {"data": dict(CLEAN, leak=text)}
    return {site: text}


@pytest.mark.parametrize("label,text", PRIVACY, ids=[c[0] for c in PRIVACY])
def test_privacy_guard_raises_on_each_blocked_pattern(tmp_path, label, text):
    for site in SITES:
        build(tmp_path / f"control_{site}.html", **_site_kwargs(site, "nothing private here"))
        with pytest.raises(ValueError):
            build(tmp_path / f"leak_{site}.html", **_site_kwargs(site, text))


def test_privacy_guard_lets_a_clean_page_through(tmp_path):
    out = tmp_path / "clean.html"
    build(out, data=CLEAN)
    page = read_page(out)
    assert same_json(json.loads(page.data_scripts()[0]["content"]), CLEAN)


# ----------------------------------------------------------------------------- determinism

_SUBPROCESS_BUILD = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from experiments.ideas_intelligence_creativity.figkit.build import build_instrument
spec = json.loads(open(sys.argv[2], encoding="utf-8").read())
build_instrument(spec["system"], spec["data"], Path(sys.argv[3]), title=spec["title"],
                 aria_label=spec["aria_label"], caption=spec["caption"])
"""


def test_same_arguments_give_byte_identical_files(tmp_path):
    out = tmp_path / "same.html"
    build(out)
    first = out.read_bytes()
    build(out)
    assert out.read_bytes() == first, "two builds with the same arguments differ"
    args = tmp_path / "args.json"
    args.write_text(json.dumps({"system": SYSTEM, "data": CLEAN, "title": TITLE, "aria_label": ARIA,
                                "caption": CAPTION}), encoding="utf-8")
    script = tmp_path / "sub_build.py"
    script.write_text(_SUBPROCESS_BUILD, encoding="utf-8")
    for seed in ("0", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([PY, str(script), str(REPO), str(args), str(out)], env=env,
                              capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr[-1500:]
        assert out.read_bytes() == first, f"a fresh process (PYTHONHASHSEED={seed}) built different bytes"
    build(out, data=dict(CLEAN, m=9.77))
    assert out.read_bytes() != first, "different data gave the same file"


# ----------------------------------------------------------------------------- valid JavaScript

def test_every_inline_script_passes_node_check(tmp_path):
    require_node()
    # negative control: the probe itself can fail
    assert node_check("function (", tmp_path, "broken").returncode != 0
    assert node_check("var ok = 1;", tmp_path, "fine").returncode == 0
    checked = 0
    for name in discovered_systems():
        page = built(tmp_path, name=f"{name}.html", system=name)
        for i, s in enumerate(page.js_scripts()):
            if not s["content"].strip():
                continue
            module = s["attrs"].get("type", "").strip().lower() == "module"
            proc = node_check(s["content"], tmp_path, f"{name}_{i}", module=module)
            assert proc.returncode == 0, f"{name} page, script {i}: node --check failed\n{proc.stderr[-1500:]}"
            checked += 1
    assert checked >= 1, "no inline JavaScript was checked"


# SPEC 10.1: names that must stay private are not written into the repo. They come from FIGKIT_PRIVATE_NAMES.
def test_privacy_guard_blocks_the_names_in_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("FIGKIT_PRIVATE_NAMES", raising=False)
    build(tmp_path / "before.html", data=dict(CLEAN, who="by Qzxname and Vwkother"))
    monkeypatch.setenv("FIGKIT_PRIVATE_NAMES", " qzxname , VWKOTHER ,")
    for i, text in enumerate(("by QZXNAME", "by vwkother", "by Qzxname")):
        with pytest.raises(ValueError):
            build(tmp_path / f"leak{i}.html", data=dict(CLEAN, who=text))
    build(tmp_path / "after.html", data=dict(CLEAN, who="by nobody private"))

