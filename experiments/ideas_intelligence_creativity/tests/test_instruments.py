"""SPEC.md 10.3: instruments.py -- the data for the two instruments, the pages, and the runner.

The oracle for every field is the recorded results JSON and the constants the spec names
(experiment3.HC_SIGMA, START_LO/START_HI, L2_ALPHA/L2_BETA, L2_CHILD_EVALS,
plots_number_systems.FLOOR_TOL, ROOT_TOL). Config-perturbation tests change the results and require the
output to move with them, so a function that returns the recorded values as literals cannot pass.
"""
import copy
import importlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.tests._iic_helpers import (  # noqa: E402
    PKG, list_files, load, require_ok, spec_float, spec_section, strict_json,
)
from experiments.ideas_intelligence_creativity.tests._fk_helpers import (  # noqa: E402
    caption_text, jargon_hits, load_results, page_data, plain, read_page, spec_embed_meta, spec_sub,
)

RESULTS = load_results("number_systems_results.json")
SEC = spec_sub("10.3")
PAGE_NAMES = re.findall(r"`(instrument_\w+\.html)`", SEC)
LINE_PAGE, ALG_PAGE = PAGE_NAMES[0], PAGE_NAMES[1]
VIEW = json.loads(re.search(r"`view`: `(\{.*?\})`", SEC).group(1))
WALKERS = int(re.search(r"`walkers`: (\d+)", SEC).group(1))
ES_NAMES = re.findall(r"`(ES_[A-Z0-9_]+)`", SEC)
EXEMPLAR_KEYS = re.findall(r"`(\w+)`", re.search(r"`exemplar`: (.*?), unchanged", SEC, re.S).group(1))
DUAL_TOL = spec_float(r"\|delta\| ≤ (1e-\d+)·max", 3)
META_NAME, _ = spec_embed_meta()


def _spec_box():
    m = re.search(r"alpha ∈ \[([^\]]+)\], beta ∈ \[([^\]]+)\]", spec_section(4))
    to = lambda s: [float(v.replace("−", "-")) for v in s.split(",")]  # noqa: E731
    return {"alpha": to(m.group(1)), "beta": to(m.group(2))}


def test_spec_parse():
    assert PAGE_NAMES == ["instrument_line.html", "instrument_algebras.html"], PAGE_NAMES
    assert VIEW == {"re": [-3.0, 4.0], "im": [-2.0, 2.0]} and WALKERS == 128
    assert ES_NAMES == ["ES_SIGMA0", "ES_UP", "ES_DOWN", "ES_SIGMA_MIN"], ES_NAMES
    assert EXEMPLAR_KEYS == ["proposals", "elite_path", "adoption_algebra", "adoption_eval"], EXEMPLAR_KEYS
    assert _spec_box() == {"alpha": [-1.0, 5.0], "beta": [-4.0, 4.0]}
    load("instruments")


def inst():
    mod = load("instruments")
    for fn in ("line_data", "algebras_data", "write_instruments"):
        if not callable(getattr(mod, fn, None)):
            pytest.fail(f"instruments.{fn} is missing", pytrace=False)
    return mod


def const(name: str, modules: tuple):
    """A constant the spec names, looked up in the module(s) it belongs to; all definitions must agree."""
    found = {}
    for modname in modules:
        try:
            mod = importlib.import_module(f"{PKG}.{modname}")
        except Exception:  # noqa: BLE001
            continue
        if hasattr(mod, name):
            found[modname] = plain(getattr(mod, name))
    if not found:
        pytest.fail(f"{name} is not defined in {' or '.join(modules)}", pytrace=False)
    vals = list(found.values())
    assert all(v == vals[0] for v in vals), f"{name} differs between modules: {found}"
    return vals[0]


def line_data(results):
    return plain(inst().line_data(copy.deepcopy(results)))


def algebras_data(results):
    return plain(inst().algebras_data(copy.deepcopy(results)))


# ----------------------------------------------------------------------------- line_data

def expected_line(results) -> dict:
    poly = results["polynomial"]
    lo = const("START_LO", ("experiment3", "number_systems"))
    hi = const("START_HI", ("experiment3", "number_systems"))
    return {
        "coeffs": poly["coeffs"],
        "m": poly["floor"]["m"],
        "x_star": poly["floor"]["x_star"],
        "minima": [[mm["x"], mm["abs_p"]] for mm in results["real_line"]["minima"]],
        "roots": [list(r) for r in poly["roots"]],
        "sigma": const("HC_SIGMA", ("experiment3",)),
        "view": VIEW,
        "start": [max(float(lo), VIEW["re"][0]), min(float(hi), VIEW["re"][1])],
        "walkers": WALKERS,
    }


def assert_line(got: dict, want: dict):
    for key, value in want.items():
        assert key in got, f"line_data has no {key!r}; keys: {sorted(got)}"
        assert got[key] == value, f"line_data[{key!r}] = {got[key]!r}, expected {value!r}"
    assert isinstance(got.get("seed"), int) and not isinstance(got["seed"], bool), \
        f"line_data['seed'] is not an int: {got.get('seed')!r}"


def test_line_data_fields_match_the_results_and_constants():
    want = expected_line(RESULTS)
    assert want["start"][0] < want["start"][1], f"the start interval does not overlap the view: {want['start']}"
    assert_line(line_data(RESULTS), want)


def test_line_data_follows_the_results():
    r2 = copy.deepcopy(RESULTS)
    r2["polynomial"]["coeffs"] = [1.0, 0.0, 2.0, 0.0, 3.0]
    r2["polynomial"]["floor"]["m"] = 3.0
    r2["polynomial"]["floor"]["x_star"] = 0.0
    r2["polynomial"]["roots"] = [[0.5, 1.5], [0.5, -1.5], [-0.5, 1.0], [-0.5, -1.0]]
    r2["real_line"]["minima"] = [{"x": 0.25, "abs_p": 3.5}, {"x": -1.75, "abs_p": 4.25},
                                 {"x": 2.5, "abs_p": 7.0}]
    want = expected_line(r2)
    assert want["coeffs"] != expected_line(RESULTS)["coeffs"]
    assert_line(line_data(r2), want)


# ----------------------------------------------------------------------------- algebras_data

def _es_value(es: dict, name: str):
    short = name[3:].lower()
    camel = lambda s: re.sub(r"_([a-z0-9])", lambda m: m.group(1).upper(), s)  # noqa: E731
    keys = list(dict.fromkeys([name, name.lower(), short, camel(short), camel(name.lower())]))
    hits = [k for k in keys if k in es]
    assert len(hits) == 1, f"es has no single key for {name} (tried {keys}); keys: {sorted(es)}"
    return es[hits[0]]


def allowed_cats(v: float, floor_tol: float, root_tol: float, clip_lo) -> set:
    """Codes a recorded log10 ratio may take. Two readings of 'within FLOOR_TOL of a ratio of 1' are
    accepted (on the ratio, or on its log10); a ratio above the floor is 0 under both (it is neither
    'below the floor' nor 'a root')."""
    lr = math.log10(root_tol)
    if abs(v - lr) <= 1e-9 or (clip_lo is not None and v <= clip_lo + 1e-12 and lr < clip_lo):
        root = {True, False}
    else:
        root = {v < lr}
    out = set()
    for r in root:
        if r:
            out.add(2)
        else:
            for at_floor in (10.0 ** v >= 1 - floor_tol, v >= -floor_tol):
                out.add(0 if at_floor else 1)
    return out


def tolerances():
    floor_tol = float(const("FLOOR_TOL", ("plots_number_systems",)))
    root_tol = float(const("ROOT_TOL", ("plots_number_systems", "experiment3", "number_systems")))
    assert 0 < root_tol < 1 - floor_tol < 1, (root_tol, floor_tol)
    return floor_tol, root_tol


def check_cat(cat, log10_ratio, clip_lo):
    floor_tol, root_tol = tolerances()
    lr = np.asarray(log10_ratio, dtype=float)
    assert isinstance(cat, list) and all(isinstance(row, list) for row in cat), "cat is not a 2-D list"
    assert np.asarray(cat, dtype=object).shape == lr.shape, \
        f"cat shape {np.asarray(cat, dtype=object).shape} != log10_ratio shape {lr.shape}"
    bad = []
    for i, row in enumerate(cat):
        for j, c in enumerate(row):
            ok_type = isinstance(c, (int, float)) and not isinstance(c, bool) and c in (0, 1, 2)
            if not ok_type or c not in allowed_cats(lr[i, j], floor_tol, root_tol, clip_lo):
                bad.append(((i, j), lr[i, j], c, sorted(allowed_cats(lr[i, j], floor_tol, root_tol, clip_lo))))
    assert bad == [], f"{len(bad)} cells with the wrong code, e.g. (cell, log10 ratio, got, allowed) {bad[:5]}"


def test_algebras_data_fields_match_the_results_and_constants():
    d = algebras_data(RESULTS)
    amap = RESULTS["algebra_map"]
    assert d["alpha"] == amap["alpha"] and d["beta"] == amap["beta"], "axes differ from algebra_map"
    assert d["m"] == RESULTS["polynomial"]["floor"]["m"]
    box = {"alpha": list(const("L2_ALPHA", ("experiment3",))), "beta": list(const("L2_BETA", ("experiment3",)))}
    assert box == _spec_box(), f"experiment3's box {box} differs from SPEC 4"
    assert d["box"] == box, d.get("box")
    assert d["dual_tol"] == DUAL_TOL, d.get("dual_tol")
    try:
        ns_tol = importlib.import_module(f"{PKG}.number_systems").__dict__.get("DUAL_TOL")
    except Exception:  # noqa: BLE001
        ns_tol = None
    if ns_tol is not None:
        assert d["dual_tol"] == ns_tol
    consts = RESULTS["params"]["constants"]
    for name in ES_NAMES:
        assert _es_value(d["es"], name) == consts[name], name
    child = const("L2_CHILD_EVALS", ("experiment3",))
    assert d["es"]["child_evals"] == child == consts["L2_CHILD_EVALS"]
    for key in EXEMPLAR_KEYS:
        assert d["exemplar"][key] == RESULTS["level2"]["exemplar"][key], f"exemplar[{key!r}] changed"


def test_cat_codes_follow_the_ratio_rules_on_the_recorded_map():
    d = algebras_data(RESULTS)
    amap = RESULTS["algebra_map"]
    check_cat(d["cat"], amap["log10_ratio"], (amap.get("clip") or [None])[0])
    flat = [c for row in d["cat"] for c in row]
    assert {0, 2} <= set(flat), f"the recorded map should have floor and root cells; codes seen {set(flat)}"


def test_cat_follows_a_perturbed_map():
    floor_tol, root_tol = tolerances()
    v_root = math.log10(root_tol) - 2.0
    v_mid = (math.log10(root_tol) + math.log10(1 - floor_tol)) / 2.0
    r2 = copy.deepcopy(RESULTS)
    r2["algebra_map"] = {
        "alpha": [-1.0, -0.5, 2.0],
        "beta": [-0.5, 0.0],
        "clip": RESULTS["algebra_map"].get("clip"),
        "log10_ratio": [[v_root, v_mid], [v_mid, 0.0], [0.0, 0.5]],
    }
    d = algebras_data(r2)
    assert d["alpha"] == [-1.0, -0.5, 2.0] and d["beta"] == [-0.5, 0.0]
    assert d["cat"] == [[2, 1], [1, 0], [0, 0]], d["cat"]


def test_theory_nonnegative_delta_cells_are_never_roots():
    d = algebras_data(RESULTS)
    alpha = np.asarray(RESULTS["algebra_map"]["alpha"], dtype=float)
    beta = np.asarray(RESULTS["algebra_map"]["beta"], dtype=float)
    cat = np.asarray(d["cat"], dtype=float)
    delta = beta[None, :] ** 2 + 4 * alpha[:, None]
    assert cat.shape == delta.shape, (cat.shape, delta.shape)
    nonneg = delta >= 0
    assert nonneg.sum() > 0 and (~nonneg).sum() > 0
    assert (cat[~nonneg] == 2).any(), "no root cell anywhere in the delta < 0 region (vacuous check)"
    offenders = np.argwhere(nonneg & (cat == 2))
    assert offenders.size == 0, \
        f"cells with delta >= 0 coded as a root: {[(alpha[i], beta[j]) for i, j in offenders[:5]]}"


def test_algebras_data_follows_the_results():
    r2 = copy.deepcopy(RESULTS)
    r2["polynomial"]["floor"]["m"] = 4.5
    for name in ES_NAMES:
        r2["params"]["constants"][name] = r2["params"]["constants"][name] * 1.25 + 0.125
    ex = r2["level2"]["exemplar"]
    ex["proposals"] = ex["proposals"][:7]
    ex["elite_path"] = ex["elite_path"][:3]
    ex["adoption_algebra"] = [-0.75, 0.25]
    ex["adoption_eval"] = ex["adoption_eval"] + 17
    d = algebras_data(r2)
    assert d["m"] == 4.5
    for name in ES_NAMES:
        assert _es_value(d["es"], name) == r2["params"]["constants"][name], name
    for key in EXEMPLAR_KEYS:
        assert d["exemplar"][key] == ex[key], f"exemplar[{key!r}] did not follow the results"


# ----------------------------------------------------------------------------- write_instruments

def _written(tmp_path):
    plots = tmp_path / "plots"
    plots.mkdir()
    paths = inst().write_instruments(copy.deepcopy(RESULTS), plots)
    return plots, paths


def assert_page_carries(page_path: Path, want: dict):
    page = read_page(page_path)
    assert page.find("meta", name=META_NAME), f"{page_path.name} was not built by build_instrument"
    got = page_data(page)
    for key, value in want.items():
        assert key in got and got[key] == value, f"{page_path.name}: data[{key!r}] differs from the instrument data"


def test_write_instruments_writes_both_pages(tmp_path):
    plots, paths = _written(tmp_path)
    assert isinstance(paths, list) and all(isinstance(p, Path) for p in paths), paths
    assert sorted(p.name for p in paths) == sorted(PAGE_NAMES), paths
    assert all(p.resolve().parent == plots.resolve() for p in paths)
    assert list_files(plots) == sorted(PAGE_NAMES), list_files(plots)
    assert_page_carries(plots / LINE_PAGE, line_data(RESULTS))
    assert_page_carries(plots / ALG_PAGE, algebras_data(RESULTS))


def test_instrument_captions_use_no_jargon(tmp_path):
    plots, _ = _written(tmp_path)
    for name in PAGE_NAMES:
        cap = caption_text(read_page(plots / name))
        assert cap and cap.strip(), f"{name} has an empty caption"
        assert jargon_hits(cap) == [], f"{name} caption uses jargon {jargon_hits(cap)}: {cap!r}"


def test_instrument_captions_say_what_is_live_and_what_is_recorded(tmp_path):
    plots, _ = _written(tmp_path)
    for name in PAGE_NAMES:
        cap = caption_text(read_page(plots / name)).lower()
        assert "live" in cap and "record" in cap, f"{name} caption does not say what is live and what is recorded: {cap!r}"


# ----------------------------------------------------------------------------- the runner

def test_runner_writes_both_instruments_and_publishes_them(exp3_cli):
    res = exp3_cli.get()
    require_ok(res["r1"], "quick run")
    plots = res["r1_dir"] / "plots"
    for name in PAGE_NAMES:
        assert (plots / name).is_file(), f"{name} not in <output-dir>/plots: {list_files(res['r1_dir'])}"
    doc = strict_json((res["r1_dir"] / "number_systems_results.json").read_text(encoding="utf-8"))
    assert_page_carries(plots / LINE_PAGE, line_data(doc))
    assert_page_carries(plots / ALG_PAGE, algebras_data(doc))
    require_ok(res["pub"], "run with --publish-plots-dir")
    for name in PAGE_NAMES:
        assert name in res["published"], f"{name} was not copied to --publish-plots-dir: {sorted(res['published'])}"
        assert res["published"][name] == (res["pub_dir"] / "plots" / name).read_bytes(), name
