"""SPEC.md 10.2: figkit/systems/numsys_math.js agrees with number_systems.py.

node loads the file with `require`; Python is the oracle (number_systems.Algebra for rho and kind,
numpy.polyval for |p(re + i*im)|). 50 seeded random points per kind, plus points inside and just outside
the dual tolerance, plus a second polynomial so a JS file that hard-codes COEFFS cannot agree.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.tests._iic_helpers import (  # noqa: E402
    Lazy, load, spec_float, spec_literal,
)
from experiments.ideas_intelligence_creativity.tests._fk_helpers import (  # noqa: E402
    SYSTEMS_DIR, load_results, require_node, run_node,
)

JS_FILE = SYSTEMS_DIR / "numsys_math.js"
FUNCS = ("absPolyComplex", "rho", "kindOf")
REL = 1e-9
N_PER_KIND = 50
KINDS = ("split", "dual", "complex")
COEFFS = [float(c) for c in spec_literal("COEFFS", 3)]
OTHER_COEFFS = [1.5, -2.0, 0.25, 3.0, -1.0, 2.0]          # degree 5, so the degree is not hard-coded either
DUAL_TOL = spec_float(r"\|delta\| ≤ (1e-\d+)·max", 3)

DRIVER = r"""
const fs = require('fs'); const vm = require('vm');
const [modPath, casesPath, outPath] = process.argv.slice(2);
const out = { types: null, error: null, results: [], plain: null };
try {
  const m = require(modPath);
  out.types = {};
  for (const f of ['absPolyComplex', 'rho', 'kindOf']) out.types[f] = typeof m[f];
  const cases = JSON.parse(fs.readFileSync(casesPath, 'utf8'));
  for (const c of cases) {
    let v;
    try {
      v = m[c.fn](...c.args);
      if (typeof v === 'number' && !Number.isFinite(v)) v = { nonfinite: String(v) };
    } catch (e) { v = { error: String(e) }; }
    out.results.push(v);
  }
} catch (e) { out.error = String(e && e.stack || e); }
try {
  // a plain browser script: no module, no require; the functions become globals
  const sandbox = {}; sandbox.window = sandbox; sandbox.self = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(modPath, 'utf8'), sandbox, { filename: 'numsys_math.js' });
  const types = vm.runInContext('[typeof absPolyComplex, typeof rho, typeof kindOf]', sandbox);
  out.plain = { types: Array.from(types) };
  if (types[0] === 'function') {
    out.plain.value = vm.runInContext('absPolyComplex([1, 0, 1], 0.5, 2)', sandbox);
  }
} catch (e) { out.plain = { error: String(e) }; }
fs.writeFileSync(outPath, JSON.stringify(out));
"""


def _sample_points():
    """[(kind, alpha, beta, a, b)], N_PER_KIND per kind, kinds decided by the Python oracle."""
    ns = load("number_systems")
    rng = np.random.default_rng(20261026)
    pts = {k: [] for k in KINDS}
    while len(pts["split"]) < N_PER_KIND or len(pts["complex"]) < N_PER_KIND:
        alpha, beta = rng.uniform(-1.0, 5.0), rng.uniform(-4.0, 4.0)
        kind = ns.Algebra(alpha, beta).kind
        if kind in ("split", "complex") and len(pts[kind]) < N_PER_KIND:
            pts[kind].append((alpha, beta, rng.uniform(-3.0, 4.0), rng.uniform(-2.0, 2.0)))
    for _ in range(N_PER_KIND):                          # dual: on the parabola beta^2 + 4 alpha = 0
        beta = rng.uniform(-4.0, 4.0)
        pts["dual"].append((-beta * beta / 4.0, beta, rng.uniform(-3.0, 4.0), rng.uniform(-2.0, 2.0)))
    for kind, alpha, beta in [("dual", p[0], p[1]) for p in pts["dual"]]:
        assert ns.Algebra(alpha, beta).kind == kind, "oracle sanity: a point on the parabola is not dual"
    return [(k, *p) for k in KINDS for p in pts[k]]


def _tolerance_points():
    """Points inside and just outside the dual tolerance (relative, per SPEC 3), both signs of delta."""
    rng = np.random.default_rng(7)
    out = []
    for _ in range(20):
        beta = rng.uniform(1.0, 4.0) * rng.choice([-1.0, 1.0])
        scale = max(1.0, beta * beta)
        for factor in (0.5, -0.5, 3.0, -3.0):
            alpha = (-beta * beta + factor * DUAL_TOL * scale) / 4.0
            out.append(("near-dual", alpha, beta, rng.uniform(-3.0, 4.0), rng.uniform(-2.0, 2.0)))
    return out


def _cases():
    ns = load("number_systems")
    if not JS_FILE.is_file():
        pytest.fail(f"{JS_FILE.name} does not exist (SPEC 10.2)", pytrace=False)
    points = _sample_points() + _tolerance_points()
    cases, expect = [], []
    for coeffs in (COEFFS, OTHER_COEFFS):
        for kind, alpha, beta, a, b in points:
            alg = ns.Algebra(alpha, beta)
            cases.append({"fn": "absPolyComplex", "args": [coeffs, a, b]})
            expect.append(("absPolyComplex", kind, (alpha, beta, a, b), coeffs,
                           abs(complex(np.polyval(coeffs, complex(a, b))))))
            cases.append({"fn": "rho", "args": [coeffs, alpha, beta, a, b]})
            expect.append(("rho", kind, (alpha, beta, a, b), coeffs,
                           float(np.asarray(alg.rho((a, b), coeffs)))))
            cases.append({"fn": "kindOf", "args": [alpha, beta]})
            expect.append(("kindOf", kind, (alpha, beta), coeffs, alg.kind))
    results = load_results("number_systems_results.json")
    m = float(results["polynomial"]["floor"]["m"])
    roots = [(float(r), float(i)) for r, i in results["polynomial"]["roots"]]
    for re_, im in roots:
        cases.append({"fn": "absPolyComplex", "args": [COEFFS, re_, im]})
        expect.append(("root", "root", (re_, im), COEFFS, None))
    return cases, expect, m, roots


def _run(tmp_path_factory):
    require_node()
    cases, expect, m, roots = _cases()
    work = tmp_path_factory.mktemp("numsys_js")
    (work / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    proc = run_node(DRIVER, [JS_FILE.resolve(), work / "cases.json", work / "out.json"], work)
    assert proc.returncode == 0, f"node driver failed:\n{proc.stderr[-2000:]}"
    out = json.loads((work / "out.json").read_text(encoding="utf-8"))
    assert out["error"] is None, f"require() of numsys_math.js failed:\n{out['error']}"
    return {"out": out, "expect": expect, "m": m, "roots": roots}


@pytest.fixture(scope="module")
def js(tmp_path_factory):
    return Lazy(lambda: _run(tmp_path_factory))


def _compare(js, fn):
    got = js.get()
    out, expect = got["out"], got["expect"]
    assert len(out["results"]) == len(expect)
    checked, bad = {k: 0 for k in KINDS + ("near-dual",)}, []
    for res, (name, kind, where, coeffs, want) in zip(out["results"], expect):
        if name != fn:
            continue
        checked[kind] += 1
        if fn == "kindOf":
            if res != want:
                bad.append((kind, where, coeffs is COEFFS, res, want))
        elif not isinstance(res, (int, float)) or isinstance(res, bool) or abs(res - want) > REL * abs(want):
            bad.append((kind, where, "COEFFS" if coeffs is COEFFS else "other", res, want))
    for kind in KINDS:
        assert checked[kind] >= 2 * N_PER_KIND, f"only {checked[kind]} {kind} points checked for {fn}"
    assert bad == [], f"{fn}: {len(bad)} disagreements with Python, e.g. {bad[:4]}"


def test_require_exports_the_three_functions(js):
    types = js.get()["out"]["types"]
    assert types == {f: "function" for f in FUNCS}, types


def test_it_is_a_plain_script_that_defines_globals_without_module(js):
    plain = js.get()["out"]["plain"]
    assert "error" not in plain, f"loading as a plain script (no module, no require) failed: {plain}"
    assert plain["types"] == ["function"] * 3, plain
    assert abs(plain["value"] - abs(complex(np.polyval([1, 0, 1], complex(0.5, 2))))) <= 1e-12 * 3.75


def test_absPolyComplex_agrees_with_python(js):
    _compare(js, "absPolyComplex")


def test_rho_agrees_with_python_in_every_kind(js):
    _compare(js, "rho")


def test_kindOf_agrees_with_python_including_the_dual_tolerance(js):
    _compare(js, "kindOf")
    got = js.get()
    near = [(res, want) for res, (name, kind, *_rest, want) in zip(got["out"]["results"], got["expect"])
            if name == "kindOf" and kind == "near-dual"]
    assert {w for _, w in near} == {"dual", "split", "complex"}, "the tolerance points do not straddle it"


def test_absPolyComplex_vanishes_at_the_four_roots(js):
    got = js.get()
    m, roots = got["m"], got["roots"]
    assert len(roots) == 4, roots
    vals = [res for res, (name, *_rest) in zip(got["out"]["results"], got["expect"]) if name == "root"]
    assert len(vals) == 4
    for (re_, im), v in zip(roots, vals):
        assert isinstance(v, (int, float)) and not isinstance(v, bool), v
        assert 0 <= v < 1e-9 * m, f"|p({re_}{im:+}i)| = {v} is not below 1e-9*m = {1e-9 * m}"
