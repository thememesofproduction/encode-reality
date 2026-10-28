"""Shared helpers for the acceptance tests written from SPEC.md.

Not a test module (no ``test_`` prefix). Everything here is derived from SPEC.md; nothing here
knows how the experiment modules are implemented.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
EXP_DIR = TESTS_DIR.parent
REPO = EXP_DIR.parent.parent
SPEC_PATH = EXP_DIR / "SPEC.md"
PKG = "experiments.ideas_intelligence_creativity"
PY = sys.executable
PUBLISH_PREFIX = ".iic-publish-test-"
CLI_TIMEOUT = 240  # seconds per runner process; the spec's quick profile must take < 60 s
REFUSAL_TIMEOUT = 90  # a refusal happens before computing anything (spec 9.7), even under --profile full

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------------- imports

def load(name: str):
    """Import an experiment module, turning a missing module into a test FAILURE (not an error)."""
    try:
        return importlib.import_module(f"{PKG}.{name}")
    except Exception as exc:  # ModuleNotFoundError, or the module failing while importing
        pytest.fail(f"module {name} missing or not importable: {type(exc).__name__}: {exc}",
                    pytrace=False)


class Lazy:
    """Compute once on first use, inside a test body, so a failure is a test failure."""

    def __init__(self, fn):
        self._fn = fn
        self._done = False
        self._value = None
        self._exc = None

    def get(self):
        if not self._done:
            try:
                self._value = self._fn()
            except (Exception, pytest.fail.Exception) as exc:  # noqa: BLE001
                self._exc = exc
            self._done = True
        if self._exc is not None:
            pytest.fail(f"{type(self._exc).__name__}: {self._exc}", pytrace=False)
        return self._value


# --------------------------------------------------------------------------- the spec

def spec_text() -> str:
    return SPEC_PATH.read_text(encoding="utf-8")


def spec_section(number: int) -> str:
    m = re.search(rf"^## {number}\. .*?(?=^## \d+\. |\Z)", spec_text(), re.S | re.M)
    assert m, f"SPEC.md has no section {number}"
    return m.group(0)


def spec_literal(name: str, section: int):
    m = re.search(rf"^{name}\s*=\s*(\(.*?\))", spec_section(section), re.M)
    assert m, f"SPEC.md section {section} does not define {name}"
    return ast.literal_eval(m.group(1))


def spec_int_constants(section: int) -> dict:
    block = re.search(r"```python(.*?)```", spec_section(section), re.S).group(1)
    return {k: int(v) for k, v in re.findall(r"\b([A-Z][A-Z0-9_]*)\s*=\s*(-?\d+)\b", block)}


def spec_float(pattern: str, section: int) -> float:
    m = re.search(pattern, spec_section(section))
    assert m, f"pattern {pattern!r} not found in SPEC.md section {section}"
    return float(m.group(1).replace("−", "-"))


def spec_plot_names(section: int) -> list:
    return re.findall(r"DIR/plots/(\w+\.html)", spec_section(section))


def spec_json_name(section: int) -> str:
    return re.search(r"DIR/(\w+\.json)", spec_section(section)).group(1)


def spec_runner_section_6() -> str:
    sec = spec_section(6)
    return sec[sec.index("### `run_experiment_1.py`"):]


def spec_exp3_top_keys() -> list:
    sec = spec_section(5)
    block = sec[sec.index("**Top-level JSON keys:**"):sec.index("**Invariants:**")]
    return re.findall(r"^- `(\w+)`", block, re.M)


def spec_exp3_schema() -> str:
    return re.search(r"`schema`: `\"([^\"]+)\"`", spec_section(5)).group(1)


def spec_exp1_top_keys() -> list:
    line = re.search(r"\*\*Top-level JSON keys:\*\*(.*)", spec_runner_section_6()).group(1)
    return re.findall(r"`(\w+)`", line)


def spec_exp1_schema() -> str:
    return re.search(r"`schema` \(`\"([^\"]+)\"`\)", spec_runner_section_6()).group(1)


def spec_exp1_arms() -> list:
    sec = spec_runner_section_6()
    block = sec[sec.index("**Arms.**"):sec.index("**Seeding.**")]
    return re.findall(r"^- `(\w+)`:", block, re.M)


def spec_exp1_conditions() -> list:
    return re.search(r"--conditions ([a-z,]+)", spec_runner_section_6()).group(1).split(",")


def spec_armresult_fields() -> list:
    sec = spec_section(4)
    block = sec[sec.index("class ArmResult:"):sec.index("```", sec.index("class ArmResult:"))]
    return re.findall(r"^\s{4}(\w+):", block, re.M)


def spec_plotly_url() -> str:
    return re.search(r"https://cdn\.plot\.ly/plotly-[\d.]+\.min\.js", spec_section(7)).group(0)


def spec_colours() -> tuple:
    sec = spec_section(7)
    bg = re.search(r"background \(`(#[0-9a-fA-F]{6})`\)", sec).group(1)
    fg = re.search(r"text \(`(#[0-9a-fA-F]{6})`\)", sec).group(1)
    return bg.lower(), fg.lower()


# --------------------------------------------------------------------------- polynomial facts

def true_floor(coeffs):
    """min over the reals of |p(x)| for a real polynomial with no real root, from its critical
    points (polished by Newton). Independent of the code under test."""
    c = np.asarray(coeffs, dtype=float)
    d1 = np.polyder(c)
    d2 = np.polyder(d1)
    crit = np.roots(d1)
    cands = []
    for z in crit:
        if abs(z.imag) > 1e-6:
            continue
        x = float(z.real)
        for _ in range(100):
            h = np.polyval(d2, x)
            if h == 0:
                break
            step = np.polyval(d1, x) / h
            x -= step
            if abs(step) <= 1e-16 * max(1.0, abs(x)):
                break
        cands.append(x)
    vals = [abs(float(np.polyval(c, x))) for x in cands]
    i = int(np.argmin(vals))
    return cands[i], vals[i]


def dual_kind(alpha: float, beta: float, tol: float) -> str:
    delta = beta ** 2 + 4 * alpha
    if abs(delta) <= tol * max(1.0, beta ** 2, abs(4 * alpha)):
        return "dual"
    return "complex" if delta < 0 else "split"


# --------------------------------------------------------------------------- strict JSON

def strict_json(text: str):
    def _reject(const):
        raise ValueError(f"non-finite JSON constant {const!r}")
    return json.loads(text, parse_constant=_reject)


def walk_numbers(obj):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from walk_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_numbers(v)


# --------------------------------------------------------------------------- filesystem

def tree_digest(root: Path) -> dict:
    """relpath -> sha256 (files), 'dir' (directories), 'link:<target>' (symlinks). Ignores bytecode."""
    out = {}
    root = Path(root)
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for d in dirnames:
            p = Path(dirpath, d)
            out[p.relative_to(root).as_posix() + "/"] = "link:" + os.readlink(p) if p.is_symlink() else "dir"
        for f in sorted(filenames):
            if f.endswith(".pyc"):
                continue
            p = Path(dirpath, f)
            rel = p.relative_to(root).as_posix()
            if p.is_symlink():
                out[rel] = "link:" + os.readlink(p)
            else:
                out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def outside_snapshot() -> dict:
    return {
        "experiment dir": tree_digest(EXP_DIR),
        "static/": tree_digest(REPO / "static"),
        "content/": tree_digest(REPO / "content"),
        "repo root entries": sorted(n for n in os.listdir(REPO) if not n.startswith(PUBLISH_PREFIX)),
    }


def snapshot_diff(before: dict, after: dict, ignore_prefix: dict | None = None) -> list:
    ignore_prefix = ignore_prefix or {}
    problems = []
    for area in before:
        b, a = before[area], after[area]
        if isinstance(b, list):
            if b != a:
                problems.append(f"{area}: entries changed: +{sorted(set(a) - set(b))} -{sorted(set(b) - set(a))}")
            continue
        skip = ignore_prefix.get(area, ())
        keys = {k for k in set(a) | set(b) if not any(k.startswith(s) for s in skip)}
        changed = sorted(k for k in keys if a.get(k) != b.get(k))
        if changed:
            problems.append(f"{area}: changed/created/removed {changed[:20]}")
    return problems


def list_files(root: Path) -> list:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink())


class RestoreGuard:
    """Safety net: snapshot results/, static/ and content/ and put them back afterwards, so a
    misbehaving runner cannot leave the worktree dirty. Detection is done separately (before restore)."""

    ROOTS = (EXP_DIR / "results", REPO / "static", REPO / "content")

    def __enter__(self):
        self.snap = {}
        for root in self.ROOTS:
            files, dirs = {}, set()
            if root.exists():
                for p in root.rglob("*"):
                    rel = p.relative_to(root)
                    if p.is_symlink():
                        files[rel] = ("link", os.readlink(p))
                    elif p.is_dir():
                        dirs.add(rel)
                    else:
                        files[rel] = ("file", p.read_bytes())
            self.snap[root] = (root.exists(), files, dirs)
        return self

    def __exit__(self, *exc):
        for root, (existed, files, dirs) in self.snap.items():
            if not existed:
                if root.exists():
                    shutil.rmtree(root, ignore_errors=True)
                continue
            for p in sorted(root.rglob("*"), key=lambda q: len(q.parts), reverse=True):
                rel = p.relative_to(root)
                if p.is_symlink() or p.is_file():
                    if rel not in files:
                        p.unlink()
                elif p.is_dir() and rel not in dirs:
                    shutil.rmtree(p, ignore_errors=True)
            for d in sorted(dirs, key=lambda q: len(q.parts)):
                (root / d).mkdir(parents=True, exist_ok=True)
            for rel, (kind, payload) in files.items():
                p = root / rel
                if kind == "link":
                    if not (p.is_symlink() and os.readlink(p) == payload):
                        if p.exists() or p.is_symlink():
                            p.unlink()
                        os.symlink(payload, p)
                elif p.is_symlink() or not p.exists() or p.read_bytes() != payload:
                    if p.is_symlink():
                        p.unlink()
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(payload)
        return False


# --------------------------------------------------------------------------- running the CLIs

@dataclass
class CliRun:
    argv: list
    cwd: str
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    elapsed: float = math.inf
    timed_out: bool = False
    extra: dict = field(default_factory=dict)

    def describe(self) -> str:
        tail = (self.stderr or "")[-2500:]
        out = (self.stdout or "")[-800:]
        return (f"command: {' '.join(map(str, self.argv))}\n cwd: {self.cwd}\n exit: {self.returncode}"
                f"{' (TIMED OUT)' if self.timed_out else ''}  elapsed: {self.elapsed:.1f}s\n"
                f" stderr tail:\n{tail}\n stdout tail:\n{out}")


def require_ok(run: CliRun, what: str = "runner"):
    if run.returncode != 0:
        pytest.fail(f"{what} did not exit 0\n{run.describe()}", pytrace=False)


def _env():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_parallel(jobs: dict, timeout: int = CLI_TIMEOUT) -> dict:
    """jobs: name -> (argv, cwd). Runs all at once; returns name -> CliRun."""
    procs = {}
    for name, (argv, cwd) in jobs.items():
        Path(cwd).mkdir(parents=True, exist_ok=True)
        run = CliRun(argv=[str(a) for a in argv], cwd=str(cwd))
        try:
            p = subprocess.Popen(run.argv, cwd=cwd, env=_env(), stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
        except OSError as exc:
            run.returncode, run.stderr = -999, f"could not start: {exc}"
            procs[name] = (None, run, time.monotonic())
            continue
        procs[name] = (p, run, time.monotonic())
    deadline = time.monotonic() + timeout
    for name, (p, run, t0) in procs.items():
        if p is None:
            continue
        try:
            out, err = p.communicate(timeout=max(1.0, deadline - time.monotonic()))
            run.returncode = p.returncode
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            run.timed_out, run.returncode = True, None
        run.elapsed = time.monotonic() - t0
        run.stdout, run.stderr = out, err
    return {name: run for name, (p, run, t0) in procs.items()}


EXP1_DRIVER = r'''
"""Runs run_experiment_1.py in-process with probes on the public API (seeding.rng_for and the
optimiser constructors). Written by the acceptance tests; not part of the experiment."""
import importlib, inspect, json, runpy, sys, traceback
out_path, repo, script = sys.argv[1], sys.argv[2], sys.argv[3]
argv = sys.argv[4:]
sys.dont_write_bytecode = True
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))   # what `python script.py` does
sys.path.insert(1, repo)
import numpy as np
PKG = "experiments.ideas_intelligence_creativity"
rec = {"rng_for": [], "optimizers": [], "exit": None, "error": None}
def dump():
    with open(out_path, "w") as fh:
        json.dump(rec, fh)
try:
    seeding = importlib.import_module(PKG + ".seeding")
    _orig = seeding.rng_for
    def rng_for(root_seed, *keys):
        rec["rng_for"].append([int(root_seed)] + [int(k) for k in keys])
        return _orig(root_seed, *keys)
    seeding.rng_for = rng_for
    intel = importlib.import_module(PKG + ".intelligence")
    crea = importlib.import_module(PKG + ".creativity")
except BaseException:
    rec["error"] = "probe setup failed:\n" + traceback.format_exc()
    rec["exit"] = 3
    dump()
    sys.exit(3)
PROBES = [[0.1, -0.2], [1.3, 2.9], [-3.7, 0.4], [2.2, -4.1]]
def fp(land):
    try:
        return [round(float(land.evaluate(np.array(p, dtype=float))), 11) for p in PROBES]
    except Exception as e:
        return ["error: " + repr(e)]
def centers(land):
    try:
        return [[float(v) for v in np.asarray(pk.center, dtype=float).ravel()] for pk in land.peaks]
    except Exception:
        return None
def wrap(cls, kind):
    orig = cls.__init__
    sig = inspect.signature(orig)
    def __init__(self, *a, **kw):
        orig(self, *a, **kw)
        if type(self) is not cls:
            return
        try:
            ba = sig.bind(self, *a, **kw)
            ba.apply_defaults()
            A = ba.arguments
            strat = A.get("strategy", A.get("initial_strategy"))
            x0 = A.get("x0")
            item = {"kind": kind,
                    "x0": None if x0 is None else [float(v) for v in np.asarray(x0, dtype=float).ravel()],
                    "selection": A.get("selection"),
                    "method": getattr(strat, "method", None),
                    "step_size": float(strat.step_size) if hasattr(strat, "step_size") else None,
                    "fp": fp(A.get("landscape")),
                    "centers": centers(A.get("landscape"))}
        except Exception:
            item = {"kind": kind, "error": traceback.format_exc()}
        rec["optimizers"].append(item)
    cls.__init__ = __init__
wrap(intel.FixedStrategyOptimizer, "fixed")
wrap(crea.MetaStrategyOptimizer, "meta")
sys.argv = [script] + argv
code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
except BaseException:
    rec["error"] = traceback.format_exc()
    code = 1
rec["exit"] = code
dump()
sys.exit(code)
'''


def run_runner_suite(script_name: str, json_name: str, plot_names: list, pert_args: list,
                     base: Path, instrument: bool = False) -> dict:
    """Runs one runner under the spec's quick profile several times.

    phase 1 (parallel): r1 = quick, 1 worker (timed, snapshot-checked);
                        pub = quick, other root seed, --publish-plots-dir inside the repo;
                        pert = quick with config perturbations (and probes for Exp 1).
    phase 2 (parallel): r2 = r1's exact arguments but --workers 2 (same --output-dir path);
                        dflt = quick --no-plots with no --output-dir, run from an unrelated cwd.
    """
    script = EXP_DIR / script_name
    base = Path(base)
    cwd = base / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    same = base / "same"
    res: dict = {"script": script, "json_name": json_name, "plot_names": plot_names, "cwd": cwd}
    publish_root = Path(tempfile.mkdtemp(prefix=PUBLISH_PREFIX, dir=REPO))
    publish_dir = publish_root / "published_plots"          # does not exist yet (spec 9.7)
    quick = ["--profile", "quick"]
    with RestoreGuard():
        try:
            before = outside_snapshot()
            jobs = {
                "r1": ([PY, script, *quick, "--workers", "1", "--output-dir", same], cwd),
                "pub": ([PY, script, *quick, "--workers", "1", "--root-seed", "918273",
                         "--output-dir", base / "pub", "--publish-plots-dir", publish_dir], cwd),
            }
            if instrument:
                driver = base / "probe_driver.py"
                driver.write_text(EXP1_DRIVER, encoding="utf-8")
                jobs["pert"] = ([PY, driver, base / "probe.json", REPO, script, *quick, "--workers", "1",
                                 *pert_args, "--output-dir", base / "pert"], cwd)
            else:
                jobs["pert"] = ([PY, script, *quick, "--workers", "1", *pert_args,
                                 "--output-dir", base / "pert"], cwd)
            runs = run_parallel(jobs)
            after = outside_snapshot()
            res["phase1_outside_problems"] = snapshot_diff(before, after)
            res["published"] = {rel: (publish_dir / rel).read_bytes() for rel in list_files(publish_dir)}
            res["publish_dir_created"] = publish_dir.is_dir()
            res["publish_dir"] = publish_dir
            if instrument:
                probe = base / "probe.json"
                res["probe"] = json.loads(probe.read_text()) if probe.exists() else None
            if same.exists():
                shutil.copytree(same, base / "r1_out", symlinks=True)
                shutil.rmtree(same)
            res["r1_dir"], res["pub_dir"], res["pert_dir"] = base / "r1_out", base / "pub", base / "pert"
            res.update(runs)

            # phase 2
            default_json = EXP_DIR / "results" / json_name
            if default_json.exists():
                default_json.unlink()  # RestoreGuard puts it back afterwards
            before2 = outside_snapshot()
            cwd2 = base / "cwd2"
            runs2 = run_parallel({
                "r2": ([PY, script, *quick, "--workers", "2", "--output-dir", same], cwd),
                "dflt": ([PY, script, *quick, "--no-plots"], cwd2),
            })
            after2 = outside_snapshot()
            res["phase2_outside_problems"] = snapshot_diff(
                before2, after2, ignore_prefix={"experiment dir": ("results/",)})
            res["dflt_json"] = default_json.read_bytes() if default_json.exists() else None
            res["cwd_files"] = list_files(cwd) + list_files(cwd2)
            res["r2_dir"] = same
            res.update(runs2)
        finally:
            shutil.rmtree(publish_root, ignore_errors=True)
    return res


REFUSAL_VARIANTS = (
    "absolute path outside the repo",
    "repo path with .. escaping it",
    "symlink inside the repo pointing outside",
    "sibling dir whose path starts with the repo path",
)


def run_refusals(script_name: str, base: Path) -> dict:
    """Runs the runner with --publish-plots-dir values that resolve OUTSIDE the repository root.

    Spec 9.7: the runner refuses before computing anything, so nothing at all is written. The runs use
    --profile full with a short timeout: an up-front refusal returns in seconds whatever the profile,
    while a runner that computes first is still computing when the timeout hits. The output dirs and
    most targets do not exist beforehand, so creating them counts as a write.
    """
    script = EXP_DIR / script_name
    base = Path(base)
    targets = {}
    jobs = {}
    cleanup = []
    outside_abs = base / "outside_abs" / "plots"                       # missing
    targets[REFUSAL_VARIANTS[0]] = (outside_abs, outside_abs)
    dotdot_target = base / "outside_dotdot"                            # missing
    dotdot_arg = str(REPO) + "/" + os.path.relpath(dotdot_target, REPO)  # starts with the repo path
    targets[REFUSAL_VARIANTS[1]] = (dotdot_arg, dotdot_target)
    link_target = base / "outside_link"
    link_target.mkdir()                                                # exists, empty
    link = REPO / (PUBLISH_PREFIX + "link-" + os.urandom(4).hex())
    os.symlink(link_target, link)
    cleanup.append(link)
    targets[REFUSAL_VARIANTS[2]] = (link, link_target)
    sibling = Path(str(REPO) + "-iic-escape-" + os.urandom(4).hex())   # missing; shares the repo prefix
    cleanup.append(sibling)
    targets[REFUSAL_VARIANTS[3]] = (sibling, sibling)
    assert tuple(targets) == REFUSAL_VARIANTS
    for i, (name, (arg, _target)) in enumerate(targets.items()):
        jobs[name] = ([PY, script, "--profile", "full", "--workers", "1",
                       "--output-dir", base / f"out_{i}", "--publish-plots-dir", arg], base / f"cwd_{i}")
    with RestoreGuard():
        try:
            before = outside_snapshot()
            runs = run_parallel(jobs, timeout=REFUSAL_TIMEOUT)
            after = outside_snapshot()
            result = {}
            for i, (name, (arg, target)) in enumerate(targets.items()):
                result[name] = {
                    "run": runs[name],
                    "target_existed_before": name == REFUSAL_VARIANTS[2],
                    "target_exists": target.exists(),
                    "target_files": list_files(target),
                    "out_exists": (base / f"out_{i}").exists(),
                    "out_files": list_files(base / f"out_{i}"),
                    "cwd_files": list_files(base / f"cwd_{i}"),
                }
            result["_outside_problems"] = snapshot_diff(before, after)
            return result
        finally:
            for p in cleanup:
                if p.is_symlink() or p.is_file():
                    p.unlink()
                elif p.exists():
                    shutil.rmtree(p, ignore_errors=True)


NOT_IMPLEMENTED_SIGNS = ("unrecognized arguments", "can't open file", "No such file or directory: '"
                         + str(EXP_DIR))


def assert_refusal_is_deliberate(run: CliRun):
    """A refusal must come from a runner that exists and knows the option."""
    for sign in NOT_IMPLEMENTED_SIGNS:
        if sign in (run.stderr or ""):
            pytest.fail(f"non-zero exit is not a deliberate refusal ({sign!r})\n{run.describe()}",
                        pytrace=False)


# --------------------------------------------------------------------------- HTML figures

class _Ident(str):
    pass


def _skip_ws(s, i):
    while i < len(s) and s[i] in " \t\r\n":
        i += 1
    return i


def _skip_call(s, i):
    depth = 0
    while i < len(s):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def extract_figures(html: str) -> list:
    """Every Plotly.newPlot/react call whose data and layout are JSON literals in the page (directly,
    or through a `var/let/const NAME = <json>` assignment)."""
    dec = json.JSONDecoder()
    assigns = {}
    for m in re.finditer(r"\b(?:var|let|const)\s+([A-Za-z_$][\w$]*)\s*=\s*", html):
        try:
            assigns[m.group(1)], _ = dec.raw_decode(html, m.end())
        except ValueError:
            pass

    def resolve(v):
        if isinstance(v, _Ident):
            parts = v.split(".")
            cur = assigns.get(parts[0])
            for p in parts[1:]:
                cur = cur.get(p) if isinstance(cur, dict) else None
            return cur
        return v

    figs = []
    for m in re.finditer(r"Plotly\.(?:newPlot|react)\s*\(", html):
        pos, args = m.end(), []
        for _ in range(4):
            pos = _skip_ws(html, pos)
            try:
                val, pos = dec.raw_decode(html, pos)
                args.append(val)
            except ValueError:
                im = re.match(r"[A-Za-z_$][\w$.]*", html[pos:])
                if not im:
                    break
                pos += len(im.group(0))
                if html[pos:pos + 1] == "(":
                    pos = _skip_call(html, pos)
                args.append(_Ident(im.group(0)))
            pos = _skip_ws(html, pos)
            if html[pos:pos + 1] != ",":
                break
            pos += 1
        rest = [resolve(a) for a in args[1:]]
        if rest and isinstance(rest[0], dict) and "data" in rest[0]:
            figs.append({"data": rest[0].get("data") or [], "layout": rest[0].get("layout") or {}})
        elif rest and isinstance(rest[0], list):
            figs.append({"data": rest[0], "layout": rest[1] if len(rest) > 1 and isinstance(rest[1], dict) else {}})
    return figs


def typed_arrays(obj):
    """Decode plotly typed arrays ({'dtype':..., 'bdata':...}) found anywhere in obj."""
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            try:
                yield np.frombuffer(base64.b64decode(obj["bdata"]), dtype=np.dtype(obj["dtype"]))
            except Exception:  # noqa: BLE001
                yield None
        else:
            for v in obj.values():
                yield from typed_arrays(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from typed_arrays(v)


def walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k == "template":
                continue
            yield from walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_strings(v)


# --------------------------------------------------------------------------- Budget point counting

def patch_budget_counter(monkeypatch, Budget) -> dict:
    """Count points that search_core.Budget evaluates (successful __call__ = 1, batch = len(result)).

    Patched on the class, so every instance is seen. A per-instance guard stops double counting when
    one method of an instance is implemented through the other (batch -> __call__ or the reverse).
    """
    orig_call, orig_batch, orig_init = Budget.__call__, Budget.batch, Budget.__init__
    inside = set()
    counter = {"n": 0, "instances": set(), "created": 0}

    def init(self, *args, **kwargs):
        counter["created"] += 1
        orig_init(self, *args, **kwargs)

    def call(self, x):
        if id(self) in inside:
            return orig_call(self, x)
        inside.add(id(self))
        try:
            v = orig_call(self, x)
        finally:
            inside.discard(id(self))
        counter["n"] += 1
        counter["instances"].add(id(self))
        return v

    def batch(self, xs):
        if id(self) in inside:
            return orig_batch(self, xs)
        inside.add(id(self))
        try:
            vals = orig_batch(self, xs)
        finally:
            inside.discard(id(self))
        counter["n"] += len(vals)
        counter["instances"].add(id(self))
        return vals

    monkeypatch.setattr(Budget, "__init__", init)
    monkeypatch.setattr(Budget, "__call__", call)
    monkeypatch.setattr(Budget, "batch", batch)
    return counter


def assert_refused(variant: dict, outside_problems: list):
    """Spec 5 and 9.7: non-zero exit, refused before computing, nothing written anywhere."""
    run = variant["run"]
    if run.timed_out:
        pytest.fail(f"no refusal within {REFUSAL_TIMEOUT}s under --profile full: the runner computed "
                    f"before refusing (spec 9.7)\n{run.describe()}", pytrace=False)
    assert run.returncode not in (0, None), run.describe()
    assert_refusal_is_deliberate(run)
    if variant["target_existed_before"]:
        assert variant["target_files"] == [], "files were written to the refused directory"
    else:
        assert not variant["target_exists"], "the refused directory was created"
    assert not variant["out_exists"], f"a refused run still wrote to --output-dir: {variant['out_files']}"
    assert variant["cwd_files"] == []
    assert outside_problems == []


def assert_published(res: dict, json_name: str, plot_names: list):
    """Spec 9.7: a missing publish dir is created; only the plot HTML is copied, byte-identical."""
    assert res["publish_dir_created"], "the missing publish dir was not created"
    published = res["published"]
    assert sorted(published) == sorted(plot_names), sorted(published)
    for name in plot_names:
        assert published[name] == (res["pub_dir"] / "plots" / name).read_bytes(), name
