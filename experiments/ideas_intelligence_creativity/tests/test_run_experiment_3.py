"""SPEC.md section 5: run_experiment_3.py, the command-line runner (quick profile)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.tests._iic_helpers import (  # noqa: E402
    REFUSAL_VARIANTS, Lazy, assert_published, assert_refused, list_files, require_ok, run_refusals,
    spec_exp3_schema, spec_exp3_top_keys, spec_json_name, spec_literal, spec_plot_names, strict_json, true_floor,
)

JSON_NAME = spec_json_name(5)
PLOTS = spec_plot_names(5)
REAL_ARMS = spec_literal("REAL_ARMS", 4)
CONTROLS = spec_literal("CONTROLS", 4)
L2_ARMS = spec_literal("L2_ARMS", 4)
ALL_ARMS = REAL_ARMS + CONTROLS + L2_ARMS        # spec 9.6: the `arms` key holds every arm
FLOOR_ARMS = REAL_ARMS + ("split_two_reals",)
COEFFS = spec_literal("COEFFS", 3)
X_TRUE, M_TRUE = true_floor(COEFFS)


@pytest.fixture(scope="module")
def refusals(tmp_path_factory):
    base = tmp_path_factory.mktemp("exp3_refusals")
    return Lazy(lambda: run_refusals("run_experiment_3.py", base))


def r1_json(cli):
    res = cli.get()
    require_ok(res["r1"], "quick run (--workers 1)")
    path = res["r1_dir"] / JSON_NAME
    assert path.exists(), f"{JSON_NAME} not written to --output-dir; files: {list_files(res['r1_dir'])}"
    return strict_json(path.read_text(encoding="utf-8"))


def all_best(doc, arms):
    return {arm: np.asarray(doc["arms"][arm]["best"], dtype=float) for arm in arms}


# ----------------------------------------------------------------------------- outputs

def test_quick_run_writes_json_and_plots(exp3_cli):
    res = exp3_cli.get()
    require_ok(res["r1"])
    files = list_files(res["r1_dir"])
    assert JSON_NAME in files, files
    for name in PLOTS:
        assert f"plots/{name}" in files, files


def test_json_is_strict_with_schema_and_keys(exp3_cli):
    doc = r1_json(exp3_cli)
    assert doc["schema"] == spec_exp3_schema()
    assert set(spec_exp3_top_keys()) <= set(doc), sorted(doc)


def test_polynomial_block(exp3_cli):
    poly = r1_json(exp3_cli)["polynomial"]
    assert [float(c) for c in poly["coeffs"]] == list(COEFFS)
    roots = np.array([complex(float(re_), float(im)) for re_, im in poly["roots"]])
    assert roots.shape == (4,)
    for r in np.roots(COEFFS):
        assert np.min(np.abs(roots - r)) < 1e-9
    floor = poly["floor"]
    assert {"x_star", "m", "method", "grid_check"} <= set(floor)
    assert abs(float(floor["m"]) - M_TRUE) <= 1e-9 * M_TRUE
    assert abs(float(floor["x_star"]) - X_TRUE) < 1e-4


def test_checkpoints_and_per_arm_best(exp3_cli):
    doc = r1_json(exp3_cli)
    cps = doc["checkpoints"]
    assert isinstance(cps, list) and cps
    assert all(int(c) == c and c > 0 for c in cps)
    assert all(a < b for a, b in zip(cps, cps[1:]))
    assert set(ALL_ARMS) <= set(doc["arms"]), sorted(doc["arms"])
    n_seeds = None
    for arm, entry in doc["arms"].items():
        best = np.asarray(entry["best"], dtype=float)
        assert best.ndim == 2 and best.shape[1] == len(cps), (arm, best.shape)
        n_seeds = best.shape[0] if n_seeds is None else n_seeds
        assert best.shape[0] == n_seeds >= 1, arm
        assert np.all(np.isfinite(best)) and np.all(best >= 0), arm
        assert np.all(np.diff(best, axis=1) <= 0), f"{arm}: best-so-far increased between checkpoints"


def test_real_floor_holds_in_the_results(exp3_cli):
    doc = r1_json(exp3_cli)
    m = float(doc["polynomial"]["floor"]["m"])
    for arm, best in all_best(doc, FLOOR_ARMS).items():
        assert best.min() >= M_TRUE * (1 - 1e-12) and best.min() >= m * (1 - 1e-12), arm


def test_invariants(exp3_cli):
    doc = r1_json(exp3_cli)
    inv = doc["invariants"]
    m = float(doc["polynomial"]["floor"]["m"])
    lowest_seen = min(float(b.min()) for b in all_best(doc, FLOOR_ARMS).values()) / m
    assert inv["min_real_best_over_m"] >= 1 - 1e-12
    assert inv["min_real_best_over_m"] <= lowest_seen * (1 + 1e-12), "invariant above what the arms recorded"
    assert inv["below_floor_from_nonnegative_delta"] == 0


# ----------------------------------------------------------------------------- determinism, speed, side effects

def test_json_is_byte_identical_for_one_and_two_workers(exp3_cli):
    res = exp3_cli.get()
    require_ok(res["r1"], "quick run (--workers 1)")
    require_ok(res["r2"], "quick run (--workers 2)")
    a = (res["r1_dir"] / JSON_NAME).read_bytes()
    b = (res["r2_dir"] / JSON_NAME).read_bytes()
    assert a == b


def test_quick_profile_finishes_in_under_60_seconds(exp3_cli):
    res = exp3_cli.get()
    require_ok(res["r1"])
    assert res["r1"].elapsed < 60.0, res["r1"].elapsed


def test_nothing_is_written_outside_the_output_dir(exp3_cli):
    res = exp3_cli.get()
    for name in ("r1", "pert", "r2"):
        require_ok(res[name], name)
    assert res["phase1_outside_problems"] == []
    assert res["phase2_outside_problems"] == []
    assert res["cwd_files"] == [], "files written to the working directory"


def test_no_plots_and_n_seeds(exp3_cli):
    res = exp3_cli.get()
    require_ok(res["pert"], "--no-plots --n-seeds 2 run")
    files = list_files(res["pert_dir"])
    assert JSON_NAME in files
    assert not [f for f in files if f.endswith(".html")], files
    doc = strict_json((res["pert_dir"] / JSON_NAME).read_text(encoding="utf-8"))
    assert set(ALL_ARMS) <= set(doc["arms"]), sorted(doc["arms"])
    for arm, entry in doc["arms"].items():
        assert len(entry["best"]) == 2, arm


def test_root_seed_changes_the_results(exp3_cli):
    res = exp3_cli.get()
    require_ok(res["pub"], "--root-seed 918273 run")
    a = r1_json(exp3_cli)
    b = strict_json((res["pub_dir"] / JSON_NAME).read_text(encoding="utf-8"))
    assert any(a["arms"][arm]["best"] != b["arms"][arm]["best"] for arm in ALL_ARMS)


def test_default_output_dir_is_the_experiment_results_dir(exp3_cli):
    res = exp3_cli.get()
    require_ok(res["dflt"], "run without --output-dir")
    assert res["dflt_json"] is not None, f"{JSON_NAME} was not written to <experiment dir>/results"
    assert strict_json(res["dflt_json"].decode("utf-8"))["schema"] == spec_exp3_schema()


# ----------------------------------------------------------------------------- --publish-plots-dir

def test_publish_dir_inside_the_repo_is_accepted_and_receives_the_plots(exp3_cli):
    res = exp3_cli.get()
    require_ok(res["pub"], "run with --publish-plots-dir inside the repository (dir missing beforehand)")
    assert_published(res, JSON_NAME, PLOTS)


@pytest.mark.parametrize("variant", REFUSAL_VARIANTS)
def test_publish_dir_outside_the_repo_is_refused_before_computing(variant, exp3_cli, refusals):
    res = exp3_cli.get()
    require_ok(res["pub"], "positive control: --publish-plots-dir inside the repository")
    got = refusals.get()
    assert_refused(got[variant], got["_outside_problems"])
