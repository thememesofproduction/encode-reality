"""SPEC.md section 6: run_experiment_1.py, the repaired Experiment 1 runner (quick profile).

The perturbation run goes through a small driver that runs the script in-process with probes on
the public API (seeding.rng_for and the optimiser constructors), to observe the seeding roles and
that the four arms share each landscape and start point.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.tests._iic_helpers import (  # noqa: E402
    REFUSAL_VARIANTS, Lazy, assert_published, assert_refused, list_files, require_ok, run_refusals,
    spec_exp1_arms, spec_exp1_conditions, spec_exp1_schema, spec_exp1_top_keys, spec_int_constants,
    spec_json_name, spec_plot_names, strict_json,
)

JSON_NAME = spec_json_name(6)
PLOTS = spec_plot_names(6)
ARMS = spec_exp1_arms()                      # hc_fixed, restart_fixed, swap_random, meta_ucb
CONDITIONS = spec_exp1_conditions()
SEEDING = spec_int_constants(1)
SUCCESS = 0.99                               # "best >= 0.99 * ceiling"
PERT = {"n_landscapes": 3, "n_starts": 2, "max_evals": 300, "conditions": ["none", "corner"], "root_seed": 7}


@pytest.fixture(scope="module")
def refusals(tmp_path_factory):
    base = tmp_path_factory.mktemp("exp1_refusals")
    return Lazy(lambda: run_refusals("run_experiment_1.py", base))


def load_json(res, run_name, dir_key):
    require_ok(res[run_name], run_name)
    path = res[dir_key] / JSON_NAME
    assert path.exists(), f"{JSON_NAME} missing; files: {list_files(res[dir_key])}"
    return strict_json(path.read_text(encoding="utf-8"))


def r1_json(cli):
    return load_json(cli.get(), "r1", "r1_dir")


def finals(doc, cond, arm):
    return np.asarray([run[-1] for run in doc["conditions"][cond][arm]["normalised_best"]], dtype=float)


def test_spec_parse_and_quick_run_writes_json_and_plots(exp1_cli):
    assert ARMS == ["hc_fixed", "restart_fixed", "swap_random", "meta_ucb"]
    assert CONDITIONS == ["none", "random", "corner"]
    res = exp1_cli.get()
    require_ok(res["r1"])
    files = list_files(res["r1_dir"])
    assert JSON_NAME in files, files
    for name in PLOTS:
        assert f"plots/{name}" in files, files


def test_json_is_strict_with_schema_and_keys(exp1_cli):
    doc = r1_json(exp1_cli)
    assert doc["schema"] == spec_exp1_schema()
    assert set(spec_exp1_top_keys()) <= set(doc), sorted(doc)


def _check_conditions(doc, expect_conditions=None, expect_runs=None):
    cps = doc["checkpoints"]
    assert isinstance(cps, list) and cps and all(int(c) == c and c > 0 for c in cps)
    assert all(a < b for a, b in zip(cps, cps[1:]))
    conds = doc["conditions"]
    assert conds and set(conds) <= set(CONDITIONS), sorted(conds)
    if expect_conditions is not None:
        assert set(conds) == set(expect_conditions)
    for cond, per_arm in conds.items():
        assert set(ARMS) <= set(per_arm), (cond, sorted(per_arm))
        n_runs = {len(per_arm[a]["normalised_best"]) for a in ARMS}
        assert len(n_runs) == 1, (cond, n_runs)          # the four arms see the same runs
        if expect_runs is not None:
            assert n_runs == {expect_runs}, (cond, n_runs)
        for arm in ARMS:
            nb = np.asarray(per_arm[arm]["normalised_best"], dtype=float)
            assert nb.ndim == 2 and nb.shape[1] == len(cps), (cond, arm, nb.shape)
            assert np.all(np.isfinite(nb)) and np.all(nb >= 0) and np.all(nb <= 1 + 1e-9), (cond, arm)
            assert np.all(np.diff(nb, axis=1) >= 0), f"{cond}/{arm}: best-so-far fell between checkpoints"
    return cps


def test_conditions_hold_normalised_best_per_arm(exp1_cli):
    _check_conditions(r1_json(exp1_cli))


def test_summary_success_rate_ci_and_mcnemar(exp1_cli):
    """Spec 9.10: summary[cond][arm] == {"success_rate", "ci"}; summary[cond]["mcnemar_p"][arm] for arm != meta_ucb."""
    doc = r1_json(exp1_cli)
    summary = doc["summary"]
    assert set(summary) == set(doc["conditions"]), (sorted(summary), sorted(doc["conditions"]))
    others = [a for a in ARMS if a != "meta_ucb"]
    for cond in doc["conditions"]:
        for arm in ARMS:
            entry = summary[cond][arm]
            assert set(entry) == {"success_rate", "ci"}, (cond, arm, sorted(entry))
            fin = finals(doc, cond, arm)
            rate = entry["success_rate"]
            assert isinstance(rate, float)
            assert np.mean(fin >= SUCCESS + 1e-9) - 1e-12 <= rate <= np.mean(fin >= SUCCESS - 1e-9) + 1e-12, \
                (cond, arm, rate)
            assert isinstance(entry["ci"], list) and len(entry["ci"]) == 2
            lo, hi = (float(v) for v in entry["ci"])
            assert 0.0 <= lo <= rate <= hi <= 1.0, (cond, arm, lo, rate, hi)
        mc = summary[cond]["mcnemar_p"]
        assert set(mc) == set(others), (cond, sorted(mc))
        ucb = finals(doc, cond, "meta_ucb") >= SUCCESS
        for arm in others:
            p = mc[arm]
            assert isinstance(p, float) and 0.0 <= p <= 1.0, (cond, arm, p)
            other = finals(doc, cond, arm) >= SUCCESS
            if (ucb.all() and other.all()) or (not ucb.any() and not other.any()):
                assert p == pytest.approx(1.0), (cond, arm, p)   # no discordant pairs, under any pairing


# ----------------------------------------------------------------------------- determinism, speed, side effects

def test_json_is_byte_identical_for_one_and_two_workers(exp1_cli):
    res = exp1_cli.get()
    require_ok(res["r1"], "quick run (--workers 1)")
    require_ok(res["r2"], "quick run (--workers 2)")
    assert (res["r1_dir"] / JSON_NAME).read_bytes() == (res["r2_dir"] / JSON_NAME).read_bytes()


def test_quick_profile_finishes_in_under_60_seconds(exp1_cli):
    res = exp1_cli.get()
    require_ok(res["r1"])
    assert res["r1"].elapsed < 60.0, res["r1"].elapsed


def test_nothing_is_written_outside_the_output_dir(exp1_cli):
    res = exp1_cli.get()
    for name in ("r1", "pert", "r2"):
        require_ok(res[name], name)
    assert res["phase1_outside_problems"] == []
    assert res["phase2_outside_problems"] == []
    assert res["cwd_files"] == [], "files written to the working directory"


def test_root_seed_changes_the_results(exp1_cli):
    res = exp1_cli.get()
    a = r1_json(exp1_cli)
    b = load_json(res, "pub", "pub_dir")
    assert a["conditions"] != b["conditions"]


def test_default_output_dir_is_the_experiment_results_dir(exp1_cli):
    res = exp1_cli.get()
    require_ok(res["dflt"], "run without --output-dir")
    assert res["dflt_json"] is not None, f"{JSON_NAME} was not written to <experiment dir>/results"
    assert strict_json(res["dflt_json"].decode("utf-8"))["schema"] == spec_exp1_schema()


# ----------------------------------------------------------------------------- options are used

def test_options_change_what_is_run(exp1_cli):
    """--n-landscapes 3 --n-starts 2 --max-evals 300 --conditions none,corner --no-plots --root-seed 7"""
    res = exp1_cli.get()
    doc = load_json(res, "pert", "pert_dir")
    cps = _check_conditions(doc, expect_conditions=PERT["conditions"],
                            expect_runs=PERT["n_landscapes"] * PERT["n_starts"])
    assert max(cps) <= PERT["max_evals"]
    assert not [f for f in list_files(res["pert_dir"]) if f.endswith(".html")]


def _probe(exp1_cli):
    res = exp1_cli.get()
    probe = res.get("probe")
    assert probe is not None, f"probe driver wrote nothing\n{res['pert'].describe()}"
    assert probe["error"] is None and probe["exit"] == 0, f"{probe['error']}\n{res['pert'].describe()}"
    return probe


def test_landscapes_starts_and_searches_use_separate_seeding_roles(exp1_cli):
    """Spec 9.11: calls are rng_for(root_seed, EXP1, ROLE_*, ...), experiment first, then role."""
    calls = _probe(exp1_cli)["rng_for"]
    for role in ("ROLE_LANDSCAPE", "ROLE_START", "ROLE_SEARCH"):
        want = [PERT["root_seed"], SEEDING["EXP1"], SEEDING[role]]
        assert any(c[:3] == want for c in calls), f"no rng_for{tuple(want)} call; saw {calls[:10]}"


def test_four_arms_share_each_landscape_and_start(exp1_cli):
    opts = _probe(exp1_cli)["optimizers"]
    assert opts, "no optimiser was constructed in-process (with --workers 1)"
    assert all("error" not in o for o in opts), [o for o in opts if "error" in o][:2]
    assert all(o["x0"] is not None for o in opts), "an arm was not handed the shared start point"
    groups = {}
    for o in opts:
        groups.setdefault((tuple(o["fp"]), tuple(o["x0"])), []).append(o)
    assert len(groups) == len(PERT["conditions"]) * PERT["n_landscapes"] * PERT["n_starts"], len(groups)
    for key, members in groups.items():
        fixed = [o for o in members if o["kind"] == "fixed"]
        meta = [o for o in members if o["kind"] == "meta"]
        assert len(members) == 4 and len(fixed) == 2 and len(meta) == 2, members
        assert sorted(o["selection"] for o in meta) == ["random", "ucb"]
        gauss = [o for o in fixed if o["method"] == "gaussian"]
        assert len(gauss) == 1 and gauss[0]["step_size"] == pytest.approx(0.2), fixed
        assert sum(o["method"] == "restart" for o in fixed) == 1, fixed


def test_no_start_point_sits_on_a_peak_centre(exp1_cli):
    opts = _probe(exp1_cli)["optimizers"]
    assert opts
    for o in opts:
        assert o.get("centers"), "landscape peaks not readable"
        d = min(np.linalg.norm(np.asarray(o["x0"]) - np.asarray(c)) for c in o["centers"])
        assert d > 1e-6, (o["x0"], d)


# ----------------------------------------------------------------------------- --publish-plots-dir

def test_publish_dir_inside_the_repo_is_accepted_and_receives_the_plots(exp1_cli):
    res = exp1_cli.get()
    require_ok(res["pub"], "run with --publish-plots-dir inside the repository (dir missing beforehand)")
    assert_published(res, JSON_NAME, PLOTS)


@pytest.mark.parametrize("variant", REFUSAL_VARIANTS)
def test_publish_dir_outside_the_repo_is_refused_before_computing(variant, exp1_cli, refusals):
    res = exp1_cli.get()
    require_ok(res["pub"], "positive control: --publish-plots-dir inside the repository")
    got = refusals.get()
    assert_refused(got[variant], got["_outside_problems"])
