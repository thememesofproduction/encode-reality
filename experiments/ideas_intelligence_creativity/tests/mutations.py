"""Deliberate defects for test-discipline's mutate.py, and the probe that must go red for each.

Run from the repository root:
    IIC_PYTHON=<the project's interpreter> python3 ~/.claude/skills/test-discipline/scripts/mutate.py \
        --project . --catalogue experiments/ideas_intelligence_creativity/tests/mutations.py

The probes run pytest with IIC_PYTHON, or with the interpreter running this file when it is unset. It needs
numpy and scipy, which the system python3 may lack.

Each entry breaks one claim the essay relies on. If a mutation survives, the suite would not have noticed
that claim failing.
"""
import os
import shlex
import sys

_PY = shlex.quote(os.environ.get("IIC_PYTHON", sys.executable))
_EXP = "experiments/ideas_intelligence_creativity"


def _pytest(*targets: str) -> str:
    paths = " ".join(f"{_EXP}/tests/{t}" for t in targets)
    return (f"PYTHONDONTWRITEBYTECODE=1 {_PY} -m pytest -c pyproject.toml --rootdir . "
            f"-o addopts= -p no:cacheprovider -x -q {paths}")


MUTATIONS = [
    {
        "name": "rho-uses-the-coordinate-norm",
        "why": "The coordinate norm of p(z) fakes sub-floor residuals in dual and split algebras. If the "
               "number-system tests still pass with it, the claim that only copies of C escape the real "
               "floor is untested.",
        "file": f"{_EXP}/number_systems.py",
        "find": "        a, b = z\n        if self.kind == \"dual\":",
        "replace": "        a, b = z\n        if self.kind != \"complex\":\n"
                   "            return self.coord_norm(z, coeffs)\n        if self.kind == \"dual\":",
        "must_fail": [_pytest("test_number_systems.py")],
        "tier": 1,
        "expect": "caught",
    },
    {
        "name": "level2-scores-algebras-by-coordinate-norm",
        "why": "Level 2 judged by the coordinate norm can 'escape' the floor inside an algebra with no "
               "roots. Contract 4 (escape only through C) must notice; otherwise the headline result "
               "could be an artefact of the metric.",
        "file": f"{_EXP}/experiment3.py",
        "find": "        return self.alg.rho(z)",
        "replace": "        return self.alg.coord_norm(z)",
        "must_fail": [_pytest("test_experiment3.py::test_contract4_level2_escapes_the_floor_only_through_c")],
        "tier": 2,
        "expect": "caught",
    },
    {
        "name": "budget-batch-overruns",
        "why": "If a batch can evaluate past the budget, arms get unequal compute and the "
               "evaluation-matched comparison is void.",
        "file": f"{_EXP}/search_core.py",
        "find": "        n = min(len(xs), self.remaining)",
        "replace": "        n = len(xs)",
        "must_fail": [_pytest("test_search_core.py")],
        "tier": 1,
        "expect": "caught",
    },
    {
        "name": "optimiser-ignores-x0",
        "why": "The original bug: arms that ignore the shared start are not paired, and the trajectory "
               "plot shows a start that never happened.",
        "file": f"{_EXP}/intelligence.py",
        "find": "start = self.rng.uniform(lo, hi, size=landscape.dims) if x0 is None else x0",
        "replace": "start = self.rng.uniform(lo, hi, size=landscape.dims)",
        "must_fail": [_pytest("test_optimizers_repaired.py")],
        "tier": 1,
        "expect": "caught",
    },
    {
        "name": "meta-never-tries-new-strategies-first",
        "why": "The original no-op meta-level: untried strategies never get a turn, so 'meta-selection' "
               "is the initial strategy forever.",
        "file": f"{_EXP}/creativity.py",
        "find": "        if untried:\n            self.active_idx = untried[0]",
        "replace": "        if False:\n            self.active_idx = untried[0]",
        "must_fail": [_pytest("test_optimizers_repaired.py")],
        "tier": 1,
        "expect": "caught",
    },
    {
        "name": "meta-score-is-raw-improvement",
        "why": "Scoring by raw improvement rewards strategies that spend more evaluations; the spec says "
               "improvement per evaluation.",
        "file": f"{_EXP}/creativity.py",
        "find": "return None if self.evals == 0 else self.improvement / self.evals",
        "replace": "return None if self.evals == 0 else self.improvement",
        "must_fail": [_pytest("test_optimizers_repaired.py")],
        "tier": 1,
        "expect": "caught",
    },
    {
        "name": "ceiling-skips-local-refinement",
        "why": "Without local refinement the 'computed ceiling' is a grid or peak-centre value, which "
               "underestimates overlapping peaks, so success rates can exceed what was really achieved.",
        "file": f"{_EXP}/landscape.py",
        "find": "        for s in list(centres) + list(grid[top]):",
        "replace": "        for s in []:",
        "must_fail": [_pytest("test_landscape_repaired.py")],
        "tier": 1,
        "expect": "caught",
    },
    {
        "name": "publish-guard-accepts-anything",
        "why": "The runners once wrote straight into the published static/images. The guard is what "
               "keeps a run from writing outside the repository.",
        "file": f"{_EXP}/cli_common.py",
        "find": "    if not target.is_relative_to(REPO_ROOT.resolve()):",
        "replace": "    if False:",
        "must_fail": [_pytest("test_run_experiment_3.py::test_publish_dir_outside_the_repo_is_refused_before_computing")],
        "tier": 2,
        "expect": "caught",
    },
    {
        "name": "exp3-start-depends-on-the-arm",
        "why": "The README and the essay call Experiment 3's runs paired. If the start stream is keyed by "
               "the arm, arms with the same seed start in different places and comparisons between them are "
               "unpaired.",
        "file": f"{_EXP}/experiment3.py",
        "find": "    start = _stream(root_seed, ROLE_START, SHARED_START, seed)",
        "replace": "    start = _stream(root_seed, ROLE_START, arm_id, seed)",
        "must_fail": [_pytest("test_exp3_pairing.py")],
        "tier": 1,
        "expect": "caught",
    },
    # Round 2 (SPEC section 10): the instruments and the plain-language figures.
    {
        "name": "js-rho-treats-dual-as-split",
        "why": "The browser must compute the same closeness as the experiment; a dual algebra scored like a "
               "split one would let the instrument disagree with the essay's numbers.",
        "file": f"{_EXP}/figkit/systems/numsys_math.js",
        "find": "  if (kind === 'dual') return real(a + b * (beta / 2));",
        "replace": "  if (kind === 'dual' && false) return real(a + b * (beta / 2));",
        "must_fail": [_pytest("test_numsys_math_js.py")],
        "tier": 1,
        "expect": "caught",
    },
    {
        "name": "privacy-guard-names-skipped",
        "why": "The deploy lint blocks pages naming the owner or the repo; the build must refuse first.",
        "file": f"{_EXP}/figkit/build.py",
        "find": "        if name in lowered:\n",
        "replace": "        if name in lowered and False:\n",
        "must_fail": [_pytest("test_figkit_build.py")],
        "tier": 1,
        "expect": "caught",
    },
    {
        "name": "cell-code-roots-counted-as-below",
        "why": "The instrument's map must mark exactly the cells where a root was found.",
        "file": f"{_EXP}/instruments.py",
        "find": "    if log10_ratio < math.log10(ROOT_TOL):\n        return 2",
        "replace": "    if log10_ratio < math.log10(ROOT_TOL):\n        return 1",
        "must_fail": [_pytest("test_instruments.py")],
        "tier": 1,
        "expect": "caught",
    },
    {
        "name": "tour-button-1-shows-level-2",
        "why": "Step 1 of the tour must show only the searches on the line.",
        "file": f"{_EXP}/plots_number_systems.py",
        "find": "\"args\": [{\"visible\": line_only}, {\"annotations\": line_notes}]",
        "replace": "\"args\": [{\"visible\": new_space}, {\"annotations\": line_notes}]",
        "must_fail": [_pytest("test_plain_language.py")],
        "tier": 1,
        "expect": "caught",
    },
    {
        "name": "figure-d-shares-not-percent",
        "why": "Figure D's shares must read as percentages for a lay reader.",
        "file": f"{_EXP}/plots_landscape.py",
        "find": "    layout[\"yaxis\"].update(tickformat=\".0%\")",
        "replace": "    layout[\"yaxis\"].update(tickformat=\"\")",
        "must_fail": [_pytest("test_plain_language.py")],
        "tier": 1,
        "expect": "caught",
    },
    {
        "name": "data-can-close-its-script",
        "why": "Data containing </script> must not end the data block early and inject markup.",
        "file": f"{_EXP}/figkit/build.py",
        "find": "text.replace(\"<\", \"\\\\u003c\")",
        "replace": "text",
        "must_fail": [_pytest("test_figkit_build.py")],
        "tier": 1,
        "expect": "caught",
    },
]

if __name__ == "__main__":
    sys.exit("run this through mutate.py, not directly")
