"""Session fixtures shared by the runner and plot tests (SPEC.md sections 5, 6 and 7).

Each fixture returns a Lazy: the runner processes start on first use inside a test body, so a
missing runner is reported as failed tests, not as fixture errors.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.tests._iic_helpers import (  # noqa: E402
    Lazy, run_runner_suite, spec_json_name, spec_plot_names,
)


@pytest.fixture(scope="session")
def exp3_cli(tmp_path_factory):
    base = tmp_path_factory.mktemp("exp3_cli")
    return Lazy(lambda: run_runner_suite(
        "run_experiment_3.py", spec_json_name(5), spec_plot_names(5),
        pert_args=["--no-plots", "--n-seeds", "2"], base=base))


@pytest.fixture(scope="session")
def exp1_cli(tmp_path_factory):
    base = tmp_path_factory.mktemp("exp1_cli")
    return Lazy(lambda: run_runner_suite(
        "run_experiment_1.py", spec_json_name(6), spec_plot_names(6),
        pert_args=["--no-plots", "--n-landscapes", "3", "--n-starts", "2", "--max-evals", "300",
                   "--conditions", "none,corner", "--root-seed", "7"],
        base=base, instrument=True))
