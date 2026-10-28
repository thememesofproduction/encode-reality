"""SPEC.md section 1: seeding.py -- independent, reproducible random streams."""
import itertools
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.ideas_intelligence_creativity.tests._iic_helpers import load, spec_int_constants  # noqa: E402

CONSTS = spec_int_constants(1)          # EXP1, EXP3, ROLE_* parsed from the spec's code block
ROLES = sorted(k for k in CONSTS if k.startswith("ROLE_"))
EXPS = sorted(k for k in CONSTS if k.startswith("EXP"))


def fingerprint(gen):
    return (tuple(gen.integers(0, 2 ** 62, size=4).tolist()), tuple(gen.random(4).tolist()))


def test_constants_have_the_spec_values():
    # guards the discovery itself: the spec names two experiments and four roles
    assert len(EXPS) == 2 and len(ROLES) == 4, CONSTS
    sd = load("seeding")
    for name, value in CONSTS.items():
        assert getattr(sd, name) == value, name


def test_returns_a_numpy_generator():
    sd = load("seeding")
    assert isinstance(sd.rng_for(0, CONSTS["EXP1"], CONSTS["ROLE_LANDSCAPE"]), np.random.Generator)


def test_same_arguments_give_the_same_stream():
    sd = load("seeding")
    rng = np.random.default_rng(2024)
    for _ in range(60):
        args = [int(rng.integers(0, 2 ** 31))] + [int(v) for v in rng.integers(0, 1000, size=int(rng.integers(1, 5)))]
        a, b = sd.rng_for(*args), sd.rng_for(*args)
        assert fingerprint(a) == fingerprint(b), args
        # long streams too, not only the first draws
        np.testing.assert_array_equal(sd.rng_for(*args).normal(size=500), sd.rng_for(*args).normal(size=500))


def test_stream_does_not_depend_on_global_random_state_or_other_generators():
    sd = load("seeding")
    np.random.seed(0)
    random.seed(0)
    ref = fingerprint(sd.rng_for(5, 1, 0, 3))
    np.random.seed(12345)
    random.seed(999)
    other = sd.rng_for(5, 1, 0, 3)
    other.random(1000)                      # consuming one stream ...
    assert fingerprint(sd.rng_for(5, 1, 0, 3)) == ref   # ... does not move a fresh one


def test_changing_any_key_changes_the_stream():
    sd = load("seeding")
    rng = np.random.default_rng(7)
    for _ in range(40):
        args = [int(rng.integers(0, 2 ** 31))] + [int(v) for v in rng.integers(0, 50, size=int(rng.integers(1, 5)))]
        ref = fingerprint(sd.rng_for(*args))
        for pos in range(len(args)):
            for new in {args[pos] + 1, args[pos] + 100, (args[pos] + 12345) % 2 ** 31, int(rng.integers(0, 2 ** 31))}:
                if new == args[pos] or new < 0:
                    continue
                changed = list(args)
                changed[pos] = new
                assert fingerprint(sd.rng_for(*changed)) != ref, (args, changed)


def test_streams_are_distinct_over_a_grid_of_seeds_experiments_roles_and_indices():
    # Offsets such as root + 100*role or root + 1000*exp collide inside this grid.
    sd = load("seeding")
    roots = [0, 1, 2, 3, 42, 100, 101, 142, 1000, 1001, 1042, 10000, 10042, 2 ** 31 - 1]
    seen = {}
    for root, exp, role, idx in itertools.product(roots, EXPS, ROLES, [0, 1, 2, 3, 10, 100]):
        key = (root, CONSTS[exp], CONSTS[role], idx)
        fp = fingerprint(sd.rng_for(*key))
        assert fp not in seen, f"{key} gives the same stream as {seen.get(fp)}"
        seen[fp] = key
    assert len(seen) == len(roots) * len(EXPS) * len(ROLES) * 6


def test_streams_for_different_roles_never_coincide():
    sd = load("seeding")
    extras = [(), (0,), (1,), (2,), (0, 0), (3, 7)]
    for root in [0, 42, 7, 2 ** 20 + 3]:
        for exp in EXPS:
            for r1, r2 in itertools.combinations(ROLES, 2):
                for e1, e2 in itertools.product(extras, extras):
                    a = fingerprint(sd.rng_for(root, CONSTS[exp], CONSTS[r1], *e1))
                    b = fingerprint(sd.rng_for(root, CONSTS[exp], CONSTS[r2], *e2))
                    assert a != b, (root, exp, r1, e1, r2, e2)


def test_old_bug_landscape_and_optimiser_no_longer_share_a_stream():
    sd = load("seeding")
    e = CONSTS["EXP1"]
    land = fingerprint(sd.rng_for(42, e, CONSTS["ROLE_LANDSCAPE"]))
    for role in ("ROLE_START", "ROLE_SEARCH", "ROLE_META"):
        assert fingerprint(sd.rng_for(42, e, CONSTS[role])) != land, role
