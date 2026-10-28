"""SPEC.md section 1: independent, reproducible random streams.

Every random stream in the experiments comes from ``rng_for(root_seed, *keys)``: the experiment
number first, then the role, then any indices. The same keys always give the same stream, and
changing any key gives a different one. There is no global state and no cached generator.
"""

import numpy as np

EXP1 = 1
EXP3 = 3

ROLE_LANDSCAPE = 0
ROLE_START = 1
ROLE_SEARCH = 2
ROLE_META = 3


def rng_for(root_seed: int, *keys: int) -> np.random.Generator:
    """A fresh generator for ``(root_seed, *keys)``, built from a SeedSequence with that spawn key."""
    seq = np.random.SeedSequence(entropy=root_seed, spawn_key=tuple(int(k) for k in keys))
    return np.random.default_rng(seq)
