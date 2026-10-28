"""Local hidden-variable models and CHSH computation.

Implements several classes of LHV strategies and computes the CHSH
quantity S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2) for each.
The classical bound |S| <= 2 holds universally.
"""

import numpy as np
from itertools import product
from scipy.optimize import differential_evolution

from experiments.bells_inequality.config import (
    OPTIMAL_ANGLES, EXHAUSTION_DEFAULTS,
)


def chsh_s(E_a1b1, E_a1b2, E_a2b1, E_a2b2):
    """Compute CHSH quantity S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)."""
    return E_a1b1 - E_a1b2 + E_a2b1 + E_a2b2


def chsh_single_lambda(A0, A1, B0, B1):
    """CHSH value for a single deterministic assignment.

    A0, A1 are Alice's outputs for settings a1, a2.
    B0, B1 are Bob's outputs for settings b1, b2.
    All values in {+1, -1}.

    Returns X = A0*B0 - A0*B1 + A1*B0 + A1*B1, which equals +/-2.
    """
    return A0 * B0 - A0 * B1 + A1 * B0 + A1 * B1


# ── Deterministic strategies ──────────────────────────────────────────

def enumerate_deterministic():
    """Enumerate all 16 deterministic strategies.

    Returns array of shape (16,) with S values. Every value is +/-2.
    """
    S_values = []
    for A0, A1, B0, B1 in product([-1, 1], repeat=4):
        S_values.append(chsh_single_lambda(A0, A1, B0, B1))
    return np.array(S_values, dtype=float)


# ── Random discrete strategies ────────────────────────────────────────

def random_discrete_lhv(n_strategies, n_lambda, rng):
    """Generate random LHV models with discrete hidden variable.

    Each strategy has n_lambda hidden states. For each lambda, response
    functions A(a,lambda) and B(b,lambda) are random +/-1. S is computed
    by averaging over all lambda (uniform distribution).

    Returns array of shape (n_strategies,) with S values.
    """
    S_values = np.empty(n_strategies)
    for i in range(n_strategies):
        # Random lookup tables: shape (n_lambda, 4) for A0, A1, B0, B1
        table = rng.choice([-1, 1], size=(n_lambda, 4))
        A0, A1, B0, B1 = table[:, 0], table[:, 1], table[:, 2], table[:, 3]
        X = chsh_single_lambda(A0, A1, B0, B1)
        S_values[i] = np.mean(X)
    return S_values


# ── Random continuous strategies ──────────────────────────────────────

def random_continuous_lhv(n_strategies, n_lambda, rng):
    """Generate random LHV models with continuous hidden variable.

    lambda is sampled from various distributions. Response functions use
    random thresholds: A(a, lambda) = sign(lambda - threshold_a).

    Returns array of shape (n_strategies,) with S values.
    """
    S_values = np.empty(n_strategies)
    for i in range(n_strategies):
        # Random distribution for lambda: pick from uniform, gaussian, beta
        dist_type = rng.integers(0, 3)
        if dist_type == 0:
            lambdas = rng.uniform(0, 1, size=n_lambda)
        elif dist_type == 1:
            lambdas = rng.normal(0, 1, size=n_lambda)
        else:
            a_param, b_param = rng.uniform(0.5, 5, size=2)
            lambdas = rng.beta(a_param, b_param, size=n_lambda)

        # Random thresholds for each response function
        thresholds = rng.uniform(lambdas.min(), lambdas.max(), size=4)
        A0 = np.where(lambdas < thresholds[0], 1, -1)
        A1 = np.where(lambdas < thresholds[1], 1, -1)
        B0 = np.where(lambdas < thresholds[2], 1, -1)
        B1 = np.where(lambdas < thresholds[3], 1, -1)

        X = chsh_single_lambda(A0, A1, B0, B1)
        S_values[i] = np.mean(X)
    return S_values


# ── Adversarial optimization ─────────────────────────────────────────

def _adversarial_objective(params, n_lambda):
    """Objective for adversarial optimizer: negative |S|.

    Parameterization: 4 thresholds in [0,1] for response functions,
    lambda uniform on [0,1]. Response is A(a,lambda) = sign(lambda - t).
    """
    t_A0, t_A1, t_B0, t_B1 = params
    # Analytic: E(a,b) = 1 - 2|t_A - t_B| for uniform lambda + thresholds
    E_a1b1 = 1 - 2 * abs(t_A0 - t_B0)
    E_a1b2 = 1 - 2 * abs(t_A0 - t_B1)
    E_a2b1 = 1 - 2 * abs(t_A1 - t_B0)
    E_a2b2 = 1 - 2 * abs(t_A1 - t_B1)
    S = chsh_s(E_a1b1, E_a1b2, E_a2b1, E_a2b2)
    return -abs(S)


def adversarial_optimize(n_restarts, rng):
    """Attempt to maximize |S| over parameterized LHV models.

    Uses differential evolution with multiple restarts to search
    aggressively. Returns array of best |S| from each restart.
    """
    n_lambda = EXHAUSTION_DEFAULTS["n_lambda"]
    best_S_values = np.empty(n_restarts)

    for i in range(n_restarts):
        result = differential_evolution(
            _adversarial_objective,
            bounds=[(0, 1)] * 4,
            args=(n_lambda,),
            seed=int(rng.integers(0, 2**31)),
            maxiter=200,
            tol=1e-10,
        )
        best_S_values[i] = -result.fun  # convert back to |S|
    return best_S_values


# ── Full experiment ───────────────────────────────────────────────────

def run_exhaustion_experiment(seed=None, n_random_discrete=None,
                              n_random_continuous=None, n_adversarial=None,
                              n_lambda=None):
    """Run Experiment 1: exhaust the LHV type space.

    Returns dict with S values for each category:
        deterministic: array of shape (16,)
        random_discrete: array of shape (n_random_discrete,)
        random_continuous: array of shape (n_random_continuous,)
        adversarial: array of shape (n_adversarial,) — best |S| per restart
    """
    cfg = EXHAUSTION_DEFAULTS
    seed = seed or cfg["seed"]
    n_random_discrete = n_random_discrete or cfg["n_random_discrete"]
    n_random_continuous = n_random_continuous or cfg["n_random_continuous"]
    n_adversarial = n_adversarial or cfg["n_adversarial"]
    n_lambda = n_lambda or cfg["n_lambda"]
    rng = np.random.default_rng(seed)

    print("\n--- Experiment 1: Exhausting the Type ---")

    # 1. All deterministic strategies
    det_S = enumerate_deterministic()
    print(f"  Deterministic: {len(det_S)} strategies, all |S| = "
          f"{np.unique(np.abs(det_S))}")

    # 2. Random discrete
    print(f"  Sampling {n_random_discrete} random discrete LHV models...")
    disc_S = random_discrete_lhv(n_random_discrete, n_lambda, rng)
    print(f"  Random discrete: max |S| = {np.max(np.abs(disc_S)):.4f}")

    # 3. Random continuous
    print(f"  Sampling {n_random_continuous} random continuous LHV models...")
    cont_S = random_continuous_lhv(n_random_continuous, n_lambda, rng)
    print(f"  Random continuous: max |S| = {np.max(np.abs(cont_S)):.4f}")

    # 4. Adversarial
    print(f"  Adversarial optimization ({n_adversarial} restarts)...")
    adv_S = adversarial_optimize(n_adversarial, rng)
    print(f"  Adversarial: max |S| = {np.max(adv_S):.6f}")

    return {
        "deterministic": det_S,
        "random_discrete": disc_S,
        "random_continuous": cont_S,
        "adversarial": adv_S,
    }
