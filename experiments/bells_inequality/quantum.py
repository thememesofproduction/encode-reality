"""Quantum singlet state measurements and CHSH violation.

The singlet state |psi-> = (|01> - |10>)/sqrt(2) produces correlations
E(a,b) = -cos(theta_a - theta_b), violating the classical CHSH bound
of 2 with a maximum |S| = 2*sqrt(2) (Tsirelson's bound).
"""

import numpy as np

from experiments.bells_inequality.config import (
    OPTIMAL_ANGLES, QUANTUM_DEFAULTS, TSIRELSON_BOUND,
)


def singlet_correlation(theta_a, theta_b):
    """Exact correlation for singlet state: E(a,b) = -cos(theta_a - theta_b).

    This follows from the Born rule applied to projective spin measurements
    on the maximally entangled singlet state |psi-> = (|01> - |10>)/sqrt(2).
    """
    return -np.cos(theta_a - theta_b)


def quantum_chsh_s(a1, a2, b1, b2):
    """Compute CHSH S for singlet state at given measurement angles.

    S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)
    """
    return (singlet_correlation(a1, b1)
            - singlet_correlation(a1, b2)
            + singlet_correlation(a2, b1)
            + singlet_correlation(a2, b2))


def singlet_joint_probabilities(theta_a, theta_b):
    """Joint measurement probabilities for the singlet state.

    Returns dict with keys (a_outcome, b_outcome) -> probability,
    where outcomes are +1 or -1.

    P(+1,+1) = P(-1,-1) = sin^2(delta/2) / 2
    P(+1,-1) = P(-1,+1) = cos^2(delta/2) / 2
    where delta = theta_a - theta_b.
    """
    delta = theta_a - theta_b
    sin2 = np.sin(delta / 2) ** 2
    cos2 = np.cos(delta / 2) ** 2
    return {
        (+1, +1): sin2 / 2,
        (+1, -1): cos2 / 2,
        (-1, +1): cos2 / 2,
        (-1, -1): sin2 / 2,
    }


def simulate_measurements(theta_a, theta_b, n_trials, rng):
    """Simulate projective measurements on the singlet state.

    For each trial, sample (A, B) from the joint distribution defined
    by Born rule probabilities.

    Returns (A_outcomes, B_outcomes) each of shape (n_trials,) with values +/-1.
    """
    probs = singlet_joint_probabilities(theta_a, theta_b)
    outcomes = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
    p = [probs[o] for o in outcomes]

    indices = rng.choice(4, size=n_trials, p=p)
    A_outcomes = np.array([outcomes[i][0] for i in indices])
    B_outcomes = np.array([outcomes[i][1] for i in indices])
    return A_outcomes, B_outcomes


def estimate_correlation(A_outcomes, B_outcomes):
    """Estimate E(a,b) = <A*B> from measurement outcomes."""
    return np.mean(A_outcomes * B_outcomes)


def run_quantum_experiment(n_trials=None, seed=None):
    """Run Experiment 2: quantum CHSH violation via simulated measurements.

    Returns dict with:
        theoretical_S: exact S from E(a,b) = -cos(a-b)
        simulated_S: estimated S from n_trials measurements
        convergence_n: array of trial counts for convergence plot
        convergence_S: array of S estimates at each trial count
        correlations: dict of (setting_pair -> estimated E)
    """
    cfg = QUANTUM_DEFAULTS
    n_trials = n_trials or cfg["n_trials"]
    seed = seed or cfg["seed"]
    rng = np.random.default_rng(seed)
    angles = OPTIMAL_ANGLES

    print("\n--- Experiment 2: Quantum Violation ---")
    print(f"  Angles: a1={np.degrees(angles['a1']):.0f} "
          f"a2={np.degrees(angles['a2']):.0f} "
          f"b1={np.degrees(angles['b1']):.0f} "
          f"b2={np.degrees(angles['b2']):.0f}")

    # Theoretical
    S_theory = quantum_chsh_s(angles["a1"], angles["a2"],
                               angles["b1"], angles["b2"])
    print(f"  Theoretical S = {S_theory:.6f} (|S| = {abs(S_theory):.6f})")
    print(f"  Tsirelson bound = {TSIRELSON_BOUND:.6f}")

    # Simulate measurements for all 4 setting pairs
    pairs = [
        ("a1", "b1"), ("a1", "b2"),
        ("a2", "b1"), ("a2", "b2"),
    ]

    all_A = {}
    all_B = {}
    for a_key, b_key in pairs:
        A, B = simulate_measurements(angles[a_key], angles[b_key], n_trials, rng)
        all_A[(a_key, b_key)] = A
        all_B[(a_key, b_key)] = B

    # Compute correlations from full sample
    E = {}
    for pair in pairs:
        E[pair] = estimate_correlation(all_A[pair], all_B[pair])

    S_simulated = (E[("a1", "b1")] - E[("a1", "b2")]
                   + E[("a2", "b1")] + E[("a2", "b2")])
    print(f"  Simulated S = {S_simulated:.6f} (|S| = {abs(S_simulated):.6f})")

    # Convergence: compute S at increasing trial counts
    trial_counts = np.unique(np.geomspace(100, n_trials, num=200).astype(int))
    convergence_S = np.empty(len(trial_counts))

    for idx, n in enumerate(trial_counts):
        E_n = {}
        for pair in pairs:
            E_n[pair] = estimate_correlation(all_A[pair][:n], all_B[pair][:n])
        convergence_S[idx] = (E_n[("a1", "b1")] - E_n[("a1", "b2")]
                              + E_n[("a2", "b1")] + E_n[("a2", "b2")])

    return {
        "theoretical_S": S_theory,
        "simulated_S": S_simulated,
        "convergence_n": trial_counts,
        "convergence_S": convergence_S,
        "correlations": E,
    }
