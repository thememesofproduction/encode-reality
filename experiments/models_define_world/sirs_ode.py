"""SIRS ODE model solved with scipy."""

import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp


@dataclass
class SIRSResult:
    """Solution of the SIRS ODE system."""
    t: np.ndarray
    S: np.ndarray
    I: np.ndarray
    R: np.ndarray

    def to_dict(self) -> dict:
        return {
            "t": self.t.tolist(),
            "S": self.S.tolist(),
            "I": self.I.tolist(),
            "R": self.R.tolist(),
        }


def _sirs_deriv(t, y, beta, gamma, omega, N):
    S, I, R = y
    dSdt = -beta * S * I / N + omega * R
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I - omega * R
    return [dSdt, dIdt, dRdt]


def solve_sirs(beta: float, gamma: float, omega: float, N: int, I0: int,
               t_max: float, n_points: int = 1000) -> SIRSResult:
    """Solve the SIRS ODE system.

    Args:
        beta: Infection rate.
        gamma: Recovery rate.
        omega: Immunity loss rate.
        N: Total population.
        I0: Initial number of infected.
        t_max: Simulation duration.
        n_points: Number of time points in output.

    Returns:
        SIRSResult with t, S, I, R arrays.
    """
    S0 = N - I0
    R0_init = 0.0
    y0 = [float(S0), float(I0), float(R0_init)]
    t_eval = np.linspace(0, t_max, n_points)

    sol = solve_ivp(
        _sirs_deriv, [0, t_max], y0,
        args=(beta, gamma, omega, N),
        method="RK45",
        t_eval=t_eval,
        rtol=1e-8, atol=1e-8,
    )

    return SIRSResult(
        t=sol.t,
        S=sol.y[0],
        I=sol.y[1],
        R=sol.y[2],
    )
