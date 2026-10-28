"""Agent-Based Model core: Agent and Population."""

import numpy as np
from dataclasses import dataclass, field

S = 0  # Susceptible
I = 1  # Infected
R = 2  # Recovered


@dataclass
class Agent:
    """An individual in the population."""
    state: int
    beta: float
    gamma: float
    omega: float
    compliance: float = 0.0
    node_id: int = 0


class Population:
    """A population of agents for ABM simulation."""

    def __init__(self, N: int, I0: int, beta: float, gamma: float, omega: float,
                 seed: int, heterogeneity: bool = False, beta_std: float = 0.0):
        self.N = N
        self.I0 = I0
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        self.seed = seed
        self.heterogeneity = heterogeneity
        self.beta_std = beta_std

        rng = np.random.default_rng(seed)
        self.agents: list[Agent] = []

        for i in range(N):
            if heterogeneity and beta_std > 0:
                agent_beta = float(np.clip(rng.normal(beta, beta_std), 0.01, 1.0))
            else:
                agent_beta = beta

            state = I if i < I0 else S
            self.agents.append(Agent(
                state=state,
                beta=agent_beta,
                gamma=gamma,
                omega=omega,
                node_id=i,
            ))

    def get_counts(self) -> dict[int, int]:
        counts = {S: 0, I: 0, R: 0}
        for a in self.agents:
            counts[a.state] += 1
        return counts

    def reset(self):
        """Reset all agents to initial S/I distribution."""
        for i, a in enumerate(self.agents):
            a.state = I if i < self.I0 else S
