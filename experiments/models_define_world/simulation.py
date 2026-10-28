"""ABM simulation engine."""

import numpy as np
from dataclasses import dataclass, field
import networkx as nx

from .abm_core import Population, Agent, S, I, R
from .networks import get_neighbors


@dataclass
class SimulationHistory:
    """Recorded S, I, R counts over time."""
    S_counts: np.ndarray
    I_counts: np.ndarray
    R_counts: np.ndarray

    def to_dict(self) -> dict:
        return {
            "S": self.S_counts.tolist(),
            "I": self.I_counts.tolist(),
            "R": self.R_counts.tolist(),
        }


def run_abm(population: Population, n_steps: int, rng: np.random.Generator,
            network: nx.Graph | None = None, contact_rate: int = 10,
            dt: float = 0.1) -> SimulationHistory:
    """Run the ABM simulation.

    Args:
        population: The population of agents.
        n_steps: Number of simulation timesteps.
        rng: Numpy random generator.
        network: Contact network (None for random mixing).
        contact_rate: Number of random contacts per agent per step (ignored if network).
        dt: Time step size for Euler-Maruyama discretization.

    Returns:
        SimulationHistory with S, I, R count arrays.
    """
    agents = population.agents
    N = len(agents)

    s_counts = np.zeros(n_steps, dtype=int)
    i_counts = np.zeros(n_steps, dtype=int)
    r_counts = np.zeros(n_steps, dtype=int)

    for step in range(n_steps):
        # --- Snapshot current states for infection step ---
        new_infections = set()

        infected_indices = [i for i, a in enumerate(agents) if a.state == I]

        for idx in infected_indices:
            agent = agents[idx]

            if network is not None:
                contact_ids = get_neighbors(network, agent.node_id)
            else:
                contact_ids = rng.choice(N, size=min(contact_rate, N), replace=False).tolist()

            p_infect = 1 - np.exp(-agent.beta * dt / max(len(contact_ids), 1))

            for cid in contact_ids:
                neighbor = agents[cid]
                if neighbor.state == S:
                    effective_p = p_infect * (1 - neighbor.compliance)
                    if rng.random() < effective_p:
                        new_infections.add(cid)

        # Apply infections from snapshot
        for idx in new_infections:
            agents[idx].state = I

        # --- Recovery step (skip newly infected this step) ---
        for i, agent in enumerate(agents):
            if agent.state == I and i not in new_infections:
                if rng.random() < 1 - np.exp(-agent.gamma * dt):
                    agent.state = R

        # --- Immunity loss step ---
        for agent in agents:
            if agent.state == R:
                if rng.random() < 1 - np.exp(-agent.omega * dt):
                    agent.state = S

        # --- Record counts ---
        counts = {S: 0, I: 0, R: 0}
        for a in agents:
            counts[a.state] += 1
        s_counts[step] = counts[S]
        i_counts[step] = counts[I]
        r_counts[step] = counts[R]

    return SimulationHistory(
        S_counts=s_counts,
        I_counts=i_counts,
        R_counts=r_counts,
    )
