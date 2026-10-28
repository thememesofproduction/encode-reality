"""Intervention strategies for ABM experiments."""

import numpy as np
import networkx as nx

from .abm_core import R


def random_vaccination(population, fraction, rng):
    """Vaccinate a random fraction of susceptible agents (set to R).

    Args:
        population: Population object.
        fraction: Fraction of population to vaccinate (0-1).
        rng: Numpy random generator.

    Returns:
        Number of agents vaccinated.
    """
    n_to_vaccinate = int(len(population.agents) * fraction)
    indices = rng.choice(len(population.agents), size=n_to_vaccinate, replace=False)
    count = 0
    for idx in indices:
        if population.agents[idx].state != R:
            population.agents[idx].state = R
            count += 1
    return count


def targeted_vaccination(population, network, fraction, rng):
    """Vaccinate the highest-degree nodes first.

    Args:
        population: Population object.
        network: nx.Graph contact network.
        fraction: Fraction of population to vaccinate (0-1).
        rng: Numpy random generator (used for tiebreaking).

    Returns:
        Number of agents vaccinated.
    """
    n_to_vaccinate = int(len(population.agents) * fraction)
    # Sort by degree descending, with random tiebreaking
    degrees = dict(network.degree())
    nodes_by_degree = sorted(degrees.keys(),
                             key=lambda n: (degrees[n], rng.random()),
                             reverse=True)
    count = 0
    for node_id in nodes_by_degree:
        if count >= n_to_vaccinate:
            break
        if population.agents[node_id].state != R:
            population.agents[node_id].state = R
            count += 1
    return count


def layer_contact_reduction(network, layer, reduction, rng):
    """Remove a fraction of edges from a specific network layer.

    Args:
        network: nx.Graph with 'layer' edge attributes.
        layer: Layer name to reduce (e.g., 'workplace', 'community').
        reduction: Fraction of layer edges to remove (0-1).
        rng: Numpy random generator.

    Returns:
        A new graph with reduced edges. Original is not modified.
    """
    G_new = network.copy()
    layer_edges = [(u, v) for u, v, d in G_new.edges(data=True)
                   if d.get("layer") == layer]

    n_remove = int(len(layer_edges) * reduction)
    if n_remove > 0:
        indices = rng.choice(len(layer_edges), size=n_remove, replace=False)
        edges_to_remove = [layer_edges[i] for i in indices]
        G_new.remove_edges_from(edges_to_remove)

    return G_new
