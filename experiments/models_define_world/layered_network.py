"""Layered contact network: household + workplace + community."""

import networkx as nx
import numpy as np


def generate_layered_network(n_agents, seed=42, household_sizes=(2, 3, 4, 5, 6),
                             workplace_size=20, workplace_p=0.3, community_m=2):
    """Build a composite contact network from three layers.

    Layers:
        - Household: fully connected cliques of size 2-6
        - Workplace: Erdos-Renyi subgraphs within groups of ~20
        - Community: Barabasi-Albert random graph across all agents

    Args:
        n_agents: Total number of agents.
        seed: Random seed.
        household_sizes: Tuple of possible household sizes.
        workplace_size: Approximate size of workplace groups.
        workplace_p: Edge probability within workplace groups.
        community_m: BA attachment parameter for community layer.

    Returns:
        nx.Graph with edge attribute 'layer' in {'household', 'workplace', 'community'}.
        Also returns a dict of layer-specific subgraphs for analysis.
    """
    rng = np.random.default_rng(seed)

    G = nx.Graph()
    G.add_nodes_from(range(n_agents))

    # --- Household layer ---
    agent_ids = list(range(n_agents))
    rng.shuffle(agent_ids)
    idx = 0
    households = []
    while idx < n_agents:
        size = int(rng.choice(household_sizes))
        size = min(size, n_agents - idx)
        hh = agent_ids[idx:idx + size]
        households.append(hh)
        # Fully connected within household
        for i in range(len(hh)):
            for j in range(i + 1, len(hh)):
                G.add_edge(hh[i], hh[j], layer="household")
        idx += size

    # --- Workplace layer ---
    agent_ids_wp = list(range(n_agents))
    rng.shuffle(agent_ids_wp)
    idx = 0
    workplaces = []
    while idx < n_agents:
        size = min(workplace_size, n_agents - idx)
        wp = agent_ids_wp[idx:idx + size]
        workplaces.append(wp)
        # Erdos-Renyi within workplace
        for i in range(len(wp)):
            for j in range(i + 1, len(wp)):
                if rng.random() < workplace_p:
                    if not G.has_edge(wp[i], wp[j]):
                        G.add_edge(wp[i], wp[j], layer="workplace")
                    # If edge already exists from household, keep the household label
        idx += size

    # --- Community layer ---
    ba = nx.barabasi_albert_graph(n_agents, community_m, seed=int(seed))
    for u, v in ba.edges():
        if not G.has_edge(u, v):
            G.add_edge(u, v, layer="community")

    return G


def get_layer_subgraph(G, layer):
    """Extract edges belonging to a specific layer."""
    edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("layer") == layer]
    sub = nx.Graph()
    sub.add_nodes_from(G.nodes())
    sub.add_edges_from(edges)
    return sub


def get_layer_stats(G):
    """Return per-layer edge counts and mean degree."""
    layers = {}
    for u, v, d in G.edges(data=True):
        layer = d.get("layer", "unknown")
        layers.setdefault(layer, 0)
        layers[layer] += 1

    stats = {}
    n = G.number_of_nodes()
    for layer, count in layers.items():
        stats[layer] = {
            "edges": count,
            "mean_degree": 2 * count / n if n > 0 else 0,
        }
    stats["total"] = {
        "edges": G.number_of_edges(),
        "mean_degree": 2 * G.number_of_edges() / n if n > 0 else 0,
    }
    return stats
