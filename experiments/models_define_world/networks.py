"""Network generation for structured contact patterns."""

import networkx as nx


def generate_network(graph_type: str, n_nodes: int, seed: int, **kwargs) -> nx.Graph:
    """Generate a contact network.

    Args:
        graph_type: One of "complete", "barabasi_albert", "watts_strogatz".
        n_nodes: Number of nodes.
        seed: Random seed for reproducibility.
        **kwargs: Graph-type-specific parameters (m, k, p).

    Returns:
        A networkx Graph.
    """
    if graph_type == "complete":
        return nx.complete_graph(n_nodes)
    elif graph_type == "barabasi_albert":
        m = kwargs.get("m", 5)
        return nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    elif graph_type == "watts_strogatz":
        k = kwargs.get("k", 10)
        p = kwargs.get("p", 0.1)
        return nx.watts_strogatz_graph(n_nodes, k, p, seed=seed)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


def get_neighbors(graph: nx.Graph, node_id: int) -> list[int]:
    """Get the neighbors of a node."""
    return list(graph.neighbors(node_id))
