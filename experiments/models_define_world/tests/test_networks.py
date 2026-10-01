"""Tests for network generation."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from experiments.models_define_world.networks import generate_network, get_neighbors


class TestGenerateNetwork:
    def test_complete_graph_size(self):
        G = generate_network("complete", n_nodes=50, seed=42)
        assert G.number_of_nodes() == 50

    def test_barabasi_albert_size(self):
        G = generate_network("barabasi_albert", n_nodes=100, seed=42, m=3)
        assert G.number_of_nodes() == 100

    def test_watts_strogatz_size(self):
        G = generate_network("watts_strogatz", n_nodes=100, seed=42, k=6, p=0.1)
        assert G.number_of_nodes() == 100

    def test_deterministic_with_seed(self):
        G1 = generate_network("barabasi_albert", n_nodes=100, seed=42, m=3)
        G2 = generate_network("barabasi_albert", n_nodes=100, seed=42, m=3)
        assert list(G1.edges()) == list(G2.edges())

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            generate_network("nonexistent", n_nodes=50, seed=42)


class TestGetNeighbors:
    def test_returns_list(self):
        G = generate_network("complete", n_nodes=10, seed=42)
        neighbors = get_neighbors(G, 0)
        assert isinstance(neighbors, list)

    def test_complete_graph_all_neighbors(self):
        G = generate_network("complete", n_nodes=10, seed=42)
        neighbors = get_neighbors(G, 0)
        assert len(neighbors) == 9
