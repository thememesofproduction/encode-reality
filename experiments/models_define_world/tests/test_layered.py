"""Tests for layered network and interventions."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from experiments.models_define_world.abm_core import Population, S, I, R
from experiments.models_define_world.layered_network import (
    generate_layered_network, get_layer_subgraph, get_layer_stats,
)
from experiments.models_define_world.interventions import (
    random_vaccination, targeted_vaccination, layer_contact_reduction,
)
from experiments.models_define_world.simulation import run_abm


class TestLayeredNetwork:
    def test_node_count(self):
        G = generate_layered_network(100, seed=42)
        assert G.number_of_nodes() == 100

    def test_has_all_layers(self):
        G = generate_layered_network(200, seed=42)
        layers = {d["layer"] for _, _, d in G.edges(data=True)}
        assert "household" in layers
        assert "workplace" in layers
        assert "community" in layers

    def test_deterministic_with_seed(self):
        G1 = generate_layered_network(100, seed=42)
        G2 = generate_layered_network(100, seed=42)
        assert set(G1.edges()) == set(G2.edges())

    def test_mean_degree_reasonable(self):
        """Mean degree should be roughly 8-20 for a 1000-agent network."""
        G = generate_layered_network(1000, seed=42)
        stats = get_layer_stats(G)
        mean_deg = stats["total"]["mean_degree"]
        assert 5 < mean_deg < 25, f"Mean degree {mean_deg} outside expected range"

    def test_layer_subgraph(self):
        G = generate_layered_network(200, seed=42)
        hh = get_layer_subgraph(G, "household")
        wp = get_layer_subgraph(G, "workplace")
        cm = get_layer_subgraph(G, "community")
        assert hh.number_of_edges() > 0
        assert wp.number_of_edges() > 0
        assert cm.number_of_edges() > 0
        # Layer edges should sum to total
        assert hh.number_of_edges() + wp.number_of_edges() + cm.number_of_edges() == G.number_of_edges()


class TestInterventions:
    def test_random_vaccination_count(self):
        pop = Population(N=100, I0=5, beta=0.3, gamma=0.1, omega=0.01, seed=42)
        rng = np.random.default_rng(42)
        count = random_vaccination(pop, fraction=0.2, rng=rng)
        # Should vaccinate ~20 agents (some may already be I→R won't change state)
        assert count > 0
        r_count = sum(1 for a in pop.agents if a.state == R)
        assert r_count >= 20

    def test_targeted_vaccination_picks_high_degree(self):
        G = generate_layered_network(100, seed=42)
        pop = Population(N=100, I0=0, beta=0.3, gamma=0.1, omega=0.01, seed=42)
        rng = np.random.default_rng(42)

        degrees = dict(G.degree())
        targeted_vaccination(pop, G, fraction=0.1, rng=rng)

        vaccinated_ids = [i for i, a in enumerate(pop.agents) if a.state == R]
        vaccinated_degrees = [degrees[i] for i in vaccinated_ids]
        all_degrees = sorted(degrees.values(), reverse=True)

        # Vaccinated agents should have above-median degree
        median_deg = np.median(list(degrees.values()))
        mean_vax_deg = np.mean(vaccinated_degrees)
        assert mean_vax_deg > median_deg, \
            f"Targeted vax mean degree {mean_vax_deg} not above median {median_deg}"

    def test_layer_contact_reduction(self):
        G = generate_layered_network(200, seed=42)
        rng = np.random.default_rng(42)
        wp_edges_before = sum(1 for _, _, d in G.edges(data=True) if d["layer"] == "workplace")

        G_reduced = layer_contact_reduction(G, "workplace", reduction=0.5, rng=rng)
        wp_edges_after = sum(1 for _, _, d in G_reduced.edges(data=True) if d["layer"] == "workplace")

        # Should remove ~50% of workplace edges
        assert wp_edges_after < wp_edges_before
        expected = wp_edges_before * 0.5
        assert abs(wp_edges_after - expected) < wp_edges_before * 0.1

        # Original should be unchanged
        wp_original = sum(1 for _, _, d in G.edges(data=True) if d["layer"] == "workplace")
        assert wp_original == wp_edges_before

    def test_interventions_reduce_infection(self):
        """Vaccination should reduce peak infection vs no intervention."""
        G = generate_layered_network(200, seed=42)

        # No intervention
        pop1 = Population(N=200, I0=10, beta=0.3, gamma=0.1, omega=0.01, seed=42)
        rng1 = np.random.default_rng(42)
        h1 = run_abm(pop1, n_steps=500, rng=rng1, network=G, dt=0.1)
        peak_no_vax = np.max(h1.I_counts)

        # With 30% vaccination
        pop2 = Population(N=200, I0=10, beta=0.3, gamma=0.1, omega=0.01, seed=42)
        rng2 = np.random.default_rng(99)
        random_vaccination(pop2, fraction=0.3, rng=rng2)
        rng2b = np.random.default_rng(42)
        h2 = run_abm(pop2, n_steps=500, rng=rng2b, network=G, dt=0.1)
        peak_vax = np.max(h2.I_counts)

        assert peak_vax <= peak_no_vax, \
            f"Vaccination peak {peak_vax} should be <= no-vax peak {peak_no_vax}"
