# Ideas, Intelligence, and Creativity — Supporting Experiments

Computational experiments demonstrating the structural hierarchy of ideas, intelligence, and creativity as described in the blog post ["Beyond What Works: Ideas, Intelligence, and the Courage to Be Creative"](https://encodereality.com/posts/ideas_intelligence_creativity/).

## Experiment 1: Fitness Landscape Optimization

Demonstrates the difference between a **fixed-strategy optimizer** (Level 1 — Intelligence) and a **meta-strategy optimizer** (Level 2 — Creativity) on a rugged 2D fitness landscape.

- **Intelligence**: Hill climbing with a fixed Gaussian step size. Gets trapped in local optima.
- **Creativity**: Same base mechanism, but periodically mutates its own search strategy. Escapes local optima via strategy-level innovation.

### Run

```bash
python run_experiment_1.py --n-steps 1000 --n-peaks 20 --seed 42
```

### Key Result

The fixed system plateaus at a local optimum. The creative system exhibits discontinuous jumps when strategy mutations open access to previously unreachable regions.

## Experiment 2: The Generator Hierarchy (Agent-Based)

Demonstrates the full three-level hierarchy with evolutionary agents in a grid world:

- **Level 0 — Idea**: A fixed agent with hardcoded behavior ("move toward nearest food").
- **Level 1 — Intelligence**: A population of neural agents with a **fixed architecture**. Weights are optimized via evolutionary algorithm. The architecture (what the agent can perceive and do) is frozen.
- **Level 2 — Creativity**: A meta-system that **evolves the architecture itself** — adding new sensory channels, changing network topology, expanding action spaces. Each architecture is evaluated by running a full Level 1 optimization on it.

The environment shifts at predefined points, creating qualitatively new challenges that only Level 2 can adapt to.

### Run

```bash
python run_experiment_2.py --meta-steps 20 --weight-generations 15 --seed 42
```

### Key Result

After environment shifts, Level 1 agents (fixed architecture) fail to recover because they cannot change *what* they perceive or *how* they process information. Level 2 agents evolve new capabilities (gradient sensing, deeper networks, diagonal movement) and recover.

## File Structure

```
landscape.py        - Rugged 2D fitness landscape (sum of Gaussians)
strategies.py       - Search strategies and strategy mutation
intelligence.py     - Fixed-strategy optimizer (Level 1)
creativity.py       - Meta-strategy optimizer (Level 2)
gridworld.py        - 2D grid world environment
agents.py           - Fixed agent, neural agent, architecture definitions
evolution.py        - Weight evolution (Level 1)
neuroevolution.py   - Architecture evolution (Level 2)
plotting.py         - All visualization
config.py           - Default parameters and color schemes
run_experiment_1.py - CLI runner for Experiment 1
run_experiment_2.py - CLI runner for Experiment 2
tests/              - 60 unit tests
```

## Dependencies

All dependencies are specified in the root `pyproject.toml`: numpy, scipy, matplotlib, seaborn.
