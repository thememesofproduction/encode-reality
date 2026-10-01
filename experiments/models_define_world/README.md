# Models Define the World — Supporting Experiments

Computational experiments for the essay *"The Models We Choose Define the Worlds We Can See"*.

## Experiments

### Experiment 1 — Recovery

An agent-based model with homogeneous mixing and fixed transition probabilities reproduces the dynamics of the SIRS ODE system. An ensemble of 20 ABM runs is compared against the deterministic ODE solution.

### Experiment 2 — Divergence

The same ABM is placed on a Barabasi-Albert scale-free network with heterogeneous per-agent transmission rates. The resulting dynamics diverge qualitatively from any SIRS parameterization.

### Experiment 3 — Irreducibility

A compliance rate sweep (0.1 to 1.0) on the network ABM reveals threshold effects and nonlinear transitions that smooth ODEs cannot capture.

## Running

```bash
# All experiments
python run_experiment.py --seed 42 --n-ensemble 20

# Individual experiment
python run_experiment.py --experiment recovery
python run_experiment.py --experiment divergence
python run_experiment.py --experiment irreducibility
```

## Output

- `results/experiment_results.json` — Full results with parameters
- `static/images/models_define_world/` — Generated plots for the blog post

## File Structure

| File | Description |
|------|-------------|
| `config.py` | Shared defaults, colors, plot settings |
| `sirs_ode.py` | SIRS ODE solver (scipy) |
| `abm_core.py` | Agent and Population classes |
| `networks.py` | Network generation (complete, BA, WS) |
| `simulation.py` | ABM simulation engine |
| `plotting.py` | Dark-themed visualization |
| `run_experiment.py` | CLI runner |

## Tests

```bash
pytest experiments/models_define_world/tests/ -v
```
