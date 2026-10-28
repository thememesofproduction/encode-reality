# Ideas, Intelligence, and Creativity: the experiments

These experiments support the essay ["Beyond What Works: Ideas, Intelligence, and the Courage to Be
Creative"](https://encodereality.com/posts/ideas_intelligence_creativity/). Its claim is that
**intelligence** searches a fixed space of solutions, and **creativity** changes which solutions can be
reached at all.

There are two experiments, and they test two different kinds of failure:

| | Experiment 3: number systems | Experiment 1: a landscape of peaks |
|---|---|---|
| What stops the search | the **space**: the answer is not in it | the **search**: the answer is in the space, but hard to find |
| What fixes it | changing the space | exploring more within it |
| Can more effort in the same space fix it? | no, and this is proven: there is an exact floor | yes |

`SPEC.md` is the contract both experiments are built and tested against.

---

## Experiment 3: number systems (the main result)

**Problem.** Find a root of p(x) = 4x⁴ − 8x³ − 7x² + 20x + 25 = (4x² + 8x + 5)(x² − 4x + 5).
p has no real roots. Its roots are −1 ± ½i and 2 ± i.

**The floor.** Over the real numbers, the smallest |p| can be is m = 9.7702, at x* ≈ −0.9215. It is
computed directly from the critical points of p (where p′ = 0), and agrees with a 2,000,001-point grid to 4·10⁻¹². **No search over
ℝ, with any strategy or any budget, can go below m.** This is a fact about the space, not a property of
any optimiser.

**Level 1: search inside ℝ.** Four strategies:
- a fixed-step hill climb;
- the same hill climb with restarts;
- an evolution strategy with step-size adaptation and restarts;
- a portfolio that learns which strategy to use (exploratory meta-search).

**Level 2: search over number systems.** It searches the two-dimensional number systems e² = α + βe
in the box α ∈ [−1, 5], β ∈ [−4, 4]. Each candidate is scored by running the Level-1 evolution strategy
inside it.
- Every system with Δ = β² + 4α < 0 is a copy of ℂ, and contains roots.
- Systems with Δ = 0 are the dual numbers, and those with Δ > 0 are "two real numbers in disguise".
  Neither contains a root.
- The Δ < 0 region is 5.6% of the box. Level 2 starts outside it, so it is never handed ℂ.

**The residual.** A search inside any system is scored by the spectral radius of multiplication by p(z).
This does not depend on how the system is written down. The obvious alternative, the length of
p(z)'s coordinate vector, reports fake successes: it drops to 0.555·m in a dual-number system that has no
roots. The tests record this.

**Controls:**
- `c_oracle`: the search in ℂ, given by hand.
- `split_two_reals`: two real numbers, i.e. one more dimension but no new multiplication.
- `flat_joint`: a single search over (α, β, a, b) at once.
- Level 2 with mutation only, with uniform proposals only, and with no selection.

**Results** (30 paired seeds, 10⁶ evaluations of p per run; `results/number_systems_results.json`):

| Arm | Outcome |
|---|---|
| Every Level-1 arm in ℝ | Never below the floor: the minimum best/m over all runs is 1 − 6·10⁻¹⁶. The restart, ES and portfolio arms reach the floor in 30/30 runs; the fixed-step hill climb reaches it in 20/30 and is trapped at the other dip (32.5) in the rest. |
| Two real numbers (`split_two_reals`) | The same floor, in 30/30 runs. One more dimension does not help; the multiplication rule has to change. |
| ℂ given by hand (`c_oracle`) | A root in 30/30 runs. The median residual is 2·10⁻¹⁰·m after 1,000 evaluations. |
| Level 2 (`l2_evo`) | Adopts a ℂ-like system and finds a root in 30/30 runs, after a median of 12,264 evaluations. |
| Level 2, uniform proposals only | 30/30, median 12,530 evaluations. |
| Level 2, no selection | 30/30, median 21,520 evaluations. |
| Level 2, mutation only | 16/30, median 474,038 evaluations. |
| Flat joint search over (α, β, a, b) | A root in 30/30 runs. |

**What this does and doesn't show:**
- The floor is a property of the space. Searching harder, longer or more cleverly inside ℝ never gets
  below it, and adding a second real dimension doesn't either.
- Escaping takes a change in *what multiplication means*, which is the move the history of complex numbers
  records.
- Outside the ℂ region, the best any number system can achieve is exactly the floor. (A 500-evaluation
  search inside one sometimes stops higher, at the other dip.) No system is closer to ℂ than another, so
  nothing points toward it. A meta-search that only refines its current system (mutation only) succeeded in
  only about half the runs, and far later. Level 2 succeeds by proposing distant systems and recognising
  success when it happens.
- The menu of systems (the α, β family) was chosen by the designer. Level 2 is itself a search in a fixed
  space, one level up. "Level" is relative to the space being escaped.
- A flat search allowed to vary the number system also succeeds. What breaks the floor is being allowed to
  change the space; a separate meta-level is not required.

**Figures** (interactive, in `results/plots/`):
- `complex_plane.html`: |p| over the plane, with the real line drawn in.
- `residual_vs_evaluations.html`: best residual against evaluations, for every arm.
- `algebra_map.html`: the space of number systems.

---

## Experiment 1: a rugged landscape (repaired)

**What was wrong with the original** (the version published with the essay in March 2026):
- **Wrong start markers and curve.** A stale random starting point stayed in the first history entry,
  which is why the blue curve started at 6.2 and fell.
- **Shared seed.** The optimiser and the landscape used the same random seed.
- **A rigged peak.** The global peak was placed by hand in a corner, "so small-step optimizers are unlikely
  to reach it".
- **The meta-level never selected anything.** Strategies that had never run all scored 0, so the first one
  was always chosen.
- **The jump was not creativity.** It came from a randomly generated strategy that samples uniformly over
  the whole box, i.e. a random restart.
- **Unequal compute.** The "creative" arm used 1.78× the evaluations before its jump, and the x-axis counted
  steps, not evaluations.
- **n = 1.** One seed, and no baselines.

**The repaired design:**
- **Conditions:** three landscape types — random peaks; random peaks plus one tall narrow peak at a random
  place; random peaks plus the old corner peak.
- **Runs:** 40 landscapes × 5 starts per condition. All four arms share the landscape and the start point,
  and each run gets exactly 20,000 evaluations.
- **The ceiling:** each landscape's true maximum is computed numerically.
- **Arms:**
  - a fixed Gaussian step (σ = 0.2);
  - a fixed restart strategy;
  - a random strategy swap (the same pool dynamics with no selection);
  - meta-selection: discounted UCB over a strategy pool that grows by mutation, crossover and novel
    strategies.
- **Tuning:** the meta-selection settings were frozen from a pilot on held-out seeds (root seed 1000) before
  the reported run.

**Results** (share of 200 runs per condition that reach within 1% of the ceiling, with 95% confidence
intervals from a bootstrap over landscapes; `results/landscape_results.json`):

| Condition | Fixed step | Fixed restarts | Random swap | Meta-selection |
|---|---|---|---|---|
| Random peaks | 15% [10, 21] | 99% [98, 100] | 88% [82, 93] | 92% [87, 96] |
| + one tall random peak | 9% [6, 13] | 100% | 91% [86, 96] | 96% [92, 99] |
| + the old corner peak | 4% [2, 6] | 100% | 89% [85, 93] | 92% [89, 96] |

**McNemar tests on paired runs:**
- Meta-selection vs random swap: p = 0.16, 0.11 and 0.38 (no detectable difference).
- Fixed restarts vs meta-selection: p = 5·10⁻⁴, 4·10⁻³ and 3·10⁻⁵ (restarts are better).
- Meta-selection vs fixed step: p < 10⁻⁴⁰.

**What this shows.** Changing *how* you search rescues a stuck hill climber. But the most reliable searcher
here is a simple fixed strategy that already explores. Learning which strategy to use was not measurably
better than swapping at random: meta-selection was 3–5 points ahead, with p = 0.11–0.38. That is no
evidence of a difference, but it is not proof of equivalence either. No arm can exceed the ceiling, because every arm searches the same
space. This is the failure caused by the search that Experiment 3 is contrasted with.

**Figures** (in `results/plots/`):
- `landscape_anytime.html`: best ÷ ceiling against evaluations.
- `landscape_success.html`: the share of runs that have found the highest point.

---

## Experiment 2: the grid world (superseded)

The agent-based "three-level hierarchy" in the grid world (`gridworld.py`, `agents.py`, `evolution.py`,
`neuroevolution.py`, `run_experiment_2.py`) does **not** support the claim it was written for, and it is no
longer cited by the essay.
- **The shifts never ran.** The environment "shifts" were never applied: `shift_environment` is called only
  from tests, and a "shift" only reseeded the same layout generator.
- **Different layouts.** The three levels were scored on different layouts.
- **Level 1 never learned Level 0's rule.** The neural agents never learned the hard-coded rule ("go to the
  nearest visible food") that Level 0 uses.
- **Level 2's edge was compute.** It came from 4.5× the compute plus fresh random restarts, not from new
  capabilities.

The code is left unchanged for the record. Experiment 3 replaces it.

---

## Running the experiments

The commands run from this directory with the repository's Python environment: numpy, scipy and plotly
(plotly only for figures).

```bash
python run_experiment_3.py --profile full --workers 16   # about 2.5 min on 16 cores
python run_experiment_1.py --profile full --workers 8    # about 1 min
python run_experiment_3.py --profile quick               # a few seconds each; a smoke test
python run_experiment_1.py --profile quick
```

- **Where output goes.** Results go to `--output-dir`, which defaults to `results/`. A run writes nothing
  outside it.
- **Publishing figures.** `--publish-plots-dir DIR` also copies the figures into `DIR`. It is refused unless
  `DIR` is inside this repository.
- **Determinism.** The same arguments give a byte-identical JSON, whatever `--workers` is.
- **Figures.** They are self-contained Plotly pages, rebuilt from the JSON alone.

**Tests.**

```bash
python -m pytest -o addopts="" -p no:cacheprovider tests -q
```

- **How the acceptance tests were written.** They were written from `SPEC.md` by a separate author before
  the implementation existed, and were frozen while red.
- **Mutation catalogue.** `tests/mutations.py` lists deliberate defects in the claims the essay relies on,
  for example:
  - scoring number systems by the coordinate norm;
  - letting a batch overrun the budget;
  - ignoring the shared start;
  - the original meta-level bug.

  Each must make the suite fail. All 9 are caught.

## Files

```
SPEC.md                 the contract (what the tests check)
number_systems.py       the polynomial, its real floor, two-dimensional number systems, the residual
experiment3.py          Level-1 arms in ℝ, controls, and Level-2 search over number systems
run_experiment_3.py     command-line runner for Experiment 3
landscape.py            peaks landscape; ceiling() computes its true maximum
strategies.py           search strategies and their mutation
intelligence.py         fixed-strategy optimiser (evaluation-budgeted)
creativity.py           strategy-pool meta-optimiser (UCB with discounting, or random swap)
run_experiment_1.py     command-line runner for Experiment 1
search_core.py          Budget: the single gateway every evaluation passes through
seeding.py              independent random streams per role
cli_common.py           output-dir and publish-dir guards, strict JSON, parallel map
plots_interactive.py    shared Plotly page template and palette
plots_number_systems.py figures for Experiment 3
plots_landscape.py      figures for Experiment 1
config.py               Experiment 1 and 2 parameters
gridworld.py, agents.py, evolution.py, neuroevolution.py, run_experiment_2.py, plotting.py
                        Experiment 2 (superseded) and its matplotlib figures
results/                JSON results and interactive figures from the full runs
tests/                  acceptance tests, older unit tests, and the mutation catalogue
```

`results/experiment_1_results.json` and `results/experiment_2_results.json` are the original
single-seed runs, kept for the record.
