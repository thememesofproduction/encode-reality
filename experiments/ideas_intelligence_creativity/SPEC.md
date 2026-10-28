# Specification: experiments for "Beyond What Works"

This file is the contract for the experiment code. Tests are written from it, and the code is written to
satisfy those tests. Python 3.12, numpy, scipy and plotly. Imports are
`experiments.ideas_intelligence_creativity.<module>`, with the repository root on `sys.path`; the existing
tests show how.

The essay's claim is that **intelligence** searches a fixed space of solutions, while **creativity**
changes the space. Experiment 3 makes the limit of a space exact. Experiment 1 (repaired) shows that a
smarter search *within* a space changes how fast you reach that space's best point, never what the best point is.

---

## 1. `seeding.py`: independent, reproducible random streams

```python
rng_for(root_seed: int, *keys: int) -> numpy.random.Generator
EXP1 = 1; EXP3 = 3
ROLE_LANDSCAPE = 0; ROLE_START = 1; ROLE_SEARCH = 2; ROLE_META = 3
```

- The same `(root_seed, *keys)` always gives the same stream.
- Changing any key gives a different stream.
- Streams for different roles never coincide. In the old bug, the landscape and the optimiser shared
  seed 42.

## 2. `search_core.py`: an evaluation budget every arm must go through

```python
class BudgetExhausted(Exception): ...

class Budget:
    def __init__(self, f, max_evals: int, mode: str = "min")  # mode "min" or "max"
    def __call__(self, x) -> float        # evaluate ONE point
    def batch(self, xs) -> numpy.ndarray  # evaluate points in order until the budget runs out
    max_evals: int
    used: int                             # points evaluated so far
    remaining: int                        # max_evals - used
    best: float                           # best value so far (by mode)
    best_x                                # the point that achieved it
    trace: numpy.ndarray                  # best-so-far after each evaluation; len == used
```

- `__call__` raises `BudgetExhausted` when `used == max_evals` at the time of the call, and `f` is not
  called in that case.
- `batch` evaluates `min(len(xs), remaining)` points and returns only those values. It raises
  `BudgetExhausted` only when `remaining == 0` on entry.
- `trace` never gets worse: in "min" mode it never increases, and in "max" mode it never decreases.
  `trace[0]` is the value of the first point evaluated.
- Every evaluation in every experiment arm goes through a `Budget`. An arm run with `max_evals = B`
  returns after exactly `B` evaluations.

## 3. `number_systems.py`: polynomials and two-dimensional number systems

```python
COEFFS = (4.0, -8.0, -7.0, 20.0, 25.0)  # highest degree first: 4x^4 - 8x^3 - 7x^2 + 20x + 25

p_real(x)                                # vectorised real evaluation of p
real_floor(coeffs=COEFFS) -> (x_star, m) # min over the reals of |p(x)|, COMPUTED (not a literal)
complex_roots(coeffs=COEFFS) -> numpy.ndarray  # complex roots of p

class Algebra:
    def __init__(self, alpha: float, beta: float)
    alpha: float; beta: float
    delta: float          # beta**2 + 4*alpha
    kind: str             # "complex" if delta < 0, "dual" if delta == 0, "split" if delta > 0
    def mul(self, z, w)   # z = (a, b) means a + b*e, where e*e = alpha + beta*e; returns (a', b')
    def poly(self, z, coeffs=COEFFS)  # p(z) computed with the algebra's own arithmetic, returns (A, B)
    def rho(self, z, coeffs=COEFFS)   # residual: spectral radius of "multiply by p(z)" (see below)
    def coord_norm(self, z, coeffs=COEFFS)  # sqrt(A^2 + B^2). REJECTED as a residual; kept to document why
    def phi(self, z) -> complex       # the isomorphism to C; ValueError unless kind == "complex"
    def roots(self, coeffs=COEFFS) -> list  # all z = (a, b) with p(z) = 0; [] unless kind == "complex"
```

**Inputs.** Each component of `z` may be a float or a numpy array; arrays broadcast elementwise.

**Classifying dual numbers.** `kind == "dual"` when |delta| ≤ 1e-12·max(1, beta², |4·alpha|).
For example, (alpha, beta) = (−2.25, 3) is dual. (−2.25, 3.001) is split, and (−2.25, 2.999) is complex.

**The residual (`rho`).** Multiplying by w = u + v·e is a linear map on ℝ². `rho(z)` is the spectral
radius of that map for w = p(z), which is max |u + v·r| over the two roots r of t² − βt − α = 0. It
doesn't depend on the basis, and it is computed from the algebra's own numbers, not by converting to
complex numbers. The facts below hold for any real polynomial with no real root.
- **On the real line:** `rho((x, 0)) == |p(x)|`.
- **Split or dual (delta ≥ 0):** `rho(z) ≥ m` for every z. The real floor holds; no root exists.
- **Complex (delta < 0):** `rho(z) == |p(phi(z))|`, which is 0 at the roots.
- **Why `coord_norm` is rejected:** in the dual algebra (−2.25, 3), `coord_norm` reaches below 0.6·m
  (about 0.555·m) even though no root exists. It would report fake successes.

**Algebraic laws.**
- At (alpha, beta) = (−1, 0), `mul` is ordinary complex multiplication (a + bi)(c + di).
- For any (alpha, beta), `mul` is commutative and associative, and (1, 0) is the identity.
- `phi` respects multiplication: phi(mul(z, w)) == phi(z)·phi(w).
- `poly` agrees with `numpy.polyval` on `phi`: phi(poly(z)) == polyval(coeffs, phi(z)).

**The polynomial.** p has no real roots. Its complex roots are −1 ± 0.5i and 2 ± i. Its real floor is
m ≈ 9.7702 at x* ≈ −0.9215, and `real_floor` must agree with a dense grid minimum to a relative 1e-9.
|p| has a second, worse local minimum near x ≈ 1.54, at about 32.48.

## 4. `experiment3.py`: search arms over number systems

```python
REAL_ARMS = ("hc_fixed", "hc_restart", "es_restart", "portfolio")   # Level 1: search in ℝ
CONTROLS  = ("c_oracle", "split_two_reals", "flat_joint")
L2_ARMS   = ("l2_evo", "l2_mutation_only", "l2_uniform_only", "l2_no_selection")  # Level 2

run_arm(arm: str, max_evals: int, seed: int) -> ArmResult

@dataclass
class ArmResult:
    arm: str
    trace: numpy.ndarray       # best residual so far after each evaluation of p; len == max_evals
    n_evals: int               # == max_evals
    x0: object                 # where the search started
    records: list              # Level-2 only (else []): one dict per evaluated algebra
                               #   keys: alpha, beta, delta, kind, best_rho, evals_used, gen
    adoption_eval: int | None  # Level-2 only: first trace index with trace < m*(1 - 1e-9)
    first_complex_proposal_eval: int | None  # Level-2 only: evals used when a delta<0 algebra was first proposed
```

**Arms.**
- **`hc_fixed`:** a Gaussian hill-climb over ℝ with fixed σ = 0.05.
- **`hc_restart`:** the same, but it restarts from a uniform random point in [−5, 5] after 100
  failures in a row.
- **`es_restart`:** a (1+1) evolution strategy with step-size adaptation (the 1/5th rule) and restarts.
- **`portfolio`:** the exploratory meta-level. It picks among several real search strategies by their
  measured improvement per evaluation.
- **`c_oracle`:** the same evolution strategy, run directly in ℂ (the Level-1 limit is removed by hand).
- **`split_two_reals`:** the same strategy, run in the split algebra (alpha, beta) = (1, 0), i.e.
  "two real numbers". It gains a dimension but keeps a multiplication with no roots.
- **`flat_joint`:** a single (1+1) evolution strategy over (alpha, beta, a, b) jointly, scored by `rho`.
- **`l2_evo`:** a (μ+λ) search over algebras in the box alpha ∈ [−1, 5], beta ∈ [−4, 4].
  - It starts from algebras with delta ≥ 1, so it is never handed ℂ.
  - Each candidate algebra is scored by running `es_restart` inside it, starting on the real line (b = 0).
  - Each child is a uniform draw from the box with probability 0.5; otherwise it mutates an elite.
- **`l2_mutation_only`:** `l2_evo` with uniform draws turned off.
- **`l2_uniform_only`:** `l2_evo` with mutation turned off.
- **`l2_no_selection`:** `l2_evo` with survivors picked at random.

**Contracts.** These are the tests' targets.
1. **Budget.** For every arm, `len(trace) == n_evals == max_evals`, and the number of points passed to
   `search_core.Budget` (`__call__` points plus `batch` points) equals `max_evals`.
2. **Traces.** Every trace never increases.
3. **The real floor holds.** No arm in REAL_ARMS, and not `split_two_reals`, ever records a value
   below m·(1 − 1e−12), whatever the seed or budget.
4. **Level 2 escapes the floor only through ℂ.** Every Level-2 record with
   `best_rho < m·(1 − 1e−9)` has `delta < 0`.
5. **The ℂ oracle finds a root.** `c_oracle` with `max_evals = 2000` reaches below 1e−6·m in at least
   9 of seeds 0 to 9.
6. **Level 2 finds ℂ.** `l2_evo` with `max_evals = 200_000` adopts (`adoption_eval` is not None) in at
   least 8 of seeds 0 to 9.
7. **Determinism.** The same (arm, max_evals, seed) gives an identical trace. Different seeds give
   different traces.

## 5. `run_experiment_3.py`: command-line runner

```
python run_experiment_3.py [--output-dir DIR] [--profile full|quick] [--n-seeds N]
                           [--root-seed S] [--workers W] [--no-plots]
                           [--publish-plots-dir DIR]
```

**Outputs.**
- `DIR` defaults to `<this experiment dir>/results`.
- It writes `DIR/number_systems_results.json`.
- Unless `--no-plots` is given, it also writes `DIR/plots/complex_plane.html`,
  `DIR/plots/residual_vs_evaluations.html` and `DIR/plots/algebra_map.html`, and (round 2, §10.3) the
  instruments `DIR/plots/instrument_line.html` and `DIR/plots/instrument_algebras.html`.

**Nothing written elsewhere.** It writes nothing outside `DIR`, except that `--publish-plots-dir` also
copies the plots there.
- `--publish-plots-dir` has no default.
- It is refused, with a non-zero exit and nothing written, unless it resolves inside the repository root
  (the directory that contains `experiments/`).

**Determinism.** The same arguments give a byte-identical JSON, whatever the number of workers.
The JSON contains no timestamps and no NaN or Infinity; it must parse with a strict parser.

**Speed.** `--profile quick` finishes in under 60 s on one worker.

**Top-level JSON keys:**
- `schema`: `"iic.number_systems/1"`
- `params`
- `polynomial`: `coeffs`, `roots` as [re, im] pairs, and `floor` holding `x_star`, `m`, `method` and
  `grid_check`
- `checkpoints`
- `arms`: per arm, `best`, a list over seeds of best-at-each-checkpoint
- `level2`
- `algebra_map`
- `summary`
- `invariants`

**Invariants:**
- `invariants.min_real_best_over_m` ≥ 1 − 1e−12, computed over REAL_ARMS and `split_two_reals`.
- `invariants.below_floor_from_nonnegative_delta` == 0.

## 6. Experiment 1 repaired: the peaks landscape

### `landscape.py`

```python
FitnessLandscape(n_peaks=15, dims=2, bounds=(-5.0, 5.0), seed=42, global_peak="none")
FitnessLandscape.from_peaks(centers, heights, widths, bounds=(-5.0, 5.0))  # explicit peaks
landscape.ceiling() -> (x, f)        # the landscape's true maximum, computed numerically
landscape.get_global_optimum()       # same as ceiling()
```

**`global_peak`** takes one of three values:
- `"none"`: exactly `n_peaks` random peaks.
- `"random"`: one extra tall, narrow peak (height 8, width 0.3) at a random place.
- `"corner"`: the same extra peak at the old hand-placed corner, kept as one condition.

Any other value raises ValueError.

**`ceiling()`** must equal or exceed the landscape's value at every peak centre, and the maximum of a
401×401 grid over the bounds (allowing 1e−9). Its x must lie within the bounds. Where two peaks overlap,
the true maximum is above both centres' values, and `ceiling()` must find it.

### `intelligence.py` and `creativity.py`

```python
FixedStrategyOptimizer(landscape, strategy, seed=42, x0=None)
MetaStrategyOptimizer(landscape, initial_strategy, mutator, meta_interval=50,
                      strategy_pool_size=5, improvement_window=25,
                      novel_strategy_prob=0.2, seed=42, x0=None, replace_every=3,
                      selection="ucb")        # "ucb" (real selection) or "random" (ablation)
optimizer.run(n_steps=None, max_evals=None) -> dict
optimizer.trace -> numpy.ndarray   # best fitness so far after each evaluation; len == evaluations used
```

**Start point.**
- When `x0` is given, the search starts exactly there. `history.positions[0]` equals `x0`, and
  `history.best_fitness[0]` equals `landscape.evaluate(x0)`.
- When `x0` is None, the optimiser draws its own start.
- The evaluation of the start point counts as evaluation 1.

**Budget.** With `max_evals = B`, the run stops after exactly B evaluations, truncating the last batch of
candidates, and `len(trace) == B`.

**Meta-selection with `selection="ucb"`.**
- The pool holds records with `.strategy`, `.evals` (int) and `.improvement` (float).
- `.score` is `improvement / evals`, i.e. improvement per evaluation, and is None when `evals == 0`.
- `meta_step()` is public. After it runs:
  - If any pool record has `evals == 0` (never run), the active strategy is one of them, so untried
    strategies run first.
  - Otherwise, the active strategy is the one with the best UCB value. With equal evaluation counts,
    that is the highest `.score`.
  - The strategy dropped from the pool, if any, is never the active one and always has `evals > 0`.
  - `mutator.mutate` and `mutator.crossover` are only ever called with strategies whose pool record has
    `evals > 0`.
- `selection="random"` keeps the same pool dynamics, but the active strategy is chosen uniformly at random.

### `run_experiment_1.py`: command-line runner

```
python run_experiment_1.py [--output-dir DIR] [--profile full|quick] [--n-landscapes N]
                           [--n-starts K] [--max-evals B] [--root-seed S] [--workers W]
                           [--conditions none,random,corner] [--no-plots] [--publish-plots-dir DIR]
```

**Arms.** All four arms share the same landscape and start point for each (condition, landscape, start):
- `hc_fixed`: a fixed Gaussian strategy, σ = 0.2.
- `restart_fixed`: a fixed `restart` strategy.
- `swap_random`: `MetaStrategyOptimizer` with `selection="random"`.
- `meta_ucb`: `MetaStrategyOptimizer` with `selection="ucb"`.

**Seeding.**
- Landscapes, starts and searches use separate `seeding` roles.
- No start point coincides with a peak centre to within 1e−6.

**Outputs.**
- It writes `DIR/landscape_results.json`.
- Unless `--no-plots` is given, it also writes `DIR/plots/landscape_anytime.html` and
  `DIR/plots/landscape_success.html`.
- The output-dir, publish-dir, determinism, strict-JSON and quick-profile (under 60 s) rules from §5 apply.

**Top-level JSON keys:** `schema` (`"iic.landscape/1"`), `params`, `checkpoints`, `conditions`, `summary`.
- `conditions` holds, per condition, per arm: `normalised_best`, a list over runs of best/ceiling at
  each checkpoint.
- **Success** at a budget means best ≥ 0.99·ceiling.
- `summary` holds, per condition and arm, the final `success_rate` and a 95% `ci` [lo, hi] from a
  bootstrap over landscapes, with lo ≤ success_rate ≤ hi. It also holds McNemar `p_value`s in [0, 1] for
  `meta_ucb` against each other arm.

## 7. `plots_interactive.py`: interactive figures

Self-contained Plotly HTML pages, styled like `experiments/bells_inequality/plotting.py`:
- **Plotly script:** loaded from `https://cdn.plot.ly/plotly-2.35.2.min.js`. The site rewrites
  cdn.plot.ly links to its own copy, which is plotly.js 2.35.3, so the figure JSON must render under
  plotly.js 2.35.x. plotly.py 6 targets plotly.js 3, so avoid features that only exist in 3.x.
- **Colours:** a dark source background (`#1e1e1e`) and light text (`#e0e0e0`) set explicitly, so the
  site can re-skin them to the reader's theme.
- **Width:** no fixed `width` in the layout; the figure is responsive.
- **Data:** no `NaN` anywhere in the file.
- **Text:** Unicode, not LaTeX.

Each figure carries its own annotations (§ figures in the plan): the real floor, the region where roots
exist, and so on.

## 8. Experiment 2

It is superseded and unchanged. The README says why.

## 9. Clarifications (2026-10-26, answering the test author's questions)

These override any looser wording above.

1. **How to compute `rho`.**
   - When delta > 0, the implementation may compute it as max |p(a + b·r)| over the two real roots r of
     t² − βt − α. This is the algebra's own idempotent decomposition, not a detour through ℂ.
   - When kind == "dual" (including a delta within the dual tolerance but slightly below zero), it is
     |p(a + b·β/2)|, because the double eigenvalue of e is β/2.
   - When kind == "complex", it equals |p(phi(z))| to a relative 1e−9.
   - Floating-point margins: tests of the floor on the lines a + b·r = x* use |b| ≤ 3.
2. **One Budget per arm run.** Inner runs (Level 2's search inside each algebra, or a portfolio's
   strategies) take their evaluations from that single Budget. A sub-limit is a plain counter, never a
   second Budget, so the class-level count of points equals `max_evals`.
3. **Fields for non-Level-2 arms.** `records == []`, and `adoption_eval` and
   `first_complex_proposal_eval` are None.
4. **What a Level-2 record holds.**
   - `evals_used` is the evaluations spent on that algebra in that generation: at least 1, at most `max_evals`.
   - `gen` is 0-based, and every gen-0 algebra has delta ≥ 1.
   - Every proposed algebra lies in the box; mutation reflects into it.
5. **Unknown arm names.** `run_arm` with an unknown arm name raises ValueError.
6. **The `arms` key in the JSON.** It holds every arm: REAL_ARMS, CONTROLS and L2_ARMS.
7. **When the publish dir is refused.** The runner refuses *before computing anything*, so nothing at all
   is written anywhere.
   - When accepted, a missing publish dir is created.
   - Only the plot HTML files are copied into it, byte-identical to DIR/plots. The JSON is not copied.
8. **Names on the Exp 1 optimisers.**
   - `optimizer.pool`: a list of records.
   - `optimizer.active_idx`: an int.
   - `optimizer.active_strategy`: a property, `pool[active_idx].strategy`.
   - `optimizer.history.positions` and `optimizer.history.best_fitness`: sequences.
   - Tests use exactly these names, with no fallbacks.
9. **When `meta_step` selects and when it replaces.** `MetaStrategyOptimizer` takes
   `replace_every: int = 3`, and `meta_step(replace: bool | None = None)` does two things in order:
   - **(a) Replacement**, only when `replace` is True, or when `replace` is None and this is every
     `replace_every`-th meta-step. Drop the worst record that has data and is not active, then add one
     new strategy (mutant, crossover or novel), which has no data.
   - **(b) Selection**, always. If an untried record exists, make one active. Otherwise make the
     best-UCB record active, which with equal `evals` is the highest `.score`.
   - Selection is checked *after* the call.
   - Without (a), selection would always pick the newest untried strategy and never exploit, making the
     real meta-level a random swap.
   - `meta_step(replace=False)` lets a test check pure selection.
10. **Summary layout in `landscape_results.json`.**
    - `summary[condition][arm]` is `{"success_rate": float, "ci": [lo, hi]}`.
    - `summary[condition]["mcnemar_p"][arm]` is a float in [0, 1] for each arm ≠ "meta_ucb".
11. **How the Exp 1 runner seeds.** Calls are `rng_for(root_seed, EXP1, ROLE_*, ...)`: the experiment
    first, then the role.
12. **Second round (2026-10-26).**
    - **(a)** The *kind* decides the `rho` formula: a dual-kind algebra uses |p(a + b·β/2)| even when
      delta is slightly above zero.
    - **(b)** Replacement drops a record only when the pool is full (`len(pool) == strategy_pool_size`).
      Otherwise it only adds. "Worst" means the lowest `.score`.
    - **(c)** The `replace_every` counter counts every `meta_step` call.
    - **(d)** Every evaluation is attributed to exactly one Level-2 record, including a final, partly
      evaluated algebra, so `sum(evals_used) == max_evals`.
13. **Discounted counts (2026-10-26).** With `discount < 1`, a record's `.evals` and `.improvement` are
    discounted totals, so they are floats. A never-run record stays exactly 0, so it stays untried. The
    class default `discount = 1.0` keeps them as plain totals.

## 10. Round 2 (2026-10-26): instruments and plain-language figures

The reader this round is written for has no maths beyond school. `READER_TAKEAWAYS.md` lists what they
should take from each figure; this section is the code contract.

### 10.1 `figkit/build.py`: one self-contained page per instrument

`build_instrument(system: str, data: dict, out, *, title: str, aria_label: str, caption: str | None = None) -> Path`

- **`system`** names a file `figkit/systems/<system>.js`. An unknown name raises `ValueError`.
- **The page.** It writes one HTML file at `out` and returns its `Path`. The kit (`figkit/kit.js`,
  `figkit/kit.css`), the system file and `data` are all inlined.
- **No `src=` and no `http://` or `https://` anywhere.** The only links allowed are to the same-origin
  stylesheets `/assets/er/tokens.css` and `/assets/er/fonts.css`.
- **The `<head>` contains:**
  - `<meta name="er-embed" content="self-themed">`, exactly once, in `<head>`;
  - `<title>`, equal to `title`, HTML-escaped;
  - before any stylesheet, a script that sets `document.documentElement.dataset.theme` from
    `localStorage['er-theme']` when the stored value is `paper` or `ink`. A failure to read storage is caught.
- **Data.** It is embedded in exactly one `<script type="application/json" data-fk="data">`. Parsing that
  element's text gives back exactly `data`.
  - Any `</` inside the data is escaped, so the data can never close the script early.
  - NaN or ±inf anywhere in `data` raises `ValueError`.
- **Accessibility.** `aria_label` is the `aria-label` of the drawing stage. `caption`, when given, appears
  exactly once as the text of a `<p data-fk="caption">`, HTML-escaped.
- **The privacy guard.** It raises `ValueError` if the finished page contains any of:
  - `/srv/` or `/home/`;
  - any name listed, comma-separated, in the FIGKIT_PRIVATE_NAMES environment variable, case-insensitive. Private
    names, the repository's own included, are never written into the code;
  - `sourceMappingURL`;
  - an e-mail address;
  - an ISO-8601 time with a numeric UTC offset;
  - a run of 40 hexadecimal characters.

  These are the deploy lint's blocking patterns.
- **Determinism.** The same arguments give byte-identical files.
- **Valid JavaScript.** Every inline non-JSON `<script>` passes `node --check`.

### 10.2 `figkit/systems/numsys_math.js`: the maths the instruments run

A plain script, with no imports. It defines the following functions and exports them through
`module.exports` when `module` exists, so node can test them:
- `absPolyComplex(coeffs, re, im)`: |p(re + i·im)|. Coefficients come highest-degree first, as in the
  results JSON.
- `rho(coeffs, alpha, beta, a, b)`: the spectral radius of multiplication by p(z), for z = a + b·e in the
  algebra e² = α + βe, matching `number_systems.Algebra(alpha, beta).rho` (§3, §9.1, §9.12a). This includes
  the dual case. Whether an algebra counts as dual is decided exactly as `Algebra.kind` decides it, with
  §3's tolerance.
- `kindOf(alpha, beta)`: returns `"split"`, `"dual"` or `"complex"`, exactly as `Algebra.kind` does.

**Agreement test.** At 50 seeded random points per kind, every value agrees with the Python to a relative
1e−9. At the 4 roots of p, `absPolyComplex` is below 1e−9·m.

### 10.3 `instruments.py`: data for the two instruments, and the pages

- **`line_data(results) -> dict`**, for "The answer is off the line":
  - `coeffs`: from `results["polynomial"]["coeffs"]`;
  - `m` and `x_star`: from `results["polynomial"]["floor"]`;
  - `minima`: a list of `[x, abs_p]` from `results["real_line"]["minima"]`, in the same order;
  - `roots`: a list of `[re, im]`, from `results["polynomial"]["roots"]`;
  - `sigma`: from `experiment3.HC_SIGMA`;
  - `view`: `{"re": [-3.0, 4.0], "im": [-2.0, 2.0]}`;
  - `start`: the overlap of `[START_LO, START_HI]` with `view["re"]`;
  - `walkers`: 128;
  - `seed`: an int.
- **`algebras_data(results) -> dict`**, for "Choose a number system":
  - `alpha` and `beta`: the `algebra_map` axes;
  - `cat`: a 2-D list with the same shape as `algebra_map["log10_ratio"]`. Each cell is 0 (stayed at the floor:
    within `plots_number_systems.FLOOR_TOL` of a ratio of 1), 1 (below the floor, no root) or 2 (a root: ratio
    below `ROOT_TOL`);
  - `m`;
  - `box`: `{"alpha": L2_ALPHA, "beta": L2_BETA}`;
  - `dual_tol`;
  - `es`: the ES constants from `results["params"]["constants"]` (`ES_SIGMA0`, `ES_UP`, `ES_DOWN`,
    `ES_SIGMA_MIN`) and `child_evals = L2_CHILD_EVALS`;
  - `exemplar`: `proposals`, `elite_path`, `adoption_algebra` and `adoption_eval`, unchanged from
    `results["level2"]["exemplar"]`.
  - **What the browser search needs** (added once implemented): `coeffs`; `start` = `[START_LO, START_HI]`;
    `bound` = `ALG_BOUND`; and a `seed`.
  - A system's `readout()` may also return a `verdict` string. A change of verdict announces the readout
    at once; otherwise announcements are throttled.
- **The theory, checked against the recorded map.** A cell with Δ = β² + 4α ≥ 0 is never 2 (§3: ρ ≥ m when
  Δ ≥ 0).
- **`write_instruments(results, plots_dir) -> list[Path]`** writes `instrument_line.html` and
  `instrument_algebras.html` via `build_instrument`. Each has a caption that says what is live and what is
  recorded data. Neither caption uses any word from the jargon list in §10.4.
- **The runner.** `run_experiment_3.py` writes both instruments into `<output-dir>/plots`. `--publish-plots-dir`
  copies them too (§9.7).

### 10.4 Plain language in figures B (`residual_vs_evaluations`) and D (`landscape_success`)

- **Visible text** means: the figure title, axis titles, legend names, annotation texts, button and
  dropdown labels, and the page caption. Hover text is excluded.
- **Jargon list.** None of these may appear in the visible text of B or D: `evaluations`, `|p|`, `÷`, `ρ`,
  `residual`, `ES`, `σ`, `≅`, `UCB`, `adopts`, `ceiling`, `quartile`, `seed`.
  - Match `ES` as a whole word and case-sensitively.
  - Match the rest case-insensitively.
- **Log axes.** Every log-type axis title contains `log scale`.
- **Figure B:**
  - **x-axis title:** `guesses checked (log scale)`;
  - **y-axis title:** it contains `1 =` and `lower is better`;
  - **legend names,** declared once in `plots_number_systems.LEGEND_NAME`:

| arm | legend name |
|---|---|
| floor line | `best possible on the line` |
| hc_fixed | `hill-climber, fixed step` |
| hc_restart | `hill-climber, restarts` |
| es_restart | `evolution strategy, restarts` |
| portfolio | `portfolio of strategies` |
| l2_evo | `Level 2: searches number systems` |
| c_oracle | `complex numbers given by hand` |
| split_two_reals | `two real numbers` |
| flat_joint | `one search over everything` |
| l2_mutation_only | `Level 2: small changes only` |
| l2_uniform_only | `Level 2: random proposals only` |
| l2_no_selection | `Level 2: no selection` |

- **Figure B's guided tour.** One `updatemenus` entry of type `buttons`, whose labels in order are
  `1 the line`, `2 a new space` and `3 comparisons`. It replaces "show controls" and "hide controls".
  - **Button 1:** shows the floor line and the four real-line arms. It hides every Level-2 arm and every
    control.
  - **Button 2:** shows the button-1 traces plus `l2_evo` and its band. This is also the figure's initial
    state, in both trace visibility and annotations.
  - **Button 3:** shows every trace.
  - **Annotations.** Each button sets the layout's annotations. Button 2's set includes an annotation
    containing `square root of −1`. Every set includes the floor annotation.
- **Figure D:**
  - **y-axis:** `tickformat` `.0%`, range [0, 1.02];
  - **dropdown labels:** `random peaks`, `+ one tall random peak`, `+ one peak hidden in a corner (the original setup)`.

### 10.5 What stays

- **Figures A (`complex_plane`) and C (`algebra_map`)** are still built. The README and earlier tests refer
  to them; the post stops embedding them.
- **Everything in §1–§9** still holds. The existing tests are not edited, except where §10.4 changes a
  visible string they assert. Any such edit is recorded with `redline.py amend`, and its reason is written down.
- **§5's list of plots now includes the two instruments.** The existing runner tests read that list from
  this file, so they now expect all five pages, still compared exactly.
- **§7 governs the Plotly pages only.** The instruments follow §10.1: no Plotly and no script `src` at all.
  `test_plots_interactive.py` therefore leaves out the pages that §10 names as instruments, and checks
  that there are exactly two of them.
