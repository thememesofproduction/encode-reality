"""SPEC.md section 4: search arms over number systems (Experiment 3).

Level 1 searches the real line: hc_fixed, hc_restart, es_restart and portfolio. The controls are
c_oracle (the same evolution strategy run in C by hand), split_two_reals (run in the split algebra
(1, 0)) and flat_joint (one evolution strategy over (alpha, beta, a, b)). Level 2 (the l2_* arms)
searches over algebras: each candidate algebra is scored by an evolution strategy run inside it.

Every evaluation of every arm goes through one search_core.Budget, created by run_arm. Inner runs (an
evolution strategy inside an algebra, a portfolio epoch) take their evaluations from that Budget through
a plain counter (spec 9.2), so the number of points that went through the Budget is max_evals.

run_arm(arm, max_evals, seed, root_seed=DEFAULT_ROOT_SEED) uses root_seed 0 unless told otherwise. The
runner's --root-seed passes its value through, so every stream shifts together.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from experiments.ideas_intelligence_creativity.number_systems import COEFFS, Algebra, real_floor
from experiments.ideas_intelligence_creativity.search_core import Budget, BudgetExhausted
from experiments.ideas_intelligence_creativity.seeding import EXP3, ROLE_META, ROLE_SEARCH, ROLE_START, rng_for

REAL_ARMS = ("hc_fixed", "hc_restart", "es_restart", "portfolio")          # Level 1: search in R
CONTROLS = ("c_oracle", "split_two_reals", "flat_joint")
L2_ARMS = ("l2_evo", "l2_mutation_only", "l2_uniform_only", "l2_no_selection")   # Level 2
ALL_ARMS = REAL_ARMS + CONTROLS + L2_ARMS

# Stable ids for the seeding streams. Never renumber an arm: its streams would change.
ARM_IDS = {"hc_fixed": 1, "hc_restart": 2, "es_restart": 3, "portfolio": 4, "c_oracle": 5,
           "split_two_reals": 6, "flat_joint": 7, "l2_evo": 8, "l2_mutation_only": 9,
           "l2_uniform_only": 10, "l2_no_selection": 11}
ALGEBRA_MAP_ID = 12

DEFAULT_ROOT_SEED = 0

# --- Tunable parameters. They live here, not in config.py (SPEC section 4 leaves the values open). ---

START_LO, START_HI = -5.0, 5.0            # uniform starts on the real line, and in algebra searches
HC_SIGMA = 0.05                           # hill-climb Gaussian step (hc_fixed, hc_restart)
HC_RESTART_AFTER = 100                    # consecutive failures before a uniform restart (hc_restart)
ES_SIGMA0 = 0.5                           # (1+1)-ES initial step size
ES_UP = 1.5                               # step size after a success
ES_DOWN = 1.5 ** -0.25                    # step size after a failure (the 1/5th rule)
ES_SIGMA_MIN = 1e-10                      # below this the ES restarts from a fresh start
ALG_BOUND = 50.0                          # |a| and |b| <= ALG_BOUND in algebra searches
PORTFOLIO_EPOCH = 100                     # evaluations per portfolio epoch
PORTFOLIO_UCB_C = math.sqrt(2.0)          # UCB1 exploration constant; rewards are decades per 100 evaluations
PORTFOLIO_UNIFORM = 10.0                  # the uniform strategy samples U[-10, 10]
L2_ALPHA = (-1.0, 5.0)                    # the algebra box
L2_BETA = (-4.0, 4.0)
L2_START_DELTA = 1.0                      # initial algebras have delta >= this
L2_INIT = 8                               # initial population
L2_MU = 4                                 # elites
L2_LAMBDA = 4                             # children per generation
L2_UNIFORM_PROB = 0.5                     # q: probability that a child is a uniform draw from the box
L2_SIGMA = 0.25                           # mutation step per coordinate
L2_CHILD_EVALS = 500                      # evaluations in a new child's ES run
L2_TOPUP_EVALS = 250                      # warm-started top-up for each surviving elite, per generation
# (probability of a uniform child, whether survivors are selected by score)
L2_VARIANTS = {"l2_evo": (L2_UNIFORM_PROB, True), "l2_mutation_only": (0.0, True),
               "l2_uniform_only": (1.0, True), "l2_no_selection": (L2_UNIFORM_PROB, False)}
ADOPTION_TOL = 1e-9                       # adoption: a trace value below m (1 - ADOPTION_TOL)
CHECKPOINTS_PER_DECADE = 10

# PORTFOLIO strategies: (name, kind, parameter)
PORTFOLIO = (("hc_0.01", "hc", 0.01), ("hc_0.1", "hc", 0.1), ("hc_1", "hc", 1.0),
             ("cauchy_1", "cauchy", 1.0), ("uniform_10", "uniform", PORTFOLIO_UNIFORM), ("es", "es", None))

X_STAR, M_FLOOR = real_floor()            # the real floor of p, computed (spec 3)


@dataclass
class ArmResult:
    arm: str
    trace: np.ndarray                     # best residual so far after each evaluation of p; len == max_evals
    n_evals: int                          # == max_evals
    x0: object                            # where the search started (None for Level 2)
    records: list                         # Level 2 only: one dict per evaluated algebra per generation
    adoption_eval: int | None             # Level 2 only: first trace index with trace < m (1 - 1e-9)
    first_complex_proposal_eval: int | None   # Level 2 only: evaluations used when a delta < 0 algebra was first run
    elite_path: list = field(default_factory=list)       # Level 2 only: per generation, [alpha, beta, best/m] per elite
    adoption_algebra: tuple | None = None                 # Level 2 only: (alpha, beta) of the adopting algebra


def checkpoints(max_evals: int) -> list:
    """Ten per decade from 1 to max_evals: unique, increasing ints that include max_evals."""
    pts = {int(max_evals)}
    k = 0
    while True:
        v = int(round(10 ** (k / CHECKPOINTS_PER_DECADE)))
        if v > max_evals:
            break
        pts.add(v)
        k += 1
    return sorted(pts)


SHARED_START = 0                          # arm key of the start stream every arm shares (arm ids start at 1)


def _stream(root_seed, role, arm_id, seed, *extra):
    return rng_for(root_seed, EXP3, role, arm_id, seed, *extra)


def _abs_p(x):
    """|p(x)| for a real x, by the same Horner steps as numpy.polyval."""
    acc = 0.0
    for c in COEFFS:
        acc = acc * x + c
    return abs(acc)


def _identity(z):
    return z


def _clip(x):
    return min(max(x, -ALG_BOUND), ALG_BOUND)


def _clip_pair(z):
    return (_clip(z[0]), _clip(z[1]))


def _reflect(x, lo, hi):
    """Reflect x into [lo, hi] (spec 9.4: mutation reflects into the box)."""
    span = hi - lo
    y = (x - lo) % (2.0 * span)
    if y > span:
        y = 2.0 * span - y
    return min(max(lo + y, lo), hi)


def _real_start(rng):
    return (float(rng.uniform(START_LO, START_HI)),)


def _alg_start(rng):
    """A start on the real line of an algebra: (a, 0) with a ~ U[-5, 5]."""
    return (float(rng.uniform(START_LO, START_HI)), 0.0)


def _delta_at_least(rng, minimum):
    """An algebra in the box with delta >= minimum, by rejection sampling."""
    while True:
        a = float(rng.uniform(*L2_ALPHA))
        b = float(rng.uniform(*L2_BETA))
        if b * b + 4.0 * a >= minimum:
            return a, b


def _joint_start(rng):
    a, b = _delta_at_least(rng, L2_START_DELTA)
    return (a, b, float(rng.uniform(START_LO, START_HI)), 0.0)


def _joint_project(z):
    return (_reflect(z[0], *L2_ALPHA), _reflect(z[1], *L2_BETA), _clip(z[2]), _clip(z[3]))


def _joint_rho(z):
    return Algebra(z[0], z[1]).rho((z[2], z[3]))


def _until_spent(fn, *args):
    """Run an arm's loop until its Budget is spent (the loops have no other exit)."""
    try:
        fn(*args)
    except BudgetExhausted:
        pass


class _ES:
    """(1+1)-ES with the 1/5th rule and restarts (spec 4: es_restart).

    Points are tuples. ``start(rng)`` draws a fresh start and ``project`` keeps a proposal in bounds;
    ``first`` is used as the very first start instead of a draw. The state persists between calls to
    ``run``, so a later run continues the same search (a warm start). ``best`` and ``evals`` count every
    evaluation this search has made.
    """

    def __init__(self, rng, start, project=_identity, first=None):
        self.rng = rng
        self._start = start
        self._project = project
        self._first = first
        self.cur = None
        self.fcur = None
        self.sigma = ES_SIGMA0
        self.best = math.inf
        self.best_z = None
        self.evals = 0

    def run(self, feval, limit: int) -> int:
        """Up to ``limit`` evaluations through ``feval``. Returns the number made. feval may raise BudgetExhausted."""
        used = 0
        while used < limit:
            if self.cur is None:
                z = self._first if self._first is not None else self._start(self.rng)
                self._first = None
                f = feval(z)
                self.evals += 1
                used += 1
                self.cur, self.fcur, self.sigma = z, f, ES_SIGMA0
            elif self.sigma < ES_SIGMA_MIN:
                self.cur = None
                continue
            else:
                s = self.sigma
                z = self._project(tuple(c + s * self.rng.standard_normal() for c in self.cur))
                f = feval(z)
                self.evals += 1
                used += 1
                if f < self.fcur:
                    self.cur, self.fcur, self.sigma = z, f, s * ES_UP
                else:
                    self.sigma = s * ES_DOWN
            if f < self.best:
                self.best, self.best_z = f, z
        return used


# ----------------------------------------------------------------------------- Level 1 and the controls

def _hill_climb(budget, rng, x, restart: bool):
    """Gaussian hill climb, accepting strict improvements only (hc_fixed, hc_restart)."""
    f = budget(x)
    fails = 0
    while True:
        z = x + HC_SIGMA * rng.standard_normal()
        v = budget(z)
        if v < f:
            x, f, fails = z, v, 0
        else:
            fails += 1
            if restart and fails >= HC_RESTART_AFTER:
                x = float(rng.uniform(START_LO, START_HI))
                f = budget(x)
                fails = 0


def _portfolio_epoch(strategy, budget, rng, x, f, n, es_sigma, i):
    """One epoch of ``n`` evaluations of one portfolio strategy, starting from the shared best (x, f)."""
    _, kind, param = strategy
    if kind == "uniform":
        for _ in range(n):
            budget(float(rng.uniform(-param, param)))
        return
    cur, fc = x, f
    for _ in range(n):
        if kind == "hc":
            z = cur + param * rng.standard_normal()
        elif kind == "cauchy":
            z = cur + param * rng.standard_cauchy()
        else:
            if es_sigma[i] < ES_SIGMA_MIN:
                es_sigma[i] = ES_SIGMA0
            z = cur + es_sigma[i] * rng.standard_normal()
        v = budget(z)
        if v < fc:
            cur, fc = z, v
            if kind == "es":
                es_sigma[i] *= ES_UP
        elif kind == "es":
            es_sigma[i] *= ES_DOWN


def _portfolio(budget, rng, x0):
    """The exploratory meta-level over real strategies (spec 4). Untried strategies run first; then UCB1
    on decades of improvement per evaluation. Every strategy starts each epoch from the shared best."""
    budget(x0)
    k = len(PORTFOLIO)
    epochs = [0] * k
    reward = [0.0] * k
    es_sigma = [ES_SIGMA0] * k
    while budget.remaining > 0:
        untried = [i for i in range(k) if epochs[i] == 0]
        if untried:
            i = untried[0]
        else:
            n_total = sum(epochs)
            ucb = [reward[j] / epochs[j] + PORTFOLIO_UCB_C * math.sqrt(math.log(n_total) / epochs[j])
                   for j in range(k)]
            i = max(range(k), key=lambda j: ucb[j])
        n = min(PORTFOLIO_EPOCH, budget.remaining)
        f0 = budget.best
        _portfolio_epoch(PORTFOLIO[i], budget, rng, float(budget.best_x), f0, n, es_sigma, i)
        decades = math.log10(max(f0, 1e-300) / max(budget.best, 1e-300))
        reward[i] += decades * PORTFOLIO_EPOCH / n
        epochs[i] += 1


# ----------------------------------------------------------------------------- Level 2

class _Current:
    """Budget's point function for Level 2: the algebra that the current segment searches."""

    def __init__(self):
        self.alg = None

    def __call__(self, z):
        return self.alg.rho(z)


class _Candidate:
    """An algebra in Level 2, with the ES state that searches inside it (kept for warm top-ups)."""

    def __init__(self, alpha, beta, es):
        self.alg = Algebra(alpha, beta)
        self.es = es


class _Level2:
    """Bookkeeping for one Level-2 run: records (spec 9.4), adoption and the first complex proposal."""

    def __init__(self, budget, current):
        self.budget = budget
        self.current = current
        self.records = []
        self.elite_path = []
        self.adoption_eval = None
        self.adoption_algebra = None
        self.first_complex = None
        self._seg = 0
        self._below = M_FLOOR * (1.0 - ADOPTION_TOL)

    def _eval(self, z):
        alg = self.current.alg
        v = self.budget(z)
        if alg.kind == "complex" and self.first_complex is None:
            self.first_complex = self.budget.used - 1
        self._seg += 1
        if self.adoption_eval is None and v < self._below:
            self.adoption_eval = self.budget.used - 1
            self.adoption_algebra = (alg.alpha, alg.beta)
        return v

    def segment(self, cand, limit: int, gen: int):
        """Run cand's search for up to ``limit`` evaluations and record them. A segment cut short by the
        Budget still records the evaluations it made, so every evaluation is attributed (spec 9.12d)."""
        self.current.alg = cand.alg
        self._seg = 0
        proposal = cand.es.evals == 0
        try:
            cand.es.run(self._eval, limit)
        finally:
            if self._seg:
                self.records.append({
                    "alpha": cand.alg.alpha, "beta": cand.alg.beta, "delta": cand.alg.delta,
                    "kind": cand.alg.kind, "best_rho": float(cand.es.best),
                    "evals_used": self._seg, "gen": gen, "proposal": proposal})


def _score(cand) -> float:
    return round(math.log10(max(cand.es.best, 1e-300) / M_FLOOR), 6)


def _select(pool, rng, by_score: bool) -> list:
    """The L2_MU survivors: the best by score (ties broken at random), or at random without selection."""
    ties = [rng.random() for _ in pool]
    if by_score:
        order = sorted(range(len(pool)), key=lambda i: (_score(pool[i]), ties[i]))
    else:
        order = [int(i) for i in rng.permutation(len(pool))]
    return [pool[i] for i in order[:L2_MU]]


def _run_level2(arm: str, lvl: _Level2, meta, root_seed: int, seed: int):
    uniform_prob, by_score = L2_VARIANTS[arm]
    made = [0]

    def candidate(alpha, beta):
        made[0] += 1
        es = _ES(_stream(root_seed, ROLE_SEARCH, ARM_IDS[arm], seed, made[0]), _alg_start, _clip_pair)
        return _Candidate(alpha, beta, es)

    pop = []
    for _ in range(L2_INIT):
        a, b = _delta_at_least(meta, L2_START_DELTA)
        c = candidate(a, b)
        lvl.segment(c, L2_CHILD_EVALS, 0)
        pop.append(c)
    elites = _select(pop, meta, by_score)
    lvl.elite_path.append(_path_entry(elites))
    gen = 0
    while True:
        gen += 1
        children = []
        for _ in range(L2_LAMBDA):
            if meta.random() < uniform_prob:
                a = float(meta.uniform(*L2_ALPHA))
                b = float(meta.uniform(*L2_BETA))
            else:
                parent = elites[int(meta.integers(len(elites)))].alg
                a = _reflect(parent.alpha + L2_SIGMA * meta.standard_normal(), *L2_ALPHA)
                b = _reflect(parent.beta + L2_SIGMA * meta.standard_normal(), *L2_BETA)
            c = candidate(a, b)
            lvl.segment(c, L2_CHILD_EVALS, gen)
            children.append(c)
        for e in elites:
            lvl.segment(e, L2_TOPUP_EVALS, gen)
        elites = _select(elites + children, meta, by_score)
        lvl.elite_path.append(_path_entry(elites))


def _path_entry(elites) -> list:
    return [[c.alg.alpha, c.alg.beta, c.es.best / M_FLOOR] for c in elites]


# ----------------------------------------------------------------------------- the public entry points

def run_arm(arm: str, max_evals: int, seed: int, root_seed: int = DEFAULT_ROOT_SEED) -> ArmResult:
    """Run one arm for exactly ``max_evals`` evaluations of p, on the streams of (root_seed, arm, seed)."""
    if arm not in ARM_IDS:
        raise ValueError(f"unknown arm {arm!r}; the arms are {ALL_ARMS}")
    max_evals = int(max_evals)
    if max_evals < 1:
        raise ValueError(f"max_evals must be at least 1, not {max_evals}")
    arm_id = ARM_IDS[arm]
    start = _stream(root_seed, ROLE_START, SHARED_START, seed)    # one start stream per seed: arms are paired
    search = _stream(root_seed, ROLE_SEARCH, arm_id, seed)
    meta = _stream(root_seed, ROLE_META, arm_id, seed)

    if arm in L2_ARMS:
        current = _Current()
        budget = Budget(current, max_evals)
        lvl = _Level2(budget, current)
        _until_spent(_run_level2, arm, lvl, meta, root_seed, seed)
        return ArmResult(arm=arm, trace=budget.trace.copy(), n_evals=budget.used, x0=None,
                         records=lvl.records, adoption_eval=lvl.adoption_eval,
                         first_complex_proposal_eval=lvl.first_complex, elite_path=lvl.elite_path,
                         adoption_algebra=lvl.adoption_algebra)

    if arm == "flat_joint":
        budget = Budget(_joint_rho, max_evals)
        x0 = _joint_start(start)
        es = _ES(search, _joint_start, _joint_project, first=x0)
        _until_spent(es.run, budget, max_evals)
    elif arm in ("c_oracle", "split_two_reals"):
        alg = Algebra(-1.0, 0.0) if arm == "c_oracle" else Algebra(1.0, 0.0)
        budget = Budget(alg.rho, max_evals)
        x0 = (float(start.uniform(START_LO, START_HI)), 0.0)
        es = _ES(search, _alg_start, _clip_pair, first=x0)
        _until_spent(es.run, budget, max_evals)
    else:
        budget = Budget(_abs_p, max_evals)
        x0 = float(start.uniform(START_LO, START_HI))
        if arm in ("hc_fixed", "hc_restart"):
            _until_spent(_hill_climb, budget, search, x0, arm == "hc_restart")
        elif arm == "portfolio":
            _until_spent(_portfolio, budget, search, x0)
        else:
            es = _ES(search, _real_start, first=(x0,))
            _until_spent(es.run, lambda z: budget(z[0]), max_evals)
    return ArmResult(arm=arm, trace=budget.trace.copy(), n_evals=budget.used, x0=x0, records=[],
                     adoption_eval=None, first_complex_proposal_eval=None)


def algebra_cell_ratio(alpha: float, beta: float, evals: int, cell: int,
                       root_seed: int = DEFAULT_ROOT_SEED) -> float:
    """best rho / m after an ES of ``evals`` evaluations inside Algebra(alpha, beta). One cell of the map."""
    alg = Algebra(alpha, beta)
    budget = Budget(alg.rho, int(evals))
    es = _ES(_stream(root_seed, ROLE_SEARCH, ALGEBRA_MAP_ID, 0, cell), _alg_start, _clip_pair)
    _until_spent(es.run, budget, int(evals))
    return float(budget.best) / M_FLOOR
