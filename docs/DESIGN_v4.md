# evoforge: LLM-Guided Evolution over Formally-Grounded Symbolic Expressions

> *A unified framework for evolving verified symbolic artifacts — with theorem proofs and turbulence closures as first-class backends.*

**v4 — locked invariants, credit assignment, deterministic evaluation**

---

## 1. Thesis

Most LLM + evolutionary algorithm work operates in domains where fitness is noisy, approximate, or expensive to evaluate. This project targets a different regime: **domains where a formal system provides a deterministic, authoritative fitness signal**. A type checker either accepts a proof or it doesn't. A PDE solver either converges to match benchmark data or it doesn't.

This constraint eliminates reward hacking, makes evolution reproducible, and lets the LLM focus on **semantically informed mutation** rather than serving as a noisy fitness proxy.

**evoforge** implements a generic LLM-guided evolutionary engine with pluggable backends:

| Backend | IR Type | Fitness Oracle | LLM Role |
|---------|---------|----------------|----------|
| **lean** | `TacticSequence` | Lean server (stepwise, deterministic) | Proof strategies from goal state + credit signals |
| **cfd** | `ClosureExpr` (SymPy) | 1D sediment-transport solver (seeded, deterministic) | Physically-motivated functional forms from subtree credit |

---

## 2. System Invariants

These four properties are **non-negotiable contracts** enforced throughout the system. Every component that touches identity, evaluation, credit, or LLM output must satisfy these. Violations are bugs, not design choices.

### Invariant 1: Canonical Identity Pipeline

Every genome has **exactly one** canonical identity. All downstream systems use this identity, never the raw genome string.

```
genome (str)
  │
  ▼
parse(genome) → IR | None           # backend-specific parsing
  │
  ▼
canonicalize(ir) → IR               # normalize: whitespace, commutativity,
  │                                 #   no-ops, constant folding
  ▼
structural_hash(canonical_ir) → str # deterministic content hash
  │
  ▼
IDENTITY (used everywhere below)
```

**Enforcement points:**

| System | Uses Identity For |
|--------|-------------------|
| Evaluation cache | Cache key = `f"{backend_version}:{eval_config_hash}:{structural_hash}"` |
| Archive insertion | Deduplicate on structural_hash before storing |
| MAP-Elites | Cell occupancy keyed on (behavior_descriptor, structural_hash) |
| Population | No two individuals with same structural_hash in population |
| Logging | All metrics reference structural_hash, not raw genome |
| LLM dedup | Don't generate mutation prompts for parents whose structural_hash matches recent offspring |

**Canonicalization contracts per backend:**

Lean `TacticSequence`:
- Normalize whitespace (collapse multiple spaces, trim)
- Sort `simp` lemma lists alphabetically: `simp [b, a]` → `simp [a, b]`
- Remove proven no-op wrappers: `try { exact h }` where `h` is the goal → `exact h`
- Normalize `by` block formatting
- Collapse consecutive `skip` tactics

CFD `ClosureExpr`:
- SymPy `simplify()` + `nsimplify(rational=False)`
- Canonical ordering of commutative operations (SymPy handles this via `srepr`)
- Constant folding: `2 * 3 * x` → `6*x`
- Tolerance-aware constant normalization: constants within `1e-10` of each other are unified

**Test:** For each backend, a suite of `test_canonicalization.py` tests asserting:
```python
assert backend.canonicalize("1 + x") == backend.canonicalize("x + 1")
assert backend.canonicalize("simp [b, a]") == backend.canonicalize("simp [a, b]")
assert structural_hash(canon_a) == structural_hash(canon_b)  # when semantically equivalent
assert structural_hash(canon_a) != structural_hash(canon_c)  # when semantically different
```

### Invariant 2: Credit Assignment

Every evaluation produces **per-substructure credit**, not just whole-program fitness. This is what makes mutation informed rather than blind.

```python
@dataclass
class Credit:
    """Attribution of fitness signal to a substructure."""
    location: int                   # step index (Lean) or AST node index (CFD)
    score: float                    # positive = contributed, negative = harmful
    signal: str                     # human-readable: "goal closed", "divergence source", etc.
    confidence: float = 1.0         # 1.0 for Lean (deterministic), may be < 1 for CFD estimates

def assign_credit(ir: IRProtocol, fitness: Fitness,
                  diagnostics: Diagnostics, trace: EvaluationTrace) -> list[Credit]:
    """Backend-specific. Produces credits consumed by mutation operators."""
    ...
```

**Lean credit assignment:**

Stepwise evaluation naturally produces credit — each tactic either succeeds or fails, and we observe the goal state before and after:

```python
def assign_credit_lean(ir: TacticSequence, diag: LeanDiagnostics,
                        trace: LeanEvalTrace) -> list[Credit]:
    credits = []
    for i, step_result in enumerate(trace.step_results):
        if step_result.succeeded:
            # Credit = how many goals were closed or simplified
            goals_before = step_result.goals_before
            goals_after = step_result.goals_after
            reduction = len(goals_before) - len(goals_after)
            credits.append(Credit(
                location=i,
                score=0.3 * reduction + 0.1,  # base credit for not failing
                signal=f"closed {reduction} goals" if reduction > 0 else "maintained progress",
            ))
        else:
            credits.append(Credit(
                location=i,
                score=-0.5,
                signal=f"failed: {step_result.error_type}: {step_result.error_message[:80]}",
            ))
            break  # no credit after failure point
    return credits
```

**CFD credit assignment:**

Approximate, based on ablation — evaluate with each subtree zeroed out:

```python
def assign_credit_cfd(ir: ClosureExpr, fitness: Fitness,
                       diagnostics: CFDDiagnostics) -> list[Credit]:
    credits = []
    baseline_error = fitness.auxiliary["raw_accuracy"]
    terms = ir.additive_terms()  # decompose into additive components

    for i, term in enumerate(terms):
        # Ablation: what happens without this term?
        ablated = ir.remove_term(i)
        if ablated.complexity() == 0:
            continue
        ablated_fitness = quick_evaluate(ablated)  # cached / fast path
        delta = baseline_error - ablated_fitness.auxiliary["raw_accuracy"]
        credits.append(Credit(
            location=i,
            score=delta,
            signal=f"term '{term}' contributes {delta:+.3f} accuracy",
            confidence=0.8,  # approximate due to interaction effects
        ))
    return credits
```

**How credit feeds into mutation:**

1. **LLM mutation prompts** include a credit summary: "Steps 1–4 are solid (closed 2 goals each). Step 5 fails with type_mismatch. Focus changes on step 5 or replace steps 5+."
2. **Cheap operators** use credit to target mutations: `PrefixTruncation` cuts after the last positively-credited step. `SubtreeMutation` preferentially mutates low-credit subtrees.
3. **Crossover** selects high-credit regions from each parent.
4. **Search memory** aggregates credit across generations to identify which substructures recurrently succeed or fail.

### Invariant 3: Evaluation Determinism

Given the same canonical IR and evaluation seed, `evaluate()` must return **identical results**. This is enforced at the system boundary.

```python
class DeterministicEvaluator:
    """Wraps backend evaluation with determinism enforcement."""

    def __init__(self, backend: Backend, eval_seed: int):
        self.backend = backend
        self.eval_seed = eval_seed

    def evaluate(self, ir: IRProtocol) -> tuple[Fitness, Diagnostics, EvaluationTrace]:
        """Deterministic evaluation. Same ir + same eval_seed → same result. Always."""
        result = self.backend.evaluate(ir, seed=self.eval_seed)
        return result
```

**Per-backend determinism contracts:**

Lean:
- Lean type checking is inherently deterministic (same input → same output)
- Elaboration order is fixed (no parallelism in checker)
- `lake env lean` invoked with fixed environment variables
- **Test:** evaluate same proof 10 times, assert identical diagnostics

CFD:
- Solver seeded with `eval_seed` for any stochastic components
- Floating-point: solver uses `np.float64` throughout, no mixed precision
- Tolerance pinned: `rtol=1e-8, atol=1e-10` in all ODE integrators
- Parallelism: solver runs single-threaded (no thread-order nondeterminism)
- **Test:** evaluate same expression 100 times, assert max |Δfitness| < `1e-12`

**If true determinism is impossible** (discovered during testing):

Fall back to **controlled stochastic evaluation**:
```python
def evaluate_stochastic(self, ir: IRProtocol, n_samples: int = 3) -> StochasticResult:
    results = [self.backend.evaluate(ir, seed=self.eval_seed + i) for i in range(n_samples)]
    mean_fitness = Fitness(
        primary=np.mean([r.fitness.primary for r in results]),
        auxiliary={k: np.mean([r.fitness.auxiliary[k] for r in results])
                  for k in results[0].fitness.auxiliary},
        ...
    )
    variance = np.var([r.fitness.primary for r in results])
    return StochasticResult(fitness=mean_fitness, variance=variance, n_samples=n_samples)
```

Cache stores `(mean_fitness, variance, n_samples)`. Selection accounts for variance (Thompson sampling or UCB).

### Invariant 4: Validated LLM Generation Boundary

Every LLM output passes through a **strict validation pipeline** before entering the evolutionary system. No unvalidated LLM output ever reaches the evaluator.

```python
class ValidatedGenerator:
    """Hard boundary between LLM and evolutionary system.
    Nothing passes without validation."""

    def __init__(self, backend: Backend, llm_client, model: str,
                 max_attempts: int = 3):
        self.backend = backend
        self.client = llm_client
        self.model = model
        self.max_attempts = max_attempts

    def generate(self, prompt: str, system: str,
                 temperature: float) -> Individual | None:
        """Generate a valid individual or return None.
        None triggers fallback to cheap operator (not retry)."""
        for attempt in range(self.max_attempts):
            try:
                response = self.client.messages.create(
                    model=self.model, system=system,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                raw_text = response.content[0].text
            except APIError:
                continue  # network retry only

            # Stage 1: Extract genome from LLM output
            genome = self.backend.extract_genome(raw_text)
            if genome is None:
                continue

            # Stage 2: Parse into IR
            ir = self.backend.parse(genome)
            if ir is None:
                continue

            # Stage 3: Backend-specific structural validation
            violations = self.backend.validate_structure(ir)
            if violations:
                continue

            # Stage 4: Canonicalize + hash
            canonical = ir.canonicalize()
            genome = canonical.serialize()
            ir_hash = canonical.structural_hash()

            return Individual(genome=genome, ir=canonical, ir_hash=ir_hash)

        # All attempts failed → return None (caller uses cheap operator fallback)
        return None
```

**Backend validation (`validate_structure`):**

Lean:
- All tactic names are valid Lean tactics (whitelist)
- No embedded `sorry` (unless explicitly allowed in config)
- No infinite loops (`repeat { ... }` without termination)
- Balanced delimiters (`{}`/`⟨⟩`/`()`/`[]`)

CFD:
- Expression only uses allowed symbols and functions
- AST depth ≤ `max_ast_depth`
- No division by zero for any variable in domain
- Dimensional consistency (via SymPy dimensional analysis)

**Fallback contract:** when `generate()` returns `None`:
```python
# In MutationEnsemble:
validated = generator.generate(prompt, system, temperature)
if validated is None:
    # Fall back to cheap operator — never inject invalid individual
    op = self._cheapest_operator()
    genome = op.apply(parent, context)
    # ... parse/validate as usual (cheap ops produce valid IR by construction)
```

**Key property:** the evolutionary population **never contains unparseable or structurally invalid individuals**. Invalid LLM output is discarded, not penalized. Only *parseable but incorrect* programs (e.g., a Lean proof that type-checks but doesn't close all goals) receive low fitness.

This is a revision from v3, where invalid parses got fitness 0.05. The v3 approach pollutes the population and cache with garbage. The v4 approach keeps the population clean.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          evoforge core                               │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │                  IDENTITY PIPELINE (Invariant 1)              │     │
│  │  genome → parse → canonicalize → structural_hash → IDENTITY   │     │
│  │  (all downstream systems use IDENTITY, never raw genome)      │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  ┌────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │ Population  │  │ SelectionStrategy│  │ Archive                  │  │
│  │ Manager     │  │ (scalar, pareto, │  │ (identity-keyed cache,   │  │
│  │ (identity-  │  │  lexicase,       │  │  prefix cache, lineage,  │  │
│  │  deduplicated)│ │  map_elites)    │  │  neighbors, credit log)  │  │
│  └──────┬──────┘  └────────┬────────┘  └───────────┬──────────────┘  │
│         │                  │                       │                 │
│  ┌──────▼──────────────────▼───────────────────────▼──────────────┐  │
│  │                    Evolution Loop                               │  │
│  │  seed → [identity pipeline → deterministic eval → credit] → loop│  │
│  │    parents = strategy.select(population)                        │  │
│  │    offspring = ensemble.generate(parents, memory, credits)      │  │
│  │      ↳ LLM: ValidatedGenerator (Invariant 4)                   │  │
│  │      ↳ Cheap: operate on IR directly (credit-guided)            │  │
│  │    for ind in offspring:                                        │  │
│  │      identity pipeline (Invariant 1)                            │  │
│  │      if cache.hit(identity): reuse                              │  │
│  │      else: deterministic eval (Invariant 3) → credit (Inv 2)   │  │
│  │    dedup(offspring, population)  ← identity-based               │  │
│  │    population = survive(population, offspring)                  │  │
│  │    archive.update(offspring)                                    │  │
│  │    memory.update(offspring, credits)                            │  │
│  └────────────────────────────────────────────────────────────────┘  │
│         │                  │                │                        │
│  ┌──────▼───────┐  ┌──────▼──────┐  ┌──────▼───────┐                │
│  │ Mutation      │  │ Backend     │  │ Execution    │                │
│  │ Ensemble      │  │ Interface   │  │ Scheduler    │                │
│  │ (LLM via      │  │ (with IR,   │  │ (async pool, │                │
│  │  Validated-   │  │  stepwise,  │  │  backpressure│                │
│  │  Generator +  │  │  credit)    │  │  cost acctg) │                │
│  │  cheap ops    │  │             │  │              │                │
│  │  on IR)       │  │             │  │              │                │
│  └──────────────┘  └──────┬──────┘  └──────────────┘                │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │   lean     │ │    cfd    │ │  (future)  │
        │  backend   │ │  backend  │ │  backends  │
        └───────────┘ └───────────┘ └───────────┘
```

---

## 4. Core Data Structures (`evoforge/core/types.py`)

### 4.1 Structured Fitness

```python
@dataclass(frozen=True)
class Fitness:
    primary: float
    auxiliary: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, bool] = field(default_factory=dict)
    feasible: bool = True

    def dominates(self, other: "Fitness") -> bool:
        dominated_in_one = False
        for key in self.auxiliary:
            if key in other.auxiliary:
                if self.auxiliary[key] < other.auxiliary[key]:
                    return False
                if self.auxiliary[key] > other.auxiliary[key]:
                    dominated_in_one = True
        return dominated_in_one and self.primary >= other.primary
```

### 4.2 Credit

```python
@dataclass
class Credit:
    location: int               # step index (Lean) or AST node index (CFD)
    score: float                # positive = contributed, negative = harmful
    signal: str                 # "closed 2 goals", "divergence source at Ri_g > 0.6"
    confidence: float = 1.0     # 1.0 for Lean (deterministic), <1 for CFD estimates
```

### 4.3 Structured Diagnostics (Backend-Specific)

```python
class Diagnostics(Protocol):
    def summary(self, max_tokens: int = 500) -> str: ...
    def credit_summary(self, credits: list[Credit], max_tokens: int = 300) -> str:
        """Render credits as text for LLM mutation prompts."""
        ...

@dataclass
class LeanDiagnostics:
    success: bool
    goals_remaining: int
    goal_types: list[str]
    goal_contexts: list[str]
    error_type: str | None
    error_message: str | None
    stuck_tactic_index: int | None
    stuck_tactic: str | None
    steps_succeeded: int
    metavar_count: int

    def summary(self, max_tokens: int = 500) -> str:
        if self.success:
            return "Proof complete."
        parts = [f"Failed at step {self.stuck_tactic_index}: `{self.stuck_tactic}`"]
        parts.append(f"Error: {self.error_type}: {self.error_message}")
        parts.append(f"Goals remaining: {self.goals_remaining}")
        for i, (typ, ctx) in enumerate(zip(self.goal_types, self.goal_contexts)):
            parts.append(f"  Goal {i}: {typ}")
            if ctx:
                parts.append(f"    Context: {ctx[:100]}")
        return "\n".join(parts)

    def credit_summary(self, credits: list[Credit], max_tokens: int = 300) -> str:
        good = [c for c in credits if c.score > 0]
        bad = [c for c in credits if c.score <= 0]
        parts = []
        if good:
            parts.append(f"Steps 0–{good[-1].location} are solid: "
                         + "; ".join(c.signal for c in good[-3:]))
        if bad:
            parts.append(f"Step {bad[0].location} fails: {bad[0].signal}")
            parts.append("Focus mutations on this step or replace everything after it.")
        return "\n".join(parts)

@dataclass
class CFDDiagnostics:
    converged_cases: int
    total_cases: int
    per_case: list[CaseResult]
    worst_case: str
    worst_error: float
    divergence_info: DivergenceInfo | None

    def summary(self, max_tokens: int = 500) -> str:
        parts = [f"Converged: {self.converged_cases}/{self.total_cases}"]
        if self.divergence_info:
            d = self.divergence_info
            parts.append(f"Diverged: {self.worst_case} at t={d.time:.2f}s, "
                         f"{d.instability} at y/δ={d.location:.3f}")
        parts.append(f"Worst error: {self.worst_case} (L2={self.worst_error:.4f})")
        return "\n".join(parts)

    def credit_summary(self, credits: list[Credit], max_tokens: int = 300) -> str:
        helpful = sorted([c for c in credits if c.score > 0], key=lambda c: -c.score)
        harmful = sorted([c for c in credits if c.score < 0], key=lambda c: c.score)
        parts = []
        if helpful:
            parts.append("Helpful terms: " + "; ".join(c.signal for c in helpful[:3]))
        if harmful:
            parts.append("Harmful terms: " + "; ".join(c.signal for c in harmful[:3]))
            parts.append("Consider removing or modifying these.")
        return "\n".join(parts)
```

### 4.4 Evaluation Trace

```python
@dataclass
class EvaluationTrace:
    """Full trace of evaluation for credit assignment.
    Backend-specific inner structure."""
    pass

@dataclass
class LeanEvalTrace(EvaluationTrace):
    step_results: list[TacticStepResult]

@dataclass
class TacticStepResult:
    succeeded: bool
    goals_before: list[Goal]
    goals_after: list[Goal]
    error_type: str | None
    error_message: str | None

@dataclass
class CFDEvalTrace(EvaluationTrace):
    per_case_traces: list[CaseSolverTrace]
    # CaseSolverTrace contains time series of residuals, velocity profiles, etc.
```

### 4.5 Individual

```python
@dataclass
class Individual:
    genome: str                             # always canonical (Invariant 1)
    ir: IRProtocol | None = None            # parsed + canonicalized IR
    ir_hash: str | None = None              # structural hash (Invariant 1)
    fitness: Fitness | None = None
    diagnostics: Diagnostics | None = None
    credits: list[Credit] | None = None     # per-substructure credit (Invariant 2)
    lineage: list[str] = field(default_factory=list)
    generation: int = 0
    id: str = field(default_factory=lambda: str(uuid4()))
    behavior_descriptor: tuple | None = None
    mutation_source: str = ""
```

### 4.6 Search Memory

```python
@dataclass
class SearchMemory:
    successful_patterns: list[Pattern]
    failure_modes: list[FailureMode]
    useful_constructs: list[str]
    dead_ends: list[str]
    best_fitness_history: list[float]
    credit_aggregates: dict[str, float]     # substructure → accumulated credit

    def update(self, generation: int, offspring: list[Individual], archive: Archive):
        for ind in offspring:
            if ind.fitness and ind.fitness.primary > self._threshold():
                self._extract_patterns(ind)
            if ind.fitness and ind.fitness.primary < 0.1:
                self._record_failure(ind)
            # Aggregate credits across generations
            if ind.credits:
                for c in ind.credits:
                    key = self._credit_key(ind, c)  # backend-specific substructure identifier
                    self.credit_aggregates[key] = (
                        self.credit_aggregates.get(key, 0) + c.score
                    )
        self._detect_dead_ends(archive)
        self.best_fitness_history.append(
            max((i.fitness.primary for i in offspring if i.fitness), default=0))

    def prompt_section(self, max_tokens: int = 400) -> str:
        parts = []
        if self.successful_patterns:
            parts.append("PATTERNS THAT WORK: " + ", ".join(
                p.description for p in self.successful_patterns[:5]))
        if self.dead_ends:
            parts.append("DEAD ENDS (don't retry): " + ", ".join(self.dead_ends[:5]))
        if self.useful_constructs:
            parts.append("USEFUL BUILDING BLOCKS: " + ", ".join(self.useful_constructs[:10]))
        return "\n".join(parts)

@dataclass
class Reflection:
    strategies_to_try: list[str]
    strategies_to_avoid: list[str]
    useful_primitives: list[str]
    population_diagnosis: str
    suggested_temperature: float | None
```

---

## 5. IR Layer (`evoforge/core/ir.py`)

### 5.1 IR Protocol

```python
class IRProtocol(Protocol):
    def canonicalize(self) -> "IRProtocol":
        """Idempotent normalization. canonicalize(canonicalize(x)) == canonicalize(x)."""
        ...
    def structural_hash(self) -> str:
        """Deterministic hash of canonical form."""
        ...
    def serialize(self) -> str:
        """Roundtrip: parse(serialize(canonicalize(ir))) ≈ canonicalize(ir)."""
        ...
    def complexity(self) -> int:
        """Structural complexity (AST nodes, tactic count)."""
        ...
```

### 5.2 Lean IR: `TacticSequence`

```python
@dataclass(frozen=True)
class TacticStep:
    tactic: str
    args: tuple[str, ...]
    raw: str

@dataclass
class TacticSequence:
    steps: list[TacticStep]

    def canonicalize(self) -> "TacticSequence":
        canonical_steps = []
        for step in self.steps:
            raw = " ".join(step.raw.split())
            if step.tactic == "simp" and step.args:
                args = tuple(sorted(step.args))
            else:
                args = step.args
            # Remove no-op skip
            if step.tactic == "skip":
                continue
            canonical_steps.append(TacticStep(tactic=step.tactic, args=args, raw=raw))
        return TacticSequence(steps=canonical_steps)

    def structural_hash(self) -> str:
        canonical = self.canonicalize()
        content = "|".join(f"{s.tactic}({','.join(s.args)})" for s in canonical.steps)
        return hashlib.sha256(content.encode()).hexdigest()

    def prefix(self, k: int) -> "TacticSequence":
        return TacticSequence(steps=self.steps[:k])

    def serialize(self) -> str:
        return "\n".join(s.raw for s in self.steps)

    def complexity(self) -> int:
        return len(self.steps)
```

### 5.3 CFD IR: `ClosureExpr`

```python
class ClosureExpr:
    def __init__(self, expr: sympy.Expr):
        self.expr = expr

    def canonicalize(self) -> "ClosureExpr":
        canonical = sympy.simplify(self.expr)
        canonical = sympy.nsimplify(canonical, rational=False)
        return ClosureExpr(canonical)

    def structural_hash(self) -> str:
        canonical = self.canonicalize()
        return hashlib.sha256(sympy.srepr(canonical.expr).encode()).hexdigest()

    def serialize(self) -> str:
        return str(self.canonicalize().expr)

    def complexity(self) -> int:
        return sum(1 for _ in sympy.preorder_traversal(self.expr))

    def additive_terms(self) -> list[sympy.Expr]:
        """Decompose into additive components for credit assignment."""
        return list(sympy.Add.make_args(self.expr))

    def remove_term(self, index: int) -> "ClosureExpr":
        terms = self.additive_terms()
        remaining = [t for i, t in enumerate(terms) if i != index]
        return ClosureExpr(sympy.Add(*remaining) if remaining else sympy.Integer(0))

    def replace_subtree(self, index: int, replacement: sympy.Expr) -> "ClosureExpr":
        nodes = list(sympy.preorder_traversal(self.expr))
        if index >= len(nodes):
            return self
        target = nodes[index]
        return ClosureExpr(self.expr.subs(target, replacement))
```

---

## 6. Selection Strategies (`evoforge/core/selection.py`)

### 6.1 Strategy ABC

```python
class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, population: list[Individual], k: int) -> list[Individual]: ...
    @abstractmethod
    def survive(self, population: list[Individual],
                offspring: list[Individual], elite_k: int) -> list[Individual]: ...
```

### 6.2 Implementations

**ScalarTournament:** tournament selection on `fitness.primary`.

**ParetoNSGA2:** NSGA-II with crowding distance. For CFD: accuracy vs. complexity.

**Lexicase:** iterates over `fitness.auxiliary` keys in random order. For Lean: `goal_progress`, `steps_succeeded`, `goal_familiarity`.

**MAPElites (with explicit competition and sampling):**

```python
class MAPElites(SelectionStrategy):
    def __init__(self, behavior_space: BehaviorSpaceConfig,
                 sampling_strategy: str = "weighted"):
        self.grid: dict[tuple, Individual] = {}
        self.behavior_space = behavior_space
        self.sampling_strategy = sampling_strategy  # "uniform", "weighted", "curiosity"
        self.cell_ages: dict[tuple, int] = {}       # for aging/replacement

    def select(self, population: list[Individual], k: int) -> list[Individual]:
        occupied = list(self.grid.items())
        if self.sampling_strategy == "uniform":
            cells = [random.choice(occupied) for _ in range(k)]
            return [cell[1] for cell in cells]
        elif self.sampling_strategy == "weighted":
            # Weight by: fitness rank × novelty × inverse age
            weights = []
            for cell_key, ind in occupied:
                fitness_w = ind.fitness.primary if ind.fitness else 0
                age = self.cell_ages.get(cell_key, 0)
                age_w = 1.0 / (1.0 + age * 0.1)  # prefer fresher
                novelty_w = self._neighbor_sparsity(cell_key)
                weights.append(fitness_w * age_w * novelty_w)
            total = sum(weights) + 1e-8
            probs = [w / total for w in weights]
            indices = np.random.choice(len(occupied), size=k, p=probs)
            return [occupied[i][1] for i in indices]

    def survive(self, population, offspring, elite_k):
        for ind in offspring:
            if ind.behavior_descriptor is None:
                continue
            cell = self._discretize(ind.behavior_descriptor)
            if cell not in self.grid:
                self.grid[cell] = ind
                self.cell_ages[cell] = 0
            elif ind.fitness.primary > self.grid[cell].fitness.primary:
                self.grid[cell] = ind
                self.cell_ages[cell] = 0
            else:
                self.cell_ages[cell] = self.cell_ages.get(cell, 0) + 1
        return list(self.grid.values())

    def coverage(self) -> float:
        total_cells = 1
        for dim in self.behavior_space.dimensions:
            total_cells *= len(dim.bin_edges) - 1 if dim.type == "numeric" else len(dim.bins)
        return len(self.grid) / total_cells

    def _neighbor_sparsity(self, cell_key: tuple) -> float:
        """Novelty: how many neighboring cells are empty?"""
        neighbors = self._get_neighbors(cell_key)
        empty = sum(1 for n in neighbors if n not in self.grid)
        return empty / max(len(neighbors), 1)
```

**Separation of exploration and exploitation:**

- **Archive (MAP-Elites grid)** = exploration. Maintains diversity across behavior space.
- **Within-cell competition** = exploitation. Only the fittest individual survives in each cell.
- **Parent sampling** = bridge. Weighted sampling balances between exploring sparse regions (novelty) and refining high-fitness regions (fitness rank), with aging to prevent stale individuals from dominating.

### 6.3 Behavior Descriptor Stability

Descriptors are computed on **canonical IR** and are deterministic by construction:

Lean:
- `strategy_class`: first tactic of canonical IR → category. Mapping is a fixed dictionary.
- `proof_depth`: `len(canonical_ir.steps)`, bucketed into [1–3, 4–6, 7–10, 11+].

CFD:
- `dominant_variable`: highest partial derivative magnitude of canonical SymPy expr. SymPy differentiation is deterministic.
- `complexity`: AST node count of canonical expr.

**Test:** for each backend, assert `descriptor(canonicalize(parse(g))) == descriptor(canonicalize(parse(g)))` over 100 random genomes, 10 trials each.

---

## 7. Mutation Operators (`evoforge/core/mutation.py`)

### 7.1 Operator ABC

```python
class MutationOperator(ABC):
    name: str
    cost: str  # "cheap" or "llm"

    @abstractmethod
    def apply(self, parent: Individual, context: MutationContext) -> str:
        """Produce new genome string. Cheap ops operate on parent.ir directly.
        LLM ops go through ValidatedGenerator."""
        ...

@dataclass
class MutationContext:
    generation: int
    memory: SearchMemory
    guidance: Reflection | None
    temperature: float
    backend: Backend
    credits: list[Credit] | None    # parent's credits, for targeted mutation
```

### 7.2 Credit-Guided Cheap Operators

Cheap operators use credit to target mutations intelligently:

```python
class PrefixTruncation(MutationOperator):
    """Cut after the last positively-credited step."""
    name = "prefix_truncation"
    cost = "cheap"

    def apply(self, parent: Individual, context: MutationContext) -> str:
        ir: TacticSequence = parent.ir
        credits = context.credits or []
        # Find last positive credit
        last_good = 0
        for c in credits:
            if c.score > 0:
                last_good = c.location + 1
        return ir.prefix(max(1, last_good)).serialize()


class SubtreeMutation(MutationOperator):
    """Preferentially mutate low-credit subtrees."""
    name = "subtree_mutation"
    cost = "cheap"

    def apply(self, parent: Individual, context: MutationContext) -> str:
        ir: ClosureExpr = parent.ir
        credits = context.credits or []
        # Target the lowest-credit subtree
        if credits:
            worst = min(credits, key=lambda c: c.score)
            target_idx = worst.location
        else:
            nodes = list(sympy.preorder_traversal(ir.expr))
            target_idx = random.randrange(1, max(2, len(nodes)))
        replacement = self._random_subexpr()
        return ir.replace_subtree(target_idx, replacement).serialize()
```

### 7.3 Mutation Mix Scheduling

Mutation weights shift over the course of a run:

```python
class MutationEnsemble:
    def __init__(self, operators: list[MutationOperator],
                 initial_weights: dict[str, float],
                 adaptive: bool = True,
                 schedule: str = "adaptive"):  # "fixed", "adaptive", "phased"
        self.operators = {op.name: op for op in operators}
        self.weights = dict(initial_weights)
        self.adaptive = adaptive
        self.schedule = schedule
        self.stats: dict[str, OperatorStats] = {n: OperatorStats() for n in self.operators}

    def get_weights(self, generation: int, max_generations: int) -> dict[str, float]:
        if self.schedule == "fixed":
            return self.weights
        elif self.schedule == "phased":
            # Early: more exploration (LLM + random cheap)
            # Late: more exploitation (targeted cheap ops)
            phase = generation / max_generations
            llm_factor = max(0.2, 1.0 - phase * 0.6)
            cheap_factor = 1.0 - llm_factor
            adjusted = {}
            for name, w in self.weights.items():
                op = self.operators[name]
                if op.cost == "llm":
                    adjusted[name] = w * llm_factor
                else:
                    adjusted[name] = w * cheap_factor
            total = sum(adjusted.values())
            return {k: v / total for k, v in adjusted.items()}
        elif self.schedule == "adaptive":
            return self._adaptive_weights()

    def update_stats(self, generation_results: list[Individual]):
        for ind in generation_results:
            if ind.fitness and ind.mutation_source in self.stats:
                self.stats[ind.mutation_source].record(ind.fitness.primary)
```

---

## 8. Evaluation & Caching (`evoforge/core/evaluator.py`)

### 8.1 Multi-Level Cache (Identity-Keyed)

```python
class EvaluationCache:
    def __init__(self, archive: Archive, backend_version: str, eval_config_hash: str):
        self.archive = archive
        self.version_prefix = f"{backend_version}:{eval_config_hash}"
        self._parse_cache: dict[str, IRProtocol] = {}

    def _make_key(self, ir_hash: str) -> str:
        return f"{self.version_prefix}:{ir_hash}"

    # Level 1: Parse cache (in-memory)
    def get_parsed(self, genome: str) -> IRProtocol | None:
        return self._parse_cache.get(genome)
    def put_parsed(self, genome: str, ir: IRProtocol):
        self._parse_cache[genome] = ir

    # Level 2: Prefix cache (Lean stepwise — on disk)
    def get_prefix(self, prefix_hash: str) -> PrefixResult | None:
        return self.archive.lookup(self._make_key(f"prefix:{prefix_hash}"))
    def put_prefix(self, prefix_hash: str, result: PrefixResult):
        self.archive.store(self._make_key(f"prefix:{prefix_hash}"), result)

    # Level 3: Full evaluation cache (on disk)
    def get_full(self, ir_hash: str) -> tuple[Fitness, Diagnostics, list[Credit]] | None:
        return self.archive.lookup_fitness(self._make_key(ir_hash))
    def put_full(self, ir_hash: str, fitness: Fitness, diag: Diagnostics, credits: list[Credit]):
        self.archive.store_fitness(self._make_key(ir_hash), fitness, diag, credits)
```

### 8.2 Deterministic Async Evaluator

```python
class AsyncEvaluator:
    def __init__(self, backend: Backend, cache: EvaluationCache,
                 config: ParallelismConfig, eval_seed: int):
        self.backend = backend
        self.cache = cache
        self.max_workers = config.max_workers
        self.eval_timeout = config.eval_timeout
        self.eval_seed = eval_seed
        self._semaphore = asyncio.Semaphore(config.max_pending)

    async def evaluate_batch(self, individuals: list[Individual]):
        tasks = []
        for ind in individuals:
            # Identity pipeline already ran (parse, canonicalize, hash)
            # Check cache
            cached = self.cache.get_full(ind.ir_hash)
            if cached is not None:
                ind.fitness, ind.diagnostics, ind.credits = cached
                continue
            tasks.append(self._eval_with_backpressure(ind))
        await asyncio.gather(*tasks)

    async def _eval_with_backpressure(self, ind: Individual):
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            try:
                fitness, diagnostics, trace = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, self.backend.evaluate, ind.ir, self.eval_seed),
                    timeout=self.eval_timeout,
                )
                credits = self.backend.assign_credit(ind.ir, fitness, diagnostics, trace)
                ind.fitness = fitness
                ind.diagnostics = diagnostics
                ind.credits = credits
                self.cache.put_full(ind.ir_hash, fitness, diagnostics, credits)
            except asyncio.TimeoutError:
                ind.fitness = Fitness(primary=0.01, constraints={"timeout": True}, feasible=False)
                ind.diagnostics = None
                ind.credits = []
```

---

## 9. Backend Interface (Complete)

```python
class Backend(ABC):
    # --- Identity ---
    @abstractmethod
    def version(self) -> str: ...
    @abstractmethod
    def eval_config_hash(self) -> str: ...

    # --- IR pipeline ---
    @abstractmethod
    def parse(self, genome: str) -> IRProtocol | None: ...

    # --- Evaluation (deterministic: same ir + seed → same result) ---
    @abstractmethod
    def evaluate(self, ir: IRProtocol, seed: int) -> tuple[Fitness, Diagnostics, EvaluationTrace]: ...

    # --- Stepwise evaluation (Lean overrides) ---
    def evaluate_stepwise(self, ir: IRProtocol, seed: int,
                          cache: EvaluationCache) -> tuple[Fitness, Diagnostics, EvaluationTrace]:
        return self.evaluate(ir, seed)

    # --- Credit assignment (Invariant 2) ---
    @abstractmethod
    def assign_credit(self, ir: IRProtocol, fitness: Fitness,
                       diagnostics: Diagnostics, trace: EvaluationTrace) -> list[Credit]: ...

    # --- Structural validation (for ValidatedGenerator, Invariant 4) ---
    @abstractmethod
    def validate_structure(self, ir: IRProtocol) -> list[str]:
        """Return list of violations. Empty = valid."""
        ...

    # --- Seed population ---
    @abstractmethod
    def seed_population(self, n: int) -> list[str]: ...

    # --- Mutation operators ---
    @abstractmethod
    def mutation_operators(self) -> list[MutationOperator]: ...
    @abstractmethod
    def default_operator_weights(self) -> dict[str, float]: ...

    # --- LLM integration ---
    @abstractmethod
    def system_prompt(self) -> str: ...
    @abstractmethod
    def format_mutation_prompt(self, parent: Individual,
                                diagnostics_summary: str,
                                credit_summary: str,
                                guidance: Reflection | None,
                                memory_section: str) -> str: ...
    @abstractmethod
    def format_crossover_prompt(self, a: Individual, b: Individual,
                                 guidance: Reflection | None,
                                 memory_section: str) -> str: ...
    @abstractmethod
    def format_reflection_prompt(self, population: list[Individual],
                                  memory: SearchMemory,
                                  archive: Archive) -> str: ...
    @abstractmethod
    def extract_genome(self, llm_text: str) -> str | None:
        """Extract genome from LLM output. Returns None if extraction fails."""
        ...

    # --- Behavior space ---
    @abstractmethod
    def behavior_descriptor(self, ir: IRProtocol, fitness: Fitness,
                             diagnostics: Diagnostics) -> tuple: ...
    @abstractmethod
    def behavior_space(self) -> BehaviorSpaceConfig: ...

    # --- Selection recommendation ---
    @abstractmethod
    def recommended_selection(self) -> SelectionStrategy: ...
```

---

## 10. Lean Backend: Stepwise Evaluation with Credit

```python
class LeanStepwiseEvaluator:
    def evaluate_stepwise(self, ir: TacticSequence, seed: int,
                          cache: EvaluationCache) -> tuple[Fitness, LeanDiagnostics, LeanEvalTrace]:
        # Find longest cached prefix
        best_k = 0
        best_state = self._initial_goal_state()
        for k in range(len(ir.steps), 0, -1):
            prefix = ir.prefix(k)
            cached = cache.get_prefix(prefix.structural_hash())
            if cached is not None:
                best_k = k
                best_state = cached.goal_state
                break

        # Evaluate remaining steps
        step_results = []
        current_state = best_state
        for i in range(best_k, len(ir.steps)):
            result = self._apply_tactic(current_state, ir.steps[i])
            step_results.append(result)
            if result.succeeded:
                current_state = result.new_state
                # Cache this prefix
                prefix = ir.prefix(i + 1)
                cache.put_prefix(prefix.structural_hash(),
                    PrefixResult(goal_state=result.new_state))
            else:
                break
            if result.proof_complete:
                break

        trace = LeanEvalTrace(step_results=step_results)
        diagnostics = self._build_diagnostics(ir, step_results, current_state)
        fitness = self._compute_fitness(diagnostics)
        return fitness, diagnostics, trace

    def assign_credit(self, ir: TacticSequence, fitness: Fitness,
                       diagnostics: LeanDiagnostics, trace: LeanEvalTrace) -> list[Credit]:
        credits = []
        for i, result in enumerate(trace.step_results):
            if result.succeeded:
                reduction = len(result.goals_before) - len(result.goals_after)
                credits.append(Credit(
                    location=i,
                    score=0.3 * reduction + 0.1,
                    signal=f"closed {reduction} goals" if reduction > 0 else "maintained",
                ))
            else:
                credits.append(Credit(
                    location=i, score=-0.5,
                    signal=f"failed: {result.error_type}: {result.error_message[:80]}",
                ))
                break
        return credits
```

---

## 11. CFD Backend: Constraint-Integrated with Credit

```python
class CFDBackend(Backend):
    def evaluate(self, ir: ClosureExpr, seed: int) -> tuple[Fitness, CFDDiagnostics, CFDEvalTrace]:
        np.random.seed(seed)  # determinism (Invariant 3)

        penalty = 1.0
        constraints = {}

        dim_ok = self._check_dimensions(ir.expr)
        constraints["dimensional"] = dim_ok
        if not dim_ok: penalty *= 0.1

        unstrat_ok = self._check_unstratified_limit(ir.expr)
        constraints["unstratified_limit"] = unstrat_ok
        if not unstrat_ok: penalty *= 0.5

        suppress_ok = self._check_stratification_suppression(ir.expr)
        constraints["suppression"] = suppress_ok
        if not suppress_ok: penalty *= 0.5

        realizable = self._check_non_negative(ir.expr)
        constraints["realizable"] = realizable
        if not realizable: penalty *= 0.3

        case_results = []
        case_traces = []
        for case in self.benchmark_cases:
            try:
                result, trace = self.solver.run(
                    closure_expr=ir.expr, seed=seed, **case.params)
                err = self._l2_error(result, case.reference)
                case_results.append(CaseResult(case=case.name, error=err, converged=True))
                case_traces.append(trace)
            except SolverDivergenceError as e:
                case_results.append(CaseResult(
                    case=case.name, error=1.0, converged=False,
                    divergence=DivergenceInfo(
                        time=e.time, instability=e.instability, location=e.location)))
                case_traces.append(None)

        errors = [c.error for c in case_results]
        raw_fitness = 1.0 / (1.0 + np.mean(errors))

        diagnostics = CFDDiagnostics(
            converged_cases=sum(1 for c in case_results if c.converged),
            total_cases=len(case_results),
            per_case=case_results,
            worst_case=max(case_results, key=lambda c: c.error).case,
            worst_error=max(c.error for c in case_results),
            divergence_info=next((c.divergence for c in case_results if c.divergence), None),
        )

        return Fitness(
            primary=raw_fitness * penalty,
            auxiliary={
                "raw_accuracy": raw_fitness,
                "complexity": float(ir.complexity()),
                "penalty": penalty,
                "converged_fraction": diagnostics.converged_cases / diagnostics.total_cases,
            },
            constraints=constraints,
            feasible=all(constraints.values()),
        ), diagnostics, CFDEvalTrace(per_case_traces=case_traces)

    def assign_credit(self, ir: ClosureExpr, fitness: Fitness,
                       diagnostics: CFDDiagnostics, trace: CFDEvalTrace) -> list[Credit]:
        credits = []
        baseline = fitness.auxiliary["raw_accuracy"]
        terms = ir.additive_terms()
        for i, term in enumerate(terms):
            ablated = ir.remove_term(i)
            if ablated.complexity() == 0:
                continue
            # Quick eval (may hit cache)
            ablated_fit, _, _ = self.evaluate(ablated, seed=0)
            delta = baseline - ablated_fit.auxiliary["raw_accuracy"]
            credits.append(Credit(
                location=i, score=delta,
                signal=f"term '{term}': {delta:+.3f} accuracy impact",
                confidence=0.8,
            ))
        return credits
```

---

## 12. Execution Scheduler (`evoforge/core/scheduler.py`)

```python
class ExecutionScheduler:
    def __init__(self, config: SchedulerConfig):
        self.mode = config.mode
        self.llm_semaphore = asyncio.Semaphore(config.max_llm_concurrent)
        self.eval_semaphore = asyncio.Semaphore(config.max_eval_concurrent)
        self.llm_budget_per_gen = config.llm_budget_per_gen
        self.cost_tracker = CostTracker()

    async def dispatch_mutations(self, ensemble, parents, context):
        llm_calls = 0
        offspring = []
        weights = ensemble.get_weights(context.generation, context.max_generations)
        for parent in parents:
            op_name = weighted_choice(weights)
            op = ensemble.operators[op_name]
            if op.cost == "llm":
                if llm_calls >= self.llm_budget_per_gen:
                    op = ensemble.cheapest_operator()
                    op_name = op.name
                else:
                    async with self.llm_semaphore:
                        genome = await asyncio.to_thread(op.apply, parent, context)
                    llm_calls += 1
                    self.cost_tracker.record_llm_call()
                    offspring.append(Individual(genome=genome, mutation_source=op_name, ...))
                    continue
            genome = op.apply(parent, context)
            offspring.append(Individual(genome=genome, mutation_source=op_name, ...))
        return offspring

class CostTracker:
    """Track LLM tokens, evaluation time, wall clock."""
    def __init__(self):
        self.llm_calls = 0
        self.llm_tokens_in = 0
        self.llm_tokens_out = 0
        self.eval_time_s = 0.0
        self.wall_time_s = 0.0
        self.estimated_cost_usd = 0.0

    def record_llm_call(self, tokens_in=0, tokens_out=0):
        self.llm_calls += 1
        self.llm_tokens_in += tokens_in
        self.llm_tokens_out += tokens_out
        # Haiku pricing (approx)
        self.estimated_cost_usd += (tokens_in * 0.25 + tokens_out * 1.25) / 1e6

    def record_eval(self, duration_s: float):
        self.eval_time_s += duration_s

    def summary(self) -> str:
        return (f"LLM: {self.llm_calls} calls, ~${self.estimated_cost_usd:.2f} | "
                f"Eval: {self.eval_time_s:.1f}s | Wall: {self.wall_time_s:.1f}s")
```

**Backpressure & failure handling:**

| Condition | Action |
|-----------|--------|
| LLM slow (>10s/call) | Shift weight toward cheap ops for remainder of generation |
| Eval queue > max_pending | Block new evaluations until queue drains |
| Eval timeout | Fitness 0.01, no credit, continue |
| LLM budget exhausted | Remaining mutations use cheap operators |
| LLM output fails validation | `ValidatedGenerator` retries up to 3, then returns None → cheap fallback |
| Solver diverges <0.1s | Fast-path: skip remaining benchmark cases |

---

## 13. Configuration

```toml
[run]
name = "levy_proof_search_001"
backend = "lean"
max_generations = 200
seed = 42                           # master seed for reproducibility

[population]
size = 30
replacement = "generational"
elitism = 2

[selection]
method = "lexicase"
lexicase_keys = ["goal_progress", "steps_succeeded", "goal_familiarity"]

[mutation]
initial_weights = { llm_mutate = 0.3, llm_crossover = 0.15, prefix_truncation = 0.15, tactic_swap = 0.15, tactic_reorder = 0.1, splice_prefixes = 0.15 }
adaptive_weights = true
schedule = "phased"                 # "fixed", "adaptive", "phased"

[llm]
model = "claude-haiku-4-5-20251001"
reflection_model = "claude-sonnet-4-5-20250929"
temperature_start = 1.0
temperature_end = 0.3
temperature_schedule = "linear"
max_attempts = 3                    # ValidatedGenerator retry limit

[reflection]
interval = 10
include_top_k = 5
include_bottom_k = 5

[memory]
max_patterns = 20
max_dead_ends = 15
max_constructs = 30
stagnation_window = 20

[scheduler]
mode = "async_batch"
max_llm_concurrent = 4
max_eval_concurrent = 8
max_pending = 16
llm_budget_per_gen = 15

[evaluation]
seed = 42                           # evaluation seed (determinism invariant)
timeout = 60

[diversity]
strategy = "map_elites"
sampling = "weighted"               # "uniform", "weighted", "curiosity"

[ablation]
disable_llm = false
disable_diagnostics = false
disable_reflection = false
disable_memory = false
disable_cheap_operators = false
disable_credit = false              # new: test credit assignment value

[lean]
project_dir = "./lean_project"
target_theorem = "levy_characteristic_continuous"
target_file = "EvoForge/Evolved.lean"
mathlib_version = "v4.15.0"
stepwise = true

[cfd]
solver_module = "evoforge.backends.cfd.solver"
benchmark_dir = "./benchmarks"
max_ast_depth = 8
```

---

## 14. Experimental Design

### 14.1 Reproducibility Contract

```python
class ExperimentRunner:
    def __init__(self, config_path: str, master_seed: int):
        self.config = load_config(config_path)
        self.rng = random.Random(master_seed)
        np.random.seed(master_seed)
        # LLM calls are non-deterministic (temp > 0)
        # but all prompts + responses are logged for replay

    def run(self) -> ExperimentResult: ...
    def replay(self, archive_path: str) -> ExperimentResult:
        """Replay from logged LLM responses → fully deterministic."""
        ...
```

### 14.2 Metrics

| Metric | Description |
|--------|-------------|
| `best_fitness` | Max primary fitness |
| `mean_fitness` | Population mean |
| `diversity_entropy` | Shannon entropy over behavior space |
| `map_elites_coverage` | Fraction of grid cells occupied |
| `cache_hit_rate` | Evaluations served from cache |
| `identity_dedup_rate` | Genomes rejected as duplicates (should be >0, not too high) |
| `llm_calls` | LLM API calls this generation |
| `llm_call_efficiency` | Fitness improvement per LLM call |
| `operator_fitness_delta` | Mean fitness improvement per operator |
| `credit_utilization` | Do high-credit substructures persist? |
| `stagnation_counter` | Generations without improvement |
| `estimated_cost_usd` | Running cost tracker |

### 14.3 Ablation Matrix

| Experiment | LLM | Diag | Reflect | Memory | Cheap | Credit | Purpose |
|------------|:---:|:---:|:---:|:---:|:---:|:---:|---------|
| **Full** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Main |
| **No LLM** | ✗ | — | — | — | ✓ | ✓ | LLM value |
| **No diag** | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | Error message value |
| **No reflect** | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | Reflection value |
| **No memory** | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | Persistent memory value |
| **No credit** | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | Credit assignment value |
| **LLM only** | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | Cheap operator value |
| **Random** | ✗ | — | — | — | rand | ✗ | Sanity |

5 seeds each. Wilcoxon signed-rank for significance.

---

## 15. Repo Structure

```
evoforge/
├── README.md
├── DESIGN.md
├── pyproject.toml
├── evoforge/
│   ├── __init__.py
│   ├── core/
│   │   ├── types.py              # Fitness, Credit, Individual, Reflection, SearchMemory
│   │   ├── ir.py                 # IRProtocol, BehaviorSpaceConfig
│   │   ├── identity.py           # IdentityPipeline: parse → canonicalize → hash
│   │   ├── population.py
│   │   ├── selection.py          # SelectionStrategy ABC + 4 implementations
│   │   ├── mutation.py           # MutationOperator ABC + MutationEnsemble
│   │   ├── archive.py            # SQLite (identity-keyed cache, lineage, neighbors)
│   │   ├── evaluator.py          # EvaluationCache (3-level), AsyncEvaluator, DeterministicEvaluator
│   │   ├── generator.py          # ValidatedGenerator (Invariant 4)
│   │   ├── scheduler.py          # ExecutionScheduler, CostTracker
│   │   ├── memory.py             # SearchMemory with credit aggregation
│   │   ├── engine.py             # Main loop (enforces all 4 invariants)
│   │   └── config.py
│   ├── llm/
│   │   ├── client.py
│   │   ├── operators.py          # LLMMutate, LLMCrossover
│   │   └── templates/
│   ├── backends/
│   │   ├── base.py               # Backend ABC (with credit + validation)
│   │   ├── lean/
│   │   │   ├── backend.py
│   │   │   ├── ir.py             # TacticStep, TacticSequence
│   │   │   ├── evaluator.py      # LeanStepwiseEvaluator
│   │   │   ├── credit.py         # assign_credit_lean
│   │   │   ├── validation.py     # validate_structure (tactic whitelist, etc.)
│   │   │   ├── operators.py      # PrefixTruncation, TacticSwap, etc.
│   │   │   └── templates/
│   │   └── cfd/
│   │       ├── backend.py
│   │       ├── ir.py             # ClosureExpr
│   │       ├── constraints.py
│   │       ├── credit.py         # assign_credit_cfd (ablation-based)
│   │       ├── validation.py     # structural validation
│   │       ├── solver.py
│   │       ├── operators.py
│   │       └── templates/
│   └── viz/
│       ├── genealogy.py
│       ├── fitness_plots.py
│       ├── map_elites.py
│       ├── operator_analysis.py
│       ├── credit_analysis.py     # visualize credit flow over generations
│       └── dashboard.py
├── configs/                        # all run + ablation configs
├── lean_project/
├── benchmarks/
├── tests/
│   ├── test_core/
│   │   ├── test_identity.py       # canonicalization invariant tests
│   │   ├── test_determinism.py    # evaluation determinism tests
│   │   ├── test_credit.py
│   │   ├── test_generator.py      # validated generation boundary tests
│   │   ├── test_selection.py
│   │   ├── test_mutation.py
│   │   ├── test_cache.py
│   │   └── test_memory.py
│   ├── test_lean/
│   │   ├── test_canonicalization.py
│   │   ├── test_stepwise.py
│   │   ├── test_credit.py
│   │   └── test_operators.py
│   └── test_cfd/
│       ├── test_canonicalization.py
│       ├── test_determinism.py
│       ├── test_credit.py
│       └── test_operators.py
└── scripts/
    ├── run.py
    ├── analyze.py
    ├── ablation_sweep.py
    └── experiment_report.py
```

---

## 16. MVP Milestones

### Phase 1: Core + Lean Backend (Weeks 1–3)

| Week | Deliverable |
|------|-------------|
| 1 | `types.py`, `ir.py`, `identity.py` (pipeline + tests), `archive.py`, `selection.py` (4 strategies), `mutation.py` (operator framework), `generator.py` (ValidatedGenerator). **Test: identity invariant suite passes.** |
| 2 | Lean IR, stepwise evaluator, credit assignment, validation, cheap operators. **Test: determinism invariant passes (10 identical evals).** |
| 3 | LLM operators, SearchMemory, MutationEnsemble, Engine. First real run. **Test: full pipeline runs 10 generations without invariant violations.** |

### Phase 2: CFD Backend (Weeks 4–6)

| Week | Deliverable |
|------|-------------|
| 4 | SymPy IR (with canonicalization tests), constraints, validation, cheap operators. |
| 5 | Solver wrapper, benchmarks, CFDBackend integration, credit assignment (ablation-based). |
| 6 | NSGA-II, Pareto visualization, comparison vs. known closures. |

### Phase 3: Quality-Diversity + Analysis (Weeks 7–8)

| Week | Deliverable |
|------|-------------|
| 7 | MAP-Elites with stable descriptors. Descriptor stability tests. |
| 8 | Full ablation sweep (8 experiments × 5 seeds × 2 backends = 80 runs). Statistical analysis. |

### Phase 4: Paper & Release (Weeks 9–10)

---

## 17. Novelty Claims → Evidence

| Claim | Ablation | Expected Signal |
|-------|----------|-----------------|
| Error-informed mutation | Full vs. No Diag | Faster convergence |
| Reflection | Full vs. No Reflect | Less stagnation |
| Search memory | Full vs. No Memory | Higher cache hits, fewer wasted evals |
| Credit assignment | Full vs. No Credit | Targeted mutations > random mutations |
| LLM + cheap > either | Full vs. LLM Only vs. No LLM | Full dominates |
| Formal fitness prevents hacking | Qualitative | No degenerate individuals |
| Unified framework | Both backends | Same engine works |
| Quality-diversity for proofs | MAP-Elites coverage | Multiple distinct valid proofs |

---

## 18. Dependencies

```
python >= 3.11
anthropic >= 0.40
pydantic >= 2.0
tomli
jinja2
sqlalchemy
aiosqlite
numpy
scipy
sympy
matplotlib
networkx
plotly
pytest
pytest-asyncio
ruff
elan          # Lean
lean4 >= 4.14 # Lean
mathlib4      # Lean
```

---

*v4 — invariants locked. Implementation-ready.*
