# evoforge: LLM-Guided Evolution over Formally-Grounded Symbolic Expressions

> *A unified framework for evolving verified symbolic artifacts — with theorem proofs and turbulence closures as first-class backends.*

**v2 — revised with architectural hardening**

---

## 1. Thesis

Most LLM + evolutionary algorithm work operates in domains where fitness is noisy, approximate, or expensive to evaluate. This project targets a different regime: **domains where a formal system provides a deterministic, authoritative fitness signal**. A type checker either accepts a proof or it doesn't. A PDE solver either converges to match benchmark data or it doesn't.

This constraint is a feature. It eliminates reward hacking, makes evolution reproducible, and lets the LLM focus on **semantically informed mutation** rather than also serving as a noisy fitness proxy.

The framework, **evoforge**, implements a generic LLM-guided evolutionary engine with pluggable backends. Two backends ship initially:

| Backend | Representation | Fitness Oracle | LLM Role |
|---------|---------------|----------------|----------|
| **lean** | Lean 4 tactic sequences | Lean server (incremental) | Proposes proof strategies from error diagnostics + goal state |
| **cfd** | Symbolic closure expressions (SymPy AST) | 1D sediment-transport solver vs. benchmark data | Proposes physically-motivated functional forms |

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                       evoforge core                          │
│                                                              │
│  ┌───────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Population │  │  Selection   │  │  Archive               │ │
│  │  Manager   │  │  (tourney,   │  │  (eval cache, lineage, │ │
│  │            │  │   lexicase,  │  │   dedup, neighbors)    │ │
│  │            │  │   MAP-Elites)│  │                        │ │
│  └─────┬─────┘  └──────┬───────┘  └──────────┬─────────────┘ │
│        │               │                     │               │
│  ┌─────▼───────────────▼─────────────────────▼────────────┐  │
│  │                 Evolution Loop                          │  │
│  │  for gen in 1..max_generations:                         │  │
│  │    parents = select(population)                         │  │
│  │    offspring_genomes = mutate(parents)  ← LLM + cheap   │  │
│  │    offspring_irs = [backend.parse(g) for g in genomes]  │  │
│  │    # invalid parse → low fitness, no retry              │  │
│  │    fitness = eval_with_cache(offspring_irs)  ← backend  │  │
│  │    population = survive(population, offspring)          │  │
│  │    archive.update(offspring)                            │  │
│  │    if gen % reflection_interval == 0:                   │  │
│  │      guidance = reflect(population) → structured        │  │
│  └────────────────────────────────────────────────────────┘  │
│        │                    │                                │
│  ┌─────▼──────┐      ┌─────▼──────┐     ┌────────────────┐  │
│  │ Mutation    │      │  Backend   │     │  Async Eval    │  │
│  │ Ensemble   │      │  Interface │     │  Queue         │  │
│  │ (LLM+cheap)│      │ (with IR)  │     │  (process pool)│  │
│  └────────────┘      └─────┬──────┘     └────────────────┘  │
└────────────────────────────┼─────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │   lean     │ │    cfd    │ │  (future)  │
        │  backend   │ │  backend  │ │  backends  │
        └───────────┘ └───────────┘ └───────────┘
```

---

## 3. Core Data Structures (`evoforge/core/`)

### 3.1 Structured Fitness

Fitness is **not a float**. This is a critical early decision that prevents rewriting selection logic when the CFD backend arrives.

```python
@dataclass
class Fitness:
    """Structured fitness with multi-signal support."""
    primary: float                          # scalar for default selection
    auxiliary: dict[str, float] = field(default_factory=dict)
                                            # e.g. {"goals_remaining": 3, "complexity": 12}
    constraints: dict[str, bool] = field(default_factory=dict)
                                            # e.g. {"dimensional_consistency": True, "realizable": False}
    feasible: bool = True                   # all constraints satisfied?

    def dominates(self, other: "Fitness") -> bool:
        """Pareto dominance for multi-objective selection."""
        dominated_in_one = False
        for key in self.auxiliary:
            if key in other.auxiliary:
                if self.auxiliary[key] < other.auxiliary[key]:
                    return False
                if self.auxiliary[key] > other.auxiliary[key]:
                    dominated_in_one = True
        return dominated_in_one and self.primary >= other.primary
```

Selection strategies operate on `Fitness`:

- **Tournament (default):** compares `primary`
- **Pareto / NSGA-II:** uses `dominates()` for non-dominated sorting (CFD: error vs. complexity)
- **Lexicase:** iterates over `auxiliary` keys in random order (Lean: per-goal scoring)
- **Feasibility-first:** infeasible individuals always lose to feasible ones

### 3.2 Individual

```python
@dataclass
class Individual:
    genome: str                             # serialized representation
    ir: Any | None = None                   # parsed intermediate representation (backend-specific)
    fitness: Fitness | None = None          # None = unevaluated
    metadata: dict = field(default_factory=dict)
                                            # backend-specific diagnostics
    lineage: list[str] = field(default_factory=list)
    generation: int = 0
    id: str = field(default_factory=lambda: str(uuid4()))
    behavior_descriptor: tuple | None = None
    mutation_source: str = ""               # "llm_mutate", "llm_crossover", "splice", "perturb", etc.
```

The `mutation_source` field tracks which operator produced each individual — essential for the ablation study and for understanding which operators are productive.

### 3.3 Intermediate Representation (IR) Layer

Every backend defines a typed IR that sits between raw genome strings and evaluation. This is the single most important architectural addition over v1.

```python
class Backend(ABC):
    """Abstract interface with mandatory IR layer."""

    # --- IR pipeline (new) ---
    @abstractmethod
    def parse(self, genome: str) -> IR | None:
        """Parse genome string into typed IR. Returns None on failure.
        Must be deterministic: same genome → same IR."""
        ...

    @abstractmethod
    def serialize(self, ir: IR) -> str:
        """Serialize IR back to genome string. Inverse of parse.
        Used for canonicalization: serialize(parse(g)) is the canonical form."""
        ...

    @abstractmethod
    def canonicalize(self, genome: str) -> str | None:
        """Round-trip through IR for deduplication.
        Returns None if genome is unparseable."""
        ir = self.parse(genome)
        return self.serialize(ir) if ir is not None else None
```

**Why this matters:**

- **Deduplication:** `canonicalize("1+x")` and `canonicalize("x+1")` produce the same string → no redundant evaluation
- **Cache key:** hash the canonical form, not the raw genome
- **Cheap operators:** AST-level mutation works on the IR, not the string
- **Validation:** `parse` returning `None` is the validity check — no separate `validate_genome`

**Backend-specific IRs:**

| Backend | IR Type | Example |
|---------|---------|---------|
| lean | `list[TacticStep]` (preserving sequential structure) | `[Intro("ε", "hε"), Apply("levy_continuity_lemma"), Exact(...)]` |
| cfd | `sympy.Expr` (parsed directly into SymPy) | `exp(-α * Ri_g) * (1 - exp(-y_plus / A))` |

---

## 4. Evolution Loop (`evoforge/core/engine.py`)

### 4.1 Main Loop

```python
class Engine:
    def __init__(self, config: Config, backend: Backend, archive: Archive):
        self.config = config
        self.backend = backend
        self.archive = archive
        self.population = PopulationManager(config.population)
        self.mutator = MutationEnsemble(config.mutation, backend)
        self.evaluator = AsyncEvaluator(backend, archive, config.parallelism)
        self.guidance: Reflection | None = None

    def run(self):
        # Seed
        seeds = self.backend.seed_population(self.config.population.size)
        self.population.initialize(seeds)
        self.evaluator.evaluate_batch(self.population.all())

        for gen in range(self.config.run.max_generations):
            # Select parents
            parents = self.population.select_parents(
                k=self.config.selection.offspring_count,
                method=self.config.selection.method,
            )

            # Generate offspring via mutation ensemble
            offspring = self.mutator.generate(
                parents=parents,
                guidance=self.guidance,
                generation=gen,
            )

            # Parse → IR (invalid = None, gets low fitness)
            for ind in offspring:
                ind.ir = self.backend.parse(ind.genome)
                if ind.ir is not None:
                    ind.genome = self.backend.serialize(ind.ir)  # canonicalize

            # Evaluate (with cache, async)
            self.evaluator.evaluate_batch(offspring)

            # Survive
            self.population.replace(offspring)
            self.archive.record_generation(gen, offspring)

            # Periodic reflection
            if gen % self.config.reflection.interval == 0:
                self.guidance = self.mutator.reflect(
                    self.population.all(), self.archive
                )

            # Logging
            self._log_generation(gen)
```

### 4.2 Invalid Genome Handling (No Retries)

This is a deliberate design choice. When the LLM produces an unparseable genome:

```python
# In evaluator:
def evaluate(self, individual: Individual) -> Fitness:
    if individual.ir is None:
        return Fitness(
            primary=0.05,  # not zero — still better than nothing
            auxiliary={"parse_success": 0.0},
            constraints={"parseable": False},
            feasible=False,
        )
    # ... proceed with backend evaluation
```

**Why no retries:**

- Retries waste LLM budget (the most expensive resource)
- Evolution naturally selects away from operators/parents that produce invalid output
- Invalid genomes still carry information: they tell us what the LLM *tried* to do
- The low-but-nonzero fitness means they can still participate as parents if nothing better exists (early generations)

### 4.3 Evaluation Cache

The cache sits in front of `backend.evaluate` and is the **first thing checked**:

```python
class EvaluationCache:
    """Content-addressed cache keyed on canonical genome hash."""

    def __init__(self, archive: Archive):
        self.archive = archive

    def get(self, canonical_genome: str) -> Fitness | None:
        genome_hash = hashlib.sha256(canonical_genome.encode()).hexdigest()
        return self.archive.lookup_fitness(genome_hash)

    def put(self, canonical_genome: str, fitness: Fitness, diagnostics: dict):
        genome_hash = hashlib.sha256(canonical_genome.encode()).hexdigest()
        self.archive.store_evaluation(genome_hash, fitness, diagnostics)
```

Deduplication happens at **two stages:**

1. **Pre-mutation:** check if canonical genome already exists in archive before spending an LLM call on crossover with it
2. **Post-mutation, pre-evaluation:** check if the offspring genome (canonicalized) has already been evaluated

### 4.4 Async Evaluation Queue

```python
class AsyncEvaluator:
    """Process-pool evaluator with concurrency limits."""

    def __init__(self, backend: Backend, archive: Archive, config: ParallelismConfig):
        self.backend = backend
        self.cache = EvaluationCache(archive)
        self.max_workers = config.max_workers  # default: cpu_count for CFD, 4 for Lean
        self.eval_timeout = config.eval_timeout

    async def evaluate_batch(self, individuals: list[Individual]):
        """Evaluate a batch, hitting cache first, then parallel eval."""
        to_evaluate = []
        for ind in individuals:
            if ind.ir is None:
                ind.fitness = self._invalid_fitness()
                continue
            cached = self.cache.get(ind.genome)
            if cached is not None:
                ind.fitness = cached
                continue
            to_evaluate.append(ind)

        # Parallel evaluation of cache misses
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._eval_one, ind): ind for ind in to_evaluate
            }
            for future in as_completed(futures, timeout=self.eval_timeout):
                ind = futures[future]
                ind.fitness, ind.metadata = future.result()
                self.cache.put(ind.genome, ind.fitness, ind.metadata)
```

---

## 5. Mutation Ensemble (`evoforge/core/mutation.py`)

The LLM is **one operator among several**, not the only one. This is critical for cost control, early-generation stability, and ablation.

```python
class MutationEnsemble:
    """
    Weighted ensemble of mutation operators.
    Operator selection can be:
    - fixed weights (config)
    - adaptive (proportional to recent fitness improvement from each operator)
    """

    def __init__(self, config: MutationConfig, backend: Backend):
        self.llm_mutator = LLMMutator(config.llm, backend)
        self.cheap_operators = backend.cheap_operators()
        self.operator_weights = config.initial_weights
        self.adaptive = config.adaptive_weights
        self.operator_stats: dict[str, OperatorStats] = {}

    def generate(self, parents, guidance, generation) -> list[Individual]:
        offspring = []
        for parent_group in self._pair_parents(parents):
            # Select operator by weight
            operator = self._select_operator()
            if operator == "llm_mutate":
                genome = self.llm_mutator.mutate(parent_group[0], guidance)
                source = "llm_mutate"
            elif operator == "llm_crossover":
                genome = self.llm_mutator.crossover(parent_group[0], parent_group[1], guidance)
                source = "llm_crossover"
            else:
                # Cheap operator (backend-specific)
                genome = self.cheap_operators[operator](parent_group[0])
                source = operator
            offspring.append(Individual(genome=genome, mutation_source=source, ...))
        return offspring

    def _update_weights(self):
        """Adaptive: increase weight of operators that produce fit offspring."""
        if not self.adaptive:
            return
        # ... proportional to mean fitness improvement per operator
```

### 5.1 Cheap Operators (Backend-Specific)

**Lean backend:**

| Operator | Description |
|----------|-------------|
| `splice_prefix` | Keep first N successful tactics, replace remainder |
| `truncate_suffix` | Drop last K tactics (often the failing ones) |
| `reorder_tactics` | Swap order of independent tactic steps |
| `tactic_swap` | Replace one tactic with a random common tactic (`simp`, `ring`, `omega`, `norm_num`) |

**CFD backend:**

| Operator | Description |
|----------|-------------|
| `subtree_mutation` | Replace a random subtree of the SymPy AST with a random expression |
| `constant_perturb` | Jitter numeric constants by ±10–50% |
| `operator_swap` | Replace `exp` with `tanh`, `+` with `*`, etc. |
| `term_deletion` | Remove one additive term |
| `variable_swap` | Replace one physical variable with another |

**Why these matter:**

- 10–100x cheaper than an LLM call
- Provide meaningful baselines for the ablation study ("is the LLM actually helping?")
- Stabilize early generations when the LLM hasn't seen enough context to be useful
- Default config: 50% LLM, 50% cheap; tunable per run

### 5.2 Structured Reflection

Reflection is no longer a free-text string. It produces structured guidance that directly influences mutation prompts:

```python
@dataclass
class Reflection:
    """Structured output from population-level LLM analysis."""
    strategies_to_try: list[str]        # "try induction on the measure"
    strategies_to_avoid: list[str]      # "Finset.sum approach hits universe issues"
    useful_primitives: list[str]        # lemma names, function forms, etc.
    population_diagnosis: str           # "population is converging prematurely on..."
    suggested_temperature: float | None # LLM's self-assessment of exploration need
```

The reflection prompt includes:

- Top 5 individuals (genome + fitness + diagnostics summary)
- Bottom 5 individuals (what's failing and why)
- Diversity metrics (how spread out is the population?)
- Operator performance stats (which operators are producing improvements?)

Reflection output is parsed (with schema enforcement) and injected into subsequent mutation prompts as a `STRATEGIC GUIDANCE` section.

**Schedule:** every `reflection_interval` generations (default: 10). Configurable. Uses stronger model (Sonnet/Opus) even if routine mutations use Haiku.

### 5.3 LLM Mutation Interface

```python
class LLMMutator:
    def __init__(self, config: LLMConfig, backend: Backend):
        self.client = anthropic.Anthropic()
        self.model = config.model
        self.backend = backend
        self.config = config

    def mutate(self, parent: Individual, guidance: Reflection | None) -> str:
        prompt = self.backend.format_mutation_prompt(
            parent=parent,
            diagnostics_summary=self.backend.summarize_diagnostics(parent.metadata),
            guidance=guidance,
        )
        response = self.client.messages.create(
            model=self.model,
            system=self.backend.system_prompt(),
            messages=[{"role": "user", "content": prompt}],
            temperature=self._current_temperature(),
        )
        return self.backend.extract_genome(response.content[0].text)
```

**Critical: prompt compression.** Backends must implement:

```python
class Backend(ABC):
    @abstractmethod
    def summarize_diagnostics(self, diagnostics: dict) -> str:
        """Compress raw diagnostics to essential signal for LLM.
        Must fit within ~500 tokens."""
        ...

    @abstractmethod
    def summarize_history(self, similar_failures: list[Individual]) -> str:
        """Compress history of similar attempts to ~300 tokens."""
        ...
```

For Lean, this means reducing error logs to: first error message, current goal state, list of missing identifiers. Not the full `stderr`.

For CFD: divergence time, worst-case identifier, error magnitude at key stations. Not full convergence history.

---

## 6. Archive (`evoforge/core/archive.py`)

The archive is elevated from a logging facility to the **central data store** that the entire system queries.

```python
class Archive:
    """SQLite-backed archive: evaluation cache, lineage, replay, neighbor search."""

    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}")
        self._init_tables()

    # --- Evaluation cache (§4.3) ---
    def lookup_fitness(self, genome_hash: str) -> Fitness | None: ...
    def store_evaluation(self, genome_hash: str, fitness: Fitness, diagnostics: dict): ...

    # --- Lineage ---
    def record_generation(self, gen: int, individuals: list[Individual]): ...
    def lineage_tree(self, individual_id: str) -> nx.DiGraph: ...

    # --- Deduplication ---
    def has_been_tried(self, canonical_genome: str) -> bool: ...

    # --- Neighbor queries (for mutation context) ---
    def nearest_neighbors(self, genome: str, k: int = 5) -> list[Individual]:
        """Find k most similar previously-evaluated individuals.
        Similarity = edit distance on canonical genome for Lean,
        AST tree edit distance for CFD."""
        ...

    def similar_failures(self, diagnostics: dict, k: int = 5) -> list[Individual]:
        """Find individuals that failed in similar ways.
        Feeds 'previously tried approaches' in mutation prompt."""
        ...

    # --- Analytics ---
    def hall_of_fame(self, n: int = 10) -> list[Individual]: ...
    def operator_stats(self) -> dict[str, OperatorStats]: ...
    def diversity_over_time(self) -> list[float]: ...
```

---

## 7. Backend Interface (Complete)

```python
class Backend(ABC):
    """Complete abstract interface with IR, compression, and cheap operators."""

    # --- IR pipeline ---
    @abstractmethod
    def parse(self, genome: str) -> IR | None: ...
    @abstractmethod
    def serialize(self, ir: IR) -> str: ...
    def canonicalize(self, genome: str) -> str | None:
        ir = self.parse(genome)
        return self.serialize(ir) if ir is not None else None

    # --- Evaluation ---
    @abstractmethod
    def evaluate(self, ir: IR) -> tuple[Fitness, dict]:
        """Evaluate a PARSED IR (not raw genome). Returns (structured fitness, diagnostics)."""
        ...

    # --- Seed ---
    @abstractmethod
    def seed_population(self, n: int) -> list[str]: ...

    # --- LLM integration ---
    @abstractmethod
    def system_prompt(self) -> str: ...
    @abstractmethod
    def format_mutation_prompt(self, parent: Individual,
                                diagnostics_summary: str,
                                guidance: Reflection | None) -> str: ...
    @abstractmethod
    def format_crossover_prompt(self, a: Individual, b: Individual,
                                 guidance: Reflection | None) -> str: ...
    @abstractmethod
    def format_reflection_prompt(self, population: list[Individual],
                                  archive: Archive) -> str: ...

    # --- Compression (for prompt budget) ---
    @abstractmethod
    def summarize_diagnostics(self, diagnostics: dict) -> str: ...
    @abstractmethod
    def summarize_history(self, similar_failures: list[Individual]) -> str: ...

    # --- Cheap operators ---
    @abstractmethod
    def cheap_operators(self) -> dict[str, Callable[[Individual], str]]:
        """Return named cheap mutation operators for the ensemble."""
        ...

    # --- Genome extraction ---
    @abstractmethod
    def extract_genome(self, llm_text: str) -> str:
        """Extract genome from raw LLM response text.
        Should use strict parsing (fenced code blocks, JSON schema)
        with regex fallback, then post-parse repair."""
        ...

    # --- MAP-Elites ---
    @abstractmethod
    def behavior_descriptor(self, ir: IR, fitness: Fitness, diagnostics: dict) -> tuple: ...
    @abstractmethod
    def behavior_space(self) -> BehaviorSpaceConfig:
        """Define bin edges, normalization, and distance metric."""
        ...
```

---

## 8. Backend: Lean Proof Evolution (`evoforge/backends/lean/`)

### 8.1 Intermediate Representation

```python
@dataclass
class TacticStep:
    """One tactic invocation with its arguments."""
    tactic: str                 # "intro", "apply", "exact", "simp", "cases", ...
    args: list[str]             # arguments to the tactic
    raw: str                    # original text (for roundtrip fidelity)

# IR = list[TacticStep]
LeanIR = list[TacticStep]
```

Parsing uses a lightweight tactic tokenizer (not a full Lean parser — that's overkill). The key structural property: **tactics are an ordered sequence with implicit state transitions between them**. This enables prefix reuse, partial evaluation, and sequence-aware mutation.

**Canonicalization:** normalize whitespace, normalize `by` blocks, sort `simp` lemma lists alphabetically.

### 8.2 Evaluation via Lean Server (Incremental)

**`lake build` for every evaluation is too slow.** Instead:

```python
class LeanEvaluator:
    """Uses Lean server protocol for incremental checking."""

    def __init__(self, project_dir: str, target_file: str):
        self.project_dir = project_dir
        self.target_file = target_file
        # Pre-build everything EXCEPT the target file
        self._warm_cache()

    def _warm_cache(self):
        """Build all dependencies once. Only the evolved proof file changes."""
        subprocess.run(["lake", "build", "+EvoForge.Dependencies"], cwd=self.project_dir)

    def evaluate(self, ir: LeanIR) -> tuple[Fitness, dict]:
        # Write proof to isolated file (only this file recompiles)
        proof_text = self._render_proof(ir)
        self._write_evolved_file(proof_text)

        # Use lean --run or lake env lean for single-file check
        result = subprocess.run(
            ["lake", "env", "lean", self.target_file],
            capture_output=True, timeout=60, cwd=self.project_dir,
        )

        diagnostics = self._parse_output(result)
        fitness = self._score(diagnostics)
        return fitness, diagnostics
```

**Key optimization:** the Lean environment (mathlib, all imports) is compiled once. Only the single evolved file is re-checked each evaluation. This reduces evaluation from minutes to seconds.

**Future optimization path:** use the Lean Language Server Protocol (LSP) for even finer-grained incremental checking — send individual tactic steps and get goal states back.

### 8.3 Fitness Scoring (with goal-state extraction)

```python
def _score(self, diagnostics: dict) -> Fitness:
    errors = diagnostics.get("errors", [])
    goals = diagnostics.get("goals", [])

    if diagnostics["success"]:
        return Fitness(primary=1.0, feasible=True)

    auxiliary = {}

    # Parse success (not syntax error)
    auxiliary["parses"] = 1.0 if not diagnostics.get("syntax_error") else 0.0

    # Goal state analysis (the key gradient signal)
    if goals:
        n_goals = len(goals)
        auxiliary["goals_remaining"] = n_goals
        # Fewer goals = more progress
        auxiliary["goal_progress"] = 1.0 / (1.0 + n_goals)
        # Goal shape similarity: do remaining goals match known lemma types?
        auxiliary["goal_familiarity"] = self._goal_shape_similarity(goals)
        # Metavariable count (fewer = more constrained = closer to done)
        auxiliary["metavar_count"] = sum(g.metavar_count for g in goals)

    # Error classification
    error_types = [e.kind for e in errors]
    # Type errors are "closer" than unknown identifiers
    auxiliary["type_errors"] = sum(1 for e in error_types if e == "type_mismatch")
    auxiliary["unknown_ids"] = sum(1 for e in error_types if e == "unknown_identifier")

    # Composite primary score
    primary = (
        0.1 * auxiliary.get("parses", 0)
        + 0.4 * auxiliary.get("goal_progress", 0)
        + 0.2 * auxiliary.get("goal_familiarity", 0)
        + 0.1 * (1.0 if auxiliary.get("type_errors", 0) > auxiliary.get("unknown_ids", 0) else 0.0)
    )

    return Fitness(
        primary=min(primary, 0.99),  # reserve 1.0 for success
        auxiliary=auxiliary,
        constraints={"compiles": diagnostics["success"]},
        feasible=diagnostics["success"],
    )
```

**Goal-state extraction** is the critical addition over v1. Instead of just counting errors, we parse the Lean goal state to understand *what remains to be proved*. This gives a much smoother fitness landscape.

### 8.4 LLM Mutation Prompt (with compression + guidance)

```
You are an expert Lean 4 mathematician working with mathlib.

THEOREM TO PROVE:
{theorem_statement}

CURRENT PROOF ATTEMPT (fitness: {fitness.primary:.2f}):
```lean
{genome}
```

GOAL STATE AFTER PARTIAL PROGRESS:
{diagnostics_summary}
  ← compressed: first error, remaining goals, missing identifiers only

SIMILAR FAILED ATTEMPTS (avoid these):
{history_summary}
  ← compressed: 3-5 attempts, one line each

{%- if guidance %}
STRATEGIC GUIDANCE (from population analysis):
  Try: {guidance.strategies_to_try | join(", ")}
  Avoid: {guidance.strategies_to_avoid | join(", ")}
  Useful lemmas: {guidance.useful_primitives | join(", ")}
{%- endif %}

Propose a MODIFIED proof. Return ONLY the tactic block (after `by`),
in a ```lean fence. No explanation needed.
```

### 8.5 Behavior Space (MAP-Elites)

```python
def behavior_space(self) -> BehaviorSpaceConfig:
    return BehaviorSpaceConfig(
        dimensions=[
            BehaviorDimension(
                name="strategy_class",
                type="categorical",
                bins=["direct", "induction", "cases", "contradiction", "calc"],
                extractor=lambda ir, f, d: classify_proof_strategy(ir),
            ),
            BehaviorDimension(
                name="proof_depth",
                type="numeric",
                bin_edges=[1, 3, 5, 10, 20, 50],
                extractor=lambda ir, f, d: len(ir),
            ),
        ],
        distance_metric="hamming",  # on discretized descriptors
    )
```

---

## 9. Backend: Turbulence Closure Evolution (`evoforge/backends/cfd/`)

### 9.1 Intermediate Representation: SymPy Direct

The DSL **is** SymPy. No separate DSL that must stay in sync. Genomes are SymPy expression strings; the IR is a `sympy.Expr`.

```python
from sympy import symbols, sympify, exp, tanh, sqrt, log, Abs
from sympy.core.expr import Expr

# Physical variables (defined once)
y_plus, Ri_g, Ri_f, Re_tau, C_s, omega_t, phi_s = symbols(
    "y_plus Ri_g Ri_f Re_tau C_s omega_t phi_s", positive=True, real=True
)

# Tunable constants
alpha, A, beta = symbols("alpha A beta", positive=True, real=True)

ALLOWED_FUNCTIONS = {exp, tanh, sqrt, log, Abs}
ALLOWED_SYMBOLS = {y_plus, Ri_g, Ri_f, Re_tau, C_s, omega_t, phi_s, alpha, A, beta}

class CFDBackend(Backend):
    def parse(self, genome: str) -> Expr | None:
        try:
            expr = sympify(genome, locals={s.name: s for s in ALLOWED_SYMBOLS})
            # Validate: only allowed functions and symbols
            if not self._uses_only_allowed(expr):
                return None
            return expr
        except (SympifyError, SyntaxError, TypeError):
            return None

    def serialize(self, ir: Expr) -> str:
        return str(ir.simplify())  # canonical form via SymPy simplification
```

**Why this eliminates the DSL mismatch risk:** there is no DSL. The genome *is* a SymPy string. Parsing goes through `sympify`. Canonicalization goes through `simplify`. No divergence possible.

### 9.2 Fitness with Constraint Penalties

Constraints are **soft penalties in fitness**, not just diagnostics:

```python
def evaluate(self, ir: Expr) -> tuple[Fitness, dict]:
    diagnostics = {}
    constraints = {}
    penalty = 1.0

    # Dimensional consistency
    dim_ok = self._check_dimensions(ir)
    constraints["dimensional_consistency"] = dim_ok
    if not dim_ok:
        penalty *= 0.1

    # Asymptotic behavior
    unstrat_ok = self._approaches_van_driest(ir)
    constraints["unstratified_limit"] = unstrat_ok
    if not unstrat_ok:
        penalty *= 0.5

    suppress_ok = self._suppresses_at_high_Ri(ir)
    constraints["stratification_suppression"] = suppress_ok
    if not suppress_ok:
        penalty *= 0.5

    # Realizability (non-negative)
    realizable = not self._can_go_negative(ir)
    constraints["realizable"] = realizable
    if not realizable:
        penalty *= 0.3

    # Run solver against benchmark cases
    case_results = []
    for case in self.benchmark_cases:
        try:
            result = self.solver.run(closure_expr=ir, **case.params)
            err = self._l2_error(result, case.reference)
            case_results.append({
                "case": case.name, "error": err, "converged": True
            })
        except SolverDivergenceError as e:
            case_results.append({
                "case": case.name,
                "error": 1.0,
                "converged": False,
                "divergence_time": e.time,         # ← structured, not discarded
                "instability_type": e.instability,  # ← e.g. "negative viscosity", "CFL violation"
            })

    diagnostics["case_results"] = case_results
    errors = [c["error"] for c in case_results]
    mean_error = np.mean(errors)

    raw_fitness = 1.0 / (1.0 + mean_error)
    complexity = self._ast_node_count(ir)

    return Fitness(
        primary=raw_fitness * penalty,
        auxiliary={
            "raw_accuracy": raw_fitness,
            "complexity": complexity,
            "constraint_penalty": penalty,
            "converged_fraction": sum(1 for c in case_results if c["converged"]) / len(case_results),
        },
        constraints=constraints,
        feasible=all(constraints.values()),
    ), diagnostics
```

### 9.3 Solver Divergence: Structured Diagnostics

```python
class SolverDivergenceError(Exception):
    def __init__(self, time: float, instability: str, location: float, message: str):
        self.time = time              # simulation time at divergence
        self.instability = instability # "negative_viscosity", "cfl_violation", "nan_propagation"
        self.location = location       # y/δ at divergence
        self.message = message
```

This feeds directly into the LLM mutation prompt:

```
SOLVER DIAGNOSTICS:
  Case 03 (high concentration): DIVERGED at t=3.2s
    Instability: negative viscosity at y/δ = 0.12
    Likely cause: closure goes negative when Ri_g > 0.6
```

Much more actionable for the LLM than `error: 1.0`.

### 9.4 Behavior Space

```python
def behavior_space(self) -> BehaviorSpaceConfig:
    return BehaviorSpaceConfig(
        dimensions=[
            BehaviorDimension(
                name="dominant_variable",
                type="categorical",
                bins=["y_plus", "Ri_g", "Re_tau", "C_s", "mixed"],
                extractor=lambda ir, f, d: self._dominant_sensitivity(ir),
            ),
            BehaviorDimension(
                name="complexity",
                type="numeric",
                bin_edges=[1, 3, 6, 10, 15, 25],
                extractor=lambda ir, f, d: self._ast_node_count(ir),
            ),
        ],
        distance_metric="hamming",
    )
```

---

## 10. Configuration

```toml
[run]
name = "levy_proof_search_001"
backend = "lean"
max_generations = 200
seed = 42

[population]
size = 30
replacement = "generational"    # or "steady_state"
elitism = 2

[selection]
method = "tournament"           # "tournament", "lexicase", "nsga2", "map_elites"
tournament_size = 3
offspring_count = 20

[mutation]
initial_weights = { llm_mutate = 0.3, llm_crossover = 0.2, splice_prefix = 0.15, truncate_suffix = 0.15, tactic_swap = 0.1, reorder = 0.1 }
adaptive_weights = true         # adjust based on operator success rates

[llm]
model = "claude-sonnet-4-5-20250929"
reflection_model = "claude-sonnet-4-5-20250929"   # can be stronger for reflection
temperature_start = 1.0
temperature_end = 0.3
temperature_schedule = "linear"
max_tokens = 2048

[reflection]
interval = 10                   # every N generations
include_top_k = 5
include_bottom_k = 5

[parallelism]
max_workers = 4
eval_timeout = 120

[diversity]
strategy = "map_elites"         # or "none"

# --- Ablation flags ---
[ablation]
disable_llm = false             # all mutations become cheap operators
disable_diagnostics = false     # LLM sees only fitness score, not error messages
disable_reflection = false      # no population-level guidance
disable_cheap_operators = false # all mutations go through LLM

# --- Backend-specific ---
[lean]
project_dir = "./lean_project"
target_theorem = "levy_characteristic_continuous"
target_file = "EvoForge/Evolved.lean"
mathlib_version = "v4.15.0"

[cfd]
solver_module = "evoforge.backends.cfd.solver"
benchmark_dir = "./benchmarks"
complexity_penalty = 0.01
max_ast_depth = 8
```

---

## 11. Repo Structure

```
evoforge/
├── README.md
├── DESIGN.md                     ← this document
├── pyproject.toml
├── evoforge/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py              # Fitness, Individual, Reflection, BehaviorSpaceConfig
│   │   ├── population.py         # PopulationManager
│   │   ├── selection.py          # tournament, lexicase, NSGA-II, MAP-Elites
│   │   ├── mutation.py           # MutationEnsemble
│   │   ├── archive.py            # SQLite archive (cache + lineage + neighbors)
│   │   ├── engine.py             # main evolution loop
│   │   ├── evaluator.py          # AsyncEvaluator with cache
│   │   └── config.py             # TOML config parsing via Pydantic
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py             # LLM API wrapper (Anthropic, OpenAI, litellm)
│   │   ├── mutator.py            # LLMMutator class
│   │   └── templates/            # Jinja2 prompt templates
│   │       ├── mutate.j2
│   │       ├── crossover.j2
│   │       └── reflect.j2
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py               # Backend ABC (complete interface from §7)
│   │   ├── lean/
│   │   │   ├── __init__.py
│   │   │   ├── backend.py        # LeanBackend
│   │   │   ├── ir.py             # TacticStep, LeanIR, parser
│   │   │   ├── evaluator.py      # Lean server wrapper, goal-state extraction
│   │   │   ├── scorer.py         # partial proof scoring with goal analysis
│   │   │   ├── cheap_ops.py      # splice, truncate, reorder, swap
│   │   │   └── templates/        # Lean-specific prompt overrides
│   │   └── cfd/
│   │       ├── __init__.py
│   │       ├── backend.py        # CFDBackend
│   │       ├── ir.py             # SymPy IR, allowed symbols, validation
│   │       ├── constraints.py    # dimensional analysis, realizability, limits
│   │       ├── solver.py         # wrapper around 1D transport solver
│   │       ├── cheap_ops.py      # subtree, perturb, swap, delete
│   │       └── templates/        # CFD-specific prompt overrides
│   └── viz/
│       ├── __init__.py
│       ├── genealogy.py          # lineage tree visualization
│       ├── fitness_plots.py      # convergence curves, operator comparison
│       ├── map_elites.py         # behavior space heatmaps
│       └── dashboard.py          # live monitoring (optional, Streamlit/Panel)
├── configs/
│   ├── lean_levy.toml
│   ├── lean_levy_ablation_no_llm.toml
│   ├── lean_levy_ablation_no_diag.toml
│   ├── cfd_oscillatory.toml
│   └── cfd_steady.toml
├── lean_project/
│   ├── lakefile.lean
│   ├── lean-toolchain
│   └── EvoForge/
│       ├── Dependencies.lean     # all imports (pre-compiled)
│       ├── Target.lean           # theorem statements
│       └── Evolved.lean          # ← written by evoforge during runs
├── benchmarks/
│   ├── case_01_clear_water/
│   ├── case_02_low_concentration/
│   └── case_03_high_concentration_laminarization/
├── tests/
│   ├── test_core/
│   │   ├── test_fitness.py
│   │   ├── test_population.py
│   │   ├── test_selection.py
│   │   ├── test_archive.py
│   │   └── test_engine.py
│   ├── test_lean/
│   │   ├── test_ir.py
│   │   ├── test_scorer.py
│   │   └── test_cheap_ops.py
│   └── test_cfd/
│       ├── test_ir.py
│       ├── test_constraints.py
│       └── test_cheap_ops.py
└── scripts/
    ├── run.py                    # CLI: python -m scripts.run --config configs/lean_levy.toml
    ├── analyze.py                # post-hoc analysis
    ├── ablation_sweep.py         # run all ablation configs
    └── seed_from_mathlib.py      # mine proof templates
```

---

## 12. MVP Milestones (Revised)

### Phase 1: Core + Lean Backend (Weeks 1–3)

| Week | Deliverable |
|------|-------------|
| 1 | `types.py` (Fitness, Individual, Reflection), `archive.py` (SQLite with eval cache + dedup), `population.py`, `selection.py` (tournament only). Full unit tests. |
| 2 | Lean IR (`ir.py`), tactic parser, cheap operators, `evaluator.py` (single-file `lake env lean`), scorer with goal-state extraction. Integration test: parse → evaluate → score a known proof. |
| 3 | `LLMMutator`, `MutationEnsemble`, `Engine`. First real run targeting a trivial lemma. Verify: cache hit rate, operator stats logging, fitness curve. |

**Success criterion:** valid proof found for ≥1 non-trivial lemma within 200 generations. LLM mutation demonstrably outperforms cheap-only baseline (measured by generations to solution).

### Phase 2: CFD Backend (Weeks 4–6)

| Week | Deliverable |
|------|-------------|
| 4 | SymPy IR, constraint checker (dimensions, limits, realizability), cheap operators. Unit tests with known closure expressions. |
| 5 | Solver wrapper, benchmark cases, `CFDBackend` integration. First run: seed with known closures, evolve. |
| 6 | Multi-objective (NSGA-II selection on accuracy vs. complexity). Pareto front visualization. Compare evolved closures against van Driest and Munk-Anderson on laminarization cases. |

**Success criterion:** evolution rediscovers van Driest-class damping from random seeds within 100 generations. At least one evolved expression matches or exceeds van Driest on the stratified benchmark cases.

### Phase 3: Quality-Diversity + Analysis (Weeks 7–8)

| Week | Deliverable |
|------|-------------|
| 7 | MAP-Elites integration (behavior space configs for both backends). Behavior space heatmap visualization. |
| 8 | Full ablation sweep: ±LLM, ±diagnostics, ±reflection, ±cheap operators. Statistical analysis (multiple seeds, confidence intervals). |

### Phase 4: Paper & Release (Weeks 9–10)

| Week | Deliverable |
|------|-------------|
| 9 | Write-up: framing, experiments, ablation results, comparison to baselines. |
| 10 | Clean repo, documentation, public release. |

---

## 13. Experimental Design (Ablation Matrix)

Every claim of novelty must be empirically supported. The ablation flags (§10) produce this matrix:

| Experiment | LLM Mutation | Diagnostics in Prompt | Reflection | Cheap Ops | Purpose |
|------------|:---:|:---:|:---:|:---:|---------|
| **Full system** | ✓ | ✓ | ✓ | ✓ | Main result |
| **No LLM** | ✗ | — | — | ✓ | Is LLM mutation better than cheap operators alone? |
| **No diagnostics** | ✓ | ✗ | ✓ | ✓ | Do error messages help the LLM mutate better? |
| **No reflection** | ✓ | ✓ | ✗ | ✓ | Does population-level analysis improve search? |
| **LLM only** | ✓ | ✓ | ✓ | ✗ | Do cheap operators help stabilize / reduce cost? |
| **Random baseline** | ✗ | — | — | random | Pure random search (sanity check) |

Each experiment: 5 seeds, report mean ± std of generations-to-solution (Lean) or best-fitness-at-generation-N (CFD).

---

## 14. Technical Risks & Mitigations (Revised)

### Risk: LLM API costs
**Budget model:** ~30 individuals × 50% LLM rate × $0.003/call (Haiku) = ~$0.05/generation. 200 generations = ~$10/run. Reflection at Sonnet: +$0.50/run. Total: ~$15/run including ablations. Manageable.
**Mitigation:** Haiku for routine mutations, Sonnet for reflection. Aggressive caching. Adaptive weights shift toward cheap operators when LLM isn't helping.

### Risk: Lean partial scoring doesn't guide evolution
**Mitigation (upgraded):** Goal-state extraction provides much richer signal than error counting. If still insufficient, add LLM-as-judge: ask the LLM to rank two failing proofs by "closeness." This is cheap and calibrated for structured domains. Lexicase selection on individual auxiliary metrics may find signal that scalar fitness misses.

### Risk: CFD solver too slow
**Mitigation:** 1D solver runs in seconds. AsyncEvaluator parallelizes across CPU cores. For longer runs, implement evaluation budgeting: don't evaluate all benchmark cases for every individual — start with easy cases, only run hard cases for individuals that pass easy ones.

### Risk: LLM mutations are no better than random
**Mitigation:** This is the hypothesis under test — the ablation study measures it directly. If LLM mutations underperform, the diagnostics-in-prompt and reflection mechanisms are the levers to pull. Worst case: the paper reports a negative result (still publishable, still valuable).

### Risk: Genome parsing fragility
**Mitigation:** Strict output format enforcement (fenced code blocks only), regex fallback extraction, backend-specific post-parse repair (SymPy normalization for CFD, whitespace normalization for Lean). Invalid genomes get low fitness rather than retries.

---

## 15. What Makes This Novel (Tightened Claims)

Each claim maps to an ablation experiment:

1. **Error-message-informed mutation** → measured by "full system" vs. "no diagnostics"
2. **Structured reflection guiding mutation** → measured by "full system" vs. "no reflection"
3. **LLM mutation vs. cheap operators** → measured by "full system" vs. "no LLM" vs. "LLM only"
4. **Formally-grounded fitness (type checker / PDE solver)** → qualitative distinction from FunSearch/AlphaEvolve (code execution, not formal verification)
5. **Unified framework across domains** → demonstrated by two backends sharing identical core engine
6. **Quality-diversity for formal proofs** → MAP-Elites on Lean proofs is novel; finding multiple distinct valid proofs has mathematical value

---

## 16. Open Questions

- **Multi-model cascade for Lean:** Use Haiku for first-pass mutation, Sonnet to "debug" Haiku's output if it fails to parse? Reduces cost while keeping quality.
- **Island model:** Sub-populations with different LLM temperatures, periodic migration. Natural parallelism. Worth exploring in Phase 3.
- **Meta-evolution of prompts:** Can the system prompt itself evolve alongside the population? Deeply recursive but could improve over time.
- **Lean: tactic mode vs. term mode?** Start with tactic mode (more natural for LLMs, easier to partially score). Add term mode as a second representation in MAP-Elites behavior space.
- **Transfer learning across theorems:** Can insights from proving theorem A help prove theorem B? Archive neighbor queries could enable this.
- **CFD: multi-fidelity evaluation:** Use coarse time steps for initial screening, fine steps only for promising individuals. Standard in surrogate-assisted EA.

---

## 17. Dependencies

```
# Core
python >= 3.11
anthropic >= 0.40
pydantic >= 2.0
tomli
jinja2
sqlalchemy
aiosqlite               # async archive access

# Lean backend
elan                     # Lean version manager
lean4 >= 4.14
mathlib4

# CFD backend
numpy
scipy
sympy                    # IR + dimensional analysis + canonicalization

# Visualization
matplotlib
networkx
plotly

# Dev
pytest
pytest-asyncio
ruff
```

---

*This is a living document. Update as the project evolves.*
