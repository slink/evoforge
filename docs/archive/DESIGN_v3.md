# evoforge: LLM-Guided Evolution over Formally-Grounded Symbolic Expressions

> *A unified framework for evolving verified symbolic artifacts — with theorem proofs and turbulence closures as first-class backends.*

**v3 — execution semantics, stepwise evaluation, operator framework**

---

## 1. Thesis

Most LLM + evolutionary algorithm work operates in domains where fitness is noisy, approximate, or expensive to evaluate. This project targets a different regime: **domains where a formal system provides a deterministic, authoritative fitness signal**. A type checker either accepts a proof or it doesn't. A PDE solver either converges to match benchmark data or it doesn't.

This constraint eliminates reward hacking, makes evolution reproducible, and lets the LLM focus on **semantically informed mutation** rather than serving as a noisy fitness proxy.

**evoforge** implements a generic LLM-guided evolutionary engine with pluggable backends:

| Backend | IR Type | Fitness Oracle | LLM Role |
|---------|---------|----------------|----------|
| **lean** | `list[TacticStep]` | Lean server (stepwise) | Proof strategies from goal state + error diagnostics |
| **cfd** | `sympy.Expr` | 1D sediment-transport solver | Physically-motivated functional forms |

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        evoforge core                             │
│                                                                  │
│  ┌────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ Population  │  │ SelectionStrategy│  │ Archive              │  │
│  │ Manager     │  │ (ABC: scalar,   │  │ (eval cache, prefix  │  │
│  │             │  │  pareto, lexi,  │  │  cache, lineage,     │  │
│  │             │  │  map_elites)    │  │  neighbors, memory)  │  │
│  └──────┬──────┘  └────────┬────────┘  └──────────┬───────────┘  │
│         │                  │                      │              │
│  ┌──────▼──────────────────▼──────────────────────▼───────────┐  │
│  │                   Evolution Loop                            │  │
│  │  seed → [parse → canonicalize → evaluate_stepwise] → loop:  │  │
│  │    parents = strategy.select(population)                    │  │
│  │    offspring = mutation_ensemble.generate(parents, memory)   │  │
│  │    for ind in offspring:                                    │  │
│  │      ir = backend.parse(genome) → canonicalize → hash       │  │
│  │      if cache.hit(hash): reuse                              │  │
│  │      else: fitness = backend.evaluate_stepwise(ir)          │  │
│  │    population = survive(population, offspring)              │  │
│  │    archive.update(offspring)                                │  │
│  │    memory.update(offspring)                                 │  │
│  │    if gen % interval == 0: guidance = reflect(pop, memory)  │  │
│  └────────────────────────────────────────────────────────────┘  │
│         │                  │                │                    │
│  ┌──────▼───────┐  ┌──────▼──────┐  ┌──────▼───────┐            │
│  │ Mutation      │  │ Backend     │  │ Execution    │            │
│  │ Ensemble      │  │ Interface   │  │ Scheduler    │            │
│  │ (LLM + cheap  │  │ (with IR,   │  │ (async pool, │            │
│  │  operator ABCs)│  │  stepwise)  │  │  backpressure│            │
│  └──────────────┘  └──────┬──────┘  └──────────────┘            │
└────────────────────────────┼─────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │   lean     │ │    cfd    │ │  (future)  │
        │  backend   │ │  backend  │ │  backends  │
        └───────────┘ └───────────┘ └───────────┘
```

---

## 3. Core Data Structures (`evoforge/core/types.py`)

### 3.1 Structured Fitness

```python
@dataclass(frozen=True)
class Fitness:
    """Immutable structured fitness. Selection strategies dispatch on this."""
    primary: float
    auxiliary: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, bool] = field(default_factory=dict)
    feasible: bool = True

    def dominates(self, other: "Fitness") -> bool:
        """Pareto dominance over auxiliary objectives."""
        dominated_in_one = False
        for key in self.auxiliary:
            if key in other.auxiliary:
                if self.auxiliary[key] < other.auxiliary[key]:
                    return False
                if self.auxiliary[key] > other.auxiliary[key]:
                    dominated_in_one = True
        return dominated_in_one and self.primary >= other.primary
```

### 3.2 Structured Diagnostics

Diagnostics are **typed per-backend**, not free-form dicts. This is critical for prompt construction and programmatic analysis.

```python
# Base protocol
class Diagnostics(Protocol):
    def summary(self, max_tokens: int = 500) -> str:
        """Compressed text for LLM prompts. Must respect token budget."""
        ...

# Lean-specific
@dataclass
class LeanDiagnostics:
    success: bool
    goals_remaining: int
    goal_types: list[str]           # e.g. ["Continuous f", "∀ ε > 0, ..."]
    goal_contexts: list[str]        # local hypotheses available in each goal
    error_type: str | None          # "type_mismatch", "unknown_identifier", "tactic_failed", ...
    error_message: str | None       # first error only
    stuck_tactic_index: int | None  # which tactic step failed
    stuck_tactic: str | None        # the tactic text that failed
    steps_succeeded: int            # how many tactics executed before failure
    metavar_count: int              # unresolved metavariables

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

# CFD-specific
@dataclass
class CFDDiagnostics:
    converged_cases: int
    total_cases: int
    per_case: list[CaseResult]
    worst_case: str
    worst_error: float
    divergence_info: DivergenceInfo | None   # time, type, location

    def summary(self, max_tokens: int = 500) -> str:
        parts = [f"Converged: {self.converged_cases}/{self.total_cases}"]
        if self.divergence_info:
            d = self.divergence_info
            parts.append(f"Diverged: {self.worst_case} at t={d.time:.2f}s, "
                         f"{d.instability} at y/δ={d.location:.3f}")
        parts.append(f"Worst error: {self.worst_case} (L2={self.worst_error:.4f})")
        return "\n".join(parts)
```

### 3.3 Individual

```python
@dataclass
class Individual:
    genome: str                             # serialized canonical form
    ir: Any | None = None                   # parsed IR (backend-specific)
    ir_hash: str | None = None              # structural hash of canonical IR
    fitness: Fitness | None = None
    diagnostics: Diagnostics | None = None  # typed, not dict
    lineage: list[str] = field(default_factory=list)
    generation: int = 0
    id: str = field(default_factory=lambda: str(uuid4()))
    behavior_descriptor: tuple | None = None
    mutation_source: str = ""               # operator that produced this
```

### 3.4 Structured Reflection

```python
@dataclass
class Reflection:
    """Structured output from population-level LLM analysis."""
    strategies_to_try: list[str]
    strategies_to_avoid: list[str]
    useful_primitives: list[str]        # lemma names, function forms
    population_diagnosis: str           # "converging prematurely on..."
    suggested_temperature: float | None
```

### 3.5 Search Memory (Persistent Global State)

This is the key addition for long-run efficiency. Without it, the system relearns the same lessons every generation.

```python
@dataclass
class SearchMemory:
    """Persistent cross-generation memory. Feeds into all mutation prompts."""
    successful_patterns: list[Pattern]    # recurring structural motifs in high-fitness individuals
    failure_modes: list[FailureMode]      # recurring failure types with examples
    useful_constructs: list[str]          # lemma names, function subexpressions that appear in top individuals
    dead_ends: list[str]                  # approaches that were tried extensively and never worked
    best_fitness_history: list[float]     # for stagnation detection

    def update(self, generation: int, offspring: list[Individual], archive: Archive):
        """Called every generation. Updates all fields incrementally."""
        # Extract patterns from new high-fitness individuals
        for ind in offspring:
            if ind.fitness and ind.fitness.primary > self._threshold():
                self._extract_patterns(ind)
            if ind.fitness and ind.fitness.primary < 0.1:
                self._record_failure(ind)
        # Detect dead ends: patterns tried >N times with no improvement
        self._detect_dead_ends(archive)
        self.best_fitness_history.append(max(i.fitness.primary for i in offspring if i.fitness))

    def prompt_section(self, max_tokens: int = 400) -> str:
        """Render as text for injection into mutation prompts."""
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
class Pattern:
    description: str        # human-readable: "intro followed by simp with norm_num"
    frequency: int          # how often seen in top-k individuals
    first_seen: int         # generation
    exemplar_id: str        # individual ID

@dataclass
class FailureMode:
    description: str        # "unknown identifier errors from Finset namespace"
    count: int
    last_seen: int
```

---

## 4. IR Layer (`evoforge/core/ir.py`)

The IR is not just a "parsed representation" — it is the **semantic control layer** for the entire system.

### 4.1 IR Protocol

Every backend's IR must satisfy this protocol:

```python
class IRProtocol(Protocol):
    """Contract that all backend IRs must satisfy."""

    def canonicalize(self) -> "IRProtocol":
        """Return canonical form. Idempotent: canonicalize(canonicalize(x)) == canonicalize(x).
        Must normalize:
          - whitespace, formatting (both backends)
          - commutative reordering (CFD: a+b == b+a)
          - tactic no-ops (Lean: remove redundant `try` wrappers)
          - constant folding (CFD: 1*x → x)
        """
        ...

    def structural_hash(self) -> str:
        """Deterministic hash of canonical form. Ignores formatting.
        Two IRs with the same structural_hash are semantically equivalent
        (modulo the backend's equivalence relation)."""
        ...

    def serialize(self) -> str:
        """Convert back to genome string. Round-trip: parse(serialize(ir)) ≈ ir."""
        ...

    def complexity(self) -> int:
        """Structural complexity measure (AST nodes, tactic count, etc.)."""
        ...
```

### 4.2 Lean IR: `TacticSequence`

```python
@dataclass(frozen=True)
class TacticStep:
    tactic: str                 # "intro", "apply", "exact", "simp", ...
    args: tuple[str, ...]       # frozen for hashing
    raw: str                    # original text for roundtrip

@dataclass
class TacticSequence:
    """IR for Lean proofs. Ordered sequence with stepwise evaluation support."""
    steps: list[TacticStep]
    _canonical: bool = False

    def canonicalize(self) -> "TacticSequence":
        """Normalize tactic representation."""
        canonical_steps = []
        for step in self.steps:
            # Normalize whitespace
            raw = " ".join(step.raw.split())
            # Sort simp lemma lists alphabetically
            if step.tactic == "simp" and step.args:
                args = tuple(sorted(step.args))
            else:
                args = step.args
            # Remove no-op wrappers (e.g., `try { exact h }` when it succeeds → `exact h`)
            canonical_steps.append(TacticStep(tactic=step.tactic, args=args, raw=raw))
        result = TacticSequence(steps=canonical_steps)
        result._canonical = True
        return result

    def structural_hash(self) -> str:
        if not self._canonical:
            return self.canonicalize().structural_hash()
        content = "|".join(f"{s.tactic}({','.join(s.args)})" for s in self.steps)
        return hashlib.sha256(content.encode()).hexdigest()

    def prefix(self, k: int) -> "TacticSequence":
        """Return first k steps. For prefix caching and truncation mutation."""
        return TacticSequence(steps=self.steps[:k])

    def serialize(self) -> str:
        return "\n".join(s.raw for s in self.steps)

    def complexity(self) -> int:
        return len(self.steps)
```

### 4.3 CFD IR: SymPy-Native

```python
class ClosureExpr:
    """IR for CFD closure expressions. Wraps sympy.Expr with evoforge semantics."""

    def __init__(self, expr: sympy.Expr):
        self.expr = expr

    def canonicalize(self) -> "ClosureExpr":
        """SymPy simplification + canonical ordering."""
        canonical = sympy.simplify(self.expr)
        canonical = sympy.nsimplify(canonical, rational=False)
        return ClosureExpr(canonical)

    def structural_hash(self) -> str:
        """Hash the canonical SymPy expression tree."""
        canonical = self.canonicalize()
        # srepr gives deterministic structural representation
        return hashlib.sha256(sympy.srepr(canonical.expr).encode()).hexdigest()

    def serialize(self) -> str:
        return str(self.canonicalize().expr)

    def complexity(self) -> int:
        """Count AST nodes."""
        return sum(1 for _ in sympy.preorder_traversal(self.expr))

    def subtree_at(self, index: int) -> sympy.Expr:
        """Access subtree by index for subtree mutation."""
        nodes = list(sympy.preorder_traversal(self.expr))
        return nodes[index] if index < len(nodes) else self.expr

    def replace_subtree(self, index: int, replacement: sympy.Expr) -> "ClosureExpr":
        """Replace subtree at index. For cheap mutation operators."""
        nodes = list(sympy.preorder_traversal(self.expr))
        if index >= len(nodes):
            return self
        target = nodes[index]
        new_expr = self.expr.subs(target, replacement)
        return ClosureExpr(new_expr)
```

---

## 5. Selection Strategy Abstraction (`evoforge/core/selection.py`)

Selection is a first-class abstraction that matches the structured fitness model.

```python
class SelectionStrategy(ABC):
    """Abstract selection. Decoupled from fitness representation."""

    @abstractmethod
    def select(self, population: list[Individual], k: int) -> list[Individual]:
        """Select k individuals for reproduction."""
        ...

    @abstractmethod
    def survive(self, population: list[Individual],
                offspring: list[Individual], elite_k: int) -> list[Individual]:
        """Determine next generation from current population + offspring."""
        ...


class ScalarTournament(SelectionStrategy):
    """Tournament selection on fitness.primary."""
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select(self, population, k):
        selected = []
        for _ in range(k):
            contestants = random.sample(population, min(self.tournament_size, len(population)))
            winner = max(contestants, key=lambda i: i.fitness.primary if i.fitness else -inf)
            selected.append(winner)
        return selected

    def survive(self, population, offspring, elite_k):
        combined = population + offspring
        combined.sort(key=lambda i: i.fitness.primary if i.fitness else -inf, reverse=True)
        return combined[:len(population)]


class ParetoNSGA2(SelectionStrategy):
    """NSGA-II style. For CFD: accuracy vs. complexity."""
    def __init__(self, objectives: list[str]):
        self.objectives = objectives  # keys into fitness.auxiliary

    def select(self, population, k):
        fronts = self._fast_nondominated_sort(population)
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= k:
                selected.extend(front)
            else:
                # Crowding distance tiebreak
                front.sort(key=lambda i: self._crowding_distance(i, front), reverse=True)
                selected.extend(front[:k - len(selected)])
                break
        return selected

    # ... _fast_nondominated_sort, _crowding_distance


class Lexicase(SelectionStrategy):
    """Lexicase selection on auxiliary fitness components. Good for Lean."""
    def __init__(self, case_keys: list[str]):
        self.case_keys = case_keys  # auxiliary keys to iterate over

    def select(self, population, k):
        selected = []
        for _ in range(k):
            candidates = list(population)
            cases = list(self.case_keys)
            random.shuffle(cases)
            for case in cases:
                if len(candidates) <= 1:
                    break
                best_val = max(c.fitness.auxiliary.get(case, -inf) for c in candidates)
                candidates = [c for c in candidates
                              if c.fitness and abs(c.fitness.auxiliary.get(case, -inf) - best_val) < 1e-9]
            selected.append(random.choice(candidates))
        return selected


class MAPElites(SelectionStrategy):
    """Quality-diversity selection. Maintains behavior-space grid."""
    def __init__(self, behavior_space: BehaviorSpaceConfig):
        self.grid: dict[tuple, Individual] = {}
        self.behavior_space = behavior_space

    def select(self, population, k):
        # Uniform random from occupied cells
        occupied = list(self.grid.values())
        return [random.choice(occupied) for _ in range(min(k, len(occupied)))]

    def survive(self, population, offspring, elite_k):
        # Update grid: offspring replace current occupant if fitter
        for ind in offspring:
            if ind.behavior_descriptor is None:
                continue
            cell = self._discretize(ind.behavior_descriptor)
            if cell not in self.grid or ind.fitness.primary > self.grid[cell].fitness.primary:
                self.grid[cell] = ind
        return list(self.grid.values())

    def _discretize(self, descriptor: tuple) -> tuple:
        """Map continuous descriptor to grid cell using bin edges from config."""
        cell = []
        for dim, val in zip(self.behavior_space.dimensions, descriptor):
            if dim.type == "categorical":
                cell.append(val)
            else:
                bin_idx = np.searchsorted(dim.bin_edges, val) - 1
                cell.append(max(0, min(bin_idx, len(dim.bin_edges) - 2)))
        return tuple(cell)

    def coverage(self) -> float:
        """Fraction of grid cells occupied."""
        total = 1
        for dim in self.behavior_space.dimensions:
            total *= len(dim.bin_edges) - 1 if dim.type == "numeric" else len(dim.bins)
        return len(self.grid) / total
```

### 5.1 Behavior Descriptor Stability

Descriptors must be deterministic, low-noise, and comparable across generations.

```python
@dataclass
class BehaviorDimension:
    name: str
    type: str                                # "numeric" or "categorical"
    bins: list[str] | None = None            # for categorical
    bin_edges: list[float] | None = None     # for numeric
    extractor: Callable[[Any, Fitness, Diagnostics], Any] = None

    # Stability guarantees:
    # 1. extractor must be deterministic (same IR + fitness → same descriptor)
    # 2. extractor operates on canonical IR (not raw genome)
    # 3. numeric extractors must be bounded (bin_edges define range)
    # 4. categorical extractors must return values in `bins` (unknown → "other")
```

For Lean, descriptors are computed on the **canonical IR**, not the raw genome:

- `strategy_class`: determined by the *first tactic* (induction → "induction", intro → "direct", by_contra → "contradiction"). Stable because the canonical form normalizes the prefix.
- `proof_depth`: `len(ir.steps)` on canonical IR. Bucketed coarsely (1–3, 4–6, 7–10, 11+) to absorb minor variation.

For CFD:

- `dominant_variable`: sensitivity analysis on canonical SymPy expr (partial derivative magnitude). Stable because SymPy simplification is deterministic.
- `complexity`: AST node count on canonical expr. Inherently stable.

---

## 6. Mutation Operator Framework (`evoforge/core/mutation.py`)

Mutation operators are **typed ABCs that operate on IR, not strings**. The LLM is one operator among many.

### 6.1 Operator Protocol

```python
class MutationOperator(ABC):
    """All mutation operators implement this interface."""
    name: str
    cost: str  # "cheap" or "llm" — for budgeting

    @abstractmethod
    def apply(self, parent: Individual, context: MutationContext) -> str:
        """Produce a new genome string from parent.
        May use parent.ir (typed), parent.diagnostics (typed), context.memory, etc.
        Returns raw genome string (will be parsed/canonicalized by engine)."""
        ...

@dataclass
class MutationContext:
    """Shared context available to all operators."""
    generation: int
    memory: SearchMemory
    guidance: Reflection | None
    temperature: float
    backend: Backend
```

### 6.2 LLM Operators

```python
class LLMMutate(MutationOperator):
    name = "llm_mutate"
    cost = "llm"

    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client = client
        self.model = model

    def apply(self, parent: Individual, context: MutationContext) -> str:
        prompt = context.backend.format_mutation_prompt(
            parent=parent,
            diagnostics_summary=parent.diagnostics.summary(max_tokens=500),
            guidance=context.guidance,
            memory_section=context.memory.prompt_section(max_tokens=400),
        )
        response = self.client.messages.create(
            model=self.model,
            system=context.backend.system_prompt(),
            messages=[{"role": "user", "content": prompt}],
            temperature=context.temperature,
        )
        return context.backend.extract_genome(response.content[0].text)


class LLMCrossover(MutationOperator):
    name = "llm_crossover"
    cost = "llm"
    # Similar but takes two parents
```

### 6.3 Cheap Operators (Lean)

```python
class PrefixTruncation(MutationOperator):
    """Keep first k successful tactics, discard the rest.
    Operates on IR: uses diagnostics.steps_succeeded to know where to cut."""
    name = "prefix_truncation"
    cost = "cheap"

    def apply(self, parent: Individual, context: MutationContext) -> str:
        ir: TacticSequence = parent.ir
        diag: LeanDiagnostics = parent.diagnostics
        # Truncate at the last successful step
        k = max(1, diag.steps_succeeded)
        truncated = ir.prefix(k)
        return truncated.serialize()


class TacticSwap(MutationOperator):
    """Replace one tactic with a random common tactic."""
    name = "tactic_swap"
    cost = "cheap"

    COMMON_TACTICS = ["simp", "ring", "omega", "norm_num", "exact?", "apply?", "aesop"]

    def apply(self, parent: Individual, context: MutationContext) -> str:
        ir: TacticSequence = parent.ir
        if not ir.steps:
            return parent.genome
        idx = random.randrange(len(ir.steps))
        new_tactic = random.choice(self.COMMON_TACTICS)
        new_steps = list(ir.steps)
        new_steps[idx] = TacticStep(tactic=new_tactic, args=(), raw=new_tactic)
        return TacticSequence(steps=new_steps).serialize()


class TacticReorder(MutationOperator):
    """Swap two adjacent tactics. Valid when they operate on independent goals."""
    name = "tactic_reorder"
    cost = "cheap"
    # ...

class SplicePrefixes(MutationOperator):
    """Take prefix from parent A, suffix from parent B. Cheap crossover."""
    name = "splice_prefixes"
    cost = "cheap"
    # ...
```

### 6.4 Cheap Operators (CFD)

```python
class SubtreeMutation(MutationOperator):
    """Replace a random subtree with a random expression."""
    name = "subtree_mutation"
    cost = "cheap"

    def apply(self, parent: Individual, context: MutationContext) -> str:
        ir: ClosureExpr = parent.ir
        nodes = list(sympy.preorder_traversal(ir.expr))
        if len(nodes) < 2:
            return parent.genome
        idx = random.randrange(1, len(nodes))  # skip root
        replacement = self._random_subexpr()
        return ir.replace_subtree(idx, replacement).serialize()

    def _random_subexpr(self) -> sympy.Expr:
        templates = [
            lambda: sympy.exp(-symbols.alpha * symbols.Ri_g),
            lambda: 1 - sympy.exp(-symbols.y_plus / symbols.A),
            lambda: sympy.tanh(symbols.Ri_g),
            # ...
        ]
        return random.choice(templates)()


class ConstantPerturbation(MutationOperator):
    """Jitter numeric constants by ±10-50%."""
    name = "constant_perturb"
    cost = "cheap"

    def apply(self, parent: Individual, context: MutationContext) -> str:
        ir: ClosureExpr = parent.ir
        expr = ir.expr
        for atom in expr.atoms(sympy.Number):
            if random.random() < 0.3:  # perturb 30% of constants
                factor = random.uniform(0.5, 1.5)
                expr = expr.subs(atom, atom * factor)
        return ClosureExpr(expr).serialize()


class OperatorSwap(MutationOperator):
    """Replace one function with another (exp↔tanh, +↔*, etc.)."""
    name = "operator_swap"
    cost = "cheap"
    # ...

class TermDeletion(MutationOperator):
    """Remove one additive term from expression."""
    name = "term_deletion"
    cost = "cheap"
    # ...

class VariableSwap(MutationOperator):
    """Replace one physical variable with another."""
    name = "variable_swap"
    cost = "cheap"
    # ...
```

### 6.5 Mutation Ensemble

```python
class MutationEnsemble:
    def __init__(self, operators: list[MutationOperator],
                 weights: dict[str, float], adaptive: bool = True):
        self.operators = {op.name: op for op in operators}
        self.weights = weights
        self.adaptive = adaptive
        self.stats: dict[str, OperatorStats] = {name: OperatorStats() for name in self.operators}

    def generate(self, parents: list[Individual], context: MutationContext) -> list[Individual]:
        offspring = []
        for parent in parents:
            op_name = self._weighted_choice()
            op = self.operators[op_name]
            genome = op.apply(parent, context)
            offspring.append(Individual(
                genome=genome,
                mutation_source=op_name,
                lineage=[parent.id],
                generation=context.generation,
            ))
        return offspring

    def update_weights(self, generation_results: list[Individual]):
        """Adaptive: shift weight toward operators producing fitter offspring."""
        if not self.adaptive:
            return
        for ind in generation_results:
            if ind.fitness and ind.mutation_source in self.stats:
                self.stats[ind.mutation_source].record(ind.fitness.primary)
        # Softmax over mean fitness improvement per operator
        mean_improvements = {name: s.mean_improvement() for name, s in self.stats.items()}
        total = sum(max(0, v) for v in mean_improvements.values()) + 1e-8
        for name in self.weights:
            self.weights[name] = max(0.05, mean_improvements.get(name, 0) / total)
```

---

## 7. Evaluation & Caching (`evoforge/core/evaluator.py`)

### 7.1 Multi-Level Cache

```python
class EvaluationCache:
    """Three-level cache: parse → prefix → full evaluation."""

    def __init__(self, archive: Archive, backend_version: str, eval_config_hash: str):
        self.archive = archive
        # Cache keys include backend version + eval config to prevent stale results
        self.version_prefix = f"{backend_version}:{eval_config_hash}"

    def _make_key(self, ir_hash: str) -> str:
        return f"{self.version_prefix}:{ir_hash}"

    # Level 1: Parse cache (IR objects, in-memory)
    # Avoids re-parsing identical genome strings
    _parse_cache: dict[str, Any] = {}

    def get_parsed(self, genome: str) -> Any | None:
        return self._parse_cache.get(genome)

    def put_parsed(self, genome: str, ir: Any):
        self._parse_cache[genome] = ir

    # Level 2: Prefix cache (Lean only — partial evaluation results)
    # Key: structural hash of tactic prefix
    def get_prefix_result(self, prefix_hash: str) -> PrefixResult | None:
        key = self._make_key(f"prefix:{prefix_hash}")
        return self.archive.lookup(key)

    def put_prefix_result(self, prefix_hash: str, result: PrefixResult):
        key = self._make_key(f"prefix:{prefix_hash}")
        self.archive.store(key, result)

    # Level 3: Full evaluation cache
    def get_fitness(self, ir_hash: str) -> tuple[Fitness, Diagnostics] | None:
        key = self._make_key(ir_hash)
        return self.archive.lookup_fitness(key)

    def put_fitness(self, ir_hash: str, fitness: Fitness, diagnostics: Diagnostics):
        key = self._make_key(ir_hash)
        self.archive.store_fitness(key, fitness, diagnostics)
```

### 7.2 Async Evaluator with Backpressure

```python
class AsyncEvaluator:
    def __init__(self, backend: Backend, cache: EvaluationCache, config: ParallelismConfig):
        self.backend = backend
        self.cache = cache
        self.max_workers = config.max_workers
        self.eval_timeout = config.eval_timeout
        self.max_pending = config.max_pending     # backpressure limit
        self._semaphore = asyncio.Semaphore(self.max_pending)

    async def evaluate_batch(self, individuals: list[Individual]):
        tasks = []
        for ind in individuals:
            # Step 1: Parse (with parse cache)
            if ind.ir is None:
                cached_ir = self.cache.get_parsed(ind.genome)
                if cached_ir is not None:
                    ind.ir = cached_ir
                else:
                    ind.ir = self.backend.parse(ind.genome)
                    if ind.ir is not None:
                        self.cache.put_parsed(ind.genome, ind.ir)

            # Step 2: Canonicalize + hash
            if ind.ir is not None:
                canonical_ir = ind.ir.canonicalize()
                ind.ir = canonical_ir
                ind.genome = canonical_ir.serialize()
                ind.ir_hash = canonical_ir.structural_hash()

                # Step 3: Check full eval cache
                cached = self.cache.get_fitness(ind.ir_hash)
                if cached is not None:
                    ind.fitness, ind.diagnostics = cached
                    continue
            else:
                # Unparseable: assign low fitness, no retry
                ind.fitness = Fitness(primary=0.05, constraints={"parseable": False}, feasible=False)
                ind.diagnostics = None
                continue

            # Step 4: Queue for evaluation (with backpressure)
            tasks.append(self._eval_with_backpressure(ind))

        await asyncio.gather(*tasks)

    async def _eval_with_backpressure(self, ind: Individual):
        async with self._semaphore:  # limits concurrent evaluations
            loop = asyncio.get_event_loop()
            try:
                fitness, diagnostics = await asyncio.wait_for(
                    loop.run_in_executor(None, self.backend.evaluate, ind.ir),
                    timeout=self.eval_timeout,
                )
                ind.fitness = fitness
                ind.diagnostics = diagnostics
                self.cache.put_fitness(ind.ir_hash, fitness, diagnostics)
            except asyncio.TimeoutError:
                ind.fitness = Fitness(primary=0.01, constraints={"timeout": True}, feasible=False)
                ind.diagnostics = None
```

---

## 8. Backend Interface (Complete)

```python
class Backend(ABC):
    """Full backend contract. Backends own: IR, evaluation, operators, prompts."""

    # --- Identity (for cache keying) ---
    @abstractmethod
    def version(self) -> str:
        """Backend version string. Change triggers cache invalidation."""
        ...

    @abstractmethod
    def eval_config_hash(self) -> str:
        """Hash of evaluation config (benchmark cases, solver params, etc.).
        Change triggers cache invalidation."""
        ...

    # --- IR pipeline ---
    @abstractmethod
    def parse(self, genome: str) -> IRProtocol | None: ...

    # --- Evaluation ---
    @abstractmethod
    def evaluate(self, ir: IRProtocol) -> tuple[Fitness, Diagnostics]: ...

    # --- Stepwise evaluation (Lean) ---
    def evaluate_stepwise(self, ir: IRProtocol, cache: EvaluationCache) -> tuple[Fitness, Diagnostics]:
        """Default: delegates to evaluate(). Lean overrides for prefix caching."""
        return self.evaluate(ir)

    # --- Seed ---
    @abstractmethod
    def seed_population(self, n: int) -> list[str]: ...

    # --- Mutation operators ---
    @abstractmethod
    def mutation_operators(self) -> list[MutationOperator]:
        """Return all operators (LLM + cheap) with default weights."""
        ...

    @abstractmethod
    def default_operator_weights(self) -> dict[str, float]: ...

    # --- LLM integration ---
    @abstractmethod
    def system_prompt(self) -> str: ...
    @abstractmethod
    def format_mutation_prompt(self, parent: Individual,
                                diagnostics_summary: str,
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

    # --- Genome extraction from LLM output ---
    @abstractmethod
    def extract_genome(self, llm_text: str) -> str: ...

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

## 9. Lean Backend: Stepwise Evaluation

This is the single highest-leverage change for the Lean backend.

### 9.1 Stepwise Execution Model

Instead of "run full proof, get result," we evaluate **tactic-by-tactic**:

```python
class LeanStepwiseEvaluator:
    """Evaluates Lean proofs one tactic at a time.
    Enables: prefix caching, credit assignment, early stopping."""

    def __init__(self, project_dir: str, target_theorem: str):
        self.project_dir = project_dir
        self.target_theorem = target_theorem
        self._warm_environment()

    def evaluate_stepwise(self, ir: TacticSequence,
                          cache: EvaluationCache) -> tuple[Fitness, LeanDiagnostics]:
        """Evaluate proof step by step, reusing cached prefix results."""

        # Find longest cached prefix
        best_prefix_len = 0
        best_prefix_state = self._initial_goal_state()
        for k in range(len(ir.steps), 0, -1):
            prefix = ir.prefix(k)
            prefix_hash = prefix.structural_hash()
            cached = cache.get_prefix_result(prefix_hash)
            if cached is not None:
                best_prefix_len = k
                best_prefix_state = cached.goal_state
                break

        # Evaluate remaining steps from cached prefix
        current_state = best_prefix_state
        steps_succeeded = best_prefix_len
        for i in range(best_prefix_len, len(ir.steps)):
            step = ir.steps[i]
            result = self._apply_tactic(current_state, step)

            if result.success:
                current_state = result.new_state
                steps_succeeded = i + 1
                # Cache this prefix
                prefix = ir.prefix(i + 1)
                cache.put_prefix_result(prefix.structural_hash(),
                    PrefixResult(goal_state=result.new_state, steps=i + 1))
            else:
                # Step failed — record diagnostics and stop
                return self._score_failure(ir, i, step, result, current_state, steps_succeeded)

            if result.proof_complete:
                return self._score_success(ir, steps_succeeded)

        # All steps succeeded but proof not complete (goals remain)
        return self._score_incomplete(ir, current_state, steps_succeeded)

    def _apply_tactic(self, state: GoalState, step: TacticStep) -> TacticResult:
        """Apply a single tactic to a goal state via Lean server."""
        # Write a proof file with tactics up to this point
        # Use `lake env lean` to check, parse goal state from output
        # OR: use Lean LSP for true incremental checking
        ...

    def _score_failure(self, ir, failed_idx, step, result, state, steps_succeeded):
        goals = state.goals
        diag = LeanDiagnostics(
            success=False,
            goals_remaining=len(goals),
            goal_types=[g.type_str for g in goals],
            goal_contexts=[g.context_summary for g in goals],
            error_type=result.error_type,
            error_message=result.error_message,
            stuck_tactic_index=failed_idx,
            stuck_tactic=step.raw,
            steps_succeeded=steps_succeeded,
            metavar_count=sum(g.metavar_count for g in goals),
        )
        fitness = self._compute_fitness(diag)
        return fitness, diag

    def _compute_fitness(self, diag: LeanDiagnostics) -> Fitness:
        aux = {
            "steps_succeeded": float(diag.steps_succeeded),
            "goals_remaining": float(diag.goals_remaining),
            "goal_progress": 1.0 / (1.0 + diag.goals_remaining),
            "goal_familiarity": self._goal_shape_similarity(diag.goal_types),
            "metavar_count": float(diag.metavar_count),
        }

        primary = (
            0.1 * (1.0 if diag.stuck_tactic_index is None or diag.stuck_tactic_index > 0 else 0.0)
            + 0.4 * aux["goal_progress"]
            + 0.2 * aux["goal_familiarity"]
            + 0.2 * min(1.0, diag.steps_succeeded / 10.0)
            + 0.1 * (1.0 if diag.error_type == "type_mismatch" else 0.0)
        )

        return Fitness(
            primary=min(primary, 0.99),
            auxiliary=aux,
            constraints={"proof_complete": False},
            feasible=False,
        )
```

### 9.2 Why Stepwise Changes Everything

- **Prefix caching:** if proofs A and B share a 5-step prefix, the first 5 steps are evaluated once
- **Credit assignment:** we know *exactly which tactic* failed, not just "the proof failed"
- **Truncation mutation:** `PrefixTruncation` operator can cut at the *last successful step*, not a random point
- **Goal-state-aware mutation:** the LLM sees the *exact goal state* at the failure point, not a reconstructed approximation
- **Early stopping:** if step 1 fails, don't bother with the remaining 15 steps

---

## 10. CFD Backend: Constraint-Integrated Evaluation

### 10.1 Constraint Penalties in Fitness

```python
class CFDBackend(Backend):
    def evaluate(self, ir: ClosureExpr) -> tuple[Fitness, CFDDiagnostics]:
        penalty = 1.0
        constraints = {}

        # Dimensional consistency (via SymPy dimensional analysis)
        dim_ok = self._check_dimensions(ir.expr)
        constraints["dimensional"] = dim_ok
        if not dim_ok:
            penalty *= 0.1

        # Asymptotic: reduces to van Driest as Ri_g → 0
        unstrat_ok = self._check_unstratified_limit(ir.expr)
        constraints["unstratified_limit"] = unstrat_ok
        if not unstrat_ok:
            penalty *= 0.5

        # Suppresses turbulence as Ri_g → ∞
        suppress_ok = self._check_stratification_suppression(ir.expr)
        constraints["suppression"] = suppress_ok
        if not suppress_ok:
            penalty *= 0.5

        # Non-negative (realizable)
        realizable = self._check_non_negative(ir.expr)
        constraints["realizable"] = realizable
        if not realizable:
            penalty *= 0.3

        # Run solver
        case_results = []
        for case in self.benchmark_cases:
            try:
                result = self.solver.run(closure_expr=ir.expr, **case.params)
                err = self._l2_error(result, case.reference)
                case_results.append(CaseResult(
                    case=case.name, error=err, converged=True))
            except SolverDivergenceError as e:
                case_results.append(CaseResult(
                    case=case.name, error=1.0, converged=False,
                    divergence=DivergenceInfo(
                        time=e.time, instability=e.instability, location=e.location)))

        errors = [c.error for c in case_results]
        raw_fitness = 1.0 / (1.0 + np.mean(errors))
        complexity = ir.complexity()

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
                "complexity": float(complexity),
                "penalty": penalty,
                "converged_fraction": diagnostics.converged_cases / diagnostics.total_cases,
            },
            constraints=constraints,
            feasible=all(constraints.values()),
        ), diagnostics
```

---

## 11. Execution Model (`evoforge/core/scheduler.py`)

### 11.1 Scheduling Modes

```python
class ExecutionScheduler:
    """Controls how evaluation and LLM calls are dispatched."""

    def __init__(self, config: SchedulerConfig):
        self.mode = config.mode  # "synchronous", "async_batch", "streaming"
        self.llm_semaphore = asyncio.Semaphore(config.max_llm_concurrent)
        self.eval_semaphore = asyncio.Semaphore(config.max_eval_concurrent)
        self.llm_budget_per_gen = config.llm_budget_per_gen  # max LLM calls per generation

    async def dispatch_mutations(self, ensemble: MutationEnsemble,
                                  parents: list[Individual],
                                  context: MutationContext) -> list[Individual]:
        """Dispatch mutations with LLM rate limiting."""
        llm_calls_this_gen = 0
        offspring = []
        for parent in parents:
            op_name = ensemble._weighted_choice()
            op = ensemble.operators[op_name]
            if op.cost == "llm":
                if llm_calls_this_gen >= self.llm_budget_per_gen:
                    # Budget exceeded: fall back to cheap operator
                    op = self._cheapest_operator(ensemble)
                    op_name = op.name
                else:
                    async with self.llm_semaphore:
                        genome = await asyncio.to_thread(op.apply, parent, context)
                    llm_calls_this_gen += 1
                    offspring.append(Individual(genome=genome, mutation_source=op_name, ...))
                    continue
            genome = op.apply(parent, context)
            offspring.append(Individual(genome=genome, mutation_source=op_name, ...))
        return offspring
```

### 11.2 Backpressure Rules

| Condition | Action |
|-----------|--------|
| LLM API slow (>10s/call) | Shift weight toward cheap operators for remainder of generation |
| Eval queue > `max_pending` | Block new evaluations until queue drains |
| Eval timeout | Assign timeout fitness (0.01), continue |
| LLM budget exhausted | All remaining mutations use cheap operators |
| Solver diverges immediately (<0.1s) | Fast-path: skip remaining cases for this individual |

### 11.3 Failure Handling Summary

| Failure | Response | Retry? |
|---------|----------|--------|
| LLM returns unparseable output | Low fitness (0.05) | No |
| LLM API error (rate limit, network) | Retry with exponential backoff, max 3 | Yes |
| Lean evaluation timeout (>60s) | Low fitness (0.01), flag as timeout | No |
| CFD solver diverges | Structured diagnostics, penalty fitness | No |
| CFD solver timeout (>30s) | Low fitness (0.01) | No |
| Cache corruption | Invalidate key, re-evaluate | Yes (once) |

---

## 12. Configuration

```toml
[run]
name = "levy_proof_search_001"
backend = "lean"
max_generations = 200
seed = 42

[population]
size = 30
replacement = "generational"
elitism = 2

[selection]
method = "lexicase"            # Lean default; CFD uses "nsga2"
tournament_size = 3            # if method = "tournament"
lexicase_keys = ["goal_progress", "steps_succeeded", "goal_familiarity"]
nsga2_objectives = ["raw_accuracy", "complexity"]  # if method = "nsga2"

[mutation]
initial_weights = { llm_mutate = 0.3, llm_crossover = 0.15, prefix_truncation = 0.15, tactic_swap = 0.15, tactic_reorder = 0.1, splice_prefixes = 0.15 }
adaptive_weights = true

[llm]
model = "claude-haiku-4-5-20251001"         # routine mutations
reflection_model = "claude-sonnet-4-5-20250929"  # reflection + hard cases
temperature_start = 1.0
temperature_end = 0.3
temperature_schedule = "linear"
max_tokens = 2048

[reflection]
interval = 10
include_top_k = 5
include_bottom_k = 5

[memory]
max_patterns = 20
max_dead_ends = 15
max_constructs = 30
stagnation_window = 20          # detect stagnation over last N generations

[scheduler]
mode = "async_batch"
max_llm_concurrent = 4
max_eval_concurrent = 8         # cpu_count for CFD, 4 for Lean
llm_budget_per_gen = 15         # max LLM calls per generation

[parallelism]
max_workers = 8
eval_timeout = 60               # seconds

[diversity]
strategy = "map_elites"

[ablation]
disable_llm = false
disable_diagnostics = false
disable_reflection = false
disable_cheap_operators = false
disable_memory = false

[lean]
project_dir = "./lean_project"
target_theorem = "levy_characteristic_continuous"
target_file = "EvoForge/Evolved.lean"
mathlib_version = "v4.15.0"
stepwise = true                 # enable stepwise evaluation

[cfd]
solver_module = "evoforge.backends.cfd.solver"
benchmark_dir = "./benchmarks"
complexity_weight = 0.01        # for NSGA-II
max_ast_depth = 8
```

---

## 13. Experimental Design

### 13.1 Reproducibility Contract

```python
class ExperimentRunner:
    """Deterministic experiment execution."""

    def __init__(self, config_path: str, seed: int):
        self.config = load_config(config_path)
        self.seed = seed
        # Deterministic RNG for all stochastic decisions
        self.rng = random.Random(seed)
        np.random.seed(seed)
        # LLM calls are non-deterministic (temperature > 0)
        # but we log all prompts + responses for replay

    def run(self) -> ExperimentResult:
        ...

    def replay(self, archive_path: str) -> ExperimentResult:
        """Replay from logged LLM responses (deterministic)."""
        ...
```

### 13.2 Metrics (Beyond Fitness)

Every run logs these metrics per generation:

| Metric | Description | Why |
|--------|-------------|-----|
| `best_fitness` | Max primary fitness | Convergence speed |
| `mean_fitness` | Population mean | Overall progress |
| `diversity_entropy` | Shannon entropy over behavior space | Population health |
| `map_elites_coverage` | Fraction of grid cells occupied | Exploration breadth |
| `cache_hit_rate` | Fraction of evaluations served from cache | Efficiency |
| `llm_calls` | Number of LLM API calls this generation | Cost |
| `llm_call_efficiency` | Fitness improvement per LLM call | Is the LLM worth it? |
| `operator_fitness_delta` | Mean fitness improvement per operator | Which operators help? |
| `stagnation_counter` | Generations without improvement | When to stop |
| `unique_genomes_evaluated` | Cumulative unique evaluations | Search breadth |

### 13.3 Ablation Matrix

| Experiment | LLM | Diagnostics | Reflection | Memory | Cheap Ops | Purpose |
|------------|:---:|:---:|:---:|:---:|:---:|---------|
| **Full system** | ✓ | ✓ | ✓ | ✓ | ✓ | Main result |
| **No LLM** | ✗ | — | — | — | ✓ | Is LLM mutation better than cheap operators? |
| **No diagnostics** | ✓ | ✗ | ✓ | ✓ | ✓ | Do error messages help LLM mutate? |
| **No reflection** | ✓ | ✓ | ✗ | ✓ | ✓ | Does population analysis improve search? |
| **No memory** | ✓ | ✓ | ✓ | ✗ | ✓ | Does persistent memory help? |
| **LLM only** | ✓ | ✓ | ✓ | ✓ | ✗ | Do cheap operators stabilize / reduce cost? |
| **Random** | ✗ | — | — | — | random | Sanity check |

Each experiment: **5 seeds**, report mean ± std. Statistical significance via Wilcoxon signed-rank test.

### 13.4 Baselines as First-Class Runs

Baselines are not config flags — they are parallel run configurations shipped with the repo:

```
configs/
├── lean_levy.toml                      # full system
├── lean_levy_ablation_no_llm.toml      # cheap operators only
├── lean_levy_ablation_no_diag.toml     # LLM without diagnostics
├── lean_levy_ablation_no_reflect.toml  # no reflection
├── lean_levy_ablation_no_memory.toml   # no search memory
├── lean_levy_ablation_llm_only.toml    # no cheap operators
├── lean_levy_ablation_random.toml      # random baseline
├── cfd_oscillatory.toml                # full system
├── cfd_oscillatory_ablation_*.toml     # same ablation set
└── ablation_sweep.toml                 # meta-config: run all of the above
```

`scripts/ablation_sweep.py` runs all configs × 5 seeds and produces comparison plots.

---

## 14. Repo Structure

```
evoforge/
├── README.md
├── DESIGN.md
├── pyproject.toml
├── evoforge/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py              # Fitness, Individual, Reflection, SearchMemory, Diagnostics
│   │   ├── ir.py                 # IRProtocol, BehaviorSpaceConfig, BehaviorDimension
│   │   ├── population.py         # PopulationManager
│   │   ├── selection.py          # SelectionStrategy ABC + ScalarTournament, ParetoNSGA2, Lexicase, MAPElites
│   │   ├── mutation.py           # MutationOperator ABC + MutationEnsemble
│   │   ├── archive.py            # SQLite archive (multi-level cache, lineage, neighbors)
│   │   ├── evaluator.py          # AsyncEvaluator, EvaluationCache
│   │   ├── scheduler.py          # ExecutionScheduler (backpressure, budgeting)
│   │   ├── memory.py             # SearchMemory, Pattern, FailureMode
│   │   ├── engine.py             # main evolution loop
│   │   └── config.py             # TOML → Pydantic models
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py             # LLM API wrapper
│   │   ├── operators.py          # LLMMutate, LLMCrossover (MutationOperator subclasses)
│   │   └── templates/
│   │       ├── mutate.j2
│   │       ├── crossover.j2
│   │       └── reflect.j2
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py               # Backend ABC
│   │   ├── lean/
│   │   │   ├── __init__.py
│   │   │   ├── backend.py        # LeanBackend
│   │   │   ├── ir.py             # TacticStep, TacticSequence
│   │   │   ├── evaluator.py      # LeanStepwiseEvaluator
│   │   │   ├── scorer.py         # goal-state scoring
│   │   │   ├── operators.py      # PrefixTruncation, TacticSwap, TacticReorder, SplicePrefixes
│   │   │   └── templates/
│   │   └── cfd/
│   │       ├── __init__.py
│   │       ├── backend.py        # CFDBackend
│   │       ├── ir.py             # ClosureExpr
│   │       ├── constraints.py    # dimensional analysis, realizability, limits
│   │       ├── solver.py         # 1D solver wrapper
│   │       ├── operators.py      # SubtreeMutation, ConstantPerturbation, OperatorSwap, etc.
│   │       └── templates/
│   └── viz/
│       ├── __init__.py
│       ├── genealogy.py
│       ├── fitness_plots.py
│       ├── map_elites.py
│       ├── operator_analysis.py  # which operators are productive?
│       └── dashboard.py
├── configs/                       # all run configs including ablation baselines
├── lean_project/
│   ├── lakefile.lean
│   ├── lean-toolchain
│   └── EvoForge/
│       ├── Dependencies.lean
│       ├── Target.lean
│       └── Evolved.lean
├── benchmarks/
│   ├── case_01_clear_water/
│   ├── case_02_low_concentration/
│   └── case_03_high_concentration_laminarization/
├── tests/
│   ├── test_core/
│   │   ├── test_types.py
│   │   ├── test_ir.py
│   │   ├── test_selection.py
│   │   ├── test_mutation.py
│   │   ├── test_archive.py
│   │   ├── test_evaluator.py
│   │   └── test_memory.py
│   ├── test_lean/
│   │   ├── test_ir.py
│   │   ├── test_stepwise.py
│   │   ├── test_scorer.py
│   │   └── test_operators.py
│   └── test_cfd/
│       ├── test_ir.py
│       ├── test_constraints.py
│       └── test_operators.py
└── scripts/
    ├── run.py
    ├── analyze.py
    ├── ablation_sweep.py
    ├── seed_from_mathlib.py
    └── experiment_report.py       # generates comparison tables + plots
```

---

## 15. MVP Milestones (Unchanged from v2)

### Phase 1: Core + Lean Backend (Weeks 1–3)

| Week | Deliverable |
|------|-------------|
| 1 | `types.py`, `ir.py`, `archive.py`, `selection.py` (all four strategies), `mutation.py` (operator framework). Tests. |
| 2 | Lean IR, stepwise evaluator, scorer with goal-state extraction, cheap operators. Integration test. |
| 3 | LLM operators, `SearchMemory`, `MutationEnsemble`, `Engine`. First real run. |

### Phase 2: CFD Backend (Weeks 4–6)

| Week | Deliverable |
|------|-------------|
| 4 | SymPy IR, constraints, cheap operators. |
| 5 | Solver wrapper, benchmark cases, CFDBackend integration. First run. |
| 6 | NSGA-II selection. Pareto front visualization. Comparison vs. known closures. |

### Phase 3: Quality-Diversity + Analysis (Weeks 7–8)

| Week | Deliverable |
|------|-------------|
| 7 | MAP-Elites with stabilized descriptors. |
| 8 | Full ablation sweep. Statistical analysis. |

### Phase 4: Paper & Release (Weeks 9–10)

---

## 16. What Makes This Novel (Claims → Evidence)

| Claim | Ablation Experiment | Expected Signal |
|-------|---------------------|-----------------|
| Error-informed mutation helps | Full vs. No Diagnostics | Faster convergence with diagnostics |
| Reflection improves search | Full vs. No Reflection | Better final fitness, less stagnation |
| Search memory reduces redundancy | Full vs. No Memory | Higher cache hit rate, fewer wasted evals |
| LLM + cheap operators > either alone | Full vs. LLM Only vs. No LLM | Full system dominates both |
| Formally-grounded fitness prevents reward hacking | Qualitative (no scoring pathologies) | No degenerate high-fitness individuals |
| Unified framework works across domains | Both backends succeed | Same engine, different backends |
| Quality-diversity for proofs is valuable | MAP-Elites finds multiple distinct valid proofs | Coverage > 0 in behavior grid |

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
aiosqlite

# Lean backend
elan
lean4 >= 4.14
mathlib4

# CFD backend
numpy
scipy
sympy

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

*v3 — this document is implementation-ready.*
