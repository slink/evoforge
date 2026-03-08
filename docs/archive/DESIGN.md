# evoforge: LLM-Guided Evolution over Formally-Grounded Symbolic Expressions

> *A unified framework for evolving verified symbolic artifacts — with theorem proofs and turbulence closures as first-class backends.*

---

## 1. Thesis

Most LLM + evolutionary algorithm work operates in domains where fitness is noisy, approximate, or expensive to evaluate (natural language quality, code benchmarks, reward models). This project targets a different regime: **domains where a formal system provides a deterministic, authoritative fitness signal**. A type checker either accepts a proof or it doesn't. A PDE solver either converges to match benchmark data or it doesn't.

This constraint is a feature, not a limitation. It eliminates reward hacking, makes evolution reproducible, and lets the LLM focus on what it's actually good at — **semantically informed mutation** — rather than also serving as a noisy fitness proxy.

The framework, **evoforge**, implements a generic LLM-guided evolutionary engine with pluggable backends. Two backends ship with the initial release:

| Backend | Representation | Fitness Oracle | LLM Role |
|---------|---------------|----------------|----------|
| **lean** | Lean 4 tactic sequences | `lake build` type checker | Proposes proof strategies from error diagnostics |
| **cfd** | Symbolic closure expressions (AST) | 1D sediment-transport solver vs. benchmark data | Proposes physically-motivated functional forms |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    evoforge core                     │
│                                                     │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Population │  │  Selection   │  │  Archive /   │  │
│  │  Manager   │  │  (tourney,   │  │  Hall of     │  │
│  │            │  │   MAP-Elites)│  │  Fame        │  │
│  └─────┬─────┘  └──────┬───────┘  └──────┬───────┘  │
│        │               │                 │          │
│  ┌─────▼───────────────▼─────────────────▼───────┐  │
│  │            Evolution Loop                      │  │
│  │  for gen in 1..max_generations:                │  │
│  │    parents = select(population)                │  │
│  │    offspring = llm_mutate(parents, context)    │  │
│  │    fitness = evaluate(offspring)  ← backend    │  │
│  │    population = survive(population, offspring) │  │
│  │    archive.update(offspring)                   │  │
│  └───────────────────────────────────────────────┘  │
│        │                    │                       │
│  ┌─────▼──────┐      ┌─────▼──────┐                │
│  │ LLM Client │      │  Backend   │                │
│  │ (mutation,  │      │  Interface │                │
│  │  crossover, │      │ (abstract) │                │
│  │  analysis)  │      └─────┬──────┘                │
│  └────────────┘            │                        │
└────────────────────────────┼────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │   lean     │ │    cfd    │ │  (future)  │
        │  backend   │ │  backend  │ │  backends  │
        └───────────┘ └───────────┘ └───────────┘
```

---

## 3. Core Engine (`evoforge/core/`)

### 3.1 Individual

An `Individual` is backend-agnostic at the core level. It holds:

```python
@dataclass
class Individual:
    genome: str                    # serialized representation (backend-specific)
    fitness: float | None          # None = unevaluated
    metadata: dict                 # backend-specific diagnostics
    lineage: list[str]             # parent IDs for genealogy tracking
    generation: int
    id: str                        # UUID
    behavior_descriptor: tuple | None  # for MAP-Elites (optional)
```

The `genome` is always a **string** — this is a deliberate choice. LLMs operate on text. Whether that string is a Lean tactic proof or a symbolic math expression, the core engine doesn't need to know. Backends are responsible for parsing/validating genomes in their domain.

### 3.2 Population Manager

```python
class PopulationManager:
    def __init__(self, max_size: int, strategy: str = "generational"):
        ...

    def initialize(self, seed_genomes: list[str]) -> list[Individual]: ...
    def select_parents(self, k: int, method: str = "tournament") -> list[Individual]: ...
    def replace(self, offspring: list[Individual]) -> None: ...
    def best(self, n: int = 1) -> list[Individual]: ...
    def diversity_metric(self) -> float: ...
```

Supports both generational and steady-state replacement. For MAP-Elites style quality-diversity, the population is partitioned into a grid of niches defined by `behavior_descriptor`.

### 3.3 Selection

Standard implementations:

- **Tournament selection** (default, configurable pressure)
- **Fitness-proportionate** (roulette wheel)
- **Lexicase selection** — particularly relevant for the Lean backend where fitness may decompose into multiple subgoals
- **MAP-Elites cell selection** — uniform random from occupied cells, for quality-diversity runs

### 3.4 LLM Mutation Interface

This is the heart of the novelty. Rather than random perturbation, mutations are **semantically informed** by an LLM.

```python
class LLMMutator:
    """
    Uses an LLM to propose mutations that are informed by:
    1. The parent genome(s)
    2. Fitness diagnostics (error messages, convergence data)
    3. Population-level context (what's been tried, what works)
    4. Domain knowledge (injected via system prompt from backend)
    """

    def __init__(self, model: str, backend: Backend):
        self.client = anthropic.Anthropic()  # or openai, etc.
        self.model = model
        self.backend = backend

    def mutate(self, parent: Individual, temperature: float = 0.8) -> str:
        """Single-parent mutation: modify an existing genome."""
        prompt = self.backend.format_mutation_prompt(parent)
        response = self.client.messages.create(
            model=self.model,
            system=self.backend.system_prompt(),
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return self.backend.parse_genome(response)

    def crossover(self, parent_a: Individual, parent_b: Individual) -> str:
        """Two-parent crossover: combine strategies from two genomes."""
        prompt = self.backend.format_crossover_prompt(parent_a, parent_b)
        ...

    def reflect(self, population: list[Individual]) -> str:
        """Population-level reflection: analyze trends and suggest direction."""
        prompt = self.backend.format_reflection_prompt(population)
        ...
```

**Key design decision:** the LLM sees the **fitness diagnostics**, not just the genome. For Lean, that means the actual error message ("unknown identifier 'Finset.sum_comm'"). For CFD, that means the convergence history and where the model diverges from data. This transforms the LLM from a blind mutator into an **informed debugger/designer**.

**Temperature scheduling:** start high (exploration), anneal low (exploitation). Configurable per run.

### 3.5 Archive & Logging

Every individual ever evaluated is logged to a SQLite database with full lineage. This enables:

- Post-hoc genealogy visualization (which evolutionary paths led to solutions?)
- Deduplication (don't re-evaluate identical genomes)
- Restart from checkpoint
- Analyzing which LLM mutation strategies are most productive

```python
class Archive:
    def __init__(self, db_path: str): ...
    def record(self, individual: Individual, generation: int): ...
    def has_been_tried(self, genome: str) -> bool: ...
    def hall_of_fame(self, n: int) -> list[Individual]: ...
    def lineage_tree(self, individual_id: str) -> nx.DiGraph: ...
```

---

## 4. Backend Interface

```python
class Backend(ABC):
    """Abstract interface that each domain must implement."""

    @abstractmethod
    def system_prompt(self) -> str:
        """Domain-specific system prompt for the LLM. Establishes the
        persona, constraints, and knowledge the LLM should bring."""
        ...

    @abstractmethod
    def seed_population(self, n: int) -> list[str]:
        """Generate n initial genomes. Can be hand-written examples,
        LLM-generated, or drawn from a corpus."""
        ...

    @abstractmethod
    def evaluate(self, genome: str) -> tuple[float, dict]:
        """Run the fitness oracle. Returns (fitness_score, diagnostics).
        Diagnostics are backend-specific and fed back to the LLM."""
        ...

    @abstractmethod
    def format_mutation_prompt(self, parent: Individual) -> str:
        """Format a mutation request for the LLM, including diagnostics."""
        ...

    @abstractmethod
    def format_crossover_prompt(self, a: Individual, b: Individual) -> str:
        """Format a crossover request for the LLM."""
        ...

    @abstractmethod
    def format_reflection_prompt(self, population: list[Individual]) -> str:
        """Ask the LLM to reflect on population-level trends."""
        ...

    @abstractmethod
    def validate_genome(self, genome: str) -> bool:
        """Quick syntactic check before expensive evaluation."""
        ...

    @abstractmethod
    def behavior_descriptor(self, genome: str, diagnostics: dict) -> tuple:
        """Optional: compute a behavior descriptor for MAP-Elites."""
        ...

    @abstractmethod
    def parse_genome(self, llm_response) -> str:
        """Extract a valid genome string from raw LLM output."""
        ...
```

---

## 5. Backend: Lean Proof Evolution (`evoforge/backends/lean/`)

### 5.1 Problem Formulation

**Goal:** Given an unproven Lean 4 theorem statement (e.g., a lemma needed for the Lévy process formalization), evolve a valid proof.

**Genome:** A complete Lean 4 proof body — either a tactic block or a term-mode expression.

```lean
-- Target theorem (fixed, provided by user):
theorem levy_characteristic_continuous (μ : Measure ℝ) [IsLevyProcess μ] :
    Continuous (characteristicFunction μ) := by
  -- ← THIS IS THE GENOME (the tactic block that follows `by`)
  intro ε hε
  obtain ⟨δ, hδ, hball⟩ := levy_continuity_lemma μ ε hε
  exact ⟨δ, hδ, fun t ht => hball t ht⟩
```

### 5.2 Fitness Function

**Primary fitness (binary):** Does `lake build` succeed? (0.0 or 1.0)

**Gradient signal for partial progress** (critical for evolution to work):

```python
def evaluate(self, genome: str) -> tuple[float, dict]:
    # Write genome into .lean file
    self._write_proof_file(genome)

    # Run lake build, capture output
    result = subprocess.run(
        ["lake", "build"], capture_output=True, timeout=120, cwd=self.project_dir
    )

    diagnostics = {"stderr": result.stderr.decode(), "returncode": result.returncode}

    if result.returncode == 0:
        return 1.0, diagnostics

    # Parse Lean errors for gradient signal
    errors = self._parse_lean_errors(result.stderr.decode())
    diagnostics["errors"] = errors

    # Heuristic scoring for partial progress:
    #   - Fewer remaining goals = higher fitness
    #   - Type-correct intermediate steps = bonus
    #   - Deeper progress into proof = bonus
    score = self._partial_score(errors)
    return score, diagnostics
```

**Partial scoring heuristics:**

| Signal | Score contribution |
|--------|--------------------|
| File parses without syntax errors | +0.1 |
| Theorem statement accepted (only body fails) | +0.1 |
| N unsolved goals remaining (fewer = better) | +0.3 × (1 - N/N_max) |
| Uses relevant mathlib lemmas (heuristic) | +0.1 |
| Type errors vs. unknown identifier errors (type errors are "closer") | +0.1 |
| No errors (proof complete) | = 1.0 |

### 5.3 LLM Mutation Prompt (example)

```
You are an expert Lean 4 mathematician working with mathlib.

THEOREM TO PROVE:
{theorem_statement}

CURRENT PROOF ATTEMPT (fitness: {fitness}):
{parent_genome}

LEAN ERROR OUTPUT:
{diagnostics.stderr}

PREVIOUSLY TRIED APPROACHES THAT FAILED:
{summary_of_failed_attempts}

Propose a MODIFIED proof that fixes the errors above. Think step by step:
1. What is the error telling us?
2. What mathlib lemmas might be relevant?
3. What tactic strategy should we try?

Return ONLY the proof body (everything after `by`), wrapped in ```lean fences.
```

### 5.4 Seed Population Strategies

1. **LLM cold start:** Ask the LLM to generate N diverse proof sketches for the target theorem
2. **Tactic templates:** Common proof patterns (`intro → apply → exact`, `induction → simp → ring`, etc.)
3. **Analogy mining:** Find similar proven theorems in mathlib and adapt their proof structures
4. **sorry-scaffolds:** Start with `sorry` placeholders and evolve them out

### 5.5 Behavior Descriptors (for MAP-Elites)

To maintain diversity, characterize proofs along dimensions like:

- **Tactic vocabulary:** which top-level tactics are used (set representation)
- **Proof depth:** number of tactic steps
- **Strategy class:** direct / by-contradiction / induction / cases

This prevents the population from collapsing to minor variations of one approach.

---

## 6. Backend: Turbulence Closure Evolution (`evoforge/backends/cfd/`)

### 6.1 Problem Formulation

**Goal:** Discover improved eddy viscosity or damping function expressions for sediment-laden oscillatory boundary layer models, specifically targeting the laminarization transition regime.

**Genome:** A symbolic mathematical expression representing a closure term, stored as a string in a simple DSL:

```
# Example closure genomes (eddy viscosity damping functions):

"1 - exp(-y_plus / 26)"                          # van Driest
"(1 + tanh(Ri_g - 0.25)) / 2"                    # Richardson-based
"exp(-alpha * Ri_g) * (1 - exp(-y_plus / A))"     # hybrid proposal
```

The DSL supports: arithmetic operators, elementary functions (`exp`, `log`, `tanh`, `sin`, `sqrt`, `abs`), and a fixed set of **physical variables** (`y_plus`, `Ri_g`, `Ri_f`, `Re_tau`, `C_s`, `omega_t`, `phi_s`).

### 6.2 Fitness Function

```python
def evaluate(self, genome: str) -> tuple[float, dict]:
    # Parse symbolic expression
    expr = self._parse_expression(genome)
    if expr is None:
        return 0.0, {"error": "parse_failure"}

    # Check dimensional consistency
    if not self._check_dimensions(expr):
        return 0.05, {"error": "dimensional_inconsistency"}

    # Plug into 1D solver, run against benchmark cases
    errors = []
    for case in self.benchmark_cases:
        try:
            result = self.solver.run(
                closure_expr=expr,
                boundary_conditions=case.bc,
                parameters=case.params,
                t_final=case.t_final,
            )
            # L2 error against DNS/experimental data
            err = np.linalg.norm(result.velocity - case.reference.velocity) / \
                  np.linalg.norm(case.reference.velocity)
            errors.append(err)
        except SolverDivergenceError:
            errors.append(1.0)  # max penalty

    mean_error = np.mean(errors)
    fitness = 1.0 / (1.0 + mean_error)  # maps (0, inf) → (0, 1]

    diagnostics = {
        "mean_error": mean_error,
        "per_case_errors": errors,
        "expression_complexity": self._complexity(expr),
        "converged_cases": sum(1 for e in errors if e < 1.0),
    }
    return fitness, diagnostics
```

**Multi-objective considerations:** fitness vs. expression complexity (Occam pressure). Can use NSGA-II style Pareto ranking or a weighted sum with configurable complexity penalty.

### 6.3 LLM Mutation Prompt (example)

```
You are an expert in turbulence modeling for sediment-laden flows.

PHYSICAL CONTEXT:
We are modeling eddy viscosity damping in oscillatory boundary layers
where suspended sediment causes density stratification. The key
phenomenon is laminarization — turbulence suppression by stable
stratification at high sediment concentrations.

AVAILABLE VARIABLES:
- y_plus: wall-normal distance in wall units
- Ri_g: gradient Richardson number (stratification strength)
- Ri_f: flux Richardson number
- Re_tau: friction Reynolds number
- C_s: volumetric sediment concentration
- omega_t: oscillation frequency × time
- phi_s: sediment settling flux

CURRENT BEST CLOSURE (fitness: {fitness}):
  {parent_genome}

PERFORMANCE:
  Mean L2 error: {diagnostics.mean_error:.4f}
  Worst case: case {worst_case_idx} (error: {worst_error:.4f})
  The model particularly struggles at {description_of_failure_regime}.

CONSTRAINTS:
- Expression must be dimensionally consistent
- Should reduce to standard van Driest in unstratified limit (Ri_g → 0)
- Should suppress turbulence as Ri_g → ∞
- Prefer parsimony (fewer terms)

Propose a modified closure expression. Explain your physical reasoning
briefly, then provide the expression in backticks.
```

### 6.4 Physical Constraint Enforcement

Unlike the Lean backend (where the type checker handles validity), the CFD backend needs explicit constraint checking:

```python
def _check_constraints(self, expr) -> list[str]:
    violations = []

    # Dimensional analysis via symbolic evaluation
    if not self._dimensionally_consistent(expr):
        violations.append("dimensional_inconsistency")

    # Asymptotic limits
    if not self._approaches_van_driest(expr, Ri_g=0):
        violations.append("wrong_unstratified_limit")

    if not self._suppresses_at_high_Ri(expr):
        violations.append("no_stratification_suppression")

    # Realizability: must be non-negative
    if self._can_go_negative(expr):
        violations.append("non_realizable")

    return violations
```

Constraint violations are included in diagnostics and fed back to the LLM during mutation.

### 6.5 Behavior Descriptors (for MAP-Elites)

- **Functional form class:** exponential / polynomial / rational / trigonometric / mixed
- **Dominant variable dependence:** which physical variable has highest sensitivity
- **Complexity bucket:** number of nodes in the expression AST

---

## 7. Configuration

All runs configured via TOML:

```toml
[run]
name = "levy_proof_search_001"
backend = "lean"
max_generations = 200
population_size = 30
seed = 42

[llm]
model = "claude-sonnet-4-5-20250929"
temperature_start = 1.0
temperature_end = 0.3
temperature_schedule = "linear"
max_tokens = 2048
mutation_retries = 3          # retry if LLM output doesn't parse

[selection]
method = "tournament"
tournament_size = 3
elitism = 2                   # top N survive unconditionally

[diversity]
strategy = "map_elites"       # or "none" for standard EA
grid_dims = [5, 5]            # dimensions of behavior space grid

[lean]                         # backend-specific config
project_dir = "./lean_project"
target_theorem = "levy_characteristic_continuous"
lake_timeout = 120
mathlib_version = "v4.15.0"

[cfd]                          # backend-specific config
solver_path = "./solver"
benchmark_dir = "./benchmarks"
complexity_penalty = 0.01
max_expression_depth = 8
```

---

## 8. Repo Structure

```
evoforge/
├── README.md
├── DESIGN.md                  ← this document
├── pyproject.toml
├── evoforge/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── individual.py      # Individual dataclass
│   │   ├── population.py      # PopulationManager
│   │   ├── selection.py       # tournament, lexicase, MAP-Elites
│   │   ├── archive.py         # SQLite archive + hall of fame
│   │   ├── engine.py          # main evolution loop
│   │   └── config.py          # TOML config parsing
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py          # LLM API wrapper (Anthropic, OpenAI)
│   │   ├── mutator.py         # LLMMutator class
│   │   └── prompt_templates/  # Jinja2 templates for prompts
│   │       ├── mutate.j2
│   │       ├── crossover.j2
│   │       └── reflect.j2
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py            # Backend ABC
│   │   ├── lean/
│   │   │   ├── __init__.py
│   │   │   ├── backend.py     # LeanBackend implementation
│   │   │   ├── parser.py      # Lean error message parser
│   │   │   ├── scorer.py      # partial proof scoring
│   │   │   └── prompts/       # Lean-specific prompt templates
│   │   └── cfd/
│   │       ├── __init__.py
│   │       ├── backend.py     # CFDBackend implementation
│   │       ├── dsl.py         # symbolic expression DSL parser
│   │       ├── constraints.py # dimensional analysis, realizability
│   │       ├── solver.py      # wrapper around 1D transport solver
│   │       └── prompts/       # CFD-specific prompt templates
│   └── viz/
│       ├── __init__.py
│       ├── genealogy.py       # lineage tree visualization
│       ├── fitness_plots.py   # convergence curves
│       └── map_elites.py      # behavior space heatmaps
├── configs/
│   ├── lean_levy.toml
│   ├── cfd_oscillatory.toml
│   └── cfd_steady.toml
├── lean_project/              # Lean 4 project with lakefile
│   ├── lakefile.lean
│   ├── lean-toolchain
│   └── EvoForge/
│       ├── Target.lean        # theorem statements to prove
│       └── Evolved.lean       # ← written by evoforge during runs
├── benchmarks/                # CFD benchmark cases
│   ├── case_01_clear_water/
│   ├── case_02_low_concentration/
│   └── case_03_high_concentration_laminarization/
├── tests/
│   ├── test_core/
│   ├── test_lean_backend/
│   └── test_cfd_backend/
└── scripts/
    ├── run_evolution.py       # CLI entry point
    ├── analyze_run.py         # post-hoc analysis of archived runs
    └── seed_from_mathlib.py   # mine mathlib for proof templates
```

---

## 9. MVP Milestones

### Phase 1: Core + Lean Backend (Weeks 1–3)

**Goal:** Evolve a proof for a simple mathlib lemma that doesn't exist yet.

| Week | Deliverable |
|------|-------------|
| 1 | Core engine: `Individual`, `PopulationManager`, `LLMMutator`, basic `Engine` loop. Unit tests. |
| 2 | Lean backend: `lake build` fitness evaluation, error parsing, partial scoring, mutation prompts. Integration test against a trivially provable lemma. |
| 3 | First real run: target a nontrivial lemma from the Lévy process formalization. Implement archive, logging, basic fitness curve visualization. |

**Success criterion:** The system finds a valid proof that `lake build` accepts for at least one non-trivial lemma, within 200 generations.

### Phase 2: CFD Backend (Weeks 4–6)

**Goal:** Rediscover (and ideally improve on) known damping functions for stratified flows.

| Week | Deliverable |
|------|-------------|
| 4 | Expression DSL: parser, dimensional analysis, constraint checker. Wrap existing 1D solver as fitness evaluator. |
| 5 | CFD backend integration. Seed population with known closures (van Driest, Munk-Anderson). First evolution run. |
| 6 | Multi-objective (fitness vs. complexity). Benchmark against known closures on the laminarization test cases from the existing paper. |

**Success criterion:** Evolution rediscovers van Driest-like damping within 100 generations when started from random expressions. Bonus: finds a novel expression that outperforms van Driest on the stratified cases.

### Phase 3: Quality-Diversity + Analysis (Weeks 7–8)

**Goal:** MAP-Elites for diverse solution discovery; analysis tooling.

| Week | Deliverable |
|------|-------------|
| 7 | MAP-Elites integration for both backends. Behavior descriptor implementations. |
| 8 | Visualization: genealogy trees, fitness landscapes, behavior space heatmaps. Analysis scripts for comparing LLM mutation effectiveness vs. random baselines. |

### Phase 4: Paper & Release (Weeks 9–10)

| Week | Deliverable |
|------|-------------|
| 9 | Ablation studies: LLM mutation vs. random, reflection vs. no reflection, temperature schedules. |
| 10 | Write-up. Clean repo for public release. |

---

## 10. Key Technical Risks & Mitigations

### Risk: LLM API costs blow up
**Mitigation:** Lean backend is cheap (short genomes, fast evaluation). CFD backend can use smaller/faster models (Haiku) for mutation and reserve Sonnet/Opus for reflection. Budget ~$0.50/generation for a 30-individual population. Implement genome caching aggressively — never re-evaluate or re-mutate identical genomes.

### Risk: Lean partial scoring is too noisy to guide evolution
**Mitigation:** The partial scoring heuristic (§5.2) is a starting point. If it doesn't provide enough gradient, fall back to LLM-as-judge: ask the LLM to rank two failing proofs by "closeness to correct." This is cheap and surprisingly calibrated for structured domains.

### Risk: CFD solver too slow for evolutionary loop
**Mitigation:** The 1D solver from laminarization-transition-study runs in seconds, not hours — this is why we're targeting 1D, not 3D DNS. For longer runs, implement asynchronous parallel evaluation with a process pool.

### Risk: LLM mutations are no better than random
**Mitigation:** This is the core hypothesis to test. The ablation study (Phase 4) directly measures this. Early signal from the Lean backend (where "random" tactic sequences almost never type-check) should give confidence. If LLM mutation underperforms, the reflection mechanism (§3.4) — showing the LLM population-level trends — may recover signal.

### Risk: Search space is too large for meaningful exploration
**Mitigation:** Seed populations with known-good starting points (existing mathlib proofs, established closures). Use the LLM to generate diverse-but-plausible seeds rather than random genomes. MAP-Elites prevents population collapse.

---

## 11. What Makes This Novel

Surveying the existing landscape (EvoPrompt, FunSearch, AlphaEvolve, OpenELM, LLaMEA):

1. **Formally-grounded fitness.** FunSearch and AlphaEvolve evolve code evaluated by running it — output correctness, not formal verification. We use a type checker (proof validity) and a physics simulator (conservation law adherence). Both are deterministic and non-gameable.

2. **Error-message-informed mutation.** No existing system feeds the oracle's *diagnostic output* back to the LLM as part of the mutation prompt. This is a qualitatively different signal from "this scored 0.3."

3. **Physical constraint injection.** The CFD backend embeds dimensional analysis and asymptotic behavior requirements directly into the LLM's mutation prompt. This is domain-specific EA guidance that has no analogue in current LLM+EA work.

4. **Unified framework across domains.** Most LLM+EA systems are built for one domain. Showing the same engine works for both theorem proving and physical model discovery — two structurally similar but superficially different problems — is a contribution in itself.

5. **Quality-diversity for proofs.** MAP-Elites has never been applied to formal theorem proving. Finding *multiple distinct valid proofs* (not just one) has pedagogical and mathematical value.

---

## 12. Dependencies

```
# Core
python >= 3.11
anthropic >= 0.40        # or openai, litellm for multi-provider
pydantic >= 2.0
tomli                     # TOML config
jinja2                    # prompt templates
sqlalchemy                # archive DB

# Lean backend
elan                      # Lean version manager
lean4 >= 4.14             # via elan
mathlib4                  # via lakefile

# CFD backend
numpy
scipy
sympy                     # symbolic expression handling + dimensional analysis

# Visualization
matplotlib
networkx                  # genealogy graphs
plotly                    # interactive behavior space heatmaps

# Dev
pytest
ruff
```

---

## 13. Open Questions

- **Should the reflection mechanism (population-level LLM analysis) run every generation or on a schedule?** Every generation is expensive. Every 5–10 generations with caching may suffice.

- **Multi-model strategy:** Use a fast model (Haiku) for routine mutations and a strong model (Opus) for reflection and hard cases? This mirrors AlphaEvolve's Flash+Pro approach.

- **Can we evolve the system prompt itself?** Meta-evolution where the LLM's domain instructions co-evolve with the population. Deeply recursive but potentially powerful.

- **Island model parallelism:** Run sub-populations with different LLM temperatures or models, with periodic migration. Natural fit for the framework.

- **Lean backend: tactic-level vs. term-level proofs?** Tactic mode is more natural for LLMs and easier to partially score. Term mode is more compositional. Could support both and let evolution discover which works better.

---

*This is a living document. Update as the project evolves.*
