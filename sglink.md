# evoforge -- The Developer's Field Guide

*An evolutionary engine that uses LLMs to breed theorem proofs.*

---

## 1. What Is Evoforge?

Imagine you are trying to prove a mathematical theorem in Lean 4. You sit
down, write `intro x`, then `simp`, and Lean tells you it did not work. You
try `ring` instead. Still stuck. You reorder the tactics, try a different
opener, combine two half-working approaches. Eventually, through trial and
error and flashes of insight, you close all the goals.

Evoforge automates that entire process.

It treats a **tactic proof** as a *genome* -- a sequence of instructions that
can be parsed, evaluated, mutated, and recombined -- and runs a full-blown
evolutionary algorithm over a population of candidate proofs. Some mutations
are cheap structural shuffles (swap two adjacent tactics, truncate at the last
working step). Others call an LLM -- ask it to look at the failing proof, the
diagnostics, the credit analysis, and write a better version.

The thesis is simple: **evolution gives you exploration breadth; LLMs give you
exploitation depth.** Cheap operators keep the population churning through
structural variations at near-zero cost, while LLM operators make the
high-value jumps that pure random mutation would never find. The engine blends
both and adapts the mixture as it learns what works.

The first domain backend is Lean 4 theorem proving, but the architecture is
designed so that the core engine never touches a single Lean-specific concept.
You could plug in a Python program synthesis backend, a circuit design backend,
or anything else that can be represented as a parseable, evaluable sequence.

---

## 2. Architecture Overview -- The Map

Think of evoforge as a small city with distinct neighborhoods connected by
well-defined roads. Here is the layout:

```
evoforge/
  core/           <-- The "downtown" -- engine, types, protocols
    engine.py         Main evolutionary loop (the mayor's office)
    types.py          Fitness, Individual, Credit, Reflection
    ir.py             IRProtocol -- the contract every genome must honor
    config.py         Pydantic config models + TOML loader
    selection.py      Tournament, NSGA-II, Lexicase, MAP-Elites
    mutation.py       MutationOperator ABC + MutationEnsemble
    evaluator.py      Async evaluator with 3-level cache
    population.py     PopulationManager -- dedup, diversity, ranking
    memory.py         SearchMemory -- patterns, failures, dead ends
    archive.py        SQLite-backed persistent storage (SQLAlchemy)
    scheduler.py      Budget & concurrency control
    identity.py       Parse -> canonicalize -> hash pipeline
    generator.py      4-stage validated LLM generation pipeline

  backends/       <-- Domain-specific plug-ins
    base.py           Backend ABC (the contract)
    lean/
      backend.py      LeanBackend facade
      ir.py           TacticSequence + TacticStep
      evaluator.py    Stepwise REPL evaluator + prefix caching
      operators.py    4 cheap structural mutation operators
      credit.py       Per-tactic credit assignment
      validation.py   Tactic whitelist + structural checks
      templates/      Jinja2 prompts for LLM operators
        system_prompt.j2
        mutation_prompt.j2
        crossover_prompt.j2
        reflection_prompt.j2

  llm/            <-- LLM integration layer
    client.py         Anthropic API wrapper with retry + cost estimation
    operators.py      LLMMutate + LLMCrossover operators

tests/            <-- Mirror structure, comprehensive coverage
  test_core/          Engine, identity, cache, mutation, population...
  test_lean/          Canonicalization, credit, operators, stepwise eval
```

The key insight in this layout: **everything flows through protocols and ABCs,
never through concrete types.** The engine imports `Backend` (the abstract
class), never `LeanBackend`. The identity pipeline imports `IRProtocol` (the
structural protocol), never `TacticSequence`. This is what makes evoforge
extensible without surgery.

---

## 3. The Four Invariants

There are four rules that the entire codebase is built around. Break any one
of them and the system falls apart. Think of them as the load-bearing walls of
the architecture.

### Invariant 1: Canonical Identity

Every genome goes through the **identity pipeline** before it enters the
population:

```
raw genome string
    --> backend.parse()          -> IRProtocol node
    --> ir.canonicalize()        -> normalized IR (whitespace collapsed,
                                    simp lemma lists sorted, skip removed)
    --> canonical_ir.structural_hash()  -> SHA-256 hex digest
```

Two genomes that are syntactically different but semantically identical (like
`simp [b, a]` vs `simp [a, b]`) produce the same hash. This means the
population **never contains semantic duplicates**. Every slot in the population
is a genuinely distinct candidate.

Why does this matter? Without it, you would waste evaluation budget testing the
same proof in different clothing. The identity pipeline is the bouncer at the
door -- it checks your ID before you enter.

The implementation lives in `evoforge/core/identity.py`:

```python
class IdentityPipeline:
    def process(self, genome: str) -> Individual | None:
        ir = self._backend.parse(genome)
        if ir is None:
            return None
        canonical_ir = ir.canonicalize()
        ir_hash = canonical_ir.structural_hash()
        return Individual(genome=genome, ir=canonical_ir, ir_hash=ir_hash, generation=0)
```

### Invariant 2: Backend Opacity

The engine never knows what domain it is operating in. It calls abstract
methods on a `Backend` instance:

- `parse()` -- turn a string into an IR node
- `evaluate()` / `evaluate_stepwise()` -- score an IR node
- `assign_credit()` -- localize where the fitness came from
- `seed_population()` -- create the starting genomes
- `mutation_operators()` -- what cheap operators are available
- `behavior_descriptor()` -- classify behavior for MAP-Elites

The full contract is 20 abstract methods defined in
`evoforge/backends/base.py`. This is
the seam between "how evolution works" and "what evolution is evolving." The
Lean backend implements every method; a future Python backend would implement
them differently but satisfy the same contract.

### Invariant 3: Fitness Is Multi-Objective

Fitness is never a bare float. It is always a `Fitness` dataclass:

```python
@dataclass(frozen=True)
class Fitness:
    primary: float                    # The main objective (0.0 to 1.0)
    auxiliary: dict[str, float]       # Named secondary objectives
    constraints: dict[str, bool]      # Hard constraint satisfaction
    feasible: bool                    # Are ALL constraints met?
```

For Lean, `primary` is the fraction of tactics that succeeded.
`auxiliary` tracks `steps_succeeded`, `goals_remaining`, and
`proof_complete`. A proof is only `feasible` when it closes all goals.

This multi-objective design means selection strategies can use different
objectives for different purposes. NSGA-II uses Pareto dominance across all
objectives. Lexicase shuffles the auxiliary keys and filters by each one in
turn. MAP-Elites bins by behavior descriptors. The engine supports all four
selection strategies out of the box, and they all work because `Fitness` gives
them a common, rich surface to work with.

### Invariant 4: Every Evaluation Is Cached

Evaluating a Lean proof means spawning a REPL subprocess and sending tactics
one by one. This is expensive. The system has a **3-level cache**:

- **Level 1 (L1):** In-memory parse cache. If we have already parsed this
  exact genome string, skip the parse. This catches identical raw strings.
- **Level 2:** Prefix cache. If we have already evaluated `intro x / simp` and
  now we see `intro x / simp / ring`, we can skip the first two REPL round-trips
  and start from the cached REPL state after step 2. This is huge -- most
  mutations only change the last few tactics, so the prefix is often identical.
- **Level 3 (L3):** Full evaluation cache in SQLite. Keyed by
  `(ir_hash, backend_version, config_hash)`. If the same canonical proof was
  evaluated in the same configuration, return the cached fitness immediately.
  This survives across process restarts because it is on disk.

The caching architecture is split between
`evoforge/core/evaluator.py` (L1 + L3
logic) and
`evoforge/backends/lean/evaluator.py`
(prefix caching at L2).

---

## 4. The Core Engine Loop

The engine (`evoforge/core/engine.py`)
runs a classic evolutionary loop with several non-obvious twists. Let me walk
through it generation by generation.

### Generation 0: Seeding

```python
seed_genomes = self.backend.seed_population(self.config.population.size)
```

The Lean backend has a hand-written **seed bank** of 31 starter tactic
sequences: things like `intro x / simp`, `intro x / ring`,
`intro x / norm_num`. These are educated guesses -- the kind of opening moves
a human would try first. The engine cycles through them to fill the requested
population size.

Each seed genome runs through the identity pipeline (parse, canonicalize,
hash), gets deduplicated, then gets evaluated concurrently via the async
evaluator. After evaluation, credit assignment runs: each tactic in the
sequence gets a localized credit score based on whether it closed goals,
maintained progress, or failed.

### Generations 1..N: The Evolutionary Loop

Each generation follows this pipeline:

```
Select parents  -->  Mutate each  -->  Identity + dedup  -->  Evaluate
    -->  Credit assignment  -->  Behavior descriptors  -->  Survive
    -->  Archive  -->  Memory update  -->  Stagnation check
```

Let me explain each stage.

**1. Selection.** The engine picks `k` parents from the current population
using the configured selection strategy. The default for Lean is lexicase
selection -- it shuffles the auxiliary fitness keys and progressively filters
the population by each one, which naturally maintains diversity across
different fitness dimensions.

**2. Mutation.** For each parent, the `MutationEnsemble` picks an operator
using weighted random selection. The operator gets a `MutationContext` that
includes the current generation number, the search memory (patterns, failures,
dead ends), a guidance string (used for crossover), the current temperature,
the backend reference, and the parent's credit history. It returns a new
genome string.

There are six operators in total:

| Operator | Type | What It Does |
|---|---|---|
| `PrefixTruncation` | cheap | Cut after the last positively-credited step |
| `TacticSwap` | cheap | Swap two adjacent tactics randomly |
| `TacticReorder` | cheap | Shuffle a small window (up to 3) of tactics |
| `SplicePrefixes` | cheap | Take prefix from parent A, suffix from parent B |
| `LLMMutate` | llm | Ask an LLM to rewrite the proof given diagnostics |
| `LLMCrossover` | llm | Ask an LLM to combine two parent proofs |

The cheap operators are defined in
`evoforge/backends/lean/operators.py`.
The LLM operators are in
`evoforge/llm/operators.py`.

Notice how `PrefixTruncation` uses credit analysis: it finds the last tactic
that got a positive credit score and cuts everything after it. This is like a
surgeon removing the gangrenous tissue -- keep what works, discard what does not.

**3. Identity + Dedup.** Every offspring genome goes through the identity
pipeline. If its canonical hash matches anything already in the population, it
is silently discarded. This is critical for preventing the population from
collapsing to a single solution when a successful mutation gets copied over
and over.

**4. Evaluate.** The async evaluator runs all novel offspring through the
backend's evaluation, respecting the concurrency semaphore (default: 4
concurrent evaluations) and the timeout (default: 60 seconds).

**5. Credit Assignment.** For each evaluated offspring, the backend assigns
per-location credit. In the Lean backend, this means per-tactic-step credit:

```python
# From evoforge/backends/lean/credit.py
if step_result.succeeded:
    reduction = len(goals_before) - len(goals_after)
    score = 0.3 * reduction + 0.1
else:
    score = -0.5
    break  # Stop crediting after first failure
```

The credit signals feed back into the next generation's mutation context.
When `PrefixTruncation` runs, it uses the credits to decide where to cut.
When `LLMMutate` runs, the credit summary appears in the prompt so the LLM
knows which steps are working and which are not.

**6. Behavior Descriptors.** Each offspring gets a behavior descriptor
computed by the backend. For Lean, it is a tuple of `(strategy_class,
depth_bucket)` -- the first tactic name and whether the proof is short/
medium/long. This is used by MAP-Elites selection to maintain diversity
across behavioral niches.

**7. Survival.** The selection strategy combines the current population and
the new offspring and selects survivors. Elitism is enforced: the top
`elite_k` individuals (by primary fitness) are always preserved. This
guarantees that the best solution found so far is never lost.

**8. Archive.** Every evaluated offspring is stored in the SQLite archive,
along with its lineage edge (which parent it came from, what operator was
used, what generation). This creates a complete evolutionary history that
can be queried after the run.

**9. Memory Update.** The search memory ingests the new offspring:
  - High-fitness genomes become **patterns** (things to try again).
  - Low-fitness genomes become **failure modes** (things to avoid).
  - Credits aggregate by tactic name (which tactics are generally useful).
  - If a genome has failed 3+ times, it becomes a **dead end**.
  - The best fitness per generation is tracked for stagnation detection.

**10. Stagnation Check.** If the best fitness has not improved for
`stagnation_window` generations (default: 10), the engine triggers
**reflection**: it bumps the LLM temperature by 0.1 (up to 1.5) to
encourage more creative mutations. This is the system's mechanism for
escaping local optima -- when you are stuck, take wilder swings.

### Stopping Conditions

The loop ends when:
- `max_generations` is reached (default: 100), or
- The budget is exhausted (LLM call count or USD cost limit), or
- The population empties (all genomes are invalid -- something went wrong).

The result is an `ExperimentResult` dataclass containing the best individual,
its fitness, the number of generations run, total evaluations, cost breakdown,
archive size, and whether reflection was triggered.

---

## 5. The Lean Backend -- How Theorem Proving Plugs In

The Lean backend is the first concrete domain implementation. It lives in
`evoforge/backends/lean/` and
implements all 20 abstract methods of the `Backend` ABC. Let me highlight the
most interesting parts.

### The IR: TacticSequence

A Lean proof genome is just a multi-line string where each line is a tactic.
The IR representation is a `TacticSequence` -- a list of `TacticStep` objects,
each holding the tactic name, its arguments, and the raw text.

```python
# From evoforge/backends/lean/ir.py
@dataclass(frozen=True)
class TacticStep:
    tactic: str   # "simp"
    args: str     # "[mul_comm]"
    raw: str      # "simp [mul_comm]"
```

Canonicalization does three things:
1. Collapses whitespace (`intro   x` becomes `intro x`).
2. Sorts lemma lists inside `simp [...]` brackets alphabetically.
3. Removes `skip` tactics (they are no-ops).

The structural hash is a SHA-256 of the canonical serialized form. This means
that `simp [b, a]` and `simp [a, b]` hash to the same value, which is exactly
what we want for deduplication.

### Stepwise Evaluation

The Lean evaluator talks to a Lean REPL subprocess. Instead of sending the
entire proof at once, it sends tactics **one at a time** and records the state
after each step:

```
Step 0: "intro x"    --> 1 goal remaining, succeeded
Step 1: "simp"       --> 0 goals remaining, succeeded  --> Proof complete!
```

Or:

```
Step 0: "intro x"    --> 1 goal remaining, succeeded
Step 1: "ring"       --> Error: tactic 'ring' failed  --> Stopped here
```

This stepwise approach is crucial for two reasons:
1. **Prefix caching.** If we already know that `intro x` succeeds and leaves
   REPL state #7, we can skip that step for any proof that starts with
   `intro x`.
2. **Fine-grained credit.** We know exactly which tactic failed and how many
   goals each tactic closed. This powers the credit assignment system.

The evaluator is in
`evoforge/backends/lean/evaluator.py`.
The REPL communication uses JSON over stdin/stdout -- each command is a
`{"tactic": "...", "proofState": N}` JSON object, and the response includes
goals, errors, and the new proof state index.

### Structural Validation

Before a genome enters the population, the validation layer checks it against
several rules:

- **Tactic whitelist:** Only known Lean 4 / Mathlib tactics are allowed.
  There is a curated `frozenset` of ~80 accepted tactics.
- **No `sorry`:** The `sorry` tactic is an escape hatch in Lean that marks a
  goal as "trust me." We explicitly ban it because the whole point is to find
  real proofs.
- **No unbounded `repeat`:** A `repeat` without `maxDepth` could loop forever
  and freeze the REPL.
- **Balanced delimiters:** Parentheses, brackets, braces, and angle brackets
  must match. Unbalanced delimiters will crash the parser.
- **Max tactic count:** No more than 100 steps. This prevents runaway genomes
  from consuming evaluation budget.

### Behavior Descriptors for MAP-Elites

The Lean backend defines a 2-dimensional behavior space:

| Dimension | Bins |
|---|---|
| Strategy | `intro`, `apply`, `simp`, `other` |
| Depth | `short` (1-3 steps), `medium` (4-8), `long` (9+) |

This gives a 4 x 3 = 12-cell grid. MAP-Elites selection maintains the best
individual in each cell, encouraging the population to explore different
*kinds* of proofs rather than converging on a single strategy.

### Seed Bank

The seed bank is a hand-curated list of 31 starter proofs:

```python
_SEED_BANK = [
    "intro x\nsimp",
    "intro x\nring",
    "intro x\nnorm_num",
    "intro x\nlinarith",
    "intro x\napply le_of_eq\nsimp",
    "intro x\napply norm_nonneg",
    "intro x\nexact le_refl _",
    "intro x\nomega",
    "intro x\npositivity",
    "intro x\nsimp [mul_comm]",
    "intro x\nring_nf\nsimp",
    "intro x\npush_neg\nsimp",
]
```

These are not random -- they are the common Lean proof openers that a human
would try first. Most start with `intro x` (introduce the variable) and then
try a different automation tactic. This gives the population a diverse starting
point across the behavior space.

---

## 6. Key Design Decisions (and Why They Matter)

### Decision 1: Credit Assignment as a First-Class Concept

Most evolutionary systems treat fitness as a single scalar and leave it at
that. Evoforge goes deeper: it assigns **localized credit** to each position
in the genome. In the Lean backend, every tactic step gets a credit score:

- Succeeded and closed goals: `+0.3 * goals_closed + 0.1`
- Succeeded but no progress: `+0.1`
- Failed: `-0.5` (and crediting stops)

This credit information flows into mutations:
- `PrefixTruncation` uses it to decide where to cut.
- `LLMMutate` includes it in the prompt so the LLM knows what is working.
- `SearchMemory` aggregates it by tactic name across the whole population.

**Why this matters:** Without credit assignment, the system would know that
a proof with `primary=0.67` is worse than one with `primary=1.0`, but it
would not know *which steps* are the problem. Credit turns a blunt fitness
signal into a surgical one.

This is analogous to the difference between "this code has a bug" (unhelpful)
and "the bug is on line 47 where the loop variable is off by one" (actionable).

### Decision 2: Cheap + LLM Operator Ensemble

The mutation ensemble mixes two fundamentally different operator types:

**Cheap operators** (no LLM call, microsecond latency):
- `PrefixTruncation`: credit-guided pruning
- `TacticSwap`: random adjacent swap
- `TacticReorder`: random window shuffle
- `SplicePrefixes`: credit-guided crossover

**LLM operators** (API call, seconds latency, costs money):
- `LLMMutate`: rewrite the proof with diagnostic context
- `LLMCrossover`: combine two proofs with credit guidance

The ensemble uses weighted random selection with three scheduling modes:
- **Fixed:** Weights never change.
- **Adaptive:** Every 10 total applications, weights shift toward operators
  with higher success rates.
- **Phased:** Reserved for future curriculum-based schedules.

Adaptive scheduling is the default. It means the system starts with a uniform
distribution over all operators, then naturally gravitates toward whatever is
working. If cheap operators are finding improvements, great -- save money. If
only the LLM can make progress, the LLM gets more weight.

The blend factor is `alpha = 0.3`:
```python
new_weight = current_weight * (1 - alpha) + success_rate * alpha
```

This is gentle enough that a single lucky success does not cause a whiplash
shift, but persistent enough that real trends get amplified over ~30
applications.

### Decision 3: Multiple Selection Strategies

The engine ships with four selection strategies, each suited to different
fitness landscapes:

**ScalarTournament.** The classic. Pick `k` random individuals, keep the
fittest. Simple, fast, and effective when fitness is a single scalar. Good
for getting started.

**ParetoNSGA2.** Non-dominated sorting with crowding distance. When you have
multiple objectives (steps succeeded, goals remaining, proof completeness),
NSGA-II finds the Pareto front -- the set of solutions where improving one
objective necessarily worsens another. Crowding distance breaks ties by
preferring solutions in sparser regions of objective space, maintaining
diversity.

**Lexicase.** Shuffles the auxiliary fitness keys and progressively filters
the population by each one. An individual survives as long as it is within
epsilon of the best on each successive criterion. This is excellent for
maintaining specialists -- a proof that is great at closing goals but long
might survive because it wins on the "goals_remaining" criterion even though
it loses on "steps_succeeded."

**MAP-Elites.** Maintains a grid of niches defined by behavior descriptors.
Each cell holds the single best individual for that behavior. Selection
samples uniformly from occupied cells, which means even low-fitness
individuals survive if they occupy an otherwise empty niche. This maximizes
behavioral diversity.

The Lean backend recommends lexicase selection, but the user can override this
in the TOML config.

### Decision 4: SQLite Archive for Everything

The archive (`evoforge/core/archive.py`)
uses async SQLAlchemy with aiosqlite to store four kinds of data:

1. **Individuals** -- full genome, IR hash, fitness, behavior descriptor.
2. **Evaluations** -- fitness cache keyed by `(ir_hash, backend_version,
   config_hash)`. Survives restarts.
3. **Prefix cache** -- cached REPL states for tactic prefixes.
4. **Lineage** -- parent-child edges with operator name and generation.

Why SQLite? Because it is zero-configuration, embedded, file-based, and ACID
compliant. You do not need to set up a database server. The archive file is
just a `.db` file you can copy, inspect with `sqlite3`, or back up trivially.
And because all access is async, the event loop never blocks on disk I/O.

The `UniqueConstraint` on `(ir_hash, backend_version, config_hash)` in the
evaluations table means you cannot accidentally store duplicate results. The
archive silently deduplicates on insert.

### Decision 5: Search Memory as LLM Context

The `SearchMemory` class is not just bookkeeping -- it is a **feedback
channel to the LLM**. When the LLM mutation operator constructs its prompt,
it includes a `prompt_section()` from the search memory:

```
Successful patterns:
  - intro x / simp (freq=5, avg_fit=0.85)
  - intro x / ring (freq=3, avg_fit=0.70)

Dead ends (avoid these):
  - intro x / sorry

Credit summary:
  - closed 1 goals: 12.5
  - maintained progress: 8.2

Recent failures:
  - intro x / omega (freq=2, last_gen=7)
```

This means the LLM is not starting from scratch every time -- it inherits
the *accumulated wisdom* of the entire evolutionary run. It knows what
patterns have worked, what tactics are consistently credited, and what to
avoid. This is a form of **in-context learning** across generations, and it
is one of evoforge's most powerful ideas.

---

## 7. Technology Choices

### Python 3.11+ with `uv`

The project requires Python 3.11+ (for `tomllib`, better typing, task groups)
and uses `uv` exclusively for package management. No bare `pip`. This gives
deterministic, fast installs and clean virtual environments.

### Pydantic for Config

All configuration is modeled with Pydantic v2:

```python
class EvoforgeConfig(BaseModel):
    population: PopulationConfig = PopulationConfig()
    selection: SelectionConfig = SelectionConfig()
    mutation: MutationConfig = MutationConfig()
    llm: LLMConfig = LLMConfig()
    eval: EvalConfig = EvalConfig()
    backend: BackendConfig = BackendConfig()
    evolution: EvolutionConfig = EvolutionConfig()
```

Config is loaded from TOML using `tomllib` (standard library in 3.11+) and
validated by Pydantic. This means you get:
- Type checking at load time (not at runtime when something crashes).
- Default values for everything (you can run with zero config).
- Clear error messages when a value is wrong.

The config hierarchy is flat and obvious. No nested inheritance, no
environment variable magic, no "config of configs."

### Anthropic SDK for LLM

The LLM client wraps the Anthropic Python SDK with exponential-backoff retry
(up to 3 attempts) and per-model cost estimation. It supports both sync
(`client.messages.create`) and async (`AsyncAnthropic`) interfaces.

Cost estimation uses a simple lookup table:

```python
_MODEL_PRICING = {
    "sonnet": (3.0, 15.0),     # per million tokens: (input, output)
    "haiku": (0.25, 1.25),
    "opus": (15.0, 75.0),
}
```

The scheduler tracks cumulative cost and stops the run when the budget is
exhausted. This prevents accidental $500 API bills when you leave a run
going overnight.

### SQLAlchemy + aiosqlite for Persistence

The archive uses SQLAlchemy's async engine with aiosqlite as the backend. The
ORM models are lightweight dataclass-style mapped columns. This gives us:
- Async I/O that does not block the event loop.
- Type-safe queries.
- Easy migration path if the schema changes.
- `sqlite3` CLI for ad-hoc inspection of results.

### Jinja2 for Prompt Templates

LLM prompts are Jinja2 templates stored alongside the backend:

```
evoforge/backends/lean/templates/
    system_prompt.j2       -- "You are an expert Lean 4 mathematician..."
    mutation_prompt.j2     -- Includes genome, diagnostics, credit, memory
    crossover_prompt.j2    -- Two parent genomes + credit analysis
    reflection_prompt.j2   -- Population state for self-reflection
```

Templating keeps the prompts out of the Python code and makes them easy to
iterate on without touching business logic. A new backend would bring its own
templates in its own directory.

### Ruff for Linting, Mypy for Types

The quality gate is:
- **Ruff** with `select = ["E", "F", "I", "N", "W", "UP"]` -- errors, pyflakes,
  import sorting, naming, warnings, and Python upgrade suggestions.
- **Mypy** in strict mode with `disallow_untyped_defs = true`.
- **pytest** with `pytest-asyncio` (auto mode) and 30-second timeout per test.

Line length is 99 characters -- wide enough for readable code, narrow enough
for side-by-side diffs.

### Hatchling Build System

The project uses Hatchling as the build backend (via `pyproject.toml`). This
is the modern standard -- no `setup.py`, no `setup.cfg`, just `pyproject.toml`.

---

## 8. Lessons and Patterns

### Lesson 1: Protocols Beat Inheritance for Extensibility

The `IRProtocol` is a `@runtime_checkable` structural protocol, not an
abstract base class:

```python
@runtime_checkable
class IRProtocol(Protocol):
    def canonicalize(self) -> Self: ...
    def structural_hash(self) -> str: ...
    def serialize(self) -> str: ...
    def complexity(self) -> int: ...
```

This means a new IR type just needs to implement these four methods -- it
does not need to inherit from anything. You can test protocol conformance
with `isinstance(my_ir, IRProtocol)` at runtime. This is Go-style duck typing
in Python, and it keeps the coupling between core and backends at an absolute
minimum.

The `Backend` ABC, on the other hand, uses traditional inheritance because
backends have more methods (13) and you want the type checker to catch missing
implementations at class definition time rather than at call time.

The general pattern: **use protocols for small contracts between subsystems,
ABCs for large contracts within a subsystem.**

### Lesson 2: Frozen Dataclasses Prevent Accidental Mutation

`Fitness`, `Credit`, `TacticStep`, `Pattern`, `FailureMode`,
`BehaviorDimension`, and `BehaviorSpaceConfig` are all `frozen=True`
dataclasses. Once created, they cannot be modified.

This matters because these objects flow through many hands -- the evaluator
creates a `Fitness`, the engine reads it, the selection strategy compares it,
the archive serializes it. If any of these steps could mutate the object,
debugging would be a nightmare. Frozen dataclasses make this impossible by
construction.

`Individual` is deliberately *not* frozen because its `fitness`, `credits`,
`behavior_descriptor`, and `generation` are set at different stages of the
pipeline. The engine uses `dataclasses.replace()` for immutable updates where
possible and direct assignment where necessary.

### Lesson 3: The Sentinel Pattern for Cache Misses

The evaluation cache uses a sentinel object to distinguish "not in cache"
from "cached but the value is None":

```python
_SENTINEL = object()

def parse_cached(self, genome: str, parse_fn):
    cached = self._parse_cache.get(genome, _SENTINEL)
    if cached is not _SENTINEL:
        return cached
    result = parse_fn(genome)
    self._parse_cache[genome] = result
    return result
```

Without the sentinel, a genome that fails to parse (returns `None`) would
be re-parsed on every call because `dict.get()` would return `None` and
the code could not tell whether that `None` came from the cache or from
a missing key. The sentinel pattern is a classic Python idiom -- worth
memorizing.

### Lesson 4: Semaphore-Based Backpressure

The async evaluator uses `asyncio.Semaphore(max_concurrent)` to limit how
many evaluations run simultaneously:

```python
async def _eval_with_sem(ind: Individual) -> Individual:
    async with self._semaphore:
        return await self.evaluate(ind)

tasks = [asyncio.create_task(_eval_with_sem(ind)) for ind in individuals]
results = list(await asyncio.gather(*tasks))
```

This is the right way to do bounded concurrency in asyncio. You create all
tasks eagerly (so they are registered with the event loop) but each one
waits for a semaphore slot before doing real work. No thread pools, no
manual chunking, no complex queue management.

The scheduler adds a second layer: separate semaphores for evals and LLM
calls, plus budget-based stopping. This two-layer approach means the engine
naturally throttles expensive operations without explicit coordination.

### Lesson 5: Adaptive Weights Need a Slow Blend

The mutation ensemble's adaptive scheduling uses a blend factor of `alpha=0.3`:

```python
new_weight = current_weight * 0.7 + success_rate * 0.3
```

Early in development, the blend factor was 0.7 (aggressive). This caused
whiplash: one lucky cheap mutation would shift all weight to cheap operators,
then the population would stagnate because cheap operators cannot make
creative jumps. The LLM weight would drop so low that it would almost never
get selected.

The fix was to slow down the adaptation. At `alpha=0.3`, it takes about
30 applications for the weights to significantly shift, which gives enough
signal to distinguish real trends from noise.

The lesson: **adaptive systems need damping.** A thermostat that reacts to
every draft will oscillate wildly. One that averages over the last 10 minutes
will keep the room comfortable.

### Lesson 6: Dead End Detection Prevents Wasted Effort

Search memory marks a genome as a "dead end" after it has failed 3 or more
times. Dead ends appear in the LLM prompt with a warning:

```
Dead ends (avoid these):
  - intro x / sorry
  - intro x / decide
```

Without this, the LLM would keep suggesting the same failing approaches
because it has no memory of previous failures. The dead end list acts as a
**negative prior** -- it tells the LLM what not to do, which is often more
valuable than telling it what to do.

### Lesson 7: Stagnation Response = Temperature Bump

When the search stagnates (best fitness unchanged for 10 generations), the
engine bumps the LLM temperature:

```python
self._temperature = min(self._temperature + 0.1, 1.5)
```

Temperature controls the randomness of LLM output. At 0.7 (default), the
LLM makes reasonable, conservative suggestions. At 1.5 (max), it gets
creative -- sometimes nonsensically so, but sometimes brilliantly.

This is a simple but effective escape mechanism. When you are stuck in a
local optimum, you need to take bigger jumps. The temperature cap at 1.5
prevents the LLM from becoming completely incoherent.

A more sophisticated approach would use the reflection prompt template to
ask the LLM to diagnose the population state and suggest structural changes.
The template exists (`reflection_prompt.j2`) but the reflection pipeline is
not yet fully wired into the engine -- that is future work.

### Lesson 8: Lineage Tracking Enables Post-Hoc Analysis

Every parent-child relationship is stored in the lineage table:

```python
await self.archive.store_lineage(
    parent_hash=parent_hash,
    child_hash=child_hash,
    operator_name="mutation",
    generation=gen,
)
```

This creates a complete family tree of every individual in the evolutionary
run. After a run completes, you can query the archive to answer questions like:
- "What was the evolutionary path from the seed to the best solution?"
- "Which mutation operator produced the most successful offspring?"
- "Did LLM mutations tend to come from cheap-mutation parents?"

This is invaluable for understanding *why* evolution found what it found, not
just *what* it found.

### Lesson 9: Subprocess Buffering Will Ruin Your Day

When we first tried to talk to the Lean REPL interactively from Python, it
looked dead simple: start `lake env repl`, write JSON to stdin, read JSON from
stdout. What actually happened: the REPL processed the command instantly, wrote
the response... and the response sat in a 4KB C runtime buffer, never reaching
Python. `readline()` hung forever.

The root cause: when a C program's stdout is a **pipe** (as with
`subprocess.PIPE`), the C runtime defaults to **block buffering** -- it
accumulates output in a 4KB buffer and only flushes when the buffer is full or
the process exits. When stdout is a **terminal**, it defaults to **line
buffering** -- it flushes after every `\n`. Our `echo '...' | lake env repl`
test worked because the REPL read EOF from stdin, exited, and the exit flushed
the buffer. But interactive long-lived use? Dead silence.

The fix: create a **pseudo-terminal** (pty) for the child's stdout using
`pty.openpty()`. The child process sees a terminal, uses line buffering, and
data flows immediately. We disable echo with `termios` (otherwise the pty
echoes our input back as output, corrupting the JSON stream) and wrap the
master fd in an `asyncio.StreamReader` for ergonomic async reads.

Other things we learned along the way:
- The REPL protocol separates commands with **blank lines** (`\n\n` after JSON)
- Error responses use `{"message": "..."}`, not `{"severity": "error"}`
- `norm_num` needs Mathlib imported; `decide` and `simp` work out of the box
- The REPL binary lives at `.lake/packages/repl/.lake/build/bin/repl`, not
  `.lake/build/bin/repl`

This is a general lesson: **if you are launching a subprocess for interactive
I/O and nothing comes back, it is almost certainly buffering.** Use a pty, or
`stdbuf -oL` on Linux, or pipe through `script -q /dev/null` on macOS.

---

## 9. How to Run It

### Setup

```bash
# Clone and enter the project
cd evoforge

# Install dependencies with uv
uv sync

# Install dev dependencies
uv sync --group dev
```

### Run Tests

```bash
# Full test suite
uv run pytest

# With coverage
uv run coverage run -m pytest
uv run coverage report

# Just core tests
uv run pytest tests/test_core/

# Just Lean tests
uv run pytest tests/test_lean/
```

### Quality Gate

```bash
# Linting
uv run ruff check .

# Type checking
uv run mypy evoforge/

# All three in sequence
uv run ruff check . && uv run mypy evoforge/ && uv run pytest
```

### Configuration

Create a TOML config file:

```toml
[population]
size = 50
elite_k = 5

[selection]
strategy = "lexicase"

[mutation]
schedule = "adaptive"
llm_weight = 0.6
cheap_weight = 0.4

[llm]
model = "claude-sonnet-4-20250514"
temperature = 0.7
max_calls = 1000
max_cost_usd = 50.0

[eval]
max_concurrent = 4
timeout_seconds = 60.0

[backend]
name = "lean"
theorem_statement = "theorem my_thm : forall x : Nat, x + 0 = x"
project_dir = "/path/to/lean/project"

[evolution]
max_generations = 100
stagnation_window = 10
```

Load it in code:

```python
from evoforge.core.config import load_config

config = load_config("experiment.toml")
```

### Running with Lean (Integration Test)

The integration test needs the Lean REPL built in a sibling `LeanLevy` project:

```bash
# Build the REPL (first time downloads mathlib -- takes a few minutes)
cd ../LeanLevy
lake update && lake build repl
cd ../evoforge

# Run all tests including the real Lean integration test
uv run pytest -x -v

# Override the LeanLevy location if it's not a sibling directory
LEAN_PROJECT_DIR=/path/to/LeanLevy uv run pytest -x -v
```

The test proves `1 + 1 = 2` by sending the `decide` tactic to the real REPL
and verifying fitness = 1.0, success = true. It conditionally skips if the
REPL binary isn't found.

### Running an Experiment

```python
import asyncio
from evoforge.core.archive import Archive
from evoforge.core.config import load_config
from evoforge.core.engine import EvolutionEngine
from evoforge.backends.lean.backend import LeanBackend
from evoforge.llm.client import LLMClient

async def main():
    config = load_config("experiment.toml")

    backend = LeanBackend(
        theorem_statement=config.backend.theorem_statement,
        project_dir=config.backend.project_dir,
    )

    archive = Archive("sqlite+aiosqlite:///results.db")
    await archive.create_tables()

    llm_client = LLMClient()  # uses ANTHROPIC_API_KEY env var

    engine = EvolutionEngine(
        config=config,
        backend=backend,
        archive=archive,
        llm_client=llm_client,
    )

    result = await engine.run()
    print(f"Best fitness: {result.best_fitness}")
    print(f"Generations: {result.generations_run}")
    print(f"Total evaluations: {result.total_evaluations}")
    print(f"Reflected: {result.reflected}")

asyncio.run(main())
```

---

## 10. What's Next

### Near-Term

- **Full reflection pipeline.** The reflection prompt template exists but is
  not wired into the engine. When triggered by stagnation, the engine should
  call the LLM with a population summary and use the `Reflection` dataclass
  (strategies to try, strategies to avoid, useful primitives, suggested
  temperature) to reconfigure the search.

- **Richer crossover.** The `LLMCrossover` operator currently falls back to
  mutation when guidance is not available. The engine loop should explicitly
  select a second parent for crossover contexts.

- **Checkpoint/resume.** ✅ Done! See Section 14 below.

### Medium-Term

- **Additional backends.** The architecture is ready for Python program
  synthesis, regex construction, circuit optimization, or any other domain
  where solutions can be parsed into an IR, evaluated for fitness, and
  mutated structurally.

- **Distributed evaluation.** The async evaluator currently runs on one
  machine. For expensive backends (Lean REPL), distributing evaluations
  across multiple machines would dramatically increase throughput.

- **Hyperparameter sweeps.** The TOML config makes it easy to run parameter
  sweeps over population size, mutation weights, selection strategy, etc.
  Adding a sweep runner that schedules multiple experiments and compares
  results would be a natural extension.

### Long-Term

- **Self-improving prompts.** Instead of static Jinja2 templates, the system
  could use the LLM to improve its own prompts based on what types of
  mutations produce the best results.

- **Multi-theorem campaigns.** Run evoforge across a corpus of theorems,
  letting the search memory transfer between runs. Tactics that work on one
  theorem might transfer to related theorems.

- **Formal verification of the pipeline.** The ultimate irony would be to
  use Lean to prove properties about evoforge itself -- like "the identity
  pipeline is idempotent" or "survival preserves elites."

---

## Appendix A: Data Flow Diagram

```
                    +-----------+
                    |   TOML    |
                    |  Config   |
                    +-----+-----+
                          |
                          v
                  +-------+--------+
                  | EvoforgeConfig |
                  +-------+--------+
                          |
         +----------------+----------------+
         |                                 |
         v                                 v
  +------+------+                  +-------+-------+
  |   Backend   |                  | EvolutionEngine|
  | (LeanBackend)|                 +-------+-------+
  +------+------+                          |
         |                                 |
    +----+----+              +-------------+-------------+
    |         |              |             |             |
    v         v              v             v             v
 parse()  evaluate()   PopulationMgr  Selection   MutationEnsemble
    |         |              |         Strategy       |    |
    v         v              |             |          v    v
 IRProtocol  Fitness         |             |       cheap  LLM
    |         |              |             |     operators operators
    v         v              v             v          |
 Identity  Credit        Archive      Survivors       |
 Pipeline  Assignment       |             |           |
    |         |              |             |           |
    v         v              v             v           v
 Individual   |          SQLite        Next Gen    LLMClient
    |         |           (.db)           |      (Anthropic API)
    +----+----+                           |
         |                                |
         +----------> SearchMemory <------+
                     (patterns, failures,
                      dead ends, credits)
```

---

## Appendix B: Key File Reference

| File | Purpose | Lines |
|---|---|---|
| `evoforge/core/engine.py` | Main evolutionary loop | ~833 |
| `evoforge/core/types.py` | Core data types (Fitness, Individual, Credit) | ~97 |
| `evoforge/core/ir.py` | IRProtocol + behavior space config | ~38 |
| `evoforge/core/config.py` | Pydantic config models + TOML loader | ~166 |
| `evoforge/core/selection.py` | 4 selection strategies | ~408 |
| `evoforge/core/mutation.py` | MutationOperator ABC + ensemble | ~205 |
| `evoforge/core/evaluator.py` | Async evaluator + 3-level cache | ~168 |
| `evoforge/core/population.py` | Population manager with diversity metrics | ~111 |
| `evoforge/core/memory.py` | Search memory (patterns, failures, dead ends) | ~296 |
| `evoforge/core/archive.py` | SQLite archive (individuals, evals, lineage) | ~377 |
| `evoforge/core/scheduler.py` | Budget + concurrency control | ~107 |
| `evoforge/core/identity.py` | Parse -> canonicalize -> hash pipeline | ~43 |
| `evoforge/core/generator.py` | 4-stage validated LLM generation | ~166 |
| `evoforge/backends/base.py` | Backend ABC (20 abstract methods) | ~136 |
| `evoforge/backends/lean/backend.py` | LeanBackend facade | ~489 |
| `evoforge/backends/lean/ir.py` | TacticSequence + TacticStep | ~217 |
| `evoforge/backends/lean/evaluator.py` | Stepwise REPL evaluator | ~441 |
| `evoforge/backends/lean/operators.py` | 4 cheap mutation operators | ~126 |
| `evoforge/backends/lean/credit.py` | Per-tactic credit assignment | ~51 |
| `evoforge/backends/lean/validation.py` | Tactic whitelist + structural checks | ~150 |
| `evoforge/llm/client.py` | Anthropic API wrapper with retry | ~146 |
| `evoforge/llm/operators.py` | LLMMutate + LLMCrossover | ~106 |

---

## Appendix C: The Mental Model

If you need a single analogy to hold the whole system in your head:

**Evoforge is a breeding program for proofs.**

- The **population** is the herd.
- The **fitness function** is the judge at a livestock show.
- The **cheap operators** are random genetic variation -- most do nothing useful,
  but occasionally one hits gold.
- The **LLM operators** are like bringing in a veterinary geneticist who can
  look at the herd, understand the bloodlines, read the health reports, and
  make targeted breeding recommendations.
- The **search memory** is the breeder's notebook -- what crosses worked, what
  lines are dead ends, which traits run in which families.
- The **archive** is the stud book -- a permanent record of every animal,
  every cross, every generation.
- **Stagnation detection** is the breeder noticing that the last 10 generations
  all look the same and deciding to introduce some wildcard genetics.
- **Elitism** is keeping the best animals no matter what -- never selling your
  champion.

The goal is the same as any breeding program: start with a diverse population,
apply selection pressure, keep what works, discard what does not, and
eventually converge on individuals that satisfy your criteria. The difference
is that evoforge's "animals" are sequences of mathematical reasoning steps,
and its "criteria" is formal logical correctness.

That is evoforge. Breed better proofs.

---

## 13. The Wiring Session -- Making It Actually Run

Phase 1 built all the components in isolation: the REPL evaluator, the backend
ABC, the engine loop, the LLM operators, the archive, the config system. They
were thoroughly tested (323 tests!). But they were *disconnected*. The engine
called `backend.evaluate()` which raised `NotImplementedError` for Lean. Five
config sections (reflection, memory, scheduler, diversity, ablation) were parsed
from TOML but had zero runtime effect.

This is a common pattern in software development: you build all the parts
correctly, then discover that the last 10% of work -- the wiring -- is where
the real complexity hides.

### What We Wired

**Backend lifecycle (startup/shutdown).** The `Backend` ABC gained two new
non-abstract async methods: `startup()` and `shutdown()`. The Lean backend
implements `startup()` to create the REPL process, send an initialization
command (the theorem statement with `sorry`), and construct the
`LeanStepwiseEvaluator`. The engine calls `startup()` at the top of `run()`
and `shutdown()` in a `try/finally` block so resources are cleaned up even if
the run crashes.

**Async evaluate.** `Backend.evaluate()` and `evaluate_stepwise()` became
`async def`. This was required because the Lean REPL is an async subprocess.
The `LeanBackend.evaluate()` acquires a lock (`self._repl_lock`), then
delegates to the `LeanStepwiseEvaluator`. The lock is critical because the REPL
is a single-threaded process -- concurrent tactic commands would interleave
nonsensically.

**Config wiring.** Every TOML parameter now has a runtime effect:

| Config Section | What It Controls |
|----------------|-----------------|
| `scheduler.llm_budget_per_gen` | Per-generation LLM call cap; engine falls back to cheap ops when exhausted |
| `scheduler.max_llm_concurrent` | Semaphore size for concurrent LLM calls |
| `reflection.interval` | Periodic reflection every N generations (in addition to stagnation-triggered) |
| `reflection.include_top_k` | How many top individuals to include in reflection prompts |
| `memory.max_patterns` / `max_dead_ends` | Caps on `SearchMemory` data structures |
| `diversity.strategy` | Behavior descriptors only computed when `== "map_elites"` |
| `ablation.disable_*` | Six feature flags that surgically disable components |

**Ablation flags.** These are the jewel of the config system. When you are
running experiments to understand which component contributes what, you can flip
flags like `disable_llm = true` or `disable_credit = true`. Each flag is wired
at the exact point where that component acts:
- `disable_llm`: No LLM operators added to the ensemble, no generator created
- `disable_cheap_operators`: No cheap operators (with a fallback -- can't have
  *zero* operators)
- `disable_reflection`: Skips both stagnation-triggered and periodic reflection
- `disable_memory`: Skips `SearchMemory.update()` calls
- `disable_credit`: Skips `assign_credit()` calls
- `disable_diagnostics`: (Ready for use in mutation prompts)

### Lessons Learned

**The `try/finally` pattern is non-negotiable for resource management.** The
REPL is an external process. If the engine crashes mid-run and you don't shut
down the REPL, you leak child processes. Python's `try/finally` guarantees
cleanup regardless of how the `try` block exits.

**Lock serialization for shared resources.** The REPL accepts one command at a
time. Even though the engine evaluates individuals concurrently (via
`asyncio.gather`), each evaluation must acquire the REPL lock. This is a
bottleneck, but correctness trumps speed. A future optimization would be to
run multiple REPL instances.

**Ablation requires surgical placement.** You can't just wrap the entire engine
in an `if not ablated` check. Each feature (credit, memory, reflection, LLM)
touches different parts of the loop. The flags need to be checked at each
specific call site.

**Test count: 323 → 330.** Seven new tests cover lifecycle (startup/shutdown
called, shutdown on error), ablation flags, per-gen LLM budget, and periodic
reflection. The full suite runs in ~11 seconds.

---

## 14. Progress Tracking + Checkpoint/Resume

Long evolutionary runs (200+ generations) need two things: visual feedback so
you know it's working, and crash recovery so you don't lose hours of compute.

### Rich Progress Bar

The engine now wraps the generation loop with a `rich.Progress` bar:

```
⠋ Evolving ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 42/200 00:03:21 00:04:47 gen 42 | best=0.6700 | pop=50 | div=3.21 | evals=2100
```

The import is wrapped in `try/except` so the engine still works if `rich` is
not installed (it just falls back to logger.info lines). The progress bar shows:
spinner, "Evolving" label, bar, M/N complete, elapsed time, ETA, and a
live status string with the key metrics.

### Checkpoint System

Every `checkpoint_every` generations (default 10), the engine serializes its
entire mutable state into a JSON blob and stores it in the archive:

```python
state = {
    "generation": generation,
    "total_evaluations": self._total_evaluations,
    "reflected": self._reflected,
    "temperature": self._temperature,
    "temperature_boost": self._temperature_boost,
    "dedup_count": self._dedup_count,
    "total_offspring_attempted": self._total_offspring_attempted,
    "population_hashes": [ind.ir_hash for ind in self.population.get_all()],
    "memory": self._memory.to_dict(),
    "ensemble": self._ensemble.to_dict(),
    "cost_summary": self._scheduler.tracker.summary(),
}
```

This captures everything needed to restart the run: scalar counters, population
membership (by ir_hash), search memory state (patterns, failures, dead ends,
credit aggregates), and mutation ensemble weights/stats.

### Resume Flow

When `--resume` is passed:
1. The engine loads the latest checkpoint from the archive
2. Restores all scalar state (temperature, dedup count, etc.)
3. Restores `SearchMemory` via `from_dict()` (patterns, failures, dead ends)
4. Restores `MutationEnsemble` via `from_dict()` (weights, operator stats)
5. Rebuilds the population by looking up each ir_hash in the archive, then
   re-parsing through the identity pipeline to restore IRs
6. Sets `_start_generation = checkpoint_gen + 1`
7. **Skips seeding entirely** — the population already exists

The population rebuild is the clever part: the archive stores genomes and
fitness, but not the parsed IR (which is backend-specific and not easily
serializable). So on resume, each individual is re-parsed through
`IdentityPipeline.process()` to reconstruct its IR, then the archived fitness
and metadata are copied back onto it.

### File-Backed Archive

The CLI now defaults to a file-backed SQLite archive instead of in-memory:

```bash
# First run — creates runs/default/archive.db
bash scripts/run_with_keychain.sh --output-dir runs/exp1

# Resume after crash
bash scripts/run_with_keychain.sh --output-dir runs/exp1 --resume
```

The archive persists individuals, evaluations, lineage, prefix cache, AND
checkpoints in a single `.db` file. This means the fitness cache also survives
restarts — individuals that were already evaluated don't need re-evaluation.

### Serialization Design

Both `SearchMemory` and `MutationEnsemble` gained `to_dict()` / `from_dict()`
methods. The key design choice: these are plain-dict serialization, not pickle.
JSON is human-readable, debuggable, and doesn't break when you refactor the
classes. You can `sqlite3 archive.db` and inspect a checkpoint with `jq`.

For `MutationEnsemble`, weights are keyed by operator name (not index). This
means if you add or remove operators between runs, the resume gracefully handles
it: known operators get their saved weights restored, new operators get default
weights, and everything is renormalized.

### Lessons

**Checkpoint state must be complete.** Missing even one field (say,
`_temperature_boost`) means the resumed run behaves differently from where it
left off. The test `test_checkpoint_contains_required_fields` explicitly checks
that all 11 required keys are present.

**Population restore needs the identity pipeline.** You can't just store and
reload `Individual` objects because the `ir` field contains a backend-specific
AST that isn't JSON-serializable. Instead, store the ir_hash, look up the
genome from the archive, and re-parse it. This is slower but correct and
backend-agnostic.

**Cost tracker is intentionally NOT restored.** The checkpoint saves
`cost_summary` for audit trail, but doesn't restore it on resume. This gives
you a fresh budget for each resumed session — if you killed a $50 run at $30,
you get another $50 on resume, not $20. This is the right default for
interactive use.

**Test count: 330 → 359.** 29 new tests cover checkpoint save/load, resume
flow, memory serialization roundtrip, ensemble serialization roundtrip, and
integration scenarios (resume skips seeding, no-checkpoint returns false).

---

## 12. The False Positive Catastrophe and the Verification Gate

Here's a cautionary tale about trusting your evaluation pipeline too much.

We ran evoforge against `norm_le_one` — a theorem about positive definite
functions where `‖φ(ξ)‖ ≤ 1`. The engine reported fitness=1.0 for the proof
`ring`. Success! Except... `ring` can't possibly prove a norm inequality from
positive definiteness hypotheses. That's like claiming 2+2=fish and the
compiler nodding approvingly.

**What happened:** The Lean REPL's interactive `tactic` command and the Lean
compiler (`lake env lean`) don't always agree. The REPL might report "no goals
remaining" in an intermediate proof state that doesn't actually constitute a
valid proof when compiled as a standalone file. This is the REPL equivalent of
a "works on my machine" bug.

**The fix — a two-layer verification system:**

1. **REPL evaluation** (fast, used for fitness scoring during evolution): The
   stepwise evaluator sends tactics one by one and checks if `goals == []`. Now
   with DEBUG-level logging of every raw REPL response so you can diagnose
   mismatches.

2. **Lake compilation** (slow, gold-standard): `LeanBackend.verify_proof()`
   writes the complete proof to a temp file in the Lean project and runs
   `lake env lean <file>`. Only if compilation succeeds (exit code 0) is the
   proof considered truly verified.

3. **Engine gate with Deb's feasibility constraint**: When any individual
   reaches fitness=1.0, the engine calls `backend.verify_proof()`. If
   verification fails, the individual is marked `feasible=False` — its
   primary fitness stays at 1.0 (preserving genetic material for mutation),
   but selection treats it as `-inf` via Deb's (2000) constraint dominance
   principle: any feasible solution always beats any infeasible one. The
   engine only triggers early exit on *verified, feasible* proofs.

The `verify_proof()` method lives on the `Backend` ABC with a default `True`
return (for backends without formal verification). The Lean backend overrides
it with the real lake compilation check.

**Lesson learned:** In formal verification, your evaluation function IS your
product. If you can't trust your fitness signal, evolution will happily optimize
for whatever artifact in the evaluator gives 1.0 — Goodhart's Law applied to
theorem proving.

---

## 13. Structured Proofs — Beyond Flat Tactic Lines

Real Lean 4 proofs aren't flat sequences of tactics. They use nesting:

```lean
by_cases h : φ ξ = 0
· simp [h]          -- focused block for the zero case
· have := ...       -- focused block for the nonzero case
  linarith
```

The original parser split on every newline, treating `· simp [h]` as a
separate top-level tactic. This broke the semantics — a focused `·` block only
makes sense as a child of its parent `by_cases`.

**The block-aware parser:** `parse_tactic_sequence()` now groups lines
intelligently:
- Lines starting with whitespace or `·` are continuations of the current block
- Each top-level tactic (including its children) becomes one `TacticStep`
- The `raw` field can be multi-line: `"by_cases h : x = 0\n· simp [h]\n· ring"`

This means mutation operators (swap, reorder, truncate) work at the
**block level** — they won't accidentally pull a `· simp` out of its
`by_cases` parent. The evaluator sends the entire multi-line block as a single
tactic command to the REPL, which handles it correctly.

Canonicalization was also updated: `_normalize_block()` applies `_normalize_line()`
to each line within a block (including sorting simp lemma lists inside `·` branches),
while preserving the block structure.

---

## 14. Richer LLM Prompts — Giving the AI Eyes

The original prompts were like giving a mechanic a car with the hood welded
shut. They knew the theorem statement and the tactic names, but had no idea
what was mathematically happening or what remained to be proved.

**System prompt enhancements:**
- **Mathematical context**: If the theorem involves `IsPositiveDefinite`, the
  prompt now explains what PD functions are, their key properties (Hermitian
  symmetry, φ(0) real, |φ(x)| ≤ φ(0)), and relevant lemma names from the
  LeanLevy library.
- **Proof strategy patterns**: `by_cases`, `calc`, `suffices`, `have` — with
  concrete usage guidance for norm inequalities.
- Context is auto-derived from the theorem statement and imports, so it adapts
  to different target theorems.

**Mutation prompt enhancements:**
- **Goal state**: The prompt now shows the actual Lean goals remaining after the
  last successful tactic. This is the single most important piece of information
  for the LLM — it tells the AI *what remains to be proved*, not just "step 3
  failed." The goal types and contexts come from `LeanDiagnostics.goal_types`
  and `goal_contexts`, threaded through to the Jinja2 template.
- **Restructuring guidance**: Instead of just "fix the failing step," the prompt
  suggests alternative approaches (restructure with `have`, try `calc`, etc.).

---

## 15. Configurable Seed Bank

The hardcoded seed bank worked fine for generic theorems, but `norm_le_one`
needs domain-specific starting points. The seed bank is now configurable:

- `BackendConfig.seeds: list[str]` — an optional list of theorem-specific seeds
  in the TOML config
- If provided, config seeds are **prepended** to the default bank, so they
  appear first in the initial population
- `lean_default.toml` now includes 5 seeds specific to `norm_le_one`:
  PD matrix inequality, Hermitian symmetry, by_cases on φ(ξ)=0, etc.

The default bank also expanded from 24 to 31 seeds, adding multi-step
structured patterns: `by_cases` with focused blocks, `have` introductions,
`calc` chains, `suffices` goals.

---

## 16. CLI Verification with `--verify`

`scripts/run.py` now accepts `--verify`. When set and a proof is found
(fitness ≥ 1.0), it runs `lake env lean proof.lean` in the Lean project
directory as a final gold-standard check. This is belt-and-suspenders on top
of the engine's internal verification gate.

**Test count: 359 → 392 → 405.** 33 new tests cover proof verification (mock
subprocess), engine downgrade behavior, configurable seed bank, structured
tactic parsing (by_cases, calc, indentation), canonicalization of multi-line
blocks, mutation prompt goal state inclusion, and system prompt math context.
13 more tests cover behavior descriptors, feasibility constraints, and
population floor enforcement.

---

## 17. The Population Collapse Bug and Three Fixes

After all the infrastructure was in place, we ran the engine for real and
watched the population plummet from 30 → 5 → 2 within a few generations,
then get stuck at `pop_size=2, diversity=0.0000, best_fitness=0.9500` for
the remainder of the run. Three bugs conspired:

### Bug 1: Silent dedup death spiral

`Lexicase.survive()` returns 30 individuals by sampling with replacement.
Many of those 30 have the *same* `ir_hash`. When the engine rebuilds the
`PopulationManager` from survivors, `add()` silently rejects duplicates.
So 30 survivors become 5 unique individuals. Next generation: 5 parents
produce offspring, many of which are also duplicates. Population shrinks
further. It's a death spiral.

**Fix: `_refill_population()`** — After survive+rebuild, if the population
is below target, the engine refills in two phases:
1. **Fresh seeds** from `backend.seed_population()` — cheap, adds diversity
2. **Cheap mutations** on existing individuals — if seeds are all duplicates

Both phases share a common `_evaluate_and_fill()` pipeline that processes
genomes through evaluate → credit → verify → assign descriptors → add.

### Bug 2: Diversity always reads zero

`diversity_entropy()` computes Shannon entropy over behavior descriptors.
But descriptors were only assigned to offspring in the main loop, and only
when `config.diversity.strategy == "map_elites"`. Seeds never got them.
Survivors from the previous generation never got them. So the population
always had zero descriptors and zero entropy.

**Fix: `_assign_behavior_descriptors()` everywhere** — A new helper method
assigns descriptors to any individual with IR that doesn't already have one.
Called on seeds (gen 0), offspring (each gen), and survivors after refill.
The MAP-Elites guard was removed — descriptors are useful for entropy metrics
regardless of selection strategy.

### Bug 3: 0.95 is still the king

When a proof failed `lake` verification, its fitness was downgraded from 1.0
to 0.95. But 0.95 was still the highest fitness in the population. In a
population of 2, both individuals might be these false-positive proofs at
0.95, crowding out all exploration. Evolution stagnates because the "best"
individuals can't actually prove anything.

**Fix: Deb's feasibility constraint** — A standard technique from constrained
evolutionary optimization (Deb 2000). Instead of downgrading fitness, we mark
the individual as `feasible=False`. Selection's `_primary_fitness()` function
returns `-inf` for infeasible individuals, so they automatically rank below
*all* feasible solutions. The primary fitness stays at 1.0 — the individual's
genetic material is preserved for mutation (it might be one tactic away from a
real proof), but it can never win a tournament against feasible solutions.

This is more principled than the 0.95 hack because:
- It works regardless of the fitness landscape (no magic threshold)
- It separates "quality of the proof attempt" from "did it actually verify"
- It matches the standard EA literature on constraint handling

### The meta-lesson

Population dynamics bugs are insidious because the engine appears to work
correctly — it runs, it evaluates, it selects, it mutates. The only symptom
is that `pop_size` quietly shrinks and `diversity` quietly reads zero. If you
aren't watching those metrics, you'd think the search is just "hard" rather
than "broken."

The fix required changes across three layers: the engine (refill + descriptor
assignment), selection (feasibility ranking), and the verification gate
(feasible=False instead of fitness downgrade). This is why the system is
tested at the integration level, not just unit level — a unit test on
`Lexicase.survive()` wouldn't catch the dedup interaction with
`PopulationManager.add()`.


## 18. Verification Cache + Dead-End Tracking

**The problem:** Every `lake env lean` call takes ~13 seconds, and the engine
was re-verifying the same false-positive proofs every generation. Tactics like
`norm_num` and `ring` get fitness=1.0 from the REPL (false positives), fail
lake compilation, but keep reappearing as offspring because mutation/crossover
regenerates them. Worse, these fitness=1.0 failures were invisible to
`SearchMemory`'s dead-end detection (which only triggers at fitness < 0.1), so
the LLM never learned to avoid them.

**The fix — two changes:**

1. **Verification cache** (`_verification_cache: dict[str, bool]` on the
   engine): Before calling `backend.verify_proof()`, check if `ir_hash` has
   been verified before. Cache both successes and failures. This is essentially
   a **tabu list** from Glover's tabu search (1986) — proven-bad solutions
   are never re-explored. The cache persists across checkpoint save/load.

2. **`SearchMemory.record_verification_failure()`**: When verification fails,
   the genome is *immediately* promoted to `dead_ends` (bypassing the normal
   "3 failures" threshold, since lake verification is definitive). This makes
   the failed proof visible in the LLM's prompt section under "Dead ends
   (avoid these)", acting as a **negative exemplar** (Madaan et al. 2023
   Self-Refine) that steers generation away from known failures.

**Why this matters:** In a typical 20-generation run targeting `norm_le_one`,
proofs like `norm_num` appear as offspring in nearly every generation.
Without the cache, that's 20 × 13s = 4+ minutes wasted on a single tactic.
With the cache, the first verification is the last — subsequent encounters
are instant cache hits, and the LLM is told to avoid the tactic entirely.

**Lesson learned:** Expensive evaluation caching is standard practice in
surrogate-assisted evolutionary algorithms (Jin 2011). The key insight is
that *verification* is even more expensive than *evaluation*, and false
positives from the REPL make it a hot path. Caching at the `ir_hash` level
(not genome string) means that cosmetically different proofs that
canonicalize to the same IR still share cache entries.

**Test count: 405 → 416.** 8 new tests: 4 for `record_verification_failure()`
(immediate dead-end, prompt visibility, max cap, serialization roundtrip)
and 4 for the engine verification cache (prevents redundant calls, caches
positive results, feeds failures to memory, checkpoint roundtrip). 3 more tests cover cache-hit log silencing (DEBUG vs WARNING), redundant `record_verification_failure` prevention, and early rejection of known-bad hashes before evaluation.
