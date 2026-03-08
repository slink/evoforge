# Evoforge Bugfixes & Tree Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix three silent bugs that cripple search effectiveness, improve the fitness function to reward goal reduction, parse and use reflection output, then add a best-first tree search mode as a complementary proof strategy.

**Architecture:** Two phases. Phase A fixes bugs in existing code (crossover wiring, ensemble stats, reflection parsing, fitness formula). Phase B adds a `ProofTreeSearch` class that explores tactic alternatives at each proof state via beam/best-first search, integrated as an optional search mode alongside evolution.

**Tech Stack:** Python 3.11+, asyncio, pytest-asyncio, Pydantic, Jinja2, Lean 4 REPL

**Quality gate:** `uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v`

---

## Phase A: Bug Fixes & Fitness Improvements

### Task 1: Wire crossover — set `guidance_individual` in engine

**Problem:** `LLMCrossover` and `SplicePrefixes` are no-ops because the engine never sets `guidance_individual` on `MutationContext`, and `SplicePrefixes` parses `context.guidance` (a text string) instead of a second individual's genome.

**Files:**
- Modify: `evoforge/core/engine.py:277-298` (mutation loop)
- Modify: `evoforge/backends/lean/operators.py:102-127` (SplicePrefixes)
- Test: `tests/test_core/test_engine.py`

**Step 1: Write failing test — crossover receives guidance_individual**

Add to `tests/test_core/test_engine.py`:

```python
@pytest.mark.asyncio
async def test_crossover_gets_guidance_individual():
    """When LLMCrossover is selected, MutationContext.guidance_individual is set."""
    captured_contexts: list[Any] = []

    class SpyOperator(MutationOperator):
        @property
        def name(self) -> str:
            return "llm_crossover"

        @property
        def cost(self) -> Literal["cheap", "llm"]:
            return "llm"

        async def apply(self, parent: Individual, context: MutationContext) -> str:
            captured_contexts.append(context)
            return parent.genome

    config = _make_config(max_generations=1)
    backend = MockBackend()
    archive = Archive(engine="sqlite+aiosqlite://")
    await archive.initialize()

    spy = SpyOperator()
    engine = EvolutionEngine(config, backend, archive, llm_client=MockLLMClient())
    # Replace ensemble with one that always picks our spy
    engine._ensemble._operators = [spy]
    engine._ensemble._weights = [1.0]

    await engine.run()

    crossover_contexts = [c for c in captured_contexts if c.guidance_individual is not None]
    assert len(crossover_contexts) > 0, "LLMCrossover should receive a guidance_individual"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_engine.py::test_crossover_gets_guidance_individual -xvs`
Expected: FAIL — `guidance_individual` is always None

**Step 3: Implement — wire guidance_individual in engine mutation loop**

In `evoforge/core/engine.py`, modify the mutation loop (around line 281-298). When the selected operator's name contains "crossover", pick a second parent and set it on context:

```python
# Inside the `for parent in parents:` loop, after selecting operator:
guidance_ind = None
if "crossover" in operator.name:
    # Pick a different parent for crossover
    other_parents = [p for p in parents if p.ir_hash != parent.ir_hash]
    if other_parents:
        guidance_ind = random.choice(other_parents)
    else:
        guidance_ind = random.choice(parents)

context = MutationContext(
    generation=gen,
    memory=self._memory,
    guidance=guidance,
    temperature=self._temperature,
    backend=self.backend,
    credits=parent.credits,
    guidance_individual=guidance_ind,
)
```

Add `import random` at top of engine.py if not already present.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_engine.py::test_crossover_gets_guidance_individual -xvs`
Expected: PASS

**Step 5: Fix SplicePrefixes to use a second individual's genome**

In `evoforge/backends/lean/operators.py`, modify `SplicePrefixes.apply()`:

```python
async def apply(self, parent: Individual, context: MutationContext) -> str:
    # Use guidance_individual's genome for crossover, not the text guidance
    if context.guidance_individual is not None:
        other_genome = context.guidance_individual.genome
    else:
        return parent.genome

    seq_a = parse_tactic_sequence(parent.genome)
    seq_b = parse_tactic_sequence(other_genome)
    if seq_a is None or seq_b is None:
        return parent.genome

    keep = _credit_prefix_len(context.credits)
    prefix_steps = seq_a.steps[:keep]
    suffix_steps = seq_b.steps[keep:]
    merged = prefix_steps + suffix_steps
    if not merged:
        return parent.genome
    return TacticSequence(steps=merged).serialize()
```

**Step 6: Run existing SplicePrefixes tests + full suite**

Run: `uv run pytest tests/test_lean/test_operators.py -xvs`
Expected: PASS (update any tests that relied on old guidance-string behavior)

**Step 7: Commit**

```bash
git add evoforge/core/engine.py evoforge/backends/lean/operators.py tests/
git commit -m "Wire crossover guidance_individual in engine, fix SplicePrefixes"
```

---

### Task 2: Wire ensemble stats tracking in engine

**Problem:** `MutationEnsemble.update_stats()` is never called, so adaptive/phased weight scheduling is dead. Operators never learn which ones produce fitness improvements.

**Files:**
- Modify: `evoforge/core/engine.py` (after evaluation, before memory update)
- Test: `tests/test_core/test_engine.py`

**Step 1: Write failing test — ensemble stats are updated after evaluation**

```python
@pytest.mark.asyncio
async def test_ensemble_stats_updated():
    """After each generation, operator stats should reflect applications."""
    config = _make_config(max_generations=2)
    backend = MockBackend()
    archive = Archive(engine="sqlite+aiosqlite://")
    await archive.initialize()
    engine = EvolutionEngine(config, backend, archive)

    await engine.run()

    total_apps = sum(s.applications for s in engine._ensemble.stats.values())
    assert total_apps > 0, "Ensemble stats should track operator applications"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_engine.py::test_ensemble_stats_updated -xvs`
Expected: FAIL — total_apps == 0

**Step 3: Implement — track stats in engine**

We need to record which operator was used for each offspring and then, after evaluation, compare child fitness to parent fitness to determine success.

In `evoforge/core/engine.py`, change the `offspring_genomes` tracking to also store the operator name, and after evaluation update stats. Modify the mutation loop to track operator per offspring:

```python
# Change offspring_lineage to also track parent fitness for delta calc
# After evaluation of offspring (around line 340), add:

# Update ensemble stats with fitness deltas
parent_fitness_map: dict[str, float] = {
    ind.ir_hash: (ind.fitness.primary if ind.fitness else 0.0)
    for ind in pop_list
}
for parent_hash, child_hash, op_name in offspring_lineage:
    child_ind = next(
        (ind for ind in credited_offspring if ind.ir_hash == child_hash), None
    )
    if child_ind is None:
        continue
    child_fit = child_ind.fitness.primary if child_ind.fitness else 0.0
    parent_fit = parent_fitness_map.get(parent_hash, 0.0)
    delta = child_fit - parent_fit
    self._ensemble.update_stats(
        op_name, success=(delta > 0), fitness_delta=delta
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_engine.py::test_ensemble_stats_updated -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add evoforge/core/engine.py tests/test_core/test_engine.py
git commit -m "Wire ensemble stat tracking to enable adaptive operator weights"
```

---

### Task 3: Parse and use reflection output

**Problem:** `_reflect()` calls the LLM and gets a `Reflection`-shaped JSON back but discards it. The `Reflection` dataclass already exists in `types.py` but is never instantiated.

**Files:**
- Modify: `evoforge/core/engine.py:680-702` (_reflect method)
- Modify: `evoforge/core/memory.py` (add method to ingest reflection)
- Test: `tests/test_core/test_engine.py`
- Test: `tests/test_core/test_memory.py`

**Step 1: Write failing test — reflection updates search memory**

In `tests/test_core/test_memory.py`:

```python
def test_ingest_reflection():
    """Reflection strategies_to_avoid should be added to dead_ends."""
    from evoforge.core.types import Reflection

    mem = SearchMemory(max_patterns=10, max_dead_ends=20)
    reflection = Reflection(
        strategies_to_try=["try calc blocks"],
        strategies_to_avoid=["avoid ring on complex norms"],
        useful_primitives=["norm_nonneg", "nlinarith"],
        population_diagnosis="stagnant, need structural changes",
        suggested_temperature=0.9,
    )
    mem.ingest_reflection(reflection)

    assert "avoid ring on complex norms" in mem.dead_ends
    assert any("calc blocks" in p.description for p in mem.patterns) or \
           "try calc blocks" in {p.description for p in mem.patterns}
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_memory.py::test_ingest_reflection -xvs`
Expected: FAIL — `AttributeError: 'SearchMemory' object has no attribute 'ingest_reflection'`

**Step 3: Implement `SearchMemory.ingest_reflection()`**

In `evoforge/core/memory.py`:

```python
def ingest_reflection(self, reflection: Any) -> None:
    """Incorporate LLM reflection into memory state.

    - strategies_to_avoid → dead_ends
    - strategies_to_try → patterns (with synthetic fitness=0.5)
    - useful_primitives → patterns (with synthetic fitness=0.3)
    """
    for avoid in reflection.strategies_to_avoid:
        self.dead_ends.add(avoid.strip())
    self._cap_dead_ends()

    for strategy in reflection.strategies_to_try:
        desc = strategy.strip()
        if desc not in self._pattern_data:
            self._pattern_data[desc] = _PatternAccum(total_fitness=0.5, count=1)

    for primitive in reflection.useful_primitives:
        desc = primitive.strip()
        if desc not in self._pattern_data:
            self._pattern_data[desc] = _PatternAccum(total_fitness=0.3, count=1)

    self._rebuild_patterns()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_memory.py::test_ingest_reflection -xvs`
Expected: PASS

**Step 5: Write failing test — engine parses reflection JSON**

In `tests/test_core/test_engine.py`:

```python
@pytest.mark.asyncio
async def test_reflection_updates_memory():
    """Reflection response should be parsed and fed into search memory."""
    reflection_json = json.dumps({
        "strategies_to_try": ["use calc blocks"],
        "strategies_to_avoid": ["avoid sorry"],
        "useful_primitives": ["norm_nonneg"],
        "population_diagnosis": "needs diversity",
        "suggested_temperature": 0.8,
    })

    class ReflectionLLM:
        async def async_generate(self, prompt, system, model, temp, max_tokens):
            from types import SimpleNamespace
            return SimpleNamespace(text=reflection_json)

    config = _make_config(max_generations=1)
    config.reflection.interval = 1
    config.ablation.disable_reflection = False
    backend = MockBackend()
    archive = Archive(engine="sqlite+aiosqlite://")
    await archive.initialize()

    engine = EvolutionEngine(config, backend, archive, llm_client=ReflectionLLM())
    await engine.run()

    assert "avoid sorry" in engine._memory.dead_ends
```

**Step 6: Implement — parse reflection JSON in `_reflect()`**

In `evoforge/core/engine.py`, modify `_reflect()`:

```python
async def _reflect(self, generation: int) -> None:
    """Call the LLM for a reflection, parse the result into memory."""
    try:
        top_k = self.config.reflection.include_top_k
        pop = self.population.best(k=top_k)
        prompt = self.backend.format_reflection_prompt(
            population=pop,
            memory=self._memory,
            generation=generation,
        )
        system = self.backend.system_prompt()
        model = self.config.llm.reflection_model
        response = await self.llm_client.async_generate(
            prompt,
            system,
            model,
            self._temperature,
            self.config.llm.max_tokens,
        )
        logger.info("Reflection response received (%d chars)", len(response.text))

        # Parse structured reflection
        self._apply_reflection(response.text)
    except Exception:
        logger.warning("Reflection LLM call failed", exc_info=True)

def _apply_reflection(self, text: str) -> None:
    """Parse reflection JSON and apply to memory + temperature."""
    import json as _json
    from evoforge.core.types import Reflection

    # Extract JSON from response (may be wrapped in markdown)
    raw = text.strip()
    if "```" in raw:
        # Strip markdown code fences
        import re
        match = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()

    try:
        data = _json.loads(raw)
    except _json.JSONDecodeError:
        logger.debug("Reflection response not valid JSON: %s", text[:200])
        return

    try:
        reflection = Reflection(
            strategies_to_try=data.get("strategies_to_try", []),
            strategies_to_avoid=data.get("strategies_to_avoid", []),
            useful_primitives=data.get("useful_primitives", []),
            population_diagnosis=data.get("population_diagnosis", ""),
            suggested_temperature=float(data.get("suggested_temperature", self._temperature)),
        )
    except (TypeError, ValueError):
        logger.debug("Reflection response malformed: %s", data)
        return

    self._memory.ingest_reflection(reflection)

    # Apply suggested temperature as a soft influence
    suggested = reflection.suggested_temperature
    if 0.1 <= suggested <= 1.5:
        # Blend: 70% scheduled, 30% suggested
        self._temperature = 0.7 * self._temperature + 0.3 * suggested
```

**Step 7: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_engine.py::test_reflection_updates_memory -xvs`
Expected: PASS

**Step 8: Commit**

```bash
git add evoforge/core/engine.py evoforge/core/memory.py tests/
git commit -m "Parse reflection JSON into search memory and temperature"
```

---

### Task 4: Improve fitness function — reward goal reduction

**Problem:** Fitness is `steps_succeeded / total_steps`, which rewards long proofs and ignores goal reduction. A 5-step proof that closes 3 goals should score higher than a 5-step proof that closes 0 goals.

**Files:**
- Modify: `evoforge/backends/lean/evaluator.py:411-431` (fitness computation)
- Test: `tests/test_lean/test_stepwise.py`

**Step 1: Write failing test — goal reduction affects fitness**

In `tests/test_lean/test_stepwise.py`:

```python
def test_goal_reduction_improves_fitness():
    """A proof that reduces goals should score higher than one that doesn't."""
    from evoforge.backends.lean.evaluator import _compute_fitness

    # 3/5 steps succeeded, closed 2 goals (started with 3, ended with 1)
    fitness_good = _compute_fitness(
        steps_succeeded=3, total_steps=5,
        initial_goals=3, goals_remaining=1, proof_complete=False,
    )
    # 3/5 steps succeeded, closed 0 goals (started with 3, ended with 3)
    fitness_bad = _compute_fitness(
        steps_succeeded=3, total_steps=5,
        initial_goals=3, goals_remaining=3, proof_complete=False,
    )
    assert fitness_good.primary > fitness_bad.primary
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lean/test_stepwise.py::test_goal_reduction_improves_fitness -xvs`
Expected: FAIL — `_compute_fitness` doesn't exist yet

**Step 3: Extract and improve fitness computation**

In `evoforge/backends/lean/evaluator.py`, extract the fitness computation into a standalone function (makes it testable and reusable):

```python
def _compute_fitness(
    *,
    steps_succeeded: int,
    total_steps: int,
    initial_goals: int,
    goals_remaining: int,
    proof_complete: bool,
) -> Fitness:
    """Compute fitness from evaluation outcomes.

    Formula for incomplete proofs:
        primary = 0.4 * (steps_succeeded / total_steps)
                + 0.6 * goal_reduction_ratio

    Where goal_reduction_ratio = (initial_goals - goals_remaining) / max(initial_goals, 1)

    Complete proofs get 1.0. This rewards proofs that close goals,
    not just proofs that have many individually-valid tactics.
    """
    if proof_complete:
        primary = 1.0
    elif total_steps > 0:
        step_ratio = steps_succeeded / total_steps
        goal_reduction = (initial_goals - goals_remaining) / max(initial_goals, 1)
        # Clamp goal_reduction to [0, 1] — can't reduce below 0 remaining
        goal_reduction = max(0.0, min(1.0, goal_reduction))
        primary = 0.4 * step_ratio + 0.6 * goal_reduction
    else:
        primary = 0.0

    return Fitness(
        primary=primary,
        auxiliary={
            "steps_succeeded": float(steps_succeeded),
            "goals_remaining": float(goals_remaining),
            "goal_reduction": float(initial_goals - goals_remaining),
            "proof_complete": 1.0 if proof_complete else 0.0,
        },
        constraints={},
        feasible=proof_complete,
    )
```

Then update `LeanStepwiseEvaluator.evaluate()` to use it, passing the initial goal count. We need to capture initial_goals from the first REPL response:

```python
# In evaluate(), track initial goal count:
# After the first successful step, record len(goals_after) as a baseline.
# Or better: the initial proof state already has goals — capture from first step's goals_before.

# At the end of evaluate(), replace the inline Fitness construction:
initial_goals = 0
if trace.step_results:
    first_with_goals = next(
        (sr for sr in trace.step_results if sr.goals_before), None
    )
    if first_with_goals:
        initial_goals = len(first_with_goals.goals_before)
    elif trace.step_results[0].succeeded and trace.step_results[0].goals_after:
        # First step succeeded — its goals_after tells us the state after step 0
        # Initial goals is unknowable, estimate from first failed step or goals_after of step 0
        initial_goals = len(trace.step_results[0].goals_after) + 1  # heuristic

fitness = _compute_fitness(
    steps_succeeded=steps_succeeded,
    total_steps=total_steps,
    initial_goals=max(initial_goals, 1),
    goals_remaining=goals_remaining,
    proof_complete=proof_complete,
)
```

**Note:** The initial goal count is tricky because the REPL doesn't give us goals_before for the very first tactic (we start from the sorry proof state). The simplest approach: assume initial_goals = 1 for the theorem-level goal. The REPL returns goals after each step. If step 0 creates sub-goals (e.g. `constructor` splits into 2), goals_after > 1 means the step created work, not reduced it. So **initial_goals = 1** (one theorem-level goal) is the correct baseline.

Update to use a fixed initial:

```python
# Replace the initial_goals heuristic with:
initial_goals = 1  # We always start from one theorem-level goal
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lean/test_stepwise.py::test_goal_reduction_improves_fitness -xvs`
Expected: PASS

**Step 5: Run full stepwise test suite to check for regressions**

Run: `uv run pytest tests/test_lean/test_stepwise.py -xvs`
Expected: PASS (some tests may need fitness value updates due to new formula)

**Step 6: Commit**

```bash
git add evoforge/backends/lean/evaluator.py tests/test_lean/test_stepwise.py
git commit -m "Weight fitness toward goal reduction (60%) over step count (40%)"
```

---

### Task 5: Quality gate — run full suite after Phase A

**Step 1: Run quality gate**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v
```

Expected: All pass. Fix any regressions from Tasks 1-4.

**Step 2: Commit any fixes**

---

## Phase B: Best-First Tree Search

### Task 6: Add `ProofTreeSearch` class

**Problem:** Evolution is inefficient for exploring tactic alternatives at specific proof states. The literature (HTPS, AlphaProof, COPRA) shows that tree search over proof states dramatically outperforms linear sequence evolution for theorem proving.

**Design:** A `ProofTreeSearch` that:
1. Starts from the initial proof state
2. At each node, asks the LLM for N candidate tactics
3. Evaluates each candidate in the REPL
4. Expands the best nodes first (by cumulative score)
5. Stops when a proof is complete or budget is exhausted

This runs *alongside* evolution — the engine can dispatch tree search on the best partial proofs found by evolution.

**Files:**
- Create: `evoforge/backends/lean/tree_search.py`
- Test: `tests/test_lean/test_tree_search.py`

**Step 1: Write failing test — basic tree search finds a proof**

In `tests/test_lean/test_tree_search.py`:

```python
"""Tests for best-first proof tree search."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from evoforge.backends.lean.tree_search import ProofNode, ProofTreeSearch


class MockREPL:
    """Mock REPL that accepts specific tactics."""

    def __init__(self, accept_map: dict[tuple[int, str], dict[str, Any]]) -> None:
        self._accept_map = accept_map

    async def send_tactic(self, tactic: str, state: int = 0) -> dict[str, object]:
        key = (state, tactic.strip())
        if key in self._accept_map:
            return self._accept_map[key]
        return {"message": f"unknown tactic '{tactic}'"}


class MockTacticGenerator:
    """Mock LLM that suggests fixed tactics per goal state."""

    def __init__(self, suggestions: dict[int, list[str]]) -> None:
        self._suggestions = suggestions

    async def suggest_tactics(
        self, goal_state: str, proof_so_far: list[str], n: int
    ) -> list[str]:
        # Use proof length as a proxy for state
        state = len(proof_so_far)
        return self._suggestions.get(state, ["sorry"])[:n]


@pytest.mark.asyncio
async def test_tree_search_finds_two_step_proof():
    """Tree search should find intro x -> simp proof."""
    repl = MockREPL({
        (0, "intro x"): {"proofState": 1, "goals": ["⊢ x = x"]},
        (1, "simp"): {"proofState": 2, "goals": []},
        (0, "simp"): {"message": "simp failed"},
    })
    generator = MockTacticGenerator({
        0: ["intro x", "simp"],
        1: ["simp", "ring"],
    })
    search = ProofTreeSearch(
        repl=repl,
        tactic_generator=generator,
        initial_state=0,
        max_nodes=50,
        beam_width=3,
    )
    result = search.search()
    proof = await result

    assert proof is not None
    assert proof.tactics == ["intro x", "simp"]
    assert proof.complete


@pytest.mark.asyncio
async def test_tree_search_respects_budget():
    """Tree search stops after max_nodes expansions."""
    repl = MockREPL({})  # All tactics fail
    generator = MockTacticGenerator({
        0: ["simp", "ring", "omega"],
    })
    search = ProofTreeSearch(
        repl=repl,
        tactic_generator=generator,
        initial_state=0,
        max_nodes=5,
        beam_width=2,
    )
    result = await search.search()

    assert result is None or not result.complete
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lean/test_tree_search.py -xvs`
Expected: FAIL — module doesn't exist

**Step 3: Implement `ProofTreeSearch`**

Create `evoforge/backends/lean/tree_search.py`:

```python
"""Best-first proof tree search for Lean 4.

Explores tactic alternatives at each proof state, expanding the most
promising nodes first. Designed to complement evolutionary search by
efficiently exploring branches that evolution identifies as promising.
"""
from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ProofNode:
    """A node in the proof search tree."""

    state: int  # REPL proof state ID
    tactics: list[str]  # tactics taken to reach this node
    goals: list[str]  # remaining goals at this state
    score: float  # cumulative score (higher = better)
    depth: int = 0
    complete: bool = False
    parent: ProofNode | None = field(default=None, repr=False)

    def __lt__(self, other: ProofNode) -> bool:
        # For max-heap via negation in heapq
        return self.score > other.score


@dataclass
class SearchResult:
    """Result of a tree search."""

    tactics: list[str]
    complete: bool
    nodes_expanded: int
    score: float


class TacticGenerator(Protocol):
    """Protocol for tactic suggestion (LLM or other)."""

    async def suggest_tactics(
        self, goal_state: str, proof_so_far: list[str], n: int
    ) -> list[str]: ...


class ProofTreeSearch:
    """Best-first search over the proof tree.

    At each step:
    1. Pop the highest-scoring unexpanded node
    2. Ask the tactic generator for N candidate tactics
    3. Try each in the REPL
    4. Add successful ones as children
    5. If any child has no remaining goals, return it

    Scoring: each successful tactic adds 1.0; goal reduction adds 0.5 per goal closed.
    """

    def __init__(
        self,
        repl: Any,
        tactic_generator: TacticGenerator,
        initial_state: int = 0,
        initial_goals: list[str] | None = None,
        max_nodes: int = 200,
        beam_width: int = 5,
        prefix: list[str] | None = None,
    ) -> None:
        self._repl = repl
        self._generator = tactic_generator
        self._initial_state = initial_state
        self._initial_goals = initial_goals or []
        self._max_nodes = max_nodes
        self._beam_width = beam_width
        self._prefix = prefix or []

    async def search(self) -> SearchResult | None:
        """Run best-first search. Returns SearchResult or None if budget exhausted."""
        root = ProofNode(
            state=self._initial_state,
            tactics=list(self._prefix),
            goals=list(self._initial_goals),
            score=float(len(self._prefix)),  # credit for prefix
            depth=len(self._prefix),
        )

        frontier: list[ProofNode] = [root]
        nodes_expanded = 0
        visited_states: set[int] = set()

        while frontier and nodes_expanded < self._max_nodes:
            node = heapq.heappop(frontier)

            if node.state in visited_states:
                continue
            visited_states.add(node.state)
            nodes_expanded += 1

            # Ask generator for candidate tactics
            goal_str = "\n".join(node.goals) if node.goals else "no goals displayed"
            candidates = await self._generator.suggest_tactics(
                goal_state=goal_str,
                proof_so_far=node.tactics,
                n=self._beam_width,
            )

            for tactic in candidates:
                tactic = tactic.strip()
                if not tactic:
                    continue

                try:
                    resp = await self._repl.send_tactic(tactic, state=node.state)
                except Exception:
                    logger.debug("REPL error for tactic %s", tactic[:60], exc_info=True)
                    continue

                # Check if tactic succeeded
                is_error = ("severity" in resp and resp["severity"] == "error") or (
                    "message" in resp and "proofState" not in resp
                )
                if is_error:
                    continue

                new_state = int(resp.get("proofState", node.state + 1))
                new_goals = resp.get("goals", [])
                if not isinstance(new_goals, list):
                    new_goals = []

                # Score: base + goal reduction bonus
                goals_before = len(node.goals) if node.goals else 1
                goals_after = len(new_goals)
                goal_reduction = max(0, goals_before - goals_after)
                step_score = 1.0 + 0.5 * goal_reduction

                child = ProofNode(
                    state=new_state,
                    tactics=node.tactics + [tactic],
                    goals=[str(g) for g in new_goals],
                    score=node.score + step_score,
                    depth=node.depth + 1,
                    complete=(len(new_goals) == 0),
                    parent=node,
                )

                if child.complete:
                    logger.info(
                        "Tree search found proof in %d steps, %d nodes expanded",
                        child.depth,
                        nodes_expanded,
                    )
                    return SearchResult(
                        tactics=child.tactics,
                        complete=True,
                        nodes_expanded=nodes_expanded,
                        score=child.score,
                    )

                heapq.heappush(frontier, child)

        # Budget exhausted — return best partial if any
        if frontier:
            best = max(frontier, key=lambda n: n.score)
            return SearchResult(
                tactics=best.tactics,
                complete=False,
                nodes_expanded=nodes_expanded,
                score=best.score,
            )

        return None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lean/test_tree_search.py -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add evoforge/backends/lean/tree_search.py tests/test_lean/test_tree_search.py
git commit -m "Add best-first proof tree search over REPL tactic states"
```

---

### Task 7: Add `LLMTacticGenerator` — the LLM adapter for tree search

**Problem:** Tree search needs a `TacticGenerator` that asks the LLM "given this goal state, suggest N tactics."

**Files:**
- Create: `evoforge/backends/lean/tactic_generator.py`
- Create: `evoforge/backends/lean/templates/tactic_suggest_prompt.j2`
- Test: `tests/test_lean/test_tactic_generator.py`

**Step 1: Write failing test**

```python
"""Tests for LLM tactic generator."""
import pytest
from unittest.mock import AsyncMock
from types import SimpleNamespace

from evoforge.backends.lean.tactic_generator import LLMTacticGenerator


@pytest.mark.asyncio
async def test_extracts_multiple_tactics():
    """Generator should parse numbered tactics from LLM response."""
    llm_response = """Here are tactics to try:
1. `simp [h0]`
2. `norm_num`
3. `linarith [norm_nonneg x]`
"""
    client = AsyncMock()
    client.async_generate.return_value = SimpleNamespace(text=llm_response)

    gen = LLMTacticGenerator(
        client=client,
        model="claude-sonnet-4-5-20250929",
        system_prompt="You are a Lean expert.",
    )
    tactics = await gen.suggest_tactics(
        goal_state="⊢ ‖φ ξ‖ ≤ 1",
        proof_so_far=["intro x"],
        n=3,
    )

    assert len(tactics) == 3
    assert "simp [h0]" in tactics
    assert "norm_num" in tactics


@pytest.mark.asyncio
async def test_handles_code_block_response():
    """Generator should handle tactics in code blocks."""
    llm_response = """```lean
simp [h0]
norm_num
linarith
```"""
    client = AsyncMock()
    client.async_generate.return_value = SimpleNamespace(text=llm_response)

    gen = LLMTacticGenerator(
        client=client,
        model="claude-sonnet-4-5-20250929",
        system_prompt="You are a Lean expert.",
    )
    tactics = await gen.suggest_tactics(
        goal_state="⊢ 0 ≤ 1", proof_so_far=[], n=5,
    )

    assert len(tactics) >= 3
    assert "simp [h0]" in tactics
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lean/test_tactic_generator.py -xvs`
Expected: FAIL — module doesn't exist

**Step 3: Implement LLMTacticGenerator**

Create `evoforge/backends/lean/tactic_generator.py`:

```python
"""LLM-based tactic generator for proof tree search."""
from __future__ import annotations

import logging
import re
from typing import Any

import jinja2

from evoforge.backends.lean.tree_search import TacticGenerator

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = __import__("pathlib").Path(__file__).parent / "templates"

# Pattern to extract backtick-wrapped tactics: `tactic here`
_BACKTICK_RE = re.compile(r"`([^`]+)`")
# Pattern to extract numbered list items: 1. tactic or 1. `tactic`
_NUMBERED_RE = re.compile(r"^\d+\.\s*`?([^`\n]+)`?\s*$", re.MULTILINE)
# Lean code block
_CODE_BLOCK_RE = re.compile(r"```(?:lean)?\s*\n(.*?)```", re.DOTALL)


class LLMTacticGenerator(TacticGenerator):
    """Asks an LLM to suggest N tactics for a given proof state."""

    def __init__(
        self,
        client: Any,
        model: str,
        system_prompt: str,
        temperature: float = 0.9,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._jinja = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            keep_trailing_newline=True,
        )

    async def suggest_tactics(
        self, goal_state: str, proof_so_far: list[str], n: int
    ) -> list[str]:
        """Ask the LLM for n candidate tactics given the current proof state."""
        template = self._jinja.get_template("tactic_suggest_prompt.j2")
        prompt = template.render(
            goal_state=goal_state,
            proof_so_far="\n".join(proof_so_far) if proof_so_far else "(empty)",
            n=n,
        )

        response = await self._client.async_generate(
            prompt,
            self._system_prompt,
            self._model,
            self._temperature,
            self._max_tokens,
        )

        return self._parse_tactics(response.text, n)

    @staticmethod
    def _parse_tactics(text: str, n: int) -> list[str]:
        """Extract individual tactics from LLM response.

        Tries multiple formats: numbered list, code block, backtick-wrapped.
        """
        tactics: list[str] = []

        # Try numbered list first (most structured)
        numbered = _NUMBERED_RE.findall(text)
        if numbered:
            tactics = [t.strip() for t in numbered if t.strip()]
            return tactics[:n]

        # Try code block
        code_match = _CODE_BLOCK_RE.search(text)
        if code_match:
            lines = code_match.group(1).strip().split("\n")
            tactics = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("--")]
            return tactics[:n]

        # Try backtick extraction
        backtick = _BACKTICK_RE.findall(text)
        if backtick:
            tactics = [t.strip() for t in backtick if t.strip()]
            return tactics[:n]

        # Last resort: split on newlines, filter obvious non-tactics
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        tactics = [
            ln for ln in lines
            if not ln.startswith(("#", "//", "--", "Here", "Try", "The", "I "))
            and len(ln) < 200
        ]
        return tactics[:n]
```

Create `evoforge/backends/lean/templates/tactic_suggest_prompt.j2`:

```
Suggest exactly {{ n }} different Lean 4 tactics to try at this proof state.

## Current Goal
```
{{ goal_state }}
```

## Proof So Far
```
{{ proof_so_far }}
```

## Instructions
Return {{ n }} distinct tactics, one per line, as a numbered list.
Each tactic should be a single Lean 4 tactic invocation (not a full proof).
Prefer diverse approaches — don't suggest minor variations of the same tactic.
Do NOT use `sorry`.

Example format:
1. `simp [h0]`
2. `linarith [norm_nonneg x]`
3. `exact le_of_eq (by ring)`
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lean/test_tactic_generator.py -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add evoforge/backends/lean/tactic_generator.py evoforge/backends/lean/templates/tactic_suggest_prompt.j2 tests/test_lean/test_tactic_generator.py
git commit -m "Add LLM tactic generator for tree search step suggestions"
```

---

### Task 8: Integrate tree search into engine — hybrid mode

**Problem:** The engine only does evolution. We want it to optionally dispatch tree search on promising partial proofs.

**Design:** After each generation, if the best individual has fitness > 0 but < 1.0, launch tree search from its last successful proof state. This combines evolution's ability to find promising proof prefixes with tree search's ability to efficiently explore branches.

**Files:**
- Modify: `evoforge/core/config.py` (add tree search config)
- Modify: `evoforge/core/engine.py` (dispatch tree search)
- Modify: `evoforge/backends/lean/backend.py` (expose tree search factory)
- Test: `tests/test_core/test_engine.py`

**Step 1: Add config fields**

In `evoforge/core/config.py`, add to `EvolutionConfig`:

```python
class EvolutionConfig(BaseModel):
    """Top-level evolution loop parameters."""
    max_generations: int = 100
    stagnation_window: int = 10
    checkpoint_every: int = 10
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    resume: bool = False
    tree_search_enabled: bool = False
    tree_search_max_nodes: int = 200
    tree_search_beam_width: int = 5
    tree_search_min_fitness: float = 0.3
```

**Step 2: Add `create_tree_search` to Backend ABC and LeanBackend**

In `evoforge/backends/base.py`, add an optional method (non-abstract, returns None by default):

```python
async def create_tree_search(
    self,
    prefix: list[str],
    llm_client: Any,
    max_nodes: int = 200,
    beam_width: int = 5,
) -> Any | None:
    """Create a tree search instance seeded from a tactic prefix.

    Returns None if tree search is not supported by this backend.
    """
    return None
```

In `evoforge/backends/lean/backend.py`, override it:

```python
async def create_tree_search(
    self,
    prefix: list[str],
    llm_client: Any,
    max_nodes: int = 200,
    beam_width: int = 5,
) -> Any | None:
    """Create a proof tree search starting from a tactic prefix."""
    if self._repl is None or self._evaluator is None:
        return None

    from evoforge.backends.lean.tactic_generator import LLMTacticGenerator
    from evoforge.backends.lean.tree_search import ProofTreeSearch

    generator = LLMTacticGenerator(
        client=llm_client,
        model="claude-sonnet-4-5-20250929",  # fast model for tactic suggestions
        system_prompt=self.system_prompt(),
        temperature=0.9,
    )

    # Replay prefix to get the REPL state
    state = self._evaluator._initial_proof_state
    for tactic in prefix:
        async with self._repl_lock:
            resp = await self._repl.send_tactic(tactic, state=state)
        if "proofState" not in resp:
            break
        state = int(resp["proofState"])

    goals = resp.get("goals", []) if "proofState" in resp else []

    return ProofTreeSearch(
        repl=self._repl,
        tactic_generator=generator,
        initial_state=state,
        initial_goals=[str(g) for g in goals] if isinstance(goals, list) else [],
        max_nodes=max_nodes,
        beam_width=beam_width,
        prefix=prefix,
    )
```

**Step 3: Add tree search dispatch to engine**

In `evoforge/core/engine.py`, add a method and call it at the end of each generation:

```python
async def _try_tree_search(self, generation: int) -> None:
    """If enabled, run tree search on the best partial proof."""
    if not self.config.evolution.tree_search_enabled:
        return
    if self.llm_client is None:
        return

    best_list = self.population.best(k=1)
    if not best_list:
        return

    best = best_list[0]
    if best.fitness is None:
        return

    min_fit = self.config.evolution.tree_search_min_fitness
    if best.fitness.primary < min_fit or best.fitness.primary >= 1.0:
        return

    # Extract the successful tactic prefix
    prefix_tactics: list[str] = []
    if best.ir is not None and hasattr(best.ir, "steps"):
        # Use credited steps only
        for i, step in enumerate(best.ir.steps):
            # Check if this step succeeded via credits
            credited = any(c.location == i and c.score > 0 for c in best.credits)
            if credited:
                prefix_tactics.append(step.raw)
            else:
                break

    if not prefix_tactics:
        return

    logger.info(
        "Launching tree search from %d-step prefix (fitness=%.2f)",
        len(prefix_tactics),
        best.fitness.primary,
    )

    searcher = await self.backend.create_tree_search(
        prefix=prefix_tactics,
        llm_client=self.llm_client,
        max_nodes=self.config.evolution.tree_search_max_nodes,
        beam_width=self.config.evolution.tree_search_beam_width,
    )
    if searcher is None:
        return

    async with self._repl_lock:
        result = await searcher.search()

    if result is not None and result.complete:
        logger.info("Tree search found complete proof: %s", result.tactics)
        # Inject the proof into the population
        genome = "\n".join(result.tactics)
        individuals = self._process_genomes([genome], generation=generation)
        if individuals:
            evaluated = await self._evaluator.evaluate_batch(individuals)
            self._total_evaluations += len(evaluated)
            self._assign_credits(evaluated)
            await self._verify_perfect_individuals(evaluated)
            self._assign_behavior_descriptors(evaluated)
            self._add_to_population(evaluated)
    elif result is not None:
        logger.info(
            "Tree search exhausted budget (%d nodes), best partial: %d steps",
            result.nodes_expanded,
            len(result.tactics),
        )
        # Inject best partial as a new individual
        genome = "\n".join(result.tactics)
        individuals = self._process_genomes([genome], generation=generation)
        if individuals:
            evaluated = await self._evaluator.evaluate_batch(individuals)
            self._total_evaluations += len(evaluated)
            self._assign_credits(evaluated)
            self._assign_behavior_descriptors(evaluated)
            self._add_to_population(evaluated)
```

Call it in the main loop after stagnation check (around line 385):

```python
# After await self._check_stagnation(gen):
await self._try_tree_search(gen)
```

**Step 4: Write test — tree search injects individuals into population**

```python
@pytest.mark.asyncio
async def test_tree_search_injects_into_population():
    """When tree search finds a proof, it should appear in the population."""
    config = _make_config(max_generations=2)
    config.evolution.tree_search_enabled = True
    config.evolution.tree_search_min_fitness = 0.1

    # ... (mock backend with create_tree_search returning a complete result)
    # This test verifies the wiring — that tree search results get evaluated
    # and added to the population.
```

(Full test code will depend on the mock infrastructure already in test_engine.py — adapt the existing MockBackend to support create_tree_search.)

**Step 5: Run tests**

Run: `uv run pytest tests/test_core/test_engine.py -xvs`
Expected: PASS

**Step 6: Commit**

```bash
git add evoforge/core/config.py evoforge/core/engine.py evoforge/backends/base.py evoforge/backends/lean/backend.py tests/
git commit -m "Integrate best-first tree search as hybrid mode in engine"
```

---

### Task 9: Update config and docs

**Files:**
- Modify: `configs/lean_default.toml`
- Modify: `MEMORY.md`

**Step 1: Add tree search config to lean_default.toml**

```toml
[evolution]
max_generations = 200
stagnation_window = 10
checkpoint_every = 10
log_level = "INFO"
tree_search_enabled = true
tree_search_max_nodes = 200
tree_search_beam_width = 5
tree_search_min_fitness = 0.3
```

**Step 2: Commit**

```bash
git add configs/lean_default.toml
git commit -m "Enable tree search in default Lean config"
```

---

### Task 10: Final quality gate

**Step 1: Run full quality suite**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v
```

Expected: All pass.

**Step 2: Run a real 5-generation test**

```bash
bash scripts/run_with_keychain.sh --max-generations 5 --output-dir /tmp/evotest_tree --verify
```

Observe: tree search should trigger when best_fitness > 0.3, and you should see "Launching tree search" log messages.

---

### Task 11: Write post-mortem — why naive LLM evolutionary search didn't work

**Problem:** We need a clear, honest document explaining what went wrong, why, and what we learned. This is useful for the project README, for future contributors, and for the `sglink.md` developer journal.

**Files:**
- Create: `docs/post-mortem-naive-llm-search.md`

**Step 1: Write the post-mortem**

The document should cover:
- What we built and what we expected
- The actual results (10 gens, 476 evals, peaked at 0.7, zero verified proofs)
- Root cause analysis: the three bugs, the fitness function, the architecture gap
- What the literature told us we should have known
- What we're doing about it

Keep it concise (~800-1200 words), honest, and educational. Write it for a reader who knows some ML but not theorem proving.

**Step 2: Commit**

```bash
git add docs/post-mortem-naive-llm-search.md
git commit -m "Add post-mortem on naive LLM evolutionary proof search"
```

---

## Summary of Changes

| Task | What | Impact |
|------|------|--------|
| 1 | Wire crossover guidance_individual | Enables LLMCrossover + SplicePrefixes (2 of 6 operators were no-ops) |
| 2 | Wire ensemble stat tracking | Enables adaptive operator weights (was uniform forever) |
| 3 | Parse reflection output | Feeds LLM analysis into memory + temperature (was discarded) |
| 4 | Goal-reduction fitness | Rewards proofs that close goals, not just long sequences |
| 5 | Quality gate checkpoint | Ensure no regressions |
| 6 | ProofTreeSearch | Best-first search over tactic alternatives at each proof state |
| 7 | LLMTacticGenerator | LLM adapter for step-level tactic suggestion |
| 8 | Hybrid integration | Engine dispatches tree search on promising partial proofs |
| 9 | Config + docs | Enable in default config |
| 10 | Final validation | Full suite + real run |
| 11 | Post-mortem | Honest writeup of what went wrong and why |

## Literature References

- **HTPS** (Lample et al. 2022) — Hyper-tree proof search, best-first expansion of proof states
- **AlphaProof** (DeepMind 2024) — MCTS over proof states with value network
- **Goedel-Prover-V2** (arXiv:2508.03613) — Monte Carlo tree self-refinement for whole-proof generation
- **COPRA** (2024) — In-context learning with stateful proof assistant interaction
- **LEGO-Prover** (2024) — Proof decomposition into reusable sub-lemmas
- **ReProver** (2024) — Premise selection via embedding retrieval
- **APRIL** (arXiv:2602.02990) — Active premise retrieval with iterative refinement
- **FunSearch** (DeepMind 2023) — Island-model evolutionary search with LLM mutation
- **ELM** (2023) — Evolution through Large Models for program synthesis

## Future Work (Not in This Plan)

- **Proof decomposition** (LEGO-Prover style): structurally break proofs into `have` sub-lemmas
- **Premise retrieval**: embedding-based search for relevant lemmas from Mathlib
- **Island model** (FunSearch style): multiple independent populations with migration
- **Value network**: train a small model to score proof states for better tree search priority
- **Tactic-level backtracking in evaluator**: try alternatives when a step fails instead of just recording the failure
