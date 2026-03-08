# Phase 1 Completion — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close all Phase 1 gaps so the core framework + Lean backend matches DESIGN_v4.md 100%.

**Architecture:** Fix the Backend ABC to include all design-specified methods, align LeanBackend and the engine's crossover/reflection/temperature flows, expand the config schema to match the full TOML spec, and add per-generation LLM budget enforcement. Every change is TDD: write failing tests first.

**Tech Stack:** Python 3.11, pytest-asyncio, pydantic, jinja2, mypy strict, ruff

---

## Task 1: Add missing Backend ABC methods

**Files:**
- Modify: `evoforge/backends/base.py`
- Test: `tests/test_core/test_backend_abc.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_core/test_backend_abc.py
"""Tests that the Backend ABC declares every required abstract method."""

from __future__ import annotations

import inspect

from evoforge.backends.base import Backend


class TestBackendABCCompleteness:
    """Every method in the design doc must be abstract on Backend."""

    REQUIRED_ABSTRACT = {
        "parse",
        "evaluate",
        "evaluate_stepwise",
        "assign_credit",
        "validate_structure",
        "seed_population",
        "mutation_operators",
        "system_prompt",
        "format_mutation_prompt",
        "format_crossover_prompt",
        "extract_genome",
        "behavior_descriptor",
        "behavior_space",
        "recommended_selection",
        # NEW — must be added
        "version",
        "eval_config_hash",
        "format_reflection_prompt",
        "default_operator_weights",
    }

    def test_all_required_methods_are_abstract(self) -> None:
        abstract = set(Backend.__abstractmethods__)
        missing = self.REQUIRED_ABSTRACT - abstract
        assert not missing, f"Backend ABC missing abstract methods: {missing}"

    def test_version_returns_str(self) -> None:
        sig = inspect.signature(Backend.version)
        assert sig.return_annotation == "str"

    def test_eval_config_hash_returns_str(self) -> None:
        sig = inspect.signature(Backend.eval_config_hash)
        assert sig.return_annotation == "str"

    def test_format_reflection_prompt_signature(self) -> None:
        sig = inspect.signature(Backend.format_reflection_prompt)
        params = list(sig.parameters.keys())
        assert "population" in params
        assert "memory" in params

    def test_default_operator_weights_returns_dict(self) -> None:
        sig = inspect.signature(Backend.default_operator_weights)
        assert "dict" in str(sig.return_annotation)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_backend_abc.py -v`
Expected: FAIL — `version`, `eval_config_hash`, `format_reflection_prompt`, `default_operator_weights` not abstract

**Step 3: Write minimal implementation**

Add to `evoforge/backends/base.py`:

```python
    @abstractmethod
    def version(self) -> str:
        """Return a version string for this backend (for cache keying)."""
        ...

    @abstractmethod
    def eval_config_hash(self) -> str:
        """Return a hash of the evaluation config (for cache keying)."""
        ...

    @abstractmethod
    def format_reflection_prompt(
        self,
        population: list[Individual],
        memory: Any,
        generation: int,
    ) -> str:
        """Format a reflection prompt for the LLM to analyze population state."""
        ...

    @abstractmethod
    def default_operator_weights(self) -> dict[str, float]:
        """Return default mutation operator weights for this domain."""
        ...
```

Also fix the `format_crossover_prompt` signature to match the design:

```python
    @abstractmethod
    def format_crossover_prompt(
        self, parent_a: Individual, parent_b: Individual, context: Any
    ) -> str:
        """Format a crossover prompt for the LLM given two parent individuals."""
        ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_backend_abc.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add evoforge/backends/base.py tests/test_core/test_backend_abc.py
git commit -m "Add missing abstract methods to Backend ABC"
```

---

## Task 2: Implement missing methods on LeanBackend

**Files:**
- Modify: `evoforge/backends/lean/backend.py`
- Test: `tests/test_lean/test_lean_backend.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_lean/test_lean_backend.py
"""Tests for LeanBackend — all Backend ABC methods implemented correctly."""

from __future__ import annotations

from evoforge.backends.lean.backend import LeanBackend
from evoforge.core.types import Fitness, Individual


def _make_backend() -> LeanBackend:
    return LeanBackend(
        theorem_statement="theorem test : True := by trivial",
        project_dir="/tmp/fake",
    )


class TestLeanBackendVersion:
    def test_version_returns_nonempty_string(self) -> None:
        b = _make_backend()
        v = b.version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_version_contains_lean(self) -> None:
        b = _make_backend()
        assert "lean" in b.version().lower()


class TestLeanBackendEvalConfigHash:
    def test_eval_config_hash_returns_hex(self) -> None:
        b = _make_backend()
        h = b.eval_config_hash()
        assert isinstance(h, str)
        assert len(h) > 0

    def test_same_config_same_hash(self) -> None:
        b1 = _make_backend()
        b2 = _make_backend()
        assert b1.eval_config_hash() == b2.eval_config_hash()

    def test_different_theorem_different_hash(self) -> None:
        b1 = _make_backend()
        b2 = LeanBackend(
            theorem_statement="theorem other : False := by sorry",
            project_dir="/tmp/fake",
        )
        assert b1.eval_config_hash() != b2.eval_config_hash()


class TestLeanBackendDefaultOperatorWeights:
    def test_returns_dict_with_all_operators(self) -> None:
        b = _make_backend()
        weights = b.default_operator_weights()
        ops = b.mutation_operators()
        for op in ops:
            assert op.name in weights

    def test_weights_sum_to_one(self) -> None:
        b = _make_backend()
        weights = b.default_operator_weights()
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestLeanBackendFormatReflectionPrompt:
    def test_returns_nonempty_string(self) -> None:
        b = _make_backend()
        ind = Individual(genome="intro x\nsimp", ir=None, ir_hash="abc", generation=0)
        ind.fitness = Fitness(primary=0.5, auxiliary={}, constraints={}, feasible=True)
        prompt = b.format_reflection_prompt(
            population=[ind],
            memory=None,
            generation=5,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_includes_fitness_info(self) -> None:
        b = _make_backend()
        ind = Individual(genome="intro x\nsimp", ir=None, ir_hash="abc", generation=0)
        ind.fitness = Fitness(primary=0.75, auxiliary={}, constraints={}, feasible=True)
        prompt = b.format_reflection_prompt(
            population=[ind],
            memory=None,
            generation=5,
        )
        assert "0.75" in prompt or "fitness" in prompt.lower()


class TestLeanBackendFormatCrossoverPrompt:
    """format_crossover_prompt now takes two Individuals."""

    def test_returns_nonempty_string(self) -> None:
        b = _make_backend()
        a = Individual(genome="intro x\nsimp", ir=None, ir_hash="a1", generation=0)
        b_ind = Individual(genome="intro x\nring", ir=None, ir_hash="b1", generation=0)
        prompt = b.format_crossover_prompt(a, b_ind, context=None)
        assert isinstance(prompt, str)
        assert "intro x" in prompt

    def test_includes_both_genomes(self) -> None:
        b = _make_backend()
        a = Individual(genome="intro x\nsimp", ir=None, ir_hash="a1", generation=0)
        b_ind = Individual(genome="intro x\nring", ir=None, ir_hash="b1", generation=0)
        prompt = b.format_crossover_prompt(a, b_ind, context=None)
        assert "simp" in prompt
        assert "ring" in prompt
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lean/test_lean_backend.py -v`
Expected: FAIL — `version()`, `eval_config_hash()`, `default_operator_weights()`, `format_reflection_prompt()` not implemented, `format_crossover_prompt()` signature changed

**Step 3: Write minimal implementation**

Add to `LeanBackend`:

```python
    def version(self) -> str:
        return "lean_v1"

    def eval_config_hash(self) -> str:
        import hashlib
        content = f"{self.theorem_statement}:{self.project_dir}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def default_operator_weights(self) -> dict[str, float]:
        return {
            "prefix_truncation": 0.25,
            "tactic_swap": 0.25,
            "tactic_reorder": 0.25,
            "splice_prefixes": 0.25,
        }

    def format_reflection_prompt(
        self,
        population: list[Individual],
        memory: Any,
        generation: int,
    ) -> str:
        # Compute stats
        fitnesses = [
            ind.fitness.primary for ind in population
            if ind.fitness is not None
        ]
        best_fitness = max(fitnesses) if fitnesses else 0.0
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0

        memory_section = ""
        if memory is not None and hasattr(memory, "prompt_section"):
            memory_section = memory.prompt_section(max_tokens=400)

        top_individuals = sorted(
            [ind for ind in population if ind.fitness is not None],
            key=lambda i: i.fitness.primary,
            reverse=True,
        )[:5]

        template = self._jinja_env.get_template("reflection_prompt.j2")
        return template.render(
            best_fitness=f"{best_fitness:.4f}",
            avg_fitness=f"{avg_fitness:.4f}",
            pop_size=len(population),
            generation=generation,
            diversity="N/A",
            memory_section=memory_section,
            top_individuals=top_individuals,
        )
```

Fix `format_crossover_prompt` to accept two `Individual`s:

```python
    def format_crossover_prompt(
        self, parent_a: Individual, parent_b: Individual, context: Any
    ) -> str:
        diagnostics_a = ""
        if parent_a.diagnostics is not None and hasattr(parent_a.diagnostics, "summary"):
            diagnostics_a = parent_a.diagnostics.summary(max_tokens=500)

        credit_a = ""
        if parent_a.diagnostics is not None and hasattr(parent_a.diagnostics, "credit_summary"):
            credit_a = parent_a.diagnostics.credit_summary(parent_a.credits, max_tokens=300)

        template = self._jinja_env.get_template("crossover_prompt.j2")
        return template.render(
            genome_a=parent_a.genome,
            genome_b=parent_b.genome,
            diagnostics_a=diagnostics_a,
            credit_a=credit_a,
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lean/test_lean_backend.py -v`
Expected: PASS

**Step 5: Fix existing test_engine.py MockBackend**

The `MockBackend` in `tests/test_core/test_engine.py` must also implement the new abstract methods to avoid `TypeError: Can't instantiate abstract class`. Add:

```python
    def version(self) -> str:
        return "mock_v1"

    def eval_config_hash(self) -> str:
        return "mock_cfg_hash"

    def format_reflection_prompt(self, population: list[Individual], memory: Any, generation: int) -> str:
        return "Mock reflection prompt"

    def default_operator_weights(self) -> dict[str, float]:
        return {"mock_append": 0.5, "mock_shuffle": 0.5}

    def format_crossover_prompt(self, parent_a: Individual, parent_b: Individual, context: Any) -> str:
        return f"Crossover: {parent_a.genome} + {parent_b.genome}"
```

**Step 6: Run full test suite**

Run: `uv run pytest -x -v`
Expected: ALL 237+ tests PASS

**Step 7: Commit**

```bash
git add evoforge/backends/lean/backend.py tests/test_lean/test_lean_backend.py tests/test_core/test_engine.py
git commit -m "Implement missing Backend methods on LeanBackend"
```

---

## Task 3: Fix LLMCrossover to use correct crossover prompt signature

**Files:**
- Modify: `evoforge/llm/operators.py`
- Modify: `evoforge/core/mutation.py` (add `guidance_individual` to MutationContext)
- Test: `tests/test_core/test_llm_operators.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_core/test_llm_operators.py
"""Tests for LLM-powered mutation operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from evoforge.core.mutation import MutationContext
from evoforge.core.types import Credit, Fitness, Individual
from evoforge.llm.operators import LLMCrossover, LLMMutate


@dataclass
class _LLMResponse:
    text: str
    input_tokens: int = 10
    output_tokens: int = 20
    model: str = "test"


class _MockLLMClient:
    def __init__(self, response_text: str) -> None:
        self._text = response_text
        self.calls: list[tuple[str, str]] = []

    async def async_generate(
        self, prompt: str, system: str, model: str,
        temperature: float, max_tokens: int,
    ) -> _LLMResponse:
        self.calls.append((prompt, system))
        return _LLMResponse(text=self._text)


class _MockBackend:
    def system_prompt(self) -> str:
        return "system"

    def format_mutation_prompt(self, parent: Individual, context: Any) -> str:
        return f"mutate: {parent.genome}"

    def format_crossover_prompt(
        self, parent_a: Individual, parent_b: Individual, context: Any
    ) -> str:
        return f"cross: {parent_a.genome} + {parent_b.genome}"

    def extract_genome(self, raw_text: str) -> str | None:
        if raw_text.startswith("```lean"):
            return raw_text.replace("```lean\n", "").replace("```", "").strip()
        return raw_text if raw_text != "bad" else None


def _make_context(
    backend: _MockBackend,
    guidance_individual: Individual | None = None,
) -> MutationContext:
    return MutationContext(
        generation=1,
        memory=None,
        guidance="some guidance",
        temperature=0.7,
        backend=backend,
        credits=[],
        guidance_individual=guidance_individual,
    )


class TestLLMCrossoverUsesSecondParent:
    """LLMCrossover must pass both parents to format_crossover_prompt."""

    @pytest.mark.asyncio
    async def test_crossover_with_second_parent(self) -> None:
        client = _MockLLMClient("intro x\nring")
        backend = _MockBackend()
        op = LLMCrossover(client, "test-model")

        parent = Individual(genome="intro x\nsimp", ir=None, ir_hash="a", generation=0)
        second = Individual(genome="intro x\nlinarith", ir=None, ir_hash="b", generation=0)
        ctx = _make_context(backend, guidance_individual=second)

        result = await op.apply(parent, ctx)
        # Should have called format_crossover_prompt with both parents
        assert len(client.calls) == 1
        prompt = client.calls[0][0]
        assert "simp" in prompt
        assert "linarith" in prompt

    @pytest.mark.asyncio
    async def test_crossover_without_second_parent_falls_back(self) -> None:
        client = _MockLLMClient("intro x\nring")
        backend = _MockBackend()
        op = LLMCrossover(client, "test-model")

        parent = Individual(genome="intro x\nsimp", ir=None, ir_hash="a", generation=0)
        ctx = _make_context(backend, guidance_individual=None)

        result = await op.apply(parent, ctx)
        # Should fall back to mutation prompt
        assert len(client.calls) == 1
        prompt = client.calls[0][0]
        assert "mutate:" in prompt


class TestLLMMutateBasic:
    @pytest.mark.asyncio
    async def test_returns_extracted_genome(self) -> None:
        client = _MockLLMClient("intro x\nring")
        backend = _MockBackend()
        op = LLMMutate(client, "test-model")

        parent = Individual(genome="intro x\nsimp", ir=None, ir_hash="a", generation=0)
        ctx = _make_context(backend)

        result = await op.apply(parent, ctx)
        assert result == "intro x\nring"

    @pytest.mark.asyncio
    async def test_falls_back_to_parent_on_extraction_failure(self) -> None:
        client = _MockLLMClient("bad")
        backend = _MockBackend()
        op = LLMMutate(client, "test-model")

        parent = Individual(genome="intro x\nsimp", ir=None, ir_hash="a", generation=0)
        ctx = _make_context(backend)

        result = await op.apply(parent, ctx)
        assert result == "intro x\nsimp"  # parent genome
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_llm_operators.py -v`
Expected: FAIL — `MutationContext.__init__` doesn't accept `guidance_individual`

**Step 3: Implement**

In `evoforge/core/mutation.py`, add `guidance_individual` to `MutationContext`:

```python
@dataclass
class MutationContext:
    """Runtime context passed to every mutation operator invocation."""

    generation: int
    memory: Any
    guidance: str
    temperature: float
    backend: Any
    credits: list[Credit]
    guidance_individual: Individual | None = None
```

In `evoforge/llm/operators.py`, fix `LLMCrossover.apply`:

```python
    async def apply(self, parent: Individual, context: MutationContext) -> str:
        if context.guidance_individual is not None:
            prompt = context.backend.format_crossover_prompt(
                parent, context.guidance_individual, context
            )
        else:
            prompt = context.backend.format_mutation_prompt(parent, context)

        system = context.backend.system_prompt()
        response = await self._client.async_generate(
            prompt, system, self._model, context.temperature, self._max_tokens,
        )
        genome = context.backend.extract_genome(response.text)
        if genome is not None:
            return genome
        logger.warning("LLMCrossover: extraction failed, falling back to parent")
        return parent.genome
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_llm_operators.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `uv run pytest -x -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add evoforge/core/mutation.py evoforge/llm/operators.py tests/test_core/test_llm_operators.py
git commit -m "Fix LLMCrossover to pass both parents to format_crossover_prompt"
```

---

## Task 4: Expand config schema to match design

**Files:**
- Modify: `evoforge/core/config.py`
- Modify: `configs/lean_default.toml`
- Test: `tests/test_core/test_config.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_core/test_config.py
"""Tests for evoforge.core.config — full DESIGN_v4 schema coverage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from evoforge.core.config import (
    AblationConfig,
    DiversityConfig,
    EvoforgeConfig,
    LLMConfig,
    MemoryConfig,
    ReflectionConfig,
    RunConfig,
    SchedulerConfig,
    load_config,
)


class TestNewConfigModels:
    def test_run_config_defaults(self) -> None:
        cfg = RunConfig()
        assert cfg.name == ""
        assert cfg.seed == 42

    def test_reflection_config_defaults(self) -> None:
        cfg = ReflectionConfig()
        assert cfg.interval == 10
        assert cfg.include_top_k == 5
        assert cfg.include_bottom_k == 5

    def test_memory_config_defaults(self) -> None:
        cfg = MemoryConfig()
        assert cfg.max_patterns == 20
        assert cfg.max_dead_ends == 15
        assert cfg.stagnation_window == 20

    def test_scheduler_config_defaults(self) -> None:
        cfg = SchedulerConfig()
        assert cfg.max_llm_concurrent == 4
        assert cfg.max_eval_concurrent == 8
        assert cfg.llm_budget_per_gen == 15

    def test_diversity_config_defaults(self) -> None:
        cfg = DiversityConfig()
        assert cfg.strategy == "map_elites"
        assert cfg.sampling == "weighted"

    def test_ablation_config_all_false(self) -> None:
        cfg = AblationConfig()
        assert cfg.disable_llm is False
        assert cfg.disable_credit is False


class TestLLMConfigExpanded:
    def test_temperature_scheduling(self) -> None:
        cfg = LLMConfig()
        assert cfg.temperature_start == 1.0
        assert cfg.temperature_end == 0.3
        assert cfg.temperature_schedule == "linear"
        assert cfg.max_attempts == 3

    def test_reflection_model_default(self) -> None:
        cfg = LLMConfig()
        assert "sonnet" in cfg.reflection_model.lower()


class TestEvoforgeConfigHasAllSections:
    def test_has_run(self) -> None:
        cfg = EvoforgeConfig()
        assert hasattr(cfg, "run")

    def test_has_reflection(self) -> None:
        cfg = EvoforgeConfig()
        assert hasattr(cfg, "reflection")

    def test_has_memory(self) -> None:
        cfg = EvoforgeConfig()
        assert hasattr(cfg, "memory")

    def test_has_scheduler(self) -> None:
        cfg = EvoforgeConfig()
        assert hasattr(cfg, "scheduler")

    def test_has_diversity(self) -> None:
        cfg = EvoforgeConfig()
        assert hasattr(cfg, "diversity")

    def test_has_ablation(self) -> None:
        cfg = EvoforgeConfig()
        assert hasattr(cfg, "ablation")


class TestLoadFullConfig:
    def test_loads_lean_default(self) -> None:
        cfg = load_config("configs/lean_default.toml")
        assert cfg.backend.name == "lean"
        assert cfg.run.seed == 42
        assert cfg.reflection.interval == 10
        assert cfg.scheduler.llm_budget_per_gen == 15
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_config.py -v`
Expected: FAIL — `RunConfig`, `ReflectionConfig`, etc. don't exist

**Step 3: Implement config models**

```python
# Add these to evoforge/core/config.py

class RunConfig(BaseModel):
    """Top-level run metadata."""
    name: str = ""
    backend: str = "lean"
    seed: int = 42

class ReflectionConfig(BaseModel):
    """LLM reflection configuration."""
    interval: int = 10
    include_top_k: int = 5
    include_bottom_k: int = 5

class MemoryConfig(BaseModel):
    """Search memory configuration."""
    max_patterns: int = 20
    max_dead_ends: int = 15
    max_constructs: int = 30
    stagnation_window: int = 20

class SchedulerConfig(BaseModel):
    """Execution scheduler configuration."""
    mode: str = "async_batch"
    max_llm_concurrent: int = 4
    max_eval_concurrent: int = 8
    max_pending: int = 16
    llm_budget_per_gen: int = 15

class DiversityConfig(BaseModel):
    """Quality-diversity configuration."""
    strategy: str = "map_elites"
    sampling: str = "weighted"

class AblationConfig(BaseModel):
    """Ablation experiment toggles."""
    disable_llm: bool = False
    disable_diagnostics: bool = False
    disable_reflection: bool = False
    disable_memory: bool = False
    disable_cheap_operators: bool = False
    disable_credit: bool = False
```

Expand `LLMConfig`:

```python
class LLMConfig(BaseModel):
    model: str = "claude-haiku-4-5-20251001"
    reflection_model: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.7
    temperature_start: float = 1.0
    temperature_end: float = 0.3
    temperature_schedule: str = "linear"
    max_tokens: int = 4096
    max_calls: int = 1000
    max_cost_usd: float = 50.0
    max_attempts: int = 3
```

Add all to `EvoforgeConfig`:

```python
class EvoforgeConfig(BaseModel):
    run: RunConfig = RunConfig()
    population: PopulationConfig = PopulationConfig()
    selection: SelectionConfig = SelectionConfig()
    mutation: MutationConfig = MutationConfig()
    llm: LLMConfig = LLMConfig()
    eval: EvalConfig = EvalConfig()
    backend: BackendConfig = BackendConfig()
    evolution: EvolutionConfig = EvolutionConfig()
    reflection: ReflectionConfig = ReflectionConfig()
    memory: MemoryConfig = MemoryConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    diversity: DiversityConfig = DiversityConfig()
    ablation: AblationConfig = AblationConfig()
```

**Step 4: Update `configs/lean_default.toml`**

```toml
[run]
name = "levy_proof_search_001"
backend = "lean"
seed = 42

[population]
size = 30
elite_k = 2

[selection]
strategy = "lexicase"
tournament_size = 3
epsilon = 0.0

[mutation]
schedule = "phased"
llm_weight = 0.45
cheap_weight = 0.55
crossover_weight = 0.15

[llm]
model = "claude-haiku-4-5-20251001"
reflection_model = "claude-sonnet-4-5-20250929"
temperature = 0.7
temperature_start = 1.0
temperature_end = 0.3
temperature_schedule = "linear"
max_tokens = 4096
max_calls = 1000
max_cost_usd = 50.0
max_attempts = 3

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

[eval]
max_concurrent = 4
timeout_seconds = 60.0
seed = 42

[backend]
name = "lean"
theorem_statement = "theorem IsPositiveDefinite.norm_le_one {f : ℝ → ℂ} (hf : IsPositiveDefinite f) (h0 : f 0 = 1) : ∀ ξ, ‖f ξ‖ ≤ 1"
project_dir = "/Users/sglink/Desktop/Projects/LeanLevy"
repl_path = ""

[evolution]
max_generations = 200
stagnation_window = 20
checkpoint_every = 10
log_level = "INFO"

[diversity]
strategy = "map_elites"
sampling = "weighted"

[ablation]
disable_llm = false
disable_diagnostics = false
disable_reflection = false
disable_memory = false
disable_cheap_operators = false
disable_credit = false
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_config.py -v`
Expected: PASS

**Step 6: Run full suite — fix any breakage from renamed config fields**

Run: `uv run pytest -x -v`

Note: The `SchedulerConfig` in `config.py` now collides with `SchedulerConfig` in `scheduler.py`. Rename the config model to `SchedulerSettings` or rename the scheduler's dataclass. The cleanest fix: rename the `config.py` version since it's new. Use `SchedulerSettings` in config.py and keep `SchedulerConfig` in scheduler.py.

Expected: ALL PASS after fixing any import conflicts

**Step 7: Commit**

```bash
git add evoforge/core/config.py configs/lean_default.toml tests/test_core/test_config.py
git commit -m "Expand config schema to match full DESIGN_v4 specification"
```

---

## Task 5: Add temperature scheduling to the engine

**Files:**
- Modify: `evoforge/core/engine.py`
- Test: `tests/test_core/test_temperature.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_core/test_temperature.py
"""Tests for linear temperature scheduling in the engine."""

from __future__ import annotations

from evoforge.core.engine import EvolutionEngine


class TestTemperatureScheduling:
    def test_linear_schedule_at_start(self) -> None:
        """At generation 0, temperature should be temperature_start."""
        temp = EvolutionEngine._compute_temperature(
            generation=0, max_generations=100,
            start=1.0, end=0.3, schedule="linear",
        )
        assert abs(temp - 1.0) < 0.001

    def test_linear_schedule_at_end(self) -> None:
        """At last generation, temperature should be temperature_end."""
        temp = EvolutionEngine._compute_temperature(
            generation=100, max_generations=100,
            start=1.0, end=0.3, schedule="linear",
        )
        assert abs(temp - 0.3) < 0.001

    def test_linear_schedule_at_midpoint(self) -> None:
        """At midpoint, temperature should be halfway between start and end."""
        temp = EvolutionEngine._compute_temperature(
            generation=50, max_generations=100,
            start=1.0, end=0.3, schedule="linear",
        )
        assert abs(temp - 0.65) < 0.001

    def test_fixed_schedule_returns_start(self) -> None:
        """With schedule='fixed', temperature is always start value."""
        temp = EvolutionEngine._compute_temperature(
            generation=50, max_generations=100,
            start=0.7, end=0.3, schedule="fixed",
        )
        assert abs(temp - 0.7) < 0.001
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_temperature.py -v`
Expected: FAIL — `_compute_temperature` doesn't exist

**Step 3: Implement**

Add static method to `EvolutionEngine`:

```python
    @staticmethod
    def _compute_temperature(
        generation: int,
        max_generations: int,
        start: float,
        end: float,
        schedule: str,
    ) -> float:
        """Compute the LLM temperature for a given generation."""
        if schedule == "fixed" or max_generations <= 0:
            return start
        # Linear interpolation
        t = min(generation / max_generations, 1.0)
        return start + (end - start) * t
```

Update the `run()` method to use it:

```python
        # In the generation loop, replace self._temperature with:
        self._temperature = self._compute_temperature(
            generation=gen,
            max_generations=max_gen,
            start=self.config.llm.temperature_start,
            end=self.config.llm.temperature_end,
            schedule=self.config.llm.temperature_schedule,
        )
```

Also update the stagnation handler to bump temperature with a cap:

```python
    def _check_stagnation(self, generation: int) -> None:
        if self._memory.is_stagnant():
            logger.info("Stagnation detected at generation %d", generation)
            self._reflected = True
            # Temporarily boost temperature (capped at start value + 0.3)
            max_temp = self.config.llm.temperature_start + 0.3
            self._temperature = min(self._temperature + 0.2, max_temp)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_temperature.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `uv run pytest -x -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add evoforge/core/engine.py tests/test_core/test_temperature.py
git commit -m "Add linear temperature scheduling to evolution engine"
```

---

## Task 6: Implement reflection (LLM call on stagnation)

**Files:**
- Modify: `evoforge/core/engine.py`
- Test: `tests/test_core/test_reflection.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_core/test_reflection.py
"""Tests for the reflection loop in the engine."""

from __future__ import annotations

import json
from typing import Any

import pytest

from evoforge.core.archive import Archive
from evoforge.core.config import EvoforgeConfig, EvolutionConfig, PopulationConfig, ReflectionConfig, SelectionConfig
from evoforge.core.engine import EvolutionEngine
from evoforge.core.types import Fitness, Individual, Reflection


# Reuse ConstantFitnessBackend from test_engine.py
from tests.test_core.test_engine import ConstantFitnessBackend


class _MockReflectionLLMClient:
    """LLM client that returns a JSON reflection response."""

    def __init__(self) -> None:
        self.call_count = 0

    async def async_generate(
        self, prompt: str, system: str, model: str,
        temperature: float, max_tokens: int = 4096,
    ) -> Any:
        from dataclasses import dataclass

        @dataclass
        class Resp:
            text: str
            input_tokens: int = 100
            output_tokens: int = 200
            model: str = "test"

        self.call_count += 1
        reflection_json = json.dumps({
            "strategies_to_try": ["try omega", "try linarith"],
            "strategies_to_avoid": ["avoid sorry"],
            "useful_primitives": ["intro", "simp"],
            "population_diagnosis": "Population is stagnant",
            "suggested_temperature": 1.2,
        })
        return Resp(text=reflection_json)


@pytest.fixture
async def archive() -> Archive:
    a = Archive("sqlite+aiosqlite://")
    await a.create_tables()
    return a


class TestReflectionTriggered:
    """When stagnation is detected and LLM client is available, engine should call LLM for reflection."""

    async def test_reflection_calls_llm_on_stagnation(self, archive: Archive) -> None:
        stagnation_window = 3
        config = EvoforgeConfig(
            population=PopulationConfig(size=5, elite_k=2),
            selection=SelectionConfig(strategy="scalar_tournament", tournament_size=2),
            evolution=EvolutionConfig(
                max_generations=stagnation_window + 3,
                stagnation_window=stagnation_window,
            ),
            reflection=ReflectionConfig(interval=10),
        )
        backend = ConstantFitnessBackend()
        llm_client = _MockReflectionLLMClient()

        engine = EvolutionEngine(
            config=config, backend=backend, archive=archive, llm_client=llm_client,
        )
        result = await engine.run()

        assert result.reflected is True
        # The LLM should have been called at least once for reflection
        assert llm_client.call_count >= 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_reflection.py -v`
Expected: FAIL — engine never calls `llm_client.async_generate` for reflection

**Step 3: Implement reflection in the engine**

In `_check_stagnation`, add the actual LLM reflection call:

```python
    async def _check_stagnation(self, generation: int) -> None:
        """Check for stagnation and trigger reflection if LLM is available."""
        if not self._memory.is_stagnant():
            return

        logger.info("Stagnation detected at generation %d", generation)
        self._reflected = True

        # Boost temperature
        max_temp = self.config.llm.temperature_start + 0.3
        self._temperature = min(self._temperature + 0.2, max_temp)

        # Call LLM for reflection if available
        if self.llm_client is not None:
            await self._reflect(generation)

    async def _reflect(self, generation: int) -> None:
        """Call LLM to reflect on population state."""
        try:
            prompt = self.backend.format_reflection_prompt(
                population=self.population.get_all(),
                memory=self._memory,
                generation=generation,
            )
            system = self.backend.system_prompt()
            model = self.config.llm.reflection_model if hasattr(self.config.llm, 'reflection_model') else self.config.llm.model

            response = await self.llm_client.async_generate(
                prompt, system, model, 0.7, self.config.llm.max_tokens,
            )
            logger.info("Reflection response: %s", response.text[:200])
        except Exception:
            logger.warning("Reflection LLM call failed", exc_info=True)
```

Note: `_check_stagnation` is now `async`. Update the call site in `run()` to `await self._check_stagnation(gen)`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_reflection.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `uv run pytest -x -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add evoforge/core/engine.py tests/test_core/test_reflection.py
git commit -m "Add LLM reflection call on stagnation detection"
```

---

## Task 7: Wire backend.version() and eval_config_hash() into engine

**Files:**
- Modify: `evoforge/core/engine.py`
- Test: `tests/test_core/test_engine.py` (extend existing)

**Step 1: Write the failing test**

Add to `tests/test_core/test_engine.py`:

```python
class TestEngineUsesBackendVersion:
    """Engine should use backend.version() and eval_config_hash(), not hardcoded strings."""

    async def test_evaluator_uses_backend_version(self, archive: Archive) -> None:
        config = _make_config(max_generations=1, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)

        # The evaluator should have been initialized with backend.version()
        assert engine._evaluator._backend_version == "mock_v1"
        assert engine._evaluator._config_hash == "mock_cfg_hash"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_engine.py::TestEngineUsesBackendVersion -v`
Expected: FAIL — evaluator has hardcoded `"mock_v1"` and `"cfg_test"`

**Step 3: Fix engine constructor**

In `EvolutionEngine.__init__`, change:

```python
        self._evaluator = AsyncEvaluator(
            backend=backend,
            archive=archive,
            backend_version=backend.version(),
            config_hash=backend.eval_config_hash(),
            max_concurrent=config.eval.max_concurrent,
            timeout_seconds=config.eval.timeout_seconds,
        )
```

**Step 4: Run full suite**

Run: `uv run pytest -x -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add evoforge/core/engine.py tests/test_core/test_engine.py
git commit -m "Wire backend.version() and eval_config_hash() into engine"
```

---

## Task 8: Add per-generation LLM budget enforcement

**Files:**
- Modify: `evoforge/core/scheduler.py`
- Modify: `evoforge/core/engine.py`
- Test: `tests/test_core/test_scheduler.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_core/test_scheduler.py
"""Tests for ExecutionScheduler — per-generation LLM budget."""

from __future__ import annotations

import pytest

from evoforge.core.scheduler import ExecutionScheduler
from evoforge.core.scheduler import SchedulerConfig


class TestPerGenBudget:
    def test_llm_budget_per_gen_starts_at_zero(self) -> None:
        config = SchedulerConfig(max_llm_calls=100, llm_budget_per_gen=5)
        sched = ExecutionScheduler(config)
        assert sched.gen_llm_calls == 0

    def test_can_use_llm_when_under_budget(self) -> None:
        config = SchedulerConfig(max_llm_calls=100, llm_budget_per_gen=5)
        sched = ExecutionScheduler(config)
        assert sched.can_use_llm() is True

    def test_cannot_use_llm_when_at_budget(self) -> None:
        config = SchedulerConfig(max_llm_calls=100, llm_budget_per_gen=2)
        sched = ExecutionScheduler(config)
        sched.record_gen_llm_call()
        sched.record_gen_llm_call()
        assert sched.can_use_llm() is False

    def test_reset_gen_resets_counter(self) -> None:
        config = SchedulerConfig(max_llm_calls=100, llm_budget_per_gen=2)
        sched = ExecutionScheduler(config)
        sched.record_gen_llm_call()
        sched.record_gen_llm_call()
        assert sched.can_use_llm() is False
        sched.reset_generation()
        assert sched.can_use_llm() is True
        assert sched.gen_llm_calls == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_scheduler.py -v`
Expected: FAIL — `llm_budget_per_gen` not in `SchedulerConfig`

**Step 3: Implement**

Add to `SchedulerConfig`:

```python
    llm_budget_per_gen: int = 15
```

Add to `ExecutionScheduler`:

```python
    def __init__(self, config: SchedulerConfig) -> None:
        ...
        self._gen_llm_calls: int = 0

    @property
    def gen_llm_calls(self) -> int:
        return self._gen_llm_calls

    def can_use_llm(self) -> bool:
        return self._gen_llm_calls < self._config.llm_budget_per_gen

    def record_gen_llm_call(self) -> None:
        self._gen_llm_calls += 1

    def reset_generation(self) -> None:
        self._gen_llm_calls = 0
```

In the engine's generation loop, add:

```python
        self._scheduler.reset_generation()
```

at the top of each generation, and check `self._scheduler.can_use_llm()` before LLM operator calls.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_scheduler.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `uv run pytest -x -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add evoforge/core/scheduler.py evoforge/core/engine.py tests/test_core/test_scheduler.py
git commit -m "Add per-generation LLM budget enforcement to scheduler"
```

---

## Task 9: Expand ExperimentResult with design-spec metrics

**Files:**
- Modify: `evoforge/core/engine.py`
- Test: `tests/test_core/test_engine.py` (extend)

**Step 1: Write the failing test**

Add to `tests/test_core/test_engine.py`:

```python
class TestExperimentResultMetrics:
    """ExperimentResult should include all design-specified metrics."""

    async def test_result_has_cache_hit_rate(self, archive: Archive) -> None:
        config = _make_config(max_generations=3, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        result = await engine.run()
        assert "cache_hit_rate" in result.metrics

    async def test_result_has_identity_dedup_rate(self, archive: Archive) -> None:
        config = _make_config(max_generations=3, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        result = await engine.run()
        assert "identity_dedup_rate" in result.metrics

    async def test_result_has_stagnation_counter(self, archive: Archive) -> None:
        config = _make_config(max_generations=3, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        result = await engine.run()
        assert "stagnation_counter" in result.metrics

    async def test_result_has_estimated_cost(self, archive: Archive) -> None:
        config = _make_config(max_generations=3, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        result = await engine.run()
        assert "estimated_cost_usd" in result.metrics
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_engine.py::TestExperimentResultMetrics -v`
Expected: FAIL — `ExperimentResult` has no `metrics` field

**Step 3: Implement**

Add `metrics` field to `ExperimentResult`:

```python
@dataclass
class ExperimentResult:
    best_individual: Individual | None
    best_fitness: float
    generations_run: int
    total_evaluations: int
    cost: dict[str, float]
    archive_size: int
    reflected: bool
    metrics: dict[str, float]
```

Add tracking counters to the engine and populate in `_build_result`:

```python
    self._cache_hits = 0
    self._cache_misses = 0
    self._dedup_count = 0
    self._total_offspring_attempted = 0
```

```python
    def _build_result(self, generations_run: int) -> ExperimentResult:
        ...
        dedup_rate = (
            self._dedup_count / self._total_offspring_attempted
            if self._total_offspring_attempted > 0 else 0.0
        )
        metrics = {
            "cache_hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0 else 0.0
            ),
            "identity_dedup_rate": dedup_rate,
            "stagnation_counter": float(
                len(self._memory.best_fitness_history) - len(set(self._memory.best_fitness_history))
            ),
            "estimated_cost_usd": self._scheduler.tracker.estimated_cost_usd,
        }
        return ExperimentResult(
            ...,
            metrics=metrics,
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core/test_engine.py::TestExperimentResultMetrics -v`
Expected: PASS

**Step 5: Run full suite**

Run: `uv run pytest -x -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add evoforge/core/engine.py tests/test_core/test_engine.py
git commit -m "Add design-specified metrics to ExperimentResult"
```

---

## Task 10: Fix engine bug — `memory.update` uses wrong variable name

**Files:**
- Modify: `evoforge/core/engine.py` (line 254)

The engine has `memory.update(credited_offspring, gen)` but should be `self._memory.update(...)`.

**Step 1: This is already caught by existing tests failing — fix directly**

Line 254: change `memory.update(credited_offspring, gen)` → `self._memory.update(credited_offspring, gen)`

**Step 2: Run full suite**

Run: `uv run pytest -x -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add evoforge/core/engine.py
git commit -m "Fix memory.update to use self._memory"
```

---

## Task 11: Run quality gate and final verification

**Step 1: Run full quality gate**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v
```

Expected: ALL PASS, zero errors

**Step 2: Fix any ruff/mypy issues discovered**

Iterate until clean.

**Step 3: Final commit if any fixes were needed**

```bash
git add -u
git commit -m "Fix lint and type errors from Phase 1 completion"
```

---

## Execution Order & Parallelism

Tasks 1-2 are sequential (Task 2 depends on Task 1's new ABC).

Tasks 3, 4, 5, 8 can be parallelized after Task 2 (independent file sets):
- **Group A (Task 3):** `mutation.py`, `llm/operators.py`, `test_llm_operators.py`
- **Group B (Task 4):** `config.py`, `lean_default.toml`, `test_config.py`
- **Group C (Task 5):** `engine.py` temperature method only, `test_temperature.py`
- **Group D (Task 8):** `scheduler.py`, `test_scheduler.py`

Tasks 6, 7, 9, 10 modify `engine.py` and should be sequential after the parallel group.

Task 11 is final.

```
Task 1 → Task 2 → [Task 3 | Task 4 | Task 5 | Task 8] → Task 6 → Task 7 → Task 9 → Task 10 → Task 11
```
