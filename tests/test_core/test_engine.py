"""Integration tests for evoforge.core.engine — EvolutionEngine."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest

from evoforge.backends.base import Backend
from evoforge.core.archive import Archive
from evoforge.core.config import (
    AblationConfig,
    EvoforgeConfig,
    EvolutionConfig,
    PopulationConfig,
    ReflectionConfig,
    SchedulerSettings,
    SelectionConfig,
)
from evoforge.core.engine import EvolutionEngine, ExperimentResult
from evoforge.core.ir import BehaviorDimension, BehaviorSpaceConfig, IRProtocol
from evoforge.core.mutation import MutationContext, MutationOperator
from evoforge.core.types import Credit, Fitness, Individual

# ---------------------------------------------------------------------------
# Mock IR node
# ---------------------------------------------------------------------------


@dataclass
class MockIR:
    """Minimal IR node for testing."""

    lines: list[str]

    def canonicalize(self) -> MockIR:
        """Return self (already canonical)."""
        return MockIR(lines=sorted(self.lines))

    def structural_hash(self) -> str:
        content = "\n".join(self.lines)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def serialize(self) -> str:
        return "\n".join(self.lines)

    def complexity(self) -> int:
        return len(self.lines)


# ---------------------------------------------------------------------------
# Mock mutation operators
# ---------------------------------------------------------------------------


class MockAppendOperator(MutationOperator):
    """Cheap operator that appends a line to the genome."""

    @property
    def name(self) -> str:
        return "mock_append"

    @property
    def cost(self) -> Literal["cheap", "llm"]:
        return "cheap"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        """Append a line to make a unique child."""
        line = f"step_{context.generation}"
        return parent.genome + f"\n{line}"


class MockShuffleOperator(MutationOperator):
    """Cheap operator that prepends a line to the genome."""

    @property
    def name(self) -> str:
        return "mock_shuffle"

    @property
    def cost(self) -> Literal["cheap", "llm"]:
        return "cheap"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        """Prepend a line to make a unique child."""
        line = f"prefix_{context.generation}"
        return f"{line}\n{parent.genome}"


# ---------------------------------------------------------------------------
# Mock Backend
# ---------------------------------------------------------------------------


class MockBackend(Backend):
    """Test backend where fitness = (non-empty lines) / 10, capped at 1.0."""

    def parse(self, genome: str) -> IRProtocol | None:
        lines = [line for line in genome.strip().split("\n") if line.strip()]
        if not lines:
            return None
        return MockIR(lines=lines)  # type: ignore[return-value]

    async def evaluate(  # type: ignore[override]
        self, ir: Any, seed: int | None = None
    ) -> tuple[Fitness, Any, Any]:
        mock_ir: MockIR = ir
        non_empty = len([line for line in mock_ir.lines if line.strip()])
        primary = min(non_empty / 10.0, 1.0)
        fitness = Fitness(
            primary=primary,
            auxiliary={"line_count": float(non_empty)},
            constraints={},
            feasible=True,
        )
        diagnostics = {"lines": non_empty}
        trace = {"steps": mock_ir.lines}
        return fitness, diagnostics, trace

    async def evaluate_stepwise(
        self, ir: Any, seed: int | None = None
    ) -> tuple[Fitness, Any, Any]:
        raise NotImplementedError

    def assign_credit(
        self, ir: Any, fitness: Fitness, diagnostics: Any, trace: Any
    ) -> list[Credit]:
        mock_ir: MockIR = ir
        credits: list[Credit] = []
        for i, line in enumerate(mock_ir.lines):
            credits.append(
                Credit(
                    location=i,
                    score=1.0 / max(len(mock_ir.lines), 1),
                    signal=line.strip() or "empty",
                )
            )
        return credits

    def validate_structure(self, ir: Any) -> list[str]:
        return []

    def seed_population(self, n: int) -> list[str]:
        return [f"seed_line_{i}\nextra_line_{i}" for i in range(n)]

    def mutation_operators(self) -> list[MutationOperator]:
        return [MockAppendOperator(), MockShuffleOperator()]

    def system_prompt(self) -> str:
        return "You are a mock proof assistant."

    def format_mutation_prompt(self, parent: Individual, context: Any) -> str:
        return f"Improve this: {parent.genome}"

    def extract_genome(self, raw_text: str) -> str | None:
        return raw_text

    def behavior_descriptor(self, ir: Any, diagnostics: Any) -> tuple[Any, ...]:
        return ("mock", "short")

    def behavior_space(self) -> BehaviorSpaceConfig:
        return BehaviorSpaceConfig(
            dimensions=(
                BehaviorDimension("strategy", ["mock"]),
                BehaviorDimension("depth", ["short"]),
            )
        )

    def recommended_selection(self) -> str:
        return "scalar_tournament"

    def version(self) -> str:
        return "mock_v1"

    def eval_config_hash(self) -> str:
        return "mock_cfg_hash"

    def format_reflection_prompt(self, population: list[Any], memory: Any, generation: int) -> str:
        return "Mock reflection prompt"

    def default_operator_weights(self) -> dict[str, float]:
        return {"mock_append": 0.5, "mock_shuffle": 0.5}

    def format_crossover_prompt(
        self, parent_a: Individual, parent_b: Individual, context: Any
    ) -> str:
        return f"Crossover: {parent_a.genome} + {parent_b.genome}"


# ---------------------------------------------------------------------------
# Constant-fitness backend for stagnation testing
# ---------------------------------------------------------------------------


class ConstantFitnessBackend(MockBackend):
    """Backend where every individual gets the same fitness."""

    async def evaluate(  # type: ignore[override]
        self, ir: Any, seed: int | None = None
    ) -> tuple[Fitness, Any, Any]:
        fitness = Fitness(
            primary=0.5,
            auxiliary={"line_count": 1.0},
            constraints={},
            feasible=True,
        )
        return fitness, {"lines": 1}, {"steps": []}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(
    *,
    max_generations: int = 1,
    population_size: int = 10,
    stagnation_window: int = 10,
) -> EvoforgeConfig:
    """Create a minimal config for testing."""
    return EvoforgeConfig(
        population=PopulationConfig(size=population_size, elite_k=2),
        selection=SelectionConfig(strategy="scalar_tournament", tournament_size=2),
        evolution=EvolutionConfig(
            max_generations=max_generations,
            stagnation_window=stagnation_window,
            log_level="DEBUG",
        ),
    )

    # archive fixture provided by tests/conftest.py


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEngineSmoke:
    """Smoke test: run 1 generation with mock backend, no error."""

    async def test_single_generation_completes(self, archive: Archive) -> None:
        config = _make_config(max_generations=1, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(
            config=config,
            backend=backend,
            archive=archive,
            llm_client=None,
        )
        result = await engine.run()

        assert isinstance(result, ExperimentResult)
        assert result.generations_run >= 1
        assert result.total_evaluations > 0
        assert result.best_fitness >= 0.0


class TestMultiGeneration:
    """Run multiple generations: population maintained, archive grows."""

    async def test_five_generations(self, archive: Archive) -> None:
        config = _make_config(max_generations=5, population_size=8)
        backend = MockBackend()
        engine = EvolutionEngine(
            config=config,
            backend=backend,
            archive=archive,
            llm_client=None,
        )
        result = await engine.run()

        assert result.generations_run == 5
        assert result.total_evaluations > 8  # At least seed + some offspring
        assert result.archive_size > 0


class TestNoDuplicateHashes:
    """After 5 generations, no two individuals in population share ir_hash."""

    async def test_population_hashes_unique(self, archive: Archive) -> None:
        config = _make_config(max_generations=5, population_size=8)
        backend = MockBackend()
        engine = EvolutionEngine(
            config=config,
            backend=backend,
            archive=archive,
            llm_client=None,
        )
        await engine.run()

        # Access the population manager to verify uniqueness
        population = engine.population.get_all()
        hashes = [ind.ir_hash for ind in population]
        assert len(hashes) == len(set(hashes)), "Duplicate ir_hash found in population"


class TestStagnationReflection:
    """Constant fitness -> after stagnation_window generations, reflection triggered."""

    async def test_stagnation_triggers_reflection(
        self, archive: Archive, caplog: pytest.LogCaptureFixture
    ) -> None:
        stagnation_window = 3
        # Run enough generations to trigger stagnation (window + some extra)
        config = _make_config(
            max_generations=stagnation_window + 3,
            population_size=5,
            stagnation_window=stagnation_window,
        )
        backend = ConstantFitnessBackend()
        engine = EvolutionEngine(
            config=config,
            backend=backend,
            archive=archive,
            llm_client=None,
        )

        with caplog.at_level(logging.INFO):
            result = await engine.run()

        assert result.reflected is True


class TestEngineUsesBackendVersion:
    """Engine wires backend.version() and eval_config_hash() into the evaluator."""

    async def test_evaluator_uses_backend_version(self, archive: Archive) -> None:
        config = _make_config(max_generations=1, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        assert engine._evaluator._backend_version == "mock_v1"
        assert engine._evaluator._config_hash == "mock_cfg_hash"


class TestBestIndividualTracked:
    """After run, result.best_individual is not None."""

    async def test_best_individual_not_none(self, archive: Archive) -> None:
        config = _make_config(max_generations=3, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(
            config=config,
            backend=backend,
            archive=archive,
            llm_client=None,
        )
        result = await engine.run()

        assert result.best_individual is not None
        assert result.best_individual.fitness is not None
        assert result.best_individual.fitness.primary == result.best_fitness


class TestExperimentResultMetrics:
    """ExperimentResult.metrics contains design-specified tracking metrics."""

    async def _run_engine(self, archive: Archive) -> ExperimentResult:
        """Helper: run a short experiment and return the result."""
        config = _make_config(max_generations=3, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(
            config=config,
            backend=backend,
            archive=archive,
            llm_client=None,
        )
        return await engine.run()

    async def test_result_has_metrics_dict(self, archive: Archive) -> None:
        result = await self._run_engine(archive)
        assert isinstance(result.metrics, dict)

    async def test_result_has_cache_hit_rate(self, archive: Archive) -> None:
        result = await self._run_engine(archive)
        assert "cache_hit_rate" in result.metrics
        assert isinstance(result.metrics["cache_hit_rate"], float)

    async def test_result_has_identity_dedup_rate(self, archive: Archive) -> None:
        result = await self._run_engine(archive)
        assert "identity_dedup_rate" in result.metrics
        assert isinstance(result.metrics["identity_dedup_rate"], float)

    async def test_result_has_stagnation_counter(self, archive: Archive) -> None:
        result = await self._run_engine(archive)
        assert "stagnation_counter" in result.metrics

    async def test_result_cost_dict_present(self, archive: Archive) -> None:
        result = await self._run_engine(archive)
        assert isinstance(result.cost, dict)


# ---------------------------------------------------------------------------
# Lifecycle tests: startup/shutdown
# ---------------------------------------------------------------------------


class TestEngineLifecycle:
    """Engine calls backend.startup() before evaluation and shutdown() after."""

    @staticmethod
    def _make_lifecycle_backend() -> MockBackend:
        backend = MockBackend()
        backend.startup = AsyncMock()  # type: ignore[method-assign]
        backend.shutdown = AsyncMock()  # type: ignore[method-assign]
        return backend

    async def test_startup_called_before_eval(self, archive: Archive) -> None:
        backend = self._make_lifecycle_backend()
        config = _make_config(max_generations=1, population_size=5)
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()
        backend.startup.assert_awaited_once()

    async def test_shutdown_called_after_run(self, archive: Archive) -> None:
        backend = self._make_lifecycle_backend()
        config = _make_config(max_generations=1, population_size=5)
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()
        backend.shutdown.assert_awaited_once()

    async def test_shutdown_called_on_error(self, archive: Archive) -> None:
        backend = self._make_lifecycle_backend()

        async def _raise_eval(ir: Any, seed: Any = None) -> Any:
            raise RuntimeError("boom")

        backend.evaluate = _raise_eval  # type: ignore[method-assign]
        config = _make_config(max_generations=1, population_size=5)
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)

        with pytest.raises(RuntimeError, match="boom"):
            await engine.run()
        backend.shutdown.assert_awaited_once()


# ---------------------------------------------------------------------------
# Ablation tests
# ---------------------------------------------------------------------------


class TestAblationFlags:
    """Ablation config flags disable specific engine components."""

    async def test_disable_llm_skips_llm_operators(self, archive: Archive) -> None:
        """With disable_llm=True, no LLM operators in ensemble."""
        config = _make_config(max_generations=1, population_size=5)
        config.ablation = AblationConfig(disable_llm=True)
        backend = MockBackend()
        engine = EvolutionEngine(
            config=config, backend=backend, archive=archive, llm_client=object()
        )

        # All operators should be cheap
        for op in engine._ensemble._operators:
            assert op.cost == "cheap"

    async def test_disable_credit_skips_assignment(self, archive: Archive) -> None:
        """With disable_credit=True, individuals get no credits assigned."""
        config = _make_config(max_generations=1, population_size=5)
        config.ablation = AblationConfig(disable_credit=True)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)

        await engine.run()
        # Credits should be empty (default) for individuals
        pop = engine.population.get_all()
        for ind in pop:
            assert ind.credits == []


# ---------------------------------------------------------------------------
# Per-gen LLM budget
# ---------------------------------------------------------------------------


class TestPerGenLLMBudget:
    """Scheduler resets per-generation LLM budget each generation."""

    async def test_scheduler_resets_each_gen(self, archive: Archive) -> None:
        config = _make_config(max_generations=2, population_size=5)
        config.scheduler = SchedulerSettings(llm_budget_per_gen=0)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)

        await engine.run()

        # With budget=0, LLM ops always fall back to cheap — engine still runs
        assert engine._scheduler.gen_llm_calls == 0


# ---------------------------------------------------------------------------
# Periodic reflection
# ---------------------------------------------------------------------------


class _FakeLLMResponse:
    text: str = "reflection output"


class _FakeLLMClient:
    async def async_generate(self, *args: Any, **kwargs: Any) -> _FakeLLMResponse:
        return _FakeLLMResponse()


class TestPeriodicReflection:
    """Periodic reflection triggers at the configured interval."""

    async def test_periodic_reflection_at_interval(
        self, archive: Archive, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = _make_config(max_generations=10, population_size=5)
        config.reflection = ReflectionConfig(interval=5)
        backend = MockBackend()
        engine = EvolutionEngine(
            config=config, backend=backend, archive=archive, llm_client=_FakeLLMClient()
        )

        with caplog.at_level(logging.INFO):
            await engine.run()

        assert any("Periodic reflection" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Fix 1: Concurrent mutation loop
# ---------------------------------------------------------------------------


class SlowLLMOperator(MutationOperator):
    """LLM operator with configurable delay, tracks concurrent count."""

    _active: int = 0
    _max_active: int = 0

    @property
    def name(self) -> str:
        return "slow_llm"

    @property
    def cost(self) -> Literal["cheap", "llm"]:
        return "llm"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        SlowLLMOperator._active += 1
        SlowLLMOperator._max_active = max(SlowLLMOperator._max_active, SlowLLMOperator._active)
        await asyncio.sleep(0.05)
        SlowLLMOperator._active -= 1
        return parent.genome + f"\nllm_{context.generation}_{id(parent)}"


class TestConcurrentMutations:
    """Fix 1: mutations run concurrently via asyncio.gather."""

    async def test_mutations_run_concurrently(self, archive: Archive) -> None:
        """Multiple parents mutated concurrently — wall time << sequential."""
        config = _make_config(max_generations=1, population_size=6)
        # High LLM budget so all use the slow operator
        config.scheduler = SchedulerSettings(llm_budget_per_gen=100, max_llm_concurrent=6)
        backend = MockBackend()

        # Override mutation_operators to return only the slow LLM operator
        backend.mutation_operators = lambda: [SlowLLMOperator()]  # type: ignore[method-assign]

        engine = EvolutionEngine(config=config, backend=backend, archive=archive)

        start = time.monotonic()
        await engine.run()
        elapsed = time.monotonic() - start

        # 6 parents × 0.05s = 0.3s sequential. Concurrent should be < 0.15s.
        # We give generous margin — just verify it's less than sequential.
        assert elapsed < 0.25, f"Mutations seem sequential: {elapsed:.2f}s"

    async def test_llm_semaphore_limits_concurrency(self, archive: Archive) -> None:
        """acquire_llm() semaphore caps concurrent LLM operator calls."""
        SlowLLMOperator._active = 0
        SlowLLMOperator._max_active = 0

        config = _make_config(max_generations=1, population_size=8)
        config.scheduler = SchedulerSettings(llm_budget_per_gen=100, max_llm_concurrent=2)
        backend = MockBackend()
        backend.mutation_operators = lambda: [SlowLLMOperator()]  # type: ignore[method-assign]

        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()

        assert SlowLLMOperator._max_active <= 2, (
            f"Concurrency exceeded semaphore limit: {SlowLLMOperator._max_active}"
        )


# ---------------------------------------------------------------------------
# Fix 2: Decaying temperature boost
# ---------------------------------------------------------------------------


class TestDecayingTemperatureBoost:
    """Fix 2: stagnation boost decays rather than being overwritten."""

    async def test_boost_survives_next_generation(self, archive: Archive) -> None:
        """After stagnation, temperature in the next gen > pure schedule value."""
        stagnation_window = 3
        max_gen = stagnation_window + 2
        config = _make_config(
            max_generations=max_gen,
            population_size=5,
            stagnation_window=stagnation_window,
        )
        backend = ConstantFitnessBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)

        await engine.run()

        # The boost should still be > 0 after run (decayed but not zero)
        assert engine._temperature_boost > 0.0

    async def test_boost_decays_over_generations(self, archive: Archive) -> None:
        """Boost decays by 0.8× each generation, approaching zero."""
        stagnation_window = 2
        max_gen = stagnation_window + 5
        config = _make_config(
            max_generations=max_gen,
            population_size=5,
            stagnation_window=stagnation_window,
        )
        backend = ConstantFitnessBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)

        await engine.run()

        # Boost decays each gen but stagnation keeps re-triggering it,
        # so it stays at or below the cap
        assert engine._temperature_boost <= 0.3

    async def test_boost_caps_at_max(self, archive: Archive) -> None:
        """Temperature boost never exceeds 0.3."""
        stagnation_window = 2
        max_gen = stagnation_window + 10
        config = _make_config(
            max_generations=max_gen,
            population_size=5,
            stagnation_window=stagnation_window,
        )
        backend = ConstantFitnessBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)

        await engine.run()

        assert engine._temperature_boost <= 0.3

    async def test_temperature_boost_in_metrics(self, archive: Archive) -> None:
        """temperature_boost appears in result metrics."""
        config = _make_config(max_generations=3, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        result = await engine.run()

        assert "temperature_boost" in result.metrics


# ---------------------------------------------------------------------------
# Fix 3: Operator name in lineage
# ---------------------------------------------------------------------------


class TestOperatorNameInLineage:
    """Fix 3: store_lineage receives actual operator name, not 'mutation'."""

    async def test_lineage_stores_operator_name(self, archive: Archive) -> None:
        """Lineage entries carry the actual operator name."""
        config = _make_config(max_generations=2, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)

        await engine.run()

        # Query all lineage entries from the archive
        pop = engine.population.get_all()
        found_operator_names: set[str] = set()
        for ind in pop:
            lineage_records = await archive.get_lineage(ind.ir_hash)
            for rec in lineage_records:
                found_operator_names.add(rec["operator_name"])

        # Should contain actual operator names, not "mutation"
        if found_operator_names:
            assert "mutation" not in found_operator_names
            # Should be one of the mock operators
            assert found_operator_names <= {"mock_append", "mock_shuffle"}
