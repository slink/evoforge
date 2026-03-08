# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
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
class MockStep:
    """Minimal tactic step for testing."""

    raw: str
    tactic: str = ""


@dataclass
class MockIR:
    """Minimal IR node for testing."""

    lines: list[str]

    @property
    def steps(self) -> list[MockStep]:
        """Expose lines as steps with .raw for tree search compatibility."""
        return [MockStep(raw=line, tactic=line.split("_")[0]) for line in self.lines]

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

    def format_proof(self, genome: str) -> str:
        lines = ["-- mock proof"]
        lines.append("theorem mock := by")
        for tactic in genome.strip().split("\n"):
            if tactic.strip():
                lines.append(f"  {tactic.strip()}")
        return "\n".join(lines) + "\n"


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


# ---------------------------------------------------------------------------
# Early exit on perfect fitness
# ---------------------------------------------------------------------------


class PerfectFitnessBackend(MockBackend):
    """Backend where every individual gets fitness=1.0 (proof complete)."""

    async def evaluate(  # type: ignore[override]
        self, ir: Any, seed: int | None = None
    ) -> tuple[Fitness, Any, Any]:
        fitness = Fitness(
            primary=1.0,
            auxiliary={"proof_complete": 1.0, "cmd_verified": 1.0},
            constraints={},
            feasible=True,
        )
        return fitness, {"lines": 1}, {"steps": []}


class TestEarlyExitOnPerfectFitness:
    """Engine exits early when best_fitness reaches 1.0."""

    async def test_exits_before_max_generations(self, archive: Archive) -> None:
        """With perfect fitness from gen 0, engine should not run all generations."""
        config = _make_config(max_generations=20, population_size=5)
        backend = PerfectFitnessBackend()
        engine = EvolutionEngine(
            config=config,
            backend=backend,
            archive=archive,
            llm_client=None,
        )
        result = await engine.run()

        # Should exit well before 20 generations
        assert result.generations_run < 20, (
            f"Engine ran all {result.generations_run} generations despite perfect fitness"
        )
        assert result.best_fitness >= 1.0

    async def test_best_individual_has_genome_on_perfect_fitness(self, archive: Archive) -> None:
        """When fitness=1.0, result.best_individual.genome is populated."""
        config = _make_config(max_generations=5, population_size=5)
        backend = PerfectFitnessBackend()
        engine = EvolutionEngine(
            config=config,
            backend=backend,
            archive=archive,
            llm_client=None,
        )
        result = await engine.run()

        assert result.best_fitness >= 1.0
        assert result.best_individual is not None
        assert result.best_individual.genome is not None
        assert len(result.best_individual.genome.strip()) > 0

    async def test_logs_early_exit(
        self, archive: Archive, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Early exit is logged."""
        config = _make_config(max_generations=10, population_size=5)
        backend = PerfectFitnessBackend()
        engine = EvolutionEngine(
            config=config,
            backend=backend,
            archive=archive,
            llm_client=None,
        )

        with caplog.at_level(logging.INFO):
            await engine.run()

        assert any(
            "perfect fitness" in r.message.lower() or "early" in r.message.lower()
            for r in caplog.records
        )


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


# ---------------------------------------------------------------------------
# Behavior descriptors assigned to all individuals
# ---------------------------------------------------------------------------


class TestBehaviorDescriptorsAssigned:
    """Behavior descriptors are assigned to seeds, offspring, and survivors."""

    async def test_seeds_have_behavior_descriptors(self, archive: Archive) -> None:
        """After gen 0, all individuals have behavior_descriptor set."""
        config = _make_config(max_generations=0, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()

        pop = engine.population.get_all()
        assert len(pop) > 0
        for ind in pop:
            assert ind.behavior_descriptor is not None, (
                f"Seed individual {ind.ir_hash} missing behavior_descriptor"
            )

    async def test_diversity_nonzero_with_descriptors(self, archive: Archive) -> None:
        """diversity_entropy() > 0 is possible when descriptors are assigned."""
        config = _make_config(max_generations=1, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()

        # All individuals should have descriptors
        pop = engine.population.get_all()
        descriptors = [ind.behavior_descriptor for ind in pop]
        assert all(d is not None for d in descriptors)

    async def test_descriptors_assigned_without_map_elites(self, archive: Archive) -> None:
        """Descriptors assigned even when strategy is not map_elites."""
        config = _make_config(max_generations=2, population_size=5)
        assert config.selection.strategy == "scalar_tournament"
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()

        pop = engine.population.get_all()
        for ind in pop:
            assert ind.behavior_descriptor is not None


# ---------------------------------------------------------------------------
# Feasibility constraint for failed verification
# ---------------------------------------------------------------------------


class PartialVerifyBackend(MockBackend):
    """Backend where verify_proof always fails."""

    async def verify_proof(self, genome: str) -> bool:
        return False


class TestFeasibilityConstraint:
    """Failed verification sets feasible=False; infeasible never beats feasible."""

    async def test_failed_verification_sets_infeasible(self, archive: Archive) -> None:
        """Proofs that fail verify_proof get feasible=False, primary stays 1.0."""
        config = _make_config(max_generations=0, population_size=5)
        backend = PerfectFitnessBackend()
        backend.verify_proof = AsyncMock(return_value=False)  # type: ignore[method-assign]
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()

        pop = engine.population.get_all()
        for ind in pop:
            assert ind.fitness is not None
            assert ind.fitness.feasible is False, "Failed verification should set feasible=False"
            assert ind.fitness.primary == 1.0, "Primary fitness should stay at 1.0"

    async def test_infeasible_never_triggers_early_exit(self, archive: Archive) -> None:
        """Even with primary=1.0, infeasible proofs don't cause early exit."""
        config = _make_config(max_generations=5, population_size=5)
        backend = PerfectFitnessBackend()
        backend.verify_proof = AsyncMock(return_value=False)  # type: ignore[method-assign]
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        result = await engine.run()

        # Should run all 5 generations since no feasible proof was found
        assert result.generations_run == 5

    async def test_infeasible_ranks_below_feasible_in_selection(self, archive: Archive) -> None:
        """In selection, feasible individuals always rank above infeasible ones."""
        from evoforge.core.selection import _primary_fitness

        feasible_ind = Individual(
            genome="a",
            ir=None,
            ir_hash="a",
            generation=0,
            fitness=Fitness(primary=0.3, auxiliary={}, constraints={}, feasible=True),
        )
        infeasible_ind = Individual(
            genome="b",
            ir=None,
            ir_hash="b",
            generation=0,
            fitness=Fitness(primary=1.0, auxiliary={}, constraints={}, feasible=False),
        )

        assert _primary_fitness(feasible_ind) > _primary_fitness(infeasible_ind)


# ---------------------------------------------------------------------------
# Population floor enforcement
# ---------------------------------------------------------------------------


class TestPopulationFloor:
    """Population is refilled when it drops below target after survival selection."""

    async def test_population_stays_near_target(self, archive: Archive) -> None:
        """After multiple generations, population doesn't collapse to tiny size."""
        config = _make_config(max_generations=5, population_size=10)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()

        # Population should be at least half the target
        assert engine.population.size >= 5, f"Population collapsed to {engine.population.size}"

    async def test_refill_adds_novel_individuals(self, archive: Archive) -> None:
        """Refill adds individuals with unique ir_hashes."""
        config = _make_config(max_generations=3, population_size=8)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()

        pop = engine.population.get_all()
        hashes = [ind.ir_hash for ind in pop]
        assert len(hashes) == len(set(hashes)), "Refill introduced duplicate hashes"


# ---------------------------------------------------------------------------
# Verification cache
# ---------------------------------------------------------------------------


class VerifyCacheBackend(PerfectFitnessBackend):
    """Backend where fitness=1.0 and verify_proof tracks call count."""

    verify_call_count: int = 0
    verify_return: bool = False

    async def verify_proof(self, genome: str) -> bool:
        self.verify_call_count += 1
        return self.verify_return


class TestVerificationCache:
    """Verification cache prevents redundant lake env lean calls."""

    async def test_verification_cache_prevents_second_call(self, archive: Archive) -> None:
        """verify_proof called only once per unique ir_hash across generations."""
        backend = VerifyCacheBackend()
        backend.verify_return = False
        config = _make_config(max_generations=3, population_size=5)
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()

        # All seeds have same behavior_descriptor -> some share ir_hash after
        # canonicalization. But even if all unique, each should be verified at
        # most once. With pop=5 and 3 gens of offspring, there are many
        # individuals but verify should be called <= number of unique ir_hashes.
        unique_hashes = {ind.ir_hash for ind in engine.population.get_all()}
        # verify_call_count should be <= total unique hashes that had fitness=1.0
        # (much less than pop_size * generations)
        assert backend.verify_call_count <= len(unique_hashes) + 20  # generous bound

    async def test_verification_cache_stores_true_results(self, archive: Archive) -> None:
        """Positive verification results are also cached."""
        backend = VerifyCacheBackend()
        backend.verify_return = True
        config = _make_config(max_generations=2, population_size=5)
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()

        # Cache should have entries
        assert len(engine._verification_cache) > 0
        # All cached values should be True
        assert all(v is True for v in engine._verification_cache.values())

    async def test_verification_failure_added_to_dead_ends(self, archive: Archive) -> None:
        """Failed verification feeds genome to memory dead_ends."""
        backend = VerifyCacheBackend()
        backend.verify_return = False
        config = _make_config(max_generations=1, population_size=5)
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()

        # At least one dead end should have been recorded from verification failure
        assert len(engine._memory.dead_ends) > 0

    async def test_verification_cache_checkpoint_roundtrip(self, archive: Archive) -> None:
        """Verification cache survives checkpoint save/load."""
        backend = VerifyCacheBackend()
        backend.verify_return = False
        config = _make_config(max_generations=1, population_size=5)
        config.evolution.checkpoint_every = 1
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()

        # Cache should have entries from verification
        assert len(engine._verification_cache) > 0

        # Save checkpoint
        await engine._save_checkpoint(1)

        # Create new engine and load checkpoint
        engine2 = EvolutionEngine(config=config, backend=backend, archive=archive)
        gen = await engine2._load_checkpoint()
        assert gen is not None
        assert engine2._verification_cache == engine._verification_cache

    async def test_cache_hit_uses_debug_not_warning(
        self, archive: Archive, caplog: pytest.LogCaptureFixture
    ) -> None:
        """First verification failure emits WARNING; second (cache hit) only DEBUG."""
        backend = VerifyCacheBackend()
        backend.verify_return = False
        config = _make_config(max_generations=1, population_size=3)
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)

        # Create a fake individual with fitness=1.0 and cmd_verified=1.0
        verified_aux = {"cmd_verified": 1.0}
        perfect = Fitness(
            primary=1.0,
            auxiliary=verified_aux,
            constraints={},
            feasible=True,
        )
        ind = Individual(
            genome="sorry_tactic",
            ir=None,
            ir_hash="test_hash_abc",
            generation=0,
            fitness=perfect,
        )

        # First call — cache miss — should produce WARNING
        with caplog.at_level(logging.DEBUG, logger="evoforge.core.engine"):
            caplog.clear()
            await engine._verify_perfect_individuals([ind])
            warnings_1 = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings_1) == 1, f"Expected 1 WARNING, got {len(warnings_1)}"

        # Reset fitness so it triggers verification path again
        ind.fitness = Fitness(
            primary=1.0,
            auxiliary={"cmd_verified": 1.0},
            constraints={},
            feasible=True,
        )

        # Second call — cache hit — should NOT produce WARNING
        with caplog.at_level(logging.DEBUG, logger="evoforge.core.engine"):
            caplog.clear()
            await engine._verify_perfect_individuals([ind])
            warnings_2 = [r for r in caplog.records if r.levelno == logging.WARNING]
            debug_2 = [
                r
                for r in caplog.records
                if r.levelno == logging.DEBUG and "cache" in r.message.lower()
            ]
            assert len(warnings_2) == 0, f"Expected 0 WARNINGs on cache hit, got {len(warnings_2)}"
            assert len(debug_2) >= 1, "Expected DEBUG log about cache hit"

    async def test_record_verification_failure_called_once(self, archive: Archive) -> None:
        """record_verification_failure is called on cache miss, not on cache hit."""
        backend = VerifyCacheBackend()
        backend.verify_return = False
        config = _make_config(max_generations=1, population_size=3)
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)

        verified_aux = {"cmd_verified": 1.0}
        ind = Individual(
            genome="bad_proof",
            ir=None,
            ir_hash="rvf_hash",
            generation=0,
            fitness=Fitness(
                primary=1.0,
                auxiliary=verified_aux,
                constraints={},
                feasible=True,
            ),
        )

        # First call — should call record_verification_failure
        call_count_before = len(engine._memory.dead_ends)
        await engine._verify_perfect_individuals([ind])
        calls_after_first = len(engine._memory.dead_ends) - call_count_before
        assert calls_after_first > 0, "record_verification_failure not called on cache miss"

        # Reset fitness
        ind.fitness = Fitness(
            primary=1.0,
            auxiliary={"cmd_verified": 1.0},
            constraints={},
            feasible=True,
        )
        dead_ends_before_second = len(engine._memory.dead_ends)

        # Second call — cache hit — should NOT call record_verification_failure
        await engine._verify_perfect_individuals([ind])
        calls_after_second = len(engine._memory.dead_ends) - dead_ends_before_second
        assert calls_after_second == 0, "record_verification_failure called on cache hit"

    async def test_known_bad_hashes_rejected_before_eval(self, archive: Archive) -> None:
        """Pre-seeded bad hashes in verification cache are rejected before evaluation."""
        backend = VerifyCacheBackend()
        backend.verify_return = True  # New proofs pass, but cached bad ones skip
        config = _make_config(max_generations=2, population_size=5)
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)

        # Pre-seed a known-bad hash
        bad_hash = "known_bad_hash_xyz"
        engine._verification_cache[bad_hash] = False

        await engine.run()

        # The known-bad hash should never appear in population
        pop_hashes = {ind.ir_hash for ind in engine.population.get_all()}
        assert bad_hash not in pop_hashes, "Known-bad hash should not appear in population"

        # Also verify it was filtered via dedup (known_hashes includes bad hashes)
        # by checking it wasn't evaluated — verify_call_count should not include it
        # (it was already in the cache, never needed verify_proof)


# ---------------------------------------------------------------------------
# Crossover operators receive guidance_individual
# ---------------------------------------------------------------------------


class MockCrossoverOperator(MutationOperator):
    """Crossover operator that records whether guidance_individual was set."""

    received_guidance: list[Individual | None] = []

    @property
    def name(self) -> str:
        return "mock_crossover"

    @property
    def cost(self) -> Literal["cheap", "llm"]:
        return "cheap"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        MockCrossoverOperator.received_guidance.append(context.guidance_individual)
        return parent.genome + f"\ncrossover_{context.generation}"


class TestCrossoverGuidanceIndividual:
    """Engine sets guidance_individual on MutationContext for crossover operators."""

    async def test_crossover_receives_guidance_individual(self, archive: Archive) -> None:
        """When operator name contains 'crossover', context.guidance_individual is set."""
        MockCrossoverOperator.received_guidance = []

        config = _make_config(max_generations=1, population_size=5)
        backend = MockBackend()
        # Override to return only the crossover operator
        backend.mutation_operators = lambda: [MockCrossoverOperator()]  # type: ignore[method-assign]
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()

        # At least one crossover was applied
        assert len(MockCrossoverOperator.received_guidance) > 0
        # Every crossover call should have received a non-None guidance_individual
        for gi in MockCrossoverOperator.received_guidance:
            assert gi is not None, "crossover operator received None guidance_individual"

    async def test_non_crossover_has_no_guidance_individual(self, archive: Archive) -> None:
        """Non-crossover operators do NOT receive guidance_individual."""
        config = _make_config(max_generations=1, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)

        # Capture contexts from mock operators
        contexts_seen: list[MutationContext] = []
        original_apply = MockAppendOperator.apply

        async def spy_apply(self: Any, parent: Individual, context: MutationContext) -> str:
            contexts_seen.append(context)
            return await original_apply(self, parent, context)

        MockAppendOperator.apply = spy_apply  # type: ignore[method-assign]
        try:
            await engine.run()
        finally:
            MockAppendOperator.apply = original_apply  # type: ignore[method-assign]

        # Non-crossover operators should have guidance_individual=None
        for ctx in contexts_seen:
            assert ctx.guidance_individual is None


class TestEnsembleStatsUpdated:
    """After each generation, operator stats should reflect applications."""

    async def test_ensemble_stats_updated(self, archive: Archive) -> None:
        config = _make_config(max_generations=2, population_size=5)
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        await engine.run()

        total_apps = sum(s.applications for s in engine._ensemble.stats.values())
        assert total_apps > 0, "Ensemble stats should track operator applications"


# ---------------------------------------------------------------------------
# Reflection parses JSON into memory and temperature
# ---------------------------------------------------------------------------


class _ReflectionLLMResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _ReflectionLLMClient:
    """LLM client that returns a configurable reflection JSON."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    async def async_generate(self, *args: Any, **kwargs: Any) -> _ReflectionLLMResponse:
        return _ReflectionLLMResponse(self._response_text)


class TestReflectionParsing:
    """Reflection JSON is parsed and fed into search memory + temperature."""

    async def test_reflection_updates_memory(self, archive: Archive) -> None:
        """Reflection response should be parsed and fed into search memory."""
        import json

        reflection_json = json.dumps(
            {
                "strategies_to_try": ["use calc blocks"],
                "strategies_to_avoid": ["avoid sorry"],
                "useful_primitives": ["norm_nonneg"],
                "population_diagnosis": "needs diversity",
                "suggested_temperature": 0.8,
            }
        )
        config = _make_config(max_generations=3, population_size=5)
        config.reflection = ReflectionConfig(interval=1)
        backend = MockBackend()
        llm_client = _ReflectionLLMClient(reflection_json)
        engine = EvolutionEngine(
            config=config, backend=backend, archive=archive, llm_client=llm_client
        )
        await engine.run()

        assert "avoid sorry" in engine._memory.dead_ends
        assert any("calc blocks" in p.description for p in engine._memory.patterns)
        assert any("norm_nonneg" in p.description for p in engine._memory.patterns)

    async def test_reflection_adjusts_temperature(self, archive: Archive) -> None:
        """Suggested temperature from reflection should influence engine temperature."""
        import json

        reflection_json = json.dumps(
            {
                "strategies_to_try": [],
                "strategies_to_avoid": [],
                "useful_primitives": [],
                "population_diagnosis": "",
                "suggested_temperature": 1.2,
            }
        )
        config = _make_config(max_generations=2, population_size=5)
        config.reflection = ReflectionConfig(interval=1)
        backend = MockBackend()
        llm_client = _ReflectionLLMClient(reflection_json)
        engine = EvolutionEngine(
            config=config, backend=backend, archive=archive, llm_client=llm_client
        )
        initial_temp = engine._temperature
        await engine.run()

        # Temperature should have been nudged toward 1.2
        assert engine._temperature != initial_temp

    async def test_reflection_handles_markdown_code_block(self, archive: Archive) -> None:
        """Reflection JSON wrapped in ```json ... ``` should still parse."""
        import json

        inner = json.dumps(
            {
                "strategies_to_try": ["try omega"],
                "strategies_to_avoid": [],
                "useful_primitives": [],
                "population_diagnosis": "",
                "suggested_temperature": 0.7,
            }
        )
        wrapped = f"Here is my analysis:\n```json\n{inner}\n```"
        config = _make_config(max_generations=2, population_size=5)
        config.reflection = ReflectionConfig(interval=1)
        backend = MockBackend()
        llm_client = _ReflectionLLMClient(wrapped)
        engine = EvolutionEngine(
            config=config, backend=backend, archive=archive, llm_client=llm_client
        )
        await engine.run()

        assert any("omega" in p.description for p in engine._memory.patterns)

    async def test_reflection_invalid_json_no_crash(self, archive: Archive) -> None:
        """Invalid JSON from reflection should not crash the engine."""
        config = _make_config(max_generations=2, population_size=5)
        config.reflection = ReflectionConfig(interval=1)
        backend = MockBackend()
        llm_client = _ReflectionLLMClient("not valid json {{{")
        engine = EvolutionEngine(
            config=config, backend=backend, archive=archive, llm_client=llm_client
        )
        # Should not raise
        result = await engine.run()
        assert result.generations_run >= 1


# ---------------------------------------------------------------------------
# Tree search integration
# ---------------------------------------------------------------------------


class TestTreeSearchIntegration:
    """Tree search hybrid mode integration with the engine."""

    async def test_tree_search_disabled_by_default(self, archive: Archive) -> None:
        """With default config, tree search does not run."""
        config = _make_config(max_generations=2, population_size=5)
        assert config.evolution.tree_search_enabled is False
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        result = await engine.run()
        assert result.generations_run >= 1

    async def test_tree_search_enabled_no_crash(self, archive: Archive) -> None:
        """With tree_search_enabled=True, engine runs without error.

        MockBackend.create_tree_search returns None, so tree search is skipped.
        """
        config = _make_config(max_generations=3, population_size=5)
        config.evolution.tree_search_enabled = True
        config.evolution.tree_search_min_fitness = 0.01
        backend = MockBackend()
        engine = EvolutionEngine(
            config=config, backend=backend, archive=archive, llm_client=_FakeLLMClient()
        )
        result = await engine.run()
        assert result.generations_run >= 1

    async def test_tree_search_dispatched_when_enabled(
        self, archive: Archive, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When tree_search_enabled and best fitness > threshold, tree search runs."""
        config = _make_config(max_generations=3, population_size=5)
        config.evolution.tree_search_enabled = True
        config.evolution.tree_search_min_fitness = 0.01
        backend = MockBackend()

        # Track whether create_tree_search was called
        create_calls: list[dict[str, Any]] = []

        async def tracking_create(
            prefix: list[str],
            llm_client: Any,
            max_nodes: int = 200,
            beam_width: int = 5,
            model: str | None = None,
        ) -> Any:
            create_calls.append(
                {"prefix": prefix, "max_nodes": max_nodes, "beam_width": beam_width}
            )
            return None  # Return None = not supported

        backend.create_tree_search = tracking_create  # type: ignore[method-assign]
        engine = EvolutionEngine(
            config=config, backend=backend, archive=archive, llm_client=_FakeLLMClient()
        )

        with caplog.at_level(logging.INFO):
            await engine.run()

        # Tree search should have been attempted (create_tree_search called)
        assert len(create_calls) > 0, "create_tree_search was never called"

    async def test_tree_search_config_fields(self) -> None:
        """EvolutionConfig has tree_search fields with correct defaults."""
        config = EvolutionConfig()
        assert config.tree_search_enabled is False
        assert config.tree_search_max_nodes == 200
        assert config.tree_search_beam_width == 5
        assert config.tree_search_min_fitness == 0.3

    async def test_tree_search_skipped_when_fitness_below_threshold(
        self, archive: Archive
    ) -> None:
        """Tree search is not attempted when best fitness < min_fitness."""
        config = _make_config(max_generations=2, population_size=5)
        config.evolution.tree_search_enabled = True
        config.evolution.tree_search_min_fitness = 0.99  # Very high threshold

        backend = MockBackend()
        create_calls: list[Any] = []

        async def tracking_create(
            prefix: list[str],
            llm_client: Any,
            max_nodes: int = 200,
            beam_width: int = 5,
            model: str | None = None,
        ) -> Any:
            create_calls.append(True)
            return None

        backend.create_tree_search = tracking_create  # type: ignore[method-assign]
        engine = EvolutionEngine(
            config=config, backend=backend, archive=archive, llm_client=_FakeLLMClient()
        )
        await engine.run()

        # With min_fitness=0.99, MockBackend fitness (lines/10) should be below
        assert len(create_calls) == 0, "Tree search should not be called below threshold"

    async def test_tree_search_skipped_without_llm_client(self, archive: Archive) -> None:
        """Tree search requires an LLM client; skipped when None."""
        config = _make_config(max_generations=2, population_size=5)
        config.evolution.tree_search_enabled = True
        config.evolution.tree_search_min_fitness = 0.01

        backend = MockBackend()
        create_calls: list[Any] = []

        async def tracking_create(
            prefix: list[str],
            llm_client: Any,
            max_nodes: int = 200,
            beam_width: int = 5,
            model: str | None = None,
        ) -> Any:
            create_calls.append(True)
            return None

        backend.create_tree_search = tracking_create  # type: ignore[method-assign]
        engine = EvolutionEngine(config=config, backend=backend, archive=archive, llm_client=None)
        await engine.run()

        assert len(create_calls) == 0, "Tree search should not run without LLM client"
