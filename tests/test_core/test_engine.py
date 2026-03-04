"""Integration tests for evoforge.core.engine — EvolutionEngine."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import pytest

from evoforge.backends.base import Backend
from evoforge.core.archive import Archive
from evoforge.core.config import (
    EvoforgeConfig,
    EvolutionConfig,
    PopulationConfig,
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
    def cost(self) -> str:
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
    def cost(self) -> str:
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

    def evaluate_stepwise(self, ir: Any, seed: int | None = None) -> tuple[Fitness, Any, Any]:
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


@pytest.fixture
async def archive() -> Archive:
    """Create an in-memory archive for testing."""
    a = Archive("sqlite+aiosqlite://")
    await a.create_tables()
    return a


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
