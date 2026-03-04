"""Tests for LLM reflection on stagnation in the EvolutionEngine."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from evoforge.core.archive import Archive
from evoforge.core.config import (
    EvoforgeConfig,
    EvolutionConfig,
    LLMConfig,
    PopulationConfig,
    SelectionConfig,
)
from evoforge.core.engine import EvolutionEngine
from tests.test_core.test_engine import ConstantFitnessBackend

# ---------------------------------------------------------------------------
# Mock LLM client for tracking reflection calls
# ---------------------------------------------------------------------------


@dataclass
class _MockLLMResponse:
    """Minimal response matching LLMResponse shape."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str


class _MockReflectionLLMClient:
    """Tracks calls to async_generate, distinguishing reflection from mutation."""

    def __init__(self, reflection_model: str) -> None:
        self.reflection_model = reflection_model
        self.reflection_call_count: int = 0
        self.mutation_call_count: int = 0
        self.last_reflection_model: str | None = None
        self.last_reflection_prompt: str | None = None

    async def async_generate(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> _MockLLMResponse:
        if model == self.reflection_model:
            self.reflection_call_count += 1
            self.last_reflection_model = model
            self.last_reflection_prompt = prompt
        else:
            self.mutation_call_count += 1
        return _MockLLMResponse(
            text="Reflection: try different tactics.",
            input_tokens=100,
            output_tokens=50,
            model=model,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    max_generations: int = 6,
    population_size: int = 5,
    stagnation_window: int = 3,
    reflection_model: str = "test-reflection-model",
) -> EvoforgeConfig:
    """Create a config tuned for stagnation testing."""
    return EvoforgeConfig(
        population=PopulationConfig(size=population_size, elite_k=2),
        selection=SelectionConfig(strategy="scalar_tournament", tournament_size=2),
        evolution=EvolutionConfig(
            max_generations=max_generations,
            stagnation_window=stagnation_window,
            log_level="DEBUG",
        ),
        llm=LLMConfig(reflection_model=reflection_model),
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


class TestReflectionWithLLM:
    """When stagnation is detected and llm_client is available, engine calls LLM."""

    async def test_stagnation_calls_llm_for_reflection(
        self, archive: Archive, caplog: pytest.LogCaptureFixture
    ) -> None:
        reflection_model = "test-reflection-model"
        config = _make_config(
            stagnation_window=3, max_generations=6, reflection_model=reflection_model
        )
        backend = ConstantFitnessBackend()
        mock_llm = _MockReflectionLLMClient(reflection_model=reflection_model)

        engine = EvolutionEngine(
            config=config,
            backend=backend,
            archive=archive,
            llm_client=mock_llm,
        )

        with caplog.at_level(logging.INFO):
            result = await engine.run()

        # Stagnation should have been detected
        assert result.reflected is True
        # The LLM should have been called at least once specifically for reflection
        assert mock_llm.reflection_call_count >= 1, (
            f"Expected at least 1 reflection LLM call, got {mock_llm.reflection_call_count}"
        )

    async def test_reflection_uses_configured_model(self, archive: Archive) -> None:
        reflection_model = "my-custom-reflection-model"
        config = _make_config(
            stagnation_window=3,
            max_generations=6,
            reflection_model=reflection_model,
        )
        backend = ConstantFitnessBackend()
        mock_llm = _MockReflectionLLMClient(reflection_model=reflection_model)

        engine = EvolutionEngine(
            config=config,
            backend=backend,
            archive=archive,
            llm_client=mock_llm,
        )

        await engine.run()

        # Reflection should have used the configured model
        assert mock_llm.last_reflection_model == reflection_model


class TestReflectionWithoutLLM:
    """When stagnation is detected but llm_client is None, no crash."""

    async def test_stagnation_without_llm_no_crash(self, archive: Archive) -> None:
        config = _make_config(stagnation_window=3, max_generations=6)
        backend = ConstantFitnessBackend()

        engine = EvolutionEngine(
            config=config,
            backend=backend,
            archive=archive,
            llm_client=None,
        )

        result = await engine.run()

        # Stagnation detected, but no crash
        assert result.reflected is True
