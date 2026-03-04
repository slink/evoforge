"""Tests for LLM-powered mutation operators (LLMMutate & LLMCrossover)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evoforge.core.mutation import MutationContext
from evoforge.core.types import Credit, Individual
from evoforge.llm.operators import LLMCrossover, LLMMutate


@dataclass
class _FakeLLMResponse:
    """Minimal stand-in for LLMResponse."""

    text: str
    input_tokens: int = 10
    output_tokens: int = 20
    model: str = "test-model"


def _make_individual(genome: str = "parent_genome") -> Individual:
    return Individual(
        genome=genome,
        ir=None,
        ir_hash="abc123",
        generation=0,
    )


def _make_context(
    *,
    backend: Any = None,
    guidance_individual: Any = None,
) -> MutationContext:
    if backend is None:
        backend = MagicMock()
        backend.format_mutation_prompt.return_value = "mutation prompt"
        backend.format_crossover_prompt.return_value = "crossover prompt"
        backend.system_prompt.return_value = "system"
        backend.extract_genome.return_value = "new_genome"

    return MutationContext(
        generation=1,
        memory=MagicMock(),
        guidance="some guidance",
        temperature=0.7,
        backend=backend,
        credits=[Credit(location=0, score=1.0, signal="test")],
        guidance_individual=guidance_individual,
    )


# ------------------------------------------------------------------ #
# LLMCrossover tests
# ------------------------------------------------------------------ #


class TestLLMCrossoverWithGuidanceIndividual:
    """When guidance_individual is set, crossover passes both parents."""

    @pytest.mark.asyncio
    async def test_passes_both_parents_to_format_crossover_prompt(self) -> None:
        client = MagicMock()
        client.async_generate = AsyncMock(
            return_value=_FakeLLMResponse(text="```lean\nnew_genome\n```")
        )
        backend = MagicMock()
        backend.format_crossover_prompt.return_value = "crossover prompt"
        backend.system_prompt.return_value = "system"
        backend.extract_genome.return_value = "new_genome"

        parent_a = _make_individual("genome_a")
        parent_b = _make_individual("genome_b")
        ctx = _make_context(backend=backend, guidance_individual=parent_b)

        op = LLMCrossover(client=client, model="test-model")
        result = await op.apply(parent_a, ctx)

        # format_crossover_prompt must be called with both parents + context
        backend.format_crossover_prompt.assert_called_once_with(parent_a, parent_b, ctx)
        # mutation prompt must NOT be called
        backend.format_mutation_prompt.assert_not_called()
        assert result == "new_genome"


class TestLLMCrossoverWithoutGuidanceIndividual:
    """When guidance_individual is None, crossover falls back to mutation prompt."""

    @pytest.mark.asyncio
    async def test_falls_back_to_format_mutation_prompt(self) -> None:
        client = MagicMock()
        client.async_generate = AsyncMock(
            return_value=_FakeLLMResponse(text="```lean\nmutated\n```")
        )
        backend = MagicMock()
        backend.format_mutation_prompt.return_value = "mutation prompt"
        backend.system_prompt.return_value = "system"
        backend.extract_genome.return_value = "mutated_genome"

        parent = _make_individual("genome_a")
        ctx = _make_context(backend=backend, guidance_individual=None)

        op = LLMCrossover(client=client, model="test-model")
        result = await op.apply(parent, ctx)

        # crossover prompt must NOT be called
        backend.format_crossover_prompt.assert_not_called()
        # mutation prompt must be called
        backend.format_mutation_prompt.assert_called_once_with(parent, ctx)
        assert result == "mutated_genome"


# ------------------------------------------------------------------ #
# LLMMutate tests
# ------------------------------------------------------------------ #


class TestLLMMutateSuccess:
    """LLMMutate returns extracted genome on success."""

    @pytest.mark.asyncio
    async def test_returns_extracted_genome(self) -> None:
        client = MagicMock()
        client.async_generate = AsyncMock(
            return_value=_FakeLLMResponse(text="```lean\nnew_code\n```")
        )
        backend = MagicMock()
        backend.format_mutation_prompt.return_value = "mutation prompt"
        backend.system_prompt.return_value = "system"
        backend.extract_genome.return_value = "extracted_genome"

        parent = _make_individual("original")
        ctx = _make_context(backend=backend)

        op = LLMMutate(client=client, model="test-model")
        result = await op.apply(parent, ctx)

        assert result == "extracted_genome"
        backend.format_mutation_prompt.assert_called_once_with(parent, ctx)


class TestLLMMutateFallback:
    """LLMMutate falls back to parent genome when extraction fails."""

    @pytest.mark.asyncio
    async def test_falls_back_to_parent_genome(self) -> None:
        client = MagicMock()
        client.async_generate = AsyncMock(return_value=_FakeLLMResponse(text="garbage output"))
        backend = MagicMock()
        backend.format_mutation_prompt.return_value = "mutation prompt"
        backend.system_prompt.return_value = "system"
        backend.extract_genome.return_value = None  # extraction fails

        parent = _make_individual("original_genome")
        ctx = _make_context(backend=backend)

        op = LLMMutate(client=client, model="test-model")
        result = await op.apply(parent, ctx)

        assert result == "original_genome"  # falls back to parent
