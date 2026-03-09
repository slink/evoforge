# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for LLM-powered mutation operators (LLMMutate & LLMCrossover)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evoforge.core.mutation import MutationContext
from evoforge.core.types import Credit
from evoforge.llm.operators import LLMCrossover, LLMMutate
from tests.conftest import FakeLLMResponse, make_individual


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
            return_value=FakeLLMResponse(text="```lean\nnew_genome\n```")
        )
        backend = MagicMock()
        backend.format_crossover_prompt.return_value = "crossover prompt"
        backend.system_prompt.return_value = "system"
        backend.extract_genome.return_value = "new_genome"

        parent_a = make_individual("genome_a")
        parent_b = make_individual("genome_b")
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
            return_value=FakeLLMResponse(text="```lean\nmutated\n```")
        )
        backend = MagicMock()
        backend.format_mutation_prompt.return_value = "mutation prompt"
        backend.system_prompt.return_value = "system"
        backend.extract_genome.return_value = "mutated_genome"

        parent = make_individual("genome_a")
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
            return_value=FakeLLMResponse(text="```lean\nnew_code\n```")
        )
        backend = MagicMock()
        backend.format_mutation_prompt.return_value = "mutation prompt"
        backend.system_prompt.return_value = "system"
        backend.extract_genome.return_value = "extracted_genome"

        parent = make_individual("original")
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
        client.async_generate = AsyncMock(return_value=FakeLLMResponse(text="garbage output"))
        backend = MagicMock()
        backend.format_mutation_prompt.return_value = "mutation prompt"
        backend.system_prompt.return_value = "system"
        backend.extract_genome.return_value = None  # extraction fails

        parent = make_individual("original_genome")
        ctx = _make_context(backend=backend)

        op = LLMMutate(client=client, model="test-model")
        result = await op.apply(parent, ctx)

        assert result == "original_genome"  # falls back to parent


# ------------------------------------------------------------------ #
# LLMMutate batch-aware generate tests
# ------------------------------------------------------------------ #


class TestLLMMutateWithBatchAwareGenerate:
    @pytest.mark.asyncio
    async def test_uses_batch_aware_generate(self) -> None:
        """LLMMutate delegates to batch_aware_generate."""
        client = MagicMock()
        backend = MagicMock()
        backend.format_mutation_prompt.return_value = "mutation prompt"
        backend.system_prompt.return_value = "system"
        backend.extract_genome.return_value = "batched_genome"
        ctx = _make_context(backend=backend)
        parent = make_individual("original")

        op = LLMMutate(client=client, model="test-model")

        with patch(
            "evoforge.llm.operators.batch_aware_generate",
            new=AsyncMock(return_value=FakeLLMResponse(text="batched")),
        ):
            result = await op.apply(parent, ctx)

        assert result == "batched_genome"

    @pytest.mark.asyncio
    async def test_batch_returns_none_falls_back_to_parent(self) -> None:
        """When batch_aware_generate returns None, fall back to parent genome."""
        client = MagicMock()
        backend = MagicMock()
        backend.format_mutation_prompt.return_value = "mutation prompt"
        backend.system_prompt.return_value = "system"
        ctx = _make_context(backend=backend)
        parent = make_individual("original_genome")

        op = LLMMutate(client=client, model="test-model")

        with patch(
            "evoforge.llm.operators.batch_aware_generate",
            new=AsyncMock(return_value=None),
        ):
            result = await op.apply(parent, ctx)

        assert result == "original_genome"


# ------------------------------------------------------------------ #
# LLMCrossover batch-aware generate tests
# ------------------------------------------------------------------ #


class TestLLMCrossoverWithBatchAwareGenerate:
    @pytest.mark.asyncio
    async def test_uses_batch_aware_generate(self) -> None:
        """LLMCrossover delegates to batch_aware_generate."""
        client = MagicMock()
        backend = MagicMock()
        backend.format_crossover_prompt.return_value = "crossover prompt"
        backend.system_prompt.return_value = "system"
        backend.extract_genome.return_value = "batched_genome"

        parent_a = make_individual("genome_a")
        parent_b = make_individual("genome_b")
        ctx = _make_context(backend=backend, guidance_individual=parent_b)

        op = LLMCrossover(client=client, model="test-model")

        with patch(
            "evoforge.llm.operators.batch_aware_generate",
            new=AsyncMock(return_value=FakeLLMResponse(text="batched")),
        ):
            result = await op.apply(parent_a, ctx)

        assert result == "batched_genome"
