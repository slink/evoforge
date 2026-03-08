# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for LLM tactic generator."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from evoforge.backends.lean.tactic_generator import LLMTacticGenerator


@pytest.mark.asyncio
async def test_extracts_numbered_tactics() -> None:
    """Generator should parse numbered tactics from LLM response."""
    llm_response = "1. `simp [h0]`\n2. `norm_num`\n3. `linarith [norm_nonneg x]`"
    client = AsyncMock()
    client.async_generate.return_value = SimpleNamespace(text=llm_response)
    gen = LLMTacticGenerator(client=client, model="test", system_prompt="test")
    tactics = await gen.suggest_tactics("⊢ ‖φ ξ‖ ≤ 1", ["intro x"], n=3)
    assert len(tactics) == 3
    assert "simp [h0]" in tactics
    assert "norm_num" in tactics


@pytest.mark.asyncio
async def test_extracts_code_block_tactics() -> None:
    """Generator should handle tactics in code blocks."""
    llm_response = "```lean\nsimp [h0]\nnorm_num\nlinarith\n```"
    client = AsyncMock()
    client.async_generate.return_value = SimpleNamespace(text=llm_response)
    gen = LLMTacticGenerator(client=client, model="test", system_prompt="test")
    tactics = await gen.suggest_tactics("⊢ 0 ≤ 1", [], n=5)
    assert len(tactics) >= 3
    assert "simp [h0]" in tactics


@pytest.mark.asyncio
async def test_extracts_backtick_tactics() -> None:
    """Generator should handle inline backtick tactics."""
    llm_response = "Try `simp` or `ring` or `omega`"
    client = AsyncMock()
    client.async_generate.return_value = SimpleNamespace(text=llm_response)
    gen = LLMTacticGenerator(client=client, model="test", system_prompt="test")
    tactics = await gen.suggest_tactics("⊢ 0 = 0", [], n=3)
    assert len(tactics) == 3
    assert "simp" in tactics


@pytest.mark.asyncio
async def test_respects_n_limit() -> None:
    """Generator should return at most n tactics."""
    llm_response = "1. `a`\n2. `b`\n3. `c`\n4. `d`\n5. `e`"
    client = AsyncMock()
    client.async_generate.return_value = SimpleNamespace(text=llm_response)
    gen = LLMTacticGenerator(client=client, model="test", system_prompt="test")
    tactics = await gen.suggest_tactics("⊢ True", [], n=2)
    assert len(tactics) == 2


@pytest.mark.asyncio
async def test_fallback_to_lines() -> None:
    """Generator should fall back to line-based parsing."""
    llm_response = "simp\nring\nomega"
    client = AsyncMock()
    client.async_generate.return_value = SimpleNamespace(text=llm_response)
    gen = LLMTacticGenerator(client=client, model="test", system_prompt="test")
    tactics = await gen.suggest_tactics("⊢ True", [], n=5)
    assert len(tactics) == 3
    assert "simp" in tactics


@pytest.mark.asyncio
async def test_filters_prose_in_fallback() -> None:
    """Line fallback should filter out prose-like lines."""
    llm_response = "Here are some tactics:\nsimp\nTry this approach:\nring"
    client = AsyncMock()
    client.async_generate.return_value = SimpleNamespace(text=llm_response)
    gen = LLMTacticGenerator(client=client, model="test", system_prompt="test")
    tactics = await gen.suggest_tactics("⊢ True", [], n=5)
    assert "simp" in tactics
    assert "ring" in tactics
    # Prose lines should be filtered
    for t in tactics:
        assert not t.startswith("Here")
        assert not t.startswith("Try")
