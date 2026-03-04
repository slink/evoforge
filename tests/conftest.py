"""Shared test fixtures and factories for evoforge."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from evoforge.core.archive import Archive
from evoforge.core.types import Fitness, Individual


def make_individual(
    genome: str = "test_genome",
    *,
    ir: Any = None,
    ir_hash: str = "test_hash",
    generation: int = 0,
    fitness: Fitness | None = None,
    behavior_descriptor: tuple[object, ...] | None = None,
    mutation_source: str | None = None,
) -> Individual:
    """Create a minimal Individual for testing."""
    return Individual(
        genome=genome,
        ir=ir,
        ir_hash=ir_hash,
        generation=generation,
        fitness=fitness,
        behavior_descriptor=behavior_descriptor,
        mutation_source=mutation_source,
    )


@dataclass
class FakeLLMResponse:
    """Minimal stand-in for LLMResponse, shared across test files."""

    text: str
    input_tokens: int = 10
    output_tokens: int = 20
    model: str = "test-model"


@pytest.fixture
async def archive() -> AsyncIterator[Archive]:
    """Create an in-memory archive with all tables ready."""
    a = Archive("sqlite+aiosqlite://")
    await a.create_tables()
    yield a
    await a.close()
