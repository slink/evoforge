# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for the SQLite-backed archive (cache / dedup / lineage)."""

from __future__ import annotations

import pytest

from evoforge.core.archive import Archive
from evoforge.core.types import Fitness, Individual


@pytest.fixture
async def archive() -> Archive:
    """Create an in-memory archive with all tables ready."""
    arch = Archive("sqlite+aiosqlite://")
    await arch.create_tables()
    return arch


def _make_individual(
    *,
    genome: str = "x + 1",
    ir_hash: str = "abc123",
    generation: int = 0,
    fitness: Fitness | None = None,
    behavior_descriptor: tuple[object, ...] | None = None,
    mutation_source: str | None = None,
) -> Individual:
    return Individual(
        genome=genome,
        ir=None,
        ir_hash=ir_hash,
        generation=generation,
        fitness=fitness,
        behavior_descriptor=behavior_descriptor,
        mutation_source=mutation_source,
    )


def _make_fitness(primary: float = 0.42) -> Fitness:
    return Fitness(
        primary=primary,
        auxiliary={"speed": 1.0},
        constraints={"valid": True},
        feasible=True,
    )


# ---- store / retrieve ---------------------------------------------------


async def test_store_and_lookup(archive: Archive) -> None:
    ind = _make_individual(fitness=_make_fitness())
    await archive.store(ind)

    result = await archive.lookup("abc123")
    assert result is not None
    assert result.ir_hash == "abc123"
    assert result.genome == "x + 1"
    assert result.generation == 0
    assert result.fitness is not None
    assert result.fitness.primary == pytest.approx(0.42)


# ---- cache miss ----------------------------------------------------------


async def test_lookup_miss(archive: Archive) -> None:
    result = await archive.lookup("nonexistent")
    assert result is None


# ---- dedup idempotency ---------------------------------------------------


async def test_dedup_idempotency(archive: Archive) -> None:
    ind1 = _make_individual(genome="x + 1", ir_hash="dup_hash")
    ind2 = _make_individual(genome="x + 1 (copy)", ir_hash="dup_hash")

    await archive.store(ind1)
    await archive.store(ind2)  # same ir_hash — should not raise

    result = await archive.lookup("dup_hash")
    assert result is not None
    # The first stored individual should win
    assert result.genome == "x + 1"


# ---- fitness cache -------------------------------------------------------


async def test_fitness_cache_roundtrip(archive: Archive) -> None:
    fitness = _make_fitness(primary=9.5)
    await archive.store_fitness(
        ir_hash="hash1",
        backend_version="v1.0",
        config_hash="cfg_abc",
        fitness=fitness,
        diagnostics_json='{"ok": true}',
    )

    result = await archive.lookup_fitness("hash1", "v1.0", "cfg_abc")
    assert result is not None
    assert result.primary == pytest.approx(9.5)
    assert result.auxiliary == {"speed": 1.0}
    assert result.feasible is True


async def test_fitness_cache_miss(archive: Archive) -> None:
    result = await archive.lookup_fitness("no_hash", "v1.0", "cfg_abc")
    assert result is None


# ---- prefix cache --------------------------------------------------------


async def test_prefix_cache_roundtrip(archive: Archive) -> None:
    await archive.put_prefix("pfx_001", '{"step": 42}')

    result = await archive.get_prefix("pfx_001")
    assert result == '{"step": 42}'


async def test_prefix_cache_miss(archive: Archive) -> None:
    result = await archive.get_prefix("missing_pfx")
    assert result is None


# ---- lineage tracking ----------------------------------------------------


async def test_lineage_tracking(archive: Archive) -> None:
    await archive.store_lineage("parent_a", "child_x", "mutate", 3)
    await archive.store_lineage("parent_b", "child_x", "crossover", 3)

    records = await archive.get_lineage("child_x")
    assert len(records) == 2

    operators = {r["operator_name"] for r in records}
    assert operators == {"mutate", "crossover"}

    parents = {r["parent_hash"] for r in records}
    assert parents == {"parent_a", "parent_b"}

    for r in records:
        assert r["child_hash"] == "child_x"
        assert r["generation"] == 3


async def test_lineage_empty(archive: Archive) -> None:
    records = await archive.get_lineage("no_such_child")
    assert records == []
