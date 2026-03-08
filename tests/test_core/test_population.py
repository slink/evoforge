# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for evoforge.core.population – PopulationManager."""

from __future__ import annotations

import math

import pytest

from evoforge.core.population import PopulationManager
from evoforge.core.types import Fitness, Individual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ind(
    primary: float,
    *,
    ir_hash: str | None = None,
    behavior_descriptor: tuple[object, ...] | None = None,
    has_fitness: bool = True,
) -> Individual:
    """Create a minimal Individual for population tests."""
    fitness = (
        Fitness(primary=primary, auxiliary={}, constraints={}, feasible=True)
        if has_fitness
        else None
    )
    return Individual(
        genome=f"g_{primary}",
        ir=None,
        ir_hash=ir_hash or f"hash_{primary}",
        fitness=fitness,
        generation=0,
        behavior_descriptor=behavior_descriptor,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestPopulationManagerInit:
    def test_empty_on_creation(self) -> None:
        pm = PopulationManager()
        assert pm.size == 0
        assert pm.get_all() == []

    def test_custom_max_size(self) -> None:
        pm = PopulationManager(max_size=50)
        assert pm.max_size == 50


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_new_individual(self) -> None:
        pm = PopulationManager()
        ind = _make_ind(5.0)
        assert pm.add(ind) is True
        assert pm.size == 1

    def test_add_duplicate_rejected(self) -> None:
        pm = PopulationManager()
        ind = _make_ind(5.0, ir_hash="dup")
        assert pm.add(ind) is True
        ind2 = _make_ind(10.0, ir_hash="dup")
        assert pm.add(ind2) is False
        assert pm.size == 1

    def test_add_multiple_unique(self) -> None:
        pm = PopulationManager()
        for i in range(5):
            ind = _make_ind(float(i), ir_hash=f"h{i}")
            assert pm.add(ind) is True
        assert pm.size == 5


# ---------------------------------------------------------------------------
# contains
# ---------------------------------------------------------------------------


class TestContains:
    def test_contains_present(self) -> None:
        pm = PopulationManager()
        ind = _make_ind(1.0, ir_hash="abc")
        pm.add(ind)
        assert pm.contains("abc") is True

    def test_contains_absent(self) -> None:
        pm = PopulationManager()
        assert pm.contains("xyz") is False


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------


class TestGetAll:
    def test_returns_all(self) -> None:
        pm = PopulationManager()
        inds = [_make_ind(float(i), ir_hash=f"h{i}") for i in range(3)]
        for ind in inds:
            pm.add(ind)
        result = pm.get_all()
        assert len(result) == 3
        assert set(ind.ir_hash for ind in result) == {"h0", "h1", "h2"}


# ---------------------------------------------------------------------------
# best
# ---------------------------------------------------------------------------


class TestBest:
    def test_best_single(self) -> None:
        pm = PopulationManager()
        for i in range(5):
            pm.add(_make_ind(float(i), ir_hash=f"h{i}"))
        result = pm.best(k=1)
        assert len(result) == 1
        assert result[0].fitness is not None
        assert result[0].fitness.primary == 4.0

    def test_best_multiple(self) -> None:
        pm = PopulationManager()
        for i in range(5):
            pm.add(_make_ind(float(i), ir_hash=f"h{i}"))
        result = pm.best(k=3)
        assert len(result) == 3
        primaries = [ind.fitness.primary for ind in result if ind.fitness]
        assert primaries == [4.0, 3.0, 2.0]

    def test_best_with_no_fitness(self) -> None:
        """Individuals without fitness should be sorted last (worst)."""
        pm = PopulationManager()
        pm.add(_make_ind(10.0, ir_hash="fit"))
        pm.add(_make_ind(0.0, ir_hash="nofit", has_fitness=False))
        result = pm.best(k=2)
        assert result[0].fitness is not None
        assert result[0].fitness.primary == 10.0
        assert result[1].fitness is None

    def test_best_k_larger_than_population(self) -> None:
        pm = PopulationManager()
        pm.add(_make_ind(1.0, ir_hash="h1"))
        result = pm.best(k=5)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# remove_worst
# ---------------------------------------------------------------------------


class TestRemoveWorst:
    def test_remove_worst_single(self) -> None:
        pm = PopulationManager()
        for i in range(5):
            pm.add(_make_ind(float(i), ir_hash=f"h{i}"))
        removed = pm.remove_worst(k=1)
        assert len(removed) == 1
        assert removed[0].fitness is not None
        assert removed[0].fitness.primary == 0.0
        assert pm.size == 4
        assert not pm.contains("h0")

    def test_remove_worst_multiple(self) -> None:
        pm = PopulationManager()
        for i in range(5):
            pm.add(_make_ind(float(i), ir_hash=f"h{i}"))
        removed = pm.remove_worst(k=2)
        assert len(removed) == 2
        primaries = sorted(ind.fitness.primary for ind in removed if ind.fitness)
        assert primaries == [0.0, 1.0]
        assert pm.size == 3

    def test_remove_worst_prefers_no_fitness(self) -> None:
        """Individuals without fitness should be removed first."""
        pm = PopulationManager()
        pm.add(_make_ind(100.0, ir_hash="fit"))
        pm.add(_make_ind(0.0, ir_hash="nofit", has_fitness=False))
        removed = pm.remove_worst(k=1)
        assert removed[0].fitness is None
        assert pm.size == 1

    def test_remove_worst_k_larger_than_population(self) -> None:
        pm = PopulationManager()
        pm.add(_make_ind(1.0, ir_hash="h1"))
        removed = pm.remove_worst(k=5)
        assert len(removed) == 1
        assert pm.size == 0


# ---------------------------------------------------------------------------
# size property
# ---------------------------------------------------------------------------


class TestSizeProperty:
    def test_size_reflects_changes(self) -> None:
        pm = PopulationManager()
        assert pm.size == 0
        pm.add(_make_ind(1.0, ir_hash="a"))
        assert pm.size == 1
        pm.add(_make_ind(2.0, ir_hash="b"))
        assert pm.size == 2
        pm.remove_worst(k=1)
        assert pm.size == 1


# ---------------------------------------------------------------------------
# diversity_entropy
# ---------------------------------------------------------------------------


class TestDiversityEntropy:
    def test_no_behavior_descriptors_returns_zero(self) -> None:
        pm = PopulationManager()
        pm.add(_make_ind(1.0, ir_hash="a", behavior_descriptor=None))
        pm.add(_make_ind(2.0, ir_hash="b", behavior_descriptor=None))
        assert pm.diversity_entropy() == 0.0

    def test_empty_population_returns_zero(self) -> None:
        pm = PopulationManager()
        assert pm.diversity_entropy() == 0.0

    def test_single_descriptor(self) -> None:
        """All same descriptor -> entropy = 0."""
        pm = PopulationManager()
        for i in range(3):
            pm.add(_make_ind(float(i), ir_hash=f"h{i}", behavior_descriptor=("a",)))
        assert pm.diversity_entropy() == pytest.approx(0.0)

    def test_uniform_distribution(self) -> None:
        """Two distinct descriptors, equal frequency -> entropy = log(2)."""
        pm = PopulationManager()
        pm.add(_make_ind(1.0, ir_hash="h1", behavior_descriptor=("a",)))
        pm.add(_make_ind(2.0, ir_hash="h2", behavior_descriptor=("b",)))
        expected = math.log(2)
        assert pm.diversity_entropy() == pytest.approx(expected)

    def test_mixed_descriptors_some_none(self) -> None:
        """Only individuals WITH behavior_descriptor count toward entropy."""
        pm = PopulationManager()
        pm.add(_make_ind(1.0, ir_hash="h1", behavior_descriptor=("a",)))
        pm.add(_make_ind(2.0, ir_hash="h2", behavior_descriptor=("a",)))
        pm.add(_make_ind(3.0, ir_hash="h3", behavior_descriptor=None))
        # Only two individuals count, both have ("a",) -> entropy = 0
        assert pm.diversity_entropy() == pytest.approx(0.0)

    def test_three_distinct_descriptors(self) -> None:
        """Three distinct descriptors, equal frequency -> entropy = log(3)."""
        pm = PopulationManager()
        pm.add(_make_ind(1.0, ir_hash="h1", behavior_descriptor=("a",)))
        pm.add(_make_ind(2.0, ir_hash="h2", behavior_descriptor=("b",)))
        pm.add(_make_ind(3.0, ir_hash="h3", behavior_descriptor=("c",)))
        expected = math.log(3)
        assert pm.diversity_entropy() == pytest.approx(expected)
