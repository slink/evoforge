# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for evoforge.core.types – the leaf data types of the system."""

from __future__ import annotations

import dataclasses

import pytest

from evoforge.core.types import (
    Credit,
    Diagnostics,
    EvaluationTrace,
    FailureMode,
    Fitness,
    Individual,
    Pattern,
    Reflection,
)

# ---------------------------------------------------------------------------
# Fitness.dominates() correctness
# ---------------------------------------------------------------------------


class TestFitnessDominates:
    """Pareto dominance: self dominates other iff at least as good on every
    objective AND strictly better on at least one."""

    def test_strictly_better_on_all(self) -> None:
        a = Fitness(primary=10.0, auxiliary={"x": 5.0}, constraints={}, feasible=True)
        b = Fitness(primary=5.0, auxiliary={"x": 2.0}, constraints={}, feasible=True)
        assert a.dominates(b)

    def test_better_on_primary_equal_on_auxiliary(self) -> None:
        a = Fitness(primary=10.0, auxiliary={"x": 3.0}, constraints={}, feasible=True)
        b = Fitness(primary=5.0, auxiliary={"x": 3.0}, constraints={}, feasible=True)
        assert a.dominates(b)

    def test_equal_on_primary_better_on_auxiliary(self) -> None:
        a = Fitness(primary=5.0, auxiliary={"x": 4.0}, constraints={}, feasible=True)
        b = Fitness(primary=5.0, auxiliary={"x": 3.0}, constraints={}, feasible=True)
        assert a.dominates(b)

    def test_self_does_not_dominate_self(self) -> None:
        a = Fitness(primary=5.0, auxiliary={"x": 3.0}, constraints={}, feasible=True)
        assert not a.dominates(a)

    def test_equal_fitness_no_domination(self) -> None:
        a = Fitness(primary=5.0, auxiliary={"x": 3.0}, constraints={}, feasible=True)
        b = Fitness(primary=5.0, auxiliary={"x": 3.0}, constraints={}, feasible=True)
        assert not a.dominates(b)

    def test_non_domination_tradeoff(self) -> None:
        """When a is better on primary but worse on auxiliary, neither dominates."""
        a = Fitness(primary=10.0, auxiliary={"x": 1.0}, constraints={}, feasible=True)
        b = Fitness(primary=5.0, auxiliary={"x": 5.0}, constraints={}, feasible=True)
        assert not a.dominates(b)
        assert not b.dominates(a)

    def test_empty_auxiliary(self) -> None:
        """With no auxiliary objectives, only primary matters."""
        a = Fitness(primary=10.0, auxiliary={}, constraints={}, feasible=True)
        b = Fitness(primary=5.0, auxiliary={}, constraints={}, feasible=True)
        assert a.dominates(b)
        assert not b.dominates(a)

    def test_multiple_auxiliary_objectives(self) -> None:
        a = Fitness(primary=5.0, auxiliary={"x": 3.0, "y": 4.0}, constraints={}, feasible=True)
        b = Fitness(primary=5.0, auxiliary={"x": 3.0, "y": 3.0}, constraints={}, feasible=True)
        assert a.dominates(b)
        assert not b.dominates(a)


# ---------------------------------------------------------------------------
# Unique ID generation
# ---------------------------------------------------------------------------


class TestIndividualId:
    def test_unique_ids(self) -> None:
        ind1 = Individual(genome="a", ir=None, ir_hash="h1", generation=0)
        ind2 = Individual(genome="b", ir=None, ir_hash="h2", generation=0)
        assert ind1.id != ind2.id

    def test_id_is_uuid_format(self) -> None:
        ind = Individual(genome="a", ir=None, ir_hash="h1", generation=0)
        # UUID4 has 36 chars (including hyphens)
        assert len(ind.id) == 36
        assert ind.id.count("-") == 4


# ---------------------------------------------------------------------------
# Frozen Fitness immutability
# ---------------------------------------------------------------------------


class TestFitnessFrozen:
    def test_cannot_assign_primary(self) -> None:
        f = Fitness(primary=1.0, auxiliary={}, constraints={}, feasible=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.primary = 2.0  # type: ignore[misc]

    def test_cannot_assign_feasible(self) -> None:
        f = Fitness(primary=1.0, auxiliary={}, constraints={}, feasible=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.feasible = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Credit default confidence
# ---------------------------------------------------------------------------


class TestCreditDefaults:
    def test_default_confidence_is_one(self) -> None:
        c = Credit(location=0, score=1.5, signal="test")
        assert c.confidence == 1.0

    def test_custom_confidence(self) -> None:
        c = Credit(location=0, score=1.5, signal="test", confidence=0.5)
        assert c.confidence == 0.5

    def test_credit_frozen(self) -> None:
        c = Credit(location=0, score=1.5, signal="test")
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.score = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Diagnostics protocol structural subtyping
# ---------------------------------------------------------------------------


class TestDiagnosticsProtocol:
    def test_structural_subtype_satisfies_protocol(self) -> None:
        class MyDiagnostics:
            def summary(self, max_tokens: int) -> str:
                return "ok"

            def credit_summary(self, credits: list[Credit], max_tokens: int) -> str:
                return "credits ok"

        assert isinstance(MyDiagnostics(), Diagnostics)

    def test_non_conforming_class_fails(self) -> None:
        class NotDiagnostics:
            pass

        assert not isinstance(NotDiagnostics(), Diagnostics)


# ---------------------------------------------------------------------------
# Other types: basic construction smoke tests
# ---------------------------------------------------------------------------


class TestOtherTypes:
    def test_evaluation_trace_base(self) -> None:
        t = EvaluationTrace()
        assert isinstance(t, EvaluationTrace)

    def test_reflection_construction(self) -> None:
        r = Reflection(
            strategies_to_try=["a"],
            strategies_to_avoid=["b"],
            useful_primitives=["c"],
            population_diagnosis="good",
            suggested_temperature=0.7,
        )
        assert r.suggested_temperature == 0.7

    def test_pattern_frozen(self) -> None:
        p = Pattern(description="loop", frequency=5, avg_fitness=0.8)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.frequency = 10  # type: ignore[misc]

    def test_failure_mode_frozen(self) -> None:
        fm = FailureMode(description="timeout", frequency=3, last_seen=10)
        with pytest.raises(dataclasses.FrozenInstanceError):
            fm.frequency = 5  # type: ignore[misc]

    def test_individual_defaults(self) -> None:
        ind = Individual(genome="x", ir=None, ir_hash="h", generation=0)
        assert ind.fitness is None
        assert ind.diagnostics is None
        assert ind.credits == []
        assert ind.lineage == {}
        assert ind.behavior_descriptor is None
        assert ind.mutation_source is None
