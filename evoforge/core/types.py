# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Core data types for the evoforge evolutionary engine."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Fitness:
    """Fitness value with multi-objective support and constraint tracking."""

    primary: float
    auxiliary: dict[str, float | str]
    constraints: dict[str, bool]
    feasible: bool

    def dominates(self, other: Fitness) -> bool:
        """Pareto dominance: self dominates other iff at least as good on every
        objective AND strictly better on at least one."""
        numeric_keys = sorted(
            k for k in self.auxiliary if isinstance(self.auxiliary[k], (int, float))
        )
        all_values_self = [self.primary] + [
            v for k in numeric_keys if isinstance((v := self.auxiliary[k]), (int, float))
        ]
        all_values_other = [other.primary] + [
            v for k in numeric_keys if isinstance((v := other.auxiliary.get(k, 0.0)), (int, float))
        ]

        at_least_as_good = all(s >= o for s, o in zip(all_values_self, all_values_other))
        strictly_better_on_one = any(s > o for s, o in zip(all_values_self, all_values_other))

        return at_least_as_good and strictly_better_on_one


@dataclass(frozen=True)
class Credit:
    """Localized credit assignment for a genome region."""

    location: int
    score: float
    signal: str
    confidence: float = 1.0


@dataclass
class Individual:
    """A single candidate in the evolutionary population."""

    genome: str
    ir: Any
    ir_hash: str
    generation: int
    fitness: Fitness | None = None
    diagnostics: Any = None
    credits: list[Credit] = field(default_factory=list)
    lineage: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    behavior_descriptor: tuple[Any, ...] | None = None
    mutation_source: str | None = None


@runtime_checkable
class Diagnostics(Protocol):
    """Protocol for evaluation diagnostics that can summarize themselves."""

    def summary(self, max_tokens: int) -> str: ...

    def credit_summary(self, credits: list[Credit], max_tokens: int) -> str: ...


class EvaluationTrace:
    """Base class for evaluation traces. Subclasses add domain-specific fields."""


@dataclass
class Reflection:
    """LLM-generated reflection on the evolutionary population state."""

    strategies_to_try: list[str]
    strategies_to_avoid: list[str]
    useful_primitives: list[str]
    population_diagnosis: str
    suggested_temperature: float


@dataclass(frozen=True)
class Pattern:
    """A recurring pattern observed in the population."""

    description: str
    frequency: int
    avg_fitness: float


@dataclass(frozen=True)
class FailureMode:
    """A recurring failure mode observed in the population."""

    description: str
    frequency: int
    last_seen: int
