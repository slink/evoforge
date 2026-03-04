"""Mutation operators and ensemble orchestration for evoforge.

Defines the abstract base class for mutation operators, the context
passed to each mutation call, per-operator statistics tracking, and
the ensemble that manages weighted operator selection with optional
adaptive scheduling.
"""

from __future__ import annotations

import dataclasses
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from evoforge.core.types import Credit, Individual


@dataclass
class MutationContext:
    """Runtime context passed to every mutation operator invocation."""

    generation: int
    memory: Any  # SearchMemory — typed as Any to avoid circular imports
    guidance: str
    temperature: float
    backend: Any  # Backend reference — typed as Any for the same reason
    credits: list[Credit]
    guidance_individual: Any = None  # Individual | None, typed as Any to avoid circular imports


@dataclass
class OperatorStats:
    """Cumulative performance statistics for a single mutation operator."""

    applications: int = 0
    successes: int = 0
    failures: int = 0
    total_fitness_delta: float = 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of applications that produced a fitter individual."""
        if self.applications == 0:
            return 0.0
        return self.successes / self.applications


class MutationOperator(ABC):
    """Abstract base class for all mutation operators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable operator name."""
        ...

    @property
    @abstractmethod
    def cost(self) -> Literal["cheap", "llm"]:
        """Cost category: 'cheap' or 'llm'."""
        ...

    @abstractmethod
    async def apply(self, parent: Individual, context: MutationContext) -> str:
        """Produce a new genome string by mutating *parent* in *context*."""
        ...


# How often (in total applications across all operators) the adaptive
# schedule re-computes weights.
_ADAPTIVE_WINDOW: int = 10


class MutationEnsemble:
    """Manages a weighted collection of mutation operators.

    Supports three scheduling modes:
      - **fixed**: weights never change.
      - **phased**: (reserved for future phase-based curricula).
      - **adaptive**: every *_ADAPTIVE_WINDOW* total applications, weights are
        shifted toward operators with higher success rates.
    """

    def __init__(
        self,
        operators: list[MutationOperator],
        schedule: str = "fixed",
        weights: list[float] | None = None,
    ) -> None:
        if not operators:
            raise ValueError("MutationEnsemble requires at least one operator")

        self._operators = operators
        self._schedule = schedule
        self._operator_map: dict[str, MutationOperator] = {op.name: op for op in operators}

        if weights is None:
            n = len(operators)
            self._weights: list[float] = [1.0 / n] * n
        else:
            if len(weights) != len(operators):
                raise ValueError("len(weights) must match len(operators)")
            self._weights = list(weights)

        self.stats: dict[str, OperatorStats] = {op.name: OperatorStats() for op in operators}
        self._total_applications: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_operator(self) -> MutationOperator:
        """Weighted random selection of an operator."""
        (chosen,) = random.choices(self._operators, weights=self._weights, k=1)
        return chosen

    def update_stats(
        self,
        operator_name: str,
        *,
        success: bool,
        fitness_delta: float,
    ) -> None:
        """Record the outcome of applying *operator_name*."""
        if operator_name not in self.stats:
            return  # ignore unknown operators silently

        s = self.stats[operator_name]
        s.applications += 1
        if success:
            s.successes += 1
        else:
            s.failures += 1
        s.total_fitness_delta += fitness_delta

        self._total_applications += 1

        if self._schedule == "adaptive" and self._total_applications % _ADAPTIVE_WINDOW == 0:
            self._adapt_weights()

    def cheapest_operator(self) -> MutationOperator:
        """Return the first operator whose cost is 'cheap'.

        Raises ``ValueError`` if no cheap operators are registered.
        """
        for op in self._operators:
            if op.cost == "cheap":
                return op
        raise ValueError("No cheap operators available in the ensemble")

    def get_weights(self) -> dict[str, float]:
        """Return current selection weights keyed by operator name."""
        return {op.name: w for op, w in zip(self._operators, self._weights)}

    def to_dict(self) -> dict[str, Any]:
        """Serialize ensemble state (weights, stats, total_applications)."""
        return {
            "weights": {op.name: w for op, w in zip(self._operators, self._weights)},
            "stats": {name: dataclasses.asdict(s) for name, s in self.stats.items()},
            "total_applications": self._total_applications,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore ensemble state from a dict produced by :meth:`to_dict`."""
        # Restore weights
        saved_weights = data.get("weights", {})
        for i, op in enumerate(self._operators):
            if op.name in saved_weights:
                self._weights[i] = saved_weights[op.name]
        # Renormalize
        total = sum(self._weights)
        if total > 0:
            self._weights = [w / total for w in self._weights]

        # Restore stats
        saved_stats = data.get("stats", {})
        for name, sdict in saved_stats.items():
            if name in self.stats:
                self.stats[name] = OperatorStats(**sdict)

        # Restore total applications
        self._total_applications = data.get("total_applications", 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adapt_weights(self) -> None:
        """Shift weights toward operators with higher success rates.

        Uses a simple softmax-style blend: each operator's new raw weight is
        a mix of its current weight and its success rate.  The result is
        normalised so weights sum to 1.
        """
        alpha = 0.3  # blend factor: how much success_rate influences the update
        raw: list[float] = []
        for op, w in zip(self._operators, self._weights):
            sr = self.stats[op.name].success_rate
            raw.append(w * (1.0 - alpha) + sr * alpha)

        total = sum(raw)
        if total > 0:
            self._weights = [r / total for r in raw]
