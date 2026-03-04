"""Abstract base class for evoforge backends.

Every domain backend (Lean, Python, etc.) must subclass :class:`Backend`
and implement all abstract methods.  The engine dispatches through this
interface so it remains domain-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from evoforge.core.ir import BehaviorSpaceConfig, IRProtocol
from evoforge.core.types import Credit, Fitness, Individual


class Backend(ABC):
    """Domain backend interface for the evolutionary engine."""

    @abstractmethod
    def parse(self, genome: str) -> IRProtocol | None:
        """Parse a raw genome string into a domain-specific IR node."""
        ...

    @abstractmethod
    def evaluate(self, ir: Any, seed: int | None = None) -> tuple[Fitness, Any, Any]:
        """Evaluate an IR node and return (fitness, diagnostics, trace)."""
        ...

    @abstractmethod
    def evaluate_stepwise(self, ir: Any, seed: int | None = None) -> tuple[Fitness, Any, Any]:
        """Step-wise evaluation returning (fitness, diagnostics, trace)."""
        ...

    @abstractmethod
    def assign_credit(
        self, ir: Any, fitness: Fitness, diagnostics: Any, trace: Any
    ) -> list[Credit]:
        """Assign localized credit to genome regions."""
        ...

    @abstractmethod
    def validate_structure(self, ir: Any) -> list[str]:
        """Return a list of structural validation errors (empty = valid)."""
        ...

    @abstractmethod
    def seed_population(self, n: int) -> list[str]:
        """Generate *n* seed genomes for the initial population."""
        ...

    @abstractmethod
    def mutation_operators(self) -> list[Any]:
        """Return available mutation operators for this domain."""
        ...

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the LLM system prompt for this domain."""
        ...

    @abstractmethod
    def format_mutation_prompt(self, parent: Individual, context: Any) -> str:
        """Format a mutation prompt for the LLM given a parent individual."""
        ...

    @abstractmethod
    def format_crossover_prompt(
        self, parent_a: Individual, parent_b: Individual, context: Any
    ) -> str:
        """Format a crossover prompt for the LLM given two parent individuals."""
        ...

    @abstractmethod
    def extract_genome(self, raw_text: str) -> str | None:
        """Extract a genome string from raw LLM output."""
        ...

    @abstractmethod
    def behavior_descriptor(self, ir: Any, diagnostics: Any) -> tuple[Any, ...]:
        """Compute the behavior descriptor for MAP-Elites archiving."""
        ...

    @abstractmethod
    def behavior_space(self) -> BehaviorSpaceConfig:
        """Return the behavior-space configuration for this domain."""
        ...

    @abstractmethod
    def recommended_selection(self) -> str:
        """Return the name of the recommended selection strategy."""
        ...

    @abstractmethod
    def version(self) -> str:
        """Return a version string for cache-keying (e.g. ``'lean-0.3.1'``)."""
        ...

    @abstractmethod
    def eval_config_hash(self) -> str:
        """Return a hash of the evaluation config for cache-keying."""
        ...

    @abstractmethod
    def format_reflection_prompt(
        self, population: list[Individual], memory: Any, generation: int
    ) -> str:
        """Format a reflection prompt summarising population state."""
        ...

    @abstractmethod
    def default_operator_weights(self) -> dict[str, float]:
        """Return default mutation-operator weights for this domain."""
        ...
