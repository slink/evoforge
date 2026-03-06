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
    async def evaluate(self, ir: Any, seed: int | None = None) -> tuple[Fitness, Any, Any]:
        """Evaluate an IR node and return (fitness, diagnostics, trace)."""
        ...

    @abstractmethod
    async def evaluate_stepwise(
        self, ir: Any, seed: int | None = None
    ) -> tuple[Fitness, Any, Any]:
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

    @abstractmethod
    def format_proof(self, genome: str) -> str:
        """Format a genome into a complete, standalone proof for this domain."""
        ...

    async def verify_proof(self, genome: str) -> bool:
        """Verify a completed proof using the backend's gold-standard checker.

        For formal verification backends, this should compile the proof
        independently of the REPL (e.g. via ``lake env lean``).  Returns
        ``True`` by default for backends without a formal verifier.
        """
        return True

    async def create_tree_search(
        self,
        prefix: list[str],
        llm_client: Any,
        max_nodes: int = 200,
        beam_width: int = 5,
    ) -> Any | None:
        """Create a tree search instance seeded from a tactic prefix.

        Returns None if not supported by this backend.
        """
        return None

    async def startup(self) -> None:
        """Initialize backend resources (e.g. REPL process). No-op by default."""

    async def shutdown(self) -> None:
        """Release backend resources. No-op by default."""
