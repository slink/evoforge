"""Tests for the Backend ABC contract.

Verifies that the Backend abstract base class defines the complete set of
abstract methods required by the design spec, including the 4 new methods
(version, eval_config_hash, format_reflection_prompt, default_operator_weights)
and the corrected format_crossover_prompt signature.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from evoforge.backends.base import Backend
from evoforge.core.types import (
    Credit,
    Fitness,
    Individual,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _abstract_method_names(cls: type) -> set[str]:
    """Return the set of abstract method names on *cls*."""
    return {
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if getattr(getattr(cls, name), "__isabstractmethod__", False)
    }


# ---------------------------------------------------------------------------
# Tests for required abstract methods
# ---------------------------------------------------------------------------


class TestBackendABCMethods:
    """Backend ABC must declare all design-spec abstract methods."""

    def test_has_version(self) -> None:
        assert "version" in _abstract_method_names(Backend)

    def test_has_eval_config_hash(self) -> None:
        assert "eval_config_hash" in _abstract_method_names(Backend)

    def test_has_format_reflection_prompt(self) -> None:
        assert "format_reflection_prompt" in _abstract_method_names(Backend)

    def test_has_default_operator_weights(self) -> None:
        assert "default_operator_weights" in _abstract_method_names(Backend)

    def test_has_format_crossover_prompt(self) -> None:
        assert "format_crossover_prompt" in _abstract_method_names(Backend)

    def test_has_format_proof(self) -> None:
        assert "format_proof" in _abstract_method_names(Backend)

    # Existing methods that must remain present:
    @pytest.mark.parametrize(
        "method",
        [
            "parse",
            "evaluate",
            "evaluate_stepwise",
            "assign_credit",
            "validate_structure",
            "seed_population",
            "mutation_operators",
            "system_prompt",
            "format_mutation_prompt",
            "extract_genome",
            "behavior_descriptor",
            "behavior_space",
            "recommended_selection",
        ],
    )
    def test_has_existing_method(self, method: str) -> None:
        assert method in _abstract_method_names(Backend)


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------


class TestBackendSignatures:
    """Verify method signatures match the design spec."""

    def test_version_returns_str(self) -> None:
        sig = inspect.signature(Backend.version)
        assert sig.return_annotation == "str"

    def test_eval_config_hash_returns_str(self) -> None:
        sig = inspect.signature(Backend.eval_config_hash)
        assert sig.return_annotation == "str"

    def test_format_reflection_prompt_params(self) -> None:
        sig = inspect.signature(Backend.format_reflection_prompt)
        params = list(sig.parameters.keys())
        assert "population" in params
        assert "memory" in params
        assert "generation" in params
        assert sig.return_annotation == "str"

    def test_default_operator_weights_returns_dict(self) -> None:
        sig = inspect.signature(Backend.default_operator_weights)
        assert sig.return_annotation == "dict[str, float]"

    def test_format_proof_returns_str(self) -> None:
        sig = inspect.signature(Backend.format_proof)
        params = list(sig.parameters.keys())
        assert "genome" in params
        assert sig.return_annotation == "str"

    def test_format_crossover_prompt_takes_two_individuals(self) -> None:
        """format_crossover_prompt must accept parent_a, parent_b, context."""
        sig = inspect.signature(Backend.format_crossover_prompt)
        params = list(sig.parameters.keys())
        assert "parent_a" in params
        assert "parent_b" in params
        assert "context" in params
        assert sig.return_annotation == "str"


# ---------------------------------------------------------------------------
# Instantiation guard
# ---------------------------------------------------------------------------


class TestBackendCannotInstantiate:
    """Backend is abstract and must not be instantiable."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            Backend()  # type: ignore[abstract]

    def test_missing_new_methods_prevents_instantiation(self) -> None:
        """A subclass that only implements the OLD methods should still be
        abstract because it's missing the 4 new ones."""

        class IncompleteBackend(Backend):
            def parse(self, genome: str) -> Any:
                return None

            async def evaluate(self, ir: Any, seed: int | None = None) -> Any:
                return None

            async def evaluate_stepwise(self, ir: Any, seed: int | None = None) -> Any:
                return None

            def assign_credit(
                self,
                ir: Any,
                fitness: Fitness,
                diagnostics: Any,
                trace: Any,
            ) -> list[Credit]:
                return []

            def validate_structure(self, ir: Any) -> list[str]:
                return []

            def seed_population(self, n: int) -> list[str]:
                return []

            def mutation_operators(self) -> list[Any]:
                return []

            def system_prompt(self) -> str:
                return ""

            def format_mutation_prompt(self, parent: Individual, context: Any) -> str:
                return ""

            def format_crossover_prompt(
                self,
                parent_a: Individual,
                parent_b: Individual,
                context: Any,
            ) -> str:
                return ""

            def extract_genome(self, raw_text: str) -> str | None:
                return None

            def behavior_descriptor(self, ir: Any, diagnostics: Any) -> tuple[Any, ...]:
                return ()

            def behavior_space(self) -> Any:
                return None

            def recommended_selection(self) -> str:
                return ""

        with pytest.raises(TypeError):
            IncompleteBackend()  # type: ignore[abstract]
