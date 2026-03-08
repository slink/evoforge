# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for evoforge.core.identity — IdentityPipeline."""

from __future__ import annotations

import hashlib

from evoforge.core.identity import IdentityPipeline
from evoforge.core.types import Individual

# ---------------------------------------------------------------------------
# Helpers: mock IR and mock backend
# ---------------------------------------------------------------------------


class _MockIR:
    """A mock IR node implementing the IRProtocol interface."""

    def __init__(self, canonical_form: str) -> None:
        self._canonical = canonical_form

    def canonicalize(self) -> _MockIR:
        return _MockIR(self._canonical)

    def structural_hash(self) -> str:
        return hashlib.sha256(self._canonical.encode()).hexdigest()

    def serialize(self) -> str:
        return self._canonical

    def complexity(self) -> int:
        return len(self._canonical)


class _MockBackend:
    """A mock backend whose parse method returns _MockIR or None."""

    def __init__(self, mapping: dict[str, str | None]) -> None:
        # mapping: raw genome string -> canonical form (or None for unparseable)
        self._mapping = mapping

    def parse(self, genome: str) -> _MockIR | None:
        canonical = self._mapping.get(genome)
        if canonical is None:
            return None
        return _MockIR(canonical)


# ---------------------------------------------------------------------------
# IdentityPipeline.process()
# ---------------------------------------------------------------------------


class TestIdentityPipelineProcess:
    def test_same_canonical_form_gives_same_hash(self) -> None:
        """Two genomes that reduce to the same canonical form must produce
        identical ir_hash values."""
        backend = _MockBackend({"x + 0": "x", "0 + x": "x"})
        pipeline = IdentityPipeline(backend)

        result_a = pipeline.process("x + 0")
        result_b = pipeline.process("0 + x")

        assert result_a is not None
        assert result_b is not None
        assert result_a.ir_hash == result_b.ir_hash

    def test_different_canonical_forms_give_different_hashes(self) -> None:
        """Two genomes with different canonical forms must have different hashes."""
        backend = _MockBackend({"expr_a": "alpha", "expr_b": "beta"})
        pipeline = IdentityPipeline(backend)

        result_a = pipeline.process("expr_a")
        result_b = pipeline.process("expr_b")

        assert result_a is not None
        assert result_b is not None
        assert result_a.ir_hash != result_b.ir_hash

    def test_unparseable_genome_returns_none(self) -> None:
        """A genome the backend cannot parse must result in None."""
        backend = _MockBackend({"valid": "ok"})
        pipeline = IdentityPipeline(backend)

        result = pipeline.process("totally_invalid")
        assert result is None

    def test_idempotent_processing(self) -> None:
        """Processing the same genome twice must yield the same ir_hash."""
        backend = _MockBackend({"foo": "canonical_foo"})
        pipeline = IdentityPipeline(backend)

        first = pipeline.process("foo")
        second = pipeline.process("foo")

        assert first is not None
        assert second is not None
        assert first.ir_hash == second.ir_hash

    def test_returns_individual_with_correct_fields(self) -> None:
        """The returned Individual should have genome, ir, ir_hash set and
        sensible defaults for the rest."""
        backend = _MockBackend({"g": "canon"})
        pipeline = IdentityPipeline(backend)

        result = pipeline.process("g")

        assert result is not None
        assert isinstance(result, Individual)
        assert result.genome == "g"
        assert result.ir is not None
        assert isinstance(result.ir_hash, str)
        assert len(result.ir_hash) > 0
        assert result.generation == 0
        assert result.fitness is None
        assert result.credits == []
        assert result.lineage == {}


# ---------------------------------------------------------------------------
# IdentityPipeline.is_duplicate()
# ---------------------------------------------------------------------------


class TestIdentityPipelineIsDuplicate:
    def test_hash_in_set_returns_true(self) -> None:
        pipeline = IdentityPipeline(_MockBackend({}))
        known = {"abc123", "def456"}
        assert pipeline.is_duplicate("abc123", known) is True

    def test_hash_not_in_set_returns_false(self) -> None:
        pipeline = IdentityPipeline(_MockBackend({}))
        known = {"abc123", "def456"}
        assert pipeline.is_duplicate("xyz789", known) is False

    def test_empty_set_always_false(self) -> None:
        pipeline = IdentityPipeline(_MockBackend({}))
        assert pipeline.is_duplicate("anything", set()) is False
