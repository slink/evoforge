# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.

"""Tests for CFD cheap mutation operators."""

from __future__ import annotations

import random

from evoforge.backends.cfd.ir import parse_closure_expr
from evoforge.backends.cfd.operators import (
    ConstantPerturb,
    SubtreeMutate,
    TermAddRemove,
)
from evoforge.core.types import Individual


def _make_parent(genome: str) -> Individual:
    """Create a test Individual from a genome string."""
    ir = parse_closure_expr(genome)
    assert ir is not None
    return Individual(genome=genome, ir=ir, ir_hash="test", generation=0)


# ---------------------------------------------------------------------------
# ConstantPerturb
# ---------------------------------------------------------------------------


class TestConstantPerturb:
    def test_produces_different_genomes(self) -> None:
        """Running multiple times should produce at least one different genome."""
        op = ConstantPerturb()
        parent = _make_parent("1 - 4*Ri_g")
        results: set[str | None] = set()
        for i in range(20):
            random.seed(i)
            result = op.apply(parent, context=None)
            results.add(result)
        # Should produce more than one distinct result
        non_none = {r for r in results if r is not None}
        assert len(non_none) > 1, f"Expected variety, got {non_none}"

    def test_preserves_structure(self) -> None:
        """Result should still parse as a valid ClosureExpr."""
        op = ConstantPerturb()
        parent = _make_parent("exp(-5*Ri_g)")
        random.seed(42)
        result = op.apply(parent, context=None)
        assert result is not None
        ir = parse_closure_expr(result)
        assert ir is not None
        assert ir.free_symbols_ok()

    def test_no_constants_returns_none(self) -> None:
        """An expression with no numeric constants (other than 0) returns None."""
        op = ConstantPerturb()
        # Ri_g alone has no Number atoms (excluding 0)
        parent = _make_parent("Ri_g")
        result = op.apply(parent, context=None)
        # Ri_g has no Number atoms at all, so should return None
        assert result is None


# ---------------------------------------------------------------------------
# SubtreeMutate
# ---------------------------------------------------------------------------


class TestSubtreeMutate:
    def test_produces_valid_expression(self) -> None:
        """Result should be a parseable ClosureExpr."""
        op = SubtreeMutate()
        parent = _make_parent("1 - Ri_g/0.25")
        valid_count = 0
        for i in range(20):
            random.seed(i)
            result = op.apply(parent, context=None)
            if result is not None:
                ir = parse_closure_expr(result)
                assert ir is not None, f"Failed to parse: {result}"
                assert ir.free_symbols_ok()
                valid_count += 1
        assert valid_count > 0, "Expected at least one valid mutation"

    def test_single_atom_returns_none(self) -> None:
        """A single-atom expression has no non-root nodes to replace."""
        op = SubtreeMutate()
        parent = _make_parent("1")
        # Integer(1) has no children, so nodes list has length 1
        result = op.apply(parent, context=None)
        assert result is None


# ---------------------------------------------------------------------------
# TermAddRemove
# ---------------------------------------------------------------------------


class TestTermAddRemove:
    def test_sometimes_adds_terms(self) -> None:
        """With a single-term expression, should always add (can't remove)."""
        op = TermAddRemove()
        parent = _make_parent("exp(-Ri_g)")
        added_count = 0
        for i in range(20):
            random.seed(i)
            result = op.apply(parent, context=None)
            assert result is not None
            ir = parse_closure_expr(result)
            assert ir is not None
            if len(ir.additive_terms()) > 1:
                added_count += 1
        assert added_count > 0, "Expected at least one addition"

    def test_sometimes_removes_terms(self) -> None:
        """With a multi-term expression, should sometimes remove a term."""
        op = TermAddRemove()
        parent = _make_parent("1 - 4*Ri_g + Ri_g**2")
        original_terms = len(parse_closure_expr("1 - 4*Ri_g + Ri_g**2").additive_terms())  # type: ignore[union-attr]
        removed_count = 0
        for i in range(50):
            random.seed(i)
            result = op.apply(parent, context=None)
            assert result is not None
            ir = parse_closure_expr(result)
            assert ir is not None
            if len(ir.additive_terms()) < original_terms:
                removed_count += 1
        assert removed_count > 0, "Expected at least one removal"

    def test_add_produces_valid_expression(self) -> None:
        """Added terms should produce valid parseable expressions."""
        op = TermAddRemove()
        parent = _make_parent("1/(1 + 4*Ri_g)")
        for i in range(10):
            random.seed(i)
            result = op.apply(parent, context=None)
            assert result is not None
            ir = parse_closure_expr(result)
            assert ir is not None
            assert ir.free_symbols_ok()


# ---------------------------------------------------------------------------
# Integration with CFDBackend
# ---------------------------------------------------------------------------


class TestBackendWiring:
    def test_mutation_operators_returns_three(self) -> None:
        """CFDBackend.mutation_operators() should return the 3 cheap operators."""
        from evoforge.backends.cfd.backend import CFDBackend
        from evoforge.core.config import CFDBackendConfig

        config = CFDBackendConfig()
        backend = CFDBackend(config)
        ops = backend.mutation_operators()
        assert len(ops) == 3
        names = {op.name for op in ops}
        assert names == {"constant_perturb", "subtree_mutate", "term_add_remove"}
        assert all(op.cost == "cheap" for op in ops)
