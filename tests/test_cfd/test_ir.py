# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for ClosureExpr IR — CFD turbulence closure backend."""

from __future__ import annotations

import math

import sympy

from evoforge.backends.cfd.ir import ClosureExpr, parse_closure_expr
from evoforge.core.ir import IRProtocol

Ri_g = sympy.Symbol("Ri_g", nonnegative=True)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_implements_ir_protocol(self) -> None:
        expr = ClosureExpr(sympy.Integer(1) + Ri_g)
        assert isinstance(expr, IRProtocol)


# ---------------------------------------------------------------------------
# Construction & basic properties
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_sympy_expr(self) -> None:
        e = ClosureExpr(Ri_g**2 + 1)
        assert isinstance(e.expr, sympy.Expr)

    def test_free_symbols_ok_valid(self) -> None:
        e = ClosureExpr(Ri_g**2 + 1)
        assert e.free_symbols_ok()

    def test_free_symbols_ok_invalid(self) -> None:
        x = sympy.Symbol("x")
        e = ClosureExpr(x + Ri_g)
        assert not e.free_symbols_ok()

    def test_free_symbols_ok_constant(self) -> None:
        e = ClosureExpr(sympy.Integer(42))
        assert e.free_symbols_ok()


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


class TestCanonicalize:
    def test_canonical_is_deterministic(self) -> None:
        a = ClosureExpr(1 + Ri_g + Ri_g**2)
        b = ClosureExpr(Ri_g**2 + 1 + Ri_g)
        assert a.canonicalize().serialize() == b.canonicalize().serialize()

    def test_canonical_simplifies(self) -> None:
        # (Ri_g + 1)*(Ri_g - 1) == Ri_g**2 - 1
        e = ClosureExpr((Ri_g + 1) * (Ri_g - 1))
        c = e.canonicalize()
        expected = ClosureExpr(Ri_g**2 - 1).canonicalize()
        assert c.serialize() == expected.serialize()

    def test_returns_closure_expr(self) -> None:
        e = ClosureExpr(Ri_g)
        c = e.canonicalize()
        assert isinstance(c, ClosureExpr)


# ---------------------------------------------------------------------------
# Structural hash
# ---------------------------------------------------------------------------


class TestStructuralHash:
    def test_same_expr_same_hash(self) -> None:
        a = ClosureExpr(Ri_g + 1)
        b = ClosureExpr(1 + Ri_g)
        assert a.structural_hash() == b.structural_hash()

    def test_different_expr_different_hash(self) -> None:
        a = ClosureExpr(Ri_g + 1)
        b = ClosureExpr(Ri_g + 2)
        assert a.structural_hash() != b.structural_hash()

    def test_hash_is_hex_string(self) -> None:
        h = ClosureExpr(Ri_g).structural_hash()
        assert isinstance(h, str)
        int(h, 16)  # should not raise


# ---------------------------------------------------------------------------
# Serialize / parse round-trip
# ---------------------------------------------------------------------------


class TestSerialize:
    def test_round_trip(self) -> None:
        original = ClosureExpr(Ri_g**2 + 3 * Ri_g + 1)
        text = original.serialize()
        restored = parse_closure_expr(text)
        assert restored is not None
        assert restored.canonicalize().serialize() == original.canonicalize().serialize()

    def test_serialize_returns_string(self) -> None:
        assert isinstance(ClosureExpr(Ri_g).serialize(), str)


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------


class TestComplexity:
    def test_constant_complexity(self) -> None:
        assert ClosureExpr(sympy.Integer(1)).complexity() == 1

    def test_symbol_complexity(self) -> None:
        assert ClosureExpr(Ri_g).complexity() == 1

    def test_sum_complexity(self) -> None:
        e = ClosureExpr(Ri_g + 1)
        assert e.complexity() > 1

    def test_more_complex_expr(self) -> None:
        simple = ClosureExpr(Ri_g + 1)
        complex_ = ClosureExpr(Ri_g**3 + 2 * Ri_g**2 + Ri_g + 1)
        assert complex_.complexity() > simple.complexity()


# ---------------------------------------------------------------------------
# Additive terms
# ---------------------------------------------------------------------------


class TestAdditiveTerms:
    def test_single_term(self) -> None:
        e = ClosureExpr(Ri_g**2)
        terms = e.additive_terms()
        assert len(terms) == 1

    def test_sum_terms(self) -> None:
        e = ClosureExpr(Ri_g**2 + 3 * Ri_g + 1)
        terms = e.additive_terms()
        assert len(terms) == 3

    def test_terms_sum_to_original(self) -> None:
        original = Ri_g**2 + 3 * Ri_g + 1
        e = ClosureExpr(original)
        terms = e.additive_terms()
        reconstructed = sum(terms[1:], terms[0])
        assert sympy.simplify(reconstructed - original) == 0


# ---------------------------------------------------------------------------
# remove_term
# ---------------------------------------------------------------------------


class TestRemoveTerm:
    def test_remove_first_term(self) -> None:
        e = ClosureExpr(Ri_g**2 + Ri_g + 1)
        terms = e.additive_terms()
        reduced = e.remove_term(0)
        remaining_terms = reduced.additive_terms()
        # Should have one fewer term
        assert len(remaining_terms) == len(terms) - 1

    def test_remove_returns_closure_expr(self) -> None:
        e = ClosureExpr(Ri_g + 1)
        result = e.remove_term(0)
        assert isinstance(result, ClosureExpr)

    def test_remove_out_of_range_raises(self) -> None:
        e = ClosureExpr(Ri_g + 1)
        try:
            e.remove_term(5)
            assert False, "Should have raised IndexError"
        except IndexError:
            pass


# ---------------------------------------------------------------------------
# replace_subtree
# ---------------------------------------------------------------------------


class TestReplaceSubtree:
    def test_replace_symbol(self) -> None:
        e = ClosureExpr(Ri_g**2 + 1)
        result = e.replace_subtree(Ri_g**2, Ri_g**3)
        expected = ClosureExpr(Ri_g**3 + 1)
        assert result.canonicalize().serialize() == expected.canonicalize().serialize()

    def test_replace_returns_closure_expr(self) -> None:
        e = ClosureExpr(Ri_g + 1)
        result = e.replace_subtree(Ri_g, Ri_g**2)
        assert isinstance(result, ClosureExpr)


# ---------------------------------------------------------------------------
# lambdify
# ---------------------------------------------------------------------------


class TestLambdify:
    def test_callable(self) -> None:
        e = ClosureExpr(Ri_g**2 + 1)
        f = e.lambdify()
        assert callable(f)

    def test_evaluates_correctly(self) -> None:
        e = ClosureExpr(Ri_g**2 + 1)
        f = e.lambdify()
        assert math.isclose(f(2.0), 5.0)

    def test_constant(self) -> None:
        e = ClosureExpr(sympy.Integer(7))
        f = e.lambdify()
        # lambdify of a constant should still accept an argument
        result = f(999.0)
        assert math.isclose(result, 7.0)


# ---------------------------------------------------------------------------
# parse_closure_expr
# ---------------------------------------------------------------------------


class TestParseClosureExpr:
    def test_simple_expr(self) -> None:
        result = parse_closure_expr("Ri_g**2 + 1")
        assert result is not None
        assert result.free_symbols_ok()

    def test_invalid_syntax_returns_none(self) -> None:
        result = parse_closure_expr(")(invalid++")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = parse_closure_expr("")
        assert result is None

    def test_preserves_nonneg_assumption(self) -> None:
        result = parse_closure_expr("Ri_g + 1")
        assert result is not None
        symbols = result.expr.free_symbols
        for s in symbols:
            assert s.is_nonnegative  # type: ignore[attr-defined]

    def test_disallowed_symbols(self) -> None:
        result = parse_closure_expr("x + y")
        assert result is not None
        assert not result.free_symbols_ok()
