# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""ClosureExpr IR for CFD turbulence closure expressions.

Represents symbolic damping functions f(Ri_g) as SymPy expressions,
implementing the IRProtocol for use with the evoforge evolutionary engine.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from tokenize import TokenError

import sympy
from sympy import Add

# Canonical symbol used throughout the CFD backend.
Ri_g: sympy.Symbol = sympy.Symbol("Ri_g", nonnegative=True)


class ClosureExpr:
    """IR node wrapping a SymPy expression over the Ri_g symbol.

    Implements ``IRProtocol`` (canonicalize, structural_hash, serialize,
    complexity) plus CFD-specific helpers for additive decomposition,
    subtree replacement, and NumPy evaluation.
    """

    __slots__ = ("expr", "_hash_cache")

    def __init__(self, expr: sympy.Expr) -> None:
        self.expr: sympy.Expr = expr
        self._hash_cache: str | None = None

    # -- IRProtocol ----------------------------------------------------------

    def canonicalize(self) -> ClosureExpr:
        """Return a canonical form by expanding and simplifying."""
        canonical = sympy.expand(sympy.simplify(self.expr))
        return ClosureExpr(canonical)

    def structural_hash(self) -> str:
        """SHA-256 hex digest of the canonical serialization (cached)."""
        if self._hash_cache is None:
            canon = self.canonicalize()
            data = canon.serialize().encode("utf-8")
            self._hash_cache = hashlib.sha256(data).hexdigest()
        return self._hash_cache

    def serialize(self) -> str:
        """Human-readable string representation (via ``sympy.srepr`` of
        the canonical form for determinism, but we use ``str`` for
        readability and round-trip via ``parse_closure_expr``)."""
        return str(self.expr)

    def complexity(self) -> int:
        """Number of nodes in the SymPy expression tree."""
        return _count_nodes(self.expr)

    # -- CFD-specific helpers ------------------------------------------------

    def additive_terms(self) -> list[sympy.Expr]:
        """Decompose into additive components via ``sympy.Add.make_args``."""
        return list(Add.make_args(self.expr))

    def remove_term(self, index: int) -> ClosureExpr:
        """Remove the *index*-th additive term, raising ``IndexError`` if
        *index* is out of range."""
        terms = self.additive_terms()
        if index < 0 or index >= len(terms):
            msg = f"Term index {index} out of range [0, {len(terms)})"
            raise IndexError(msg)
        remaining = terms[:index] + terms[index + 1 :]
        if not remaining:
            return ClosureExpr(sympy.Integer(0))
        return ClosureExpr(Add(*remaining))

    def replace_subtree(self, target: sympy.Expr, replacement: sympy.Expr) -> ClosureExpr:
        """Substitute *target* with *replacement* in the expression."""
        new_expr = self.expr.subs(target, replacement)
        return ClosureExpr(new_expr)

    def lambdify(self) -> Callable[[float], float]:
        """Convert to a NumPy-backed callable ``f(Ri_g) -> float``.

        For constant expressions the returned callable still accepts one
        positional argument (which is ignored).
        """
        fn = sympy.lambdify(Ri_g, self.expr, modules="numpy")

        def _wrapper(val: float) -> float:
            result: float = float(fn(val))
            return result

        return _wrapper

    def free_symbols_ok(self) -> bool:
        """Return ``True`` if the expression uses only ``Ri_g`` (or no
        free symbols at all)."""
        return bool(self.expr.free_symbols <= {Ri_g})

    # -- dunder --------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ClosureExpr({self.expr!s})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClosureExpr):
            return NotImplemented
        return self.structural_hash() == other.structural_hash()

    def __hash__(self) -> int:
        return hash(self.structural_hash())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_nodes(expr: sympy.Basic) -> int:
    """Recursively count nodes in a SymPy expression tree."""
    if not expr.args:
        return 1
    return 1 + sum(_count_nodes(a) for a in expr.args)


def parse_closure_expr(text: str) -> ClosureExpr | None:
    """Parse a string into a ``ClosureExpr``, returning ``None`` on failure.

    The parser uses ``sympy.sympify`` with a local mapping that ensures
    ``Ri_g`` resolves to the canonical nonneg symbol.
    """
    if not text or not text.strip():
        return None
    try:
        parsed = sympy.parse_expr(
            text,
            local_dict={"Ri_g": Ri_g},
        )
        if not isinstance(parsed, sympy.Basic):
            return None
        # Ensure any Ri_g symbols carry the nonneg assumption by
        # substituting bare symbols that share the name.
        for sym in list(parsed.free_symbols):
            if sym.name == "Ri_g" and sym != Ri_g:
                parsed = parsed.subs(sym, Ri_g)
        return ClosureExpr(parsed)
    except (
        sympy.SympifyError,
        SyntaxError,
        TypeError,
        ValueError,
        TokenError,
    ):
        return None
