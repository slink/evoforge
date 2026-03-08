# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.

"""Cheap mutation operators for CFD closure expressions.

Three structure-aware operators that manipulate SymPy expression trees
representing turbulence damping functions f(Ri_g).
"""

from __future__ import annotations

import random
from typing import Any

import sympy
from sympy import Integer, Rational, exp, sqrt

from evoforge.backends.cfd.ir import ClosureExpr, Ri_g, parse_closure_expr
from evoforge.core.types import Individual

# Fragments used by SubtreeMutate for random subtree replacement.
_FRAGMENTS: list[sympy.Expr] = [
    Ri_g,
    Ri_g**2,
    exp(-Ri_g),
    Rational(1, 4),
    Rational(1, 2),
    Integer(1),
    sqrt(Ri_g),
]


def _collect_nodes(expr: sympy.Basic) -> list[sympy.Basic]:
    """Collect all nodes in a SymPy expression tree (pre-order)."""
    nodes: list[sympy.Basic] = [expr]
    for arg in expr.args:
        nodes.extend(_collect_nodes(arg))
    return nodes


class ConstantPerturb:
    """Perturb a random numerical constant by a Gaussian factor."""

    name: str = "constant_perturb"
    cost: str = "cheap"

    def apply(self, parent: Individual, context: Any) -> str | None:
        ir = parent.ir
        if not isinstance(ir, ClosureExpr):
            return None

        # Find all Number atoms, excluding zero
        constants = [
            atom
            for atom in ir.expr.atoms(sympy.Number)
            if atom != sympy.Integer(0)
        ]
        if not constants:
            return None

        target = random.choice(constants)
        factor = 1.0 + random.gauss(0.0, 0.15)
        new_val = sympy.nsimplify(float(target) * factor)
        new_expr = ir.expr.subs(target, new_val)
        return str(new_expr)


class SubtreeMutate:
    """Replace a random non-root subtree with a random fragment."""

    name: str = "subtree_mutate"
    cost: str = "cheap"

    def apply(self, parent: Individual, context: Any) -> str | None:
        ir = parent.ir
        if not isinstance(ir, ClosureExpr):
            return None

        nodes = _collect_nodes(ir.expr)
        # Exclude root node (index 0)
        if len(nodes) < 2:
            return None

        target = random.choice(nodes[1:])
        replacement = random.choice(_FRAGMENTS)
        new_expr = ir.expr.subs(target, replacement)

        # Validate the result
        result = parse_closure_expr(str(new_expr))
        if result is None or not result.free_symbols_ok():
            return None
        return str(new_expr)


class TermAddRemove:
    """Add or remove an additive term from the expression."""

    name: str = "term_add_remove"
    cost: str = "cheap"

    def apply(self, parent: Individual, context: Any) -> str | None:
        ir = parent.ir
        if not isinstance(ir, ClosureExpr):
            return None

        terms = ir.additive_terms()

        # 40% chance to remove a term (if more than 1)
        if len(terms) > 1 and random.random() < 0.4:
            idx = random.randrange(len(terms))
            new_ir = ir.remove_term(idx)
            return str(new_ir.expr)

        # Otherwise add a new term: coeff * fragment
        coeff = random.uniform(-0.5, 0.5)
        # Avoid near-zero coefficients
        if abs(coeff) < 0.01:
            coeff = 0.1
        fragment = random.choice(_FRAGMENTS)
        new_term = sympy.nsimplify(coeff) * fragment
        new_expr = ir.expr + new_term
        return str(new_expr)
