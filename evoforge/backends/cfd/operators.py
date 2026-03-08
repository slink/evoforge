# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.

"""Cheap mutation operators for CFD closure expressions.

Three structure-aware operators that manipulate SymPy expression trees
representing turbulence damping functions f(Ri_g).
"""

from __future__ import annotations

import random
from typing import Literal

import sympy
from sympy import Integer, Rational, exp, sqrt

from evoforge.backends.cfd.ir import ClosureExpr, Ri_g, parse_closure_expr
from evoforge.core.mutation import MutationContext, MutationOperator
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


class ConstantPerturb(MutationOperator):
    """Perturb a random numerical constant by a Gaussian factor."""

    @property
    def name(self) -> str:
        return "constant_perturb"

    @property
    def cost(self) -> Literal["cheap", "llm"]:
        return "cheap"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        ir = parent.ir
        if not isinstance(ir, ClosureExpr):
            return parent.genome

        # Find all Number atoms, excluding zero
        constants = [atom for atom in ir.expr.atoms(sympy.Number) if not atom.is_zero]
        if not constants:
            return parent.genome

        target = random.choice(constants)
        factor = 1.0 + random.gauss(0.0, 0.15)
        new_val = sympy.nsimplify(float(target) * factor)
        new_expr = ir.expr.subs(target, new_val)
        result_str = str(new_expr)
        if not _validate_mutation(result_str):
            return parent.genome
        return result_str


class SubtreeMutate(MutationOperator):
    """Replace a random non-root subtree with a random fragment."""

    @property
    def name(self) -> str:
        return "subtree_mutate"

    @property
    def cost(self) -> Literal["cheap", "llm"]:
        return "cheap"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        ir = parent.ir
        if not isinstance(ir, ClosureExpr):
            return parent.genome

        nodes = _collect_nodes(ir.expr)
        # Exclude root node (index 0)
        if len(nodes) < 2:
            return parent.genome

        target = random.choice(nodes[1:])
        replacement = random.choice(_FRAGMENTS)
        new_expr = ir.expr.subs(target, replacement)

        # Validate the result
        result = parse_closure_expr(str(new_expr))
        if result is None or not result.free_symbols_ok():
            return parent.genome
        return str(new_expr)


class TermAddRemove(MutationOperator):
    """Add or remove an additive term from the expression."""

    @property
    def name(self) -> str:
        return "term_add_remove"

    @property
    def cost(self) -> Literal["cheap", "llm"]:
        return "cheap"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        ir = parent.ir
        if not isinstance(ir, ClosureExpr):
            return parent.genome

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
        result_str = str(new_expr)
        if not _validate_mutation(result_str):
            return parent.genome
        return result_str


def _validate_mutation(genome_str: str) -> bool:
    """Validate that a mutated expression is physically plausible.

    Checks that the expression parses, uses only Ri_g, and f(0) ≈ 1.
    """
    ir = parse_closure_expr(genome_str)
    if ir is None or not ir.free_symbols_ok():
        return False
    try:
        fn = ir.lambdify()
        f0 = fn(0.0)
        if not (isinstance(f0, (int, float)) and abs(f0 - 1.0) < 0.2):
            return False
    except Exception:
        return False
    return True
