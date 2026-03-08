# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.

"""Ablation-based credit assignment for CFD turbulence closures.

For each additive term in a :class:`ClosureExpr`, we remove that term,
re-evaluate the ablated expression, and compute the credit as the accuracy
delta: ``baseline_accuracy - ablated_accuracy``.  A positive score means
the term is beneficial.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from evoforge.backends.cfd.ir import ClosureExpr
from evoforge.core.types import Credit, Fitness

# Interaction effects make single-term ablation an approximation.
_DEFAULT_CONFIDENCE: float = 0.8


async def assign_credit_cfd(
    ir: ClosureExpr,
    fitness: Fitness,
    quick_eval: Callable[[ClosureExpr], Coroutine[Any, Any, Fitness]],
) -> list[Credit]:
    """Ablation-based credit assignment for additive terms.

    Parameters
    ----------
    ir:
        The full closure expression (baseline).
    fitness:
        The already-computed fitness of *ir*.
    quick_eval:
        An async callable that evaluates a :class:`ClosureExpr` and returns
        its :class:`Fitness`.  Typically a thin wrapper around
        ``CFDBackend.evaluate()``.

    Returns
    -------
    list[Credit]
        One :class:`Credit` per additive term, with ``score`` equal to
        ``baseline_accuracy - ablated_accuracy`` (positive = helpful).
    """
    terms = ir.additive_terms()
    if len(terms) <= 1:
        # Single-term expression: no meaningful ablation possible.
        # Return a single credit with score 0 (no delta measurable).
        return [
            Credit(
                location=0,
                score=0.0,
                signal="single_term_no_ablation",
                confidence=_DEFAULT_CONFIDENCE,
            )
        ]

    baseline_accuracy = fitness.primary
    credits: list[Credit] = []

    for i in range(len(terms)):
        ablated = ir.remove_term(i)
        # Skip degenerate ablations (e.g. removing leaves nothing useful).
        if ablated.complexity() == 0:
            credits.append(
                Credit(
                    location=i,
                    score=0.0,
                    signal="ablated_to_zero",
                    confidence=_DEFAULT_CONFIDENCE,
                )
            )
            continue

        ablated_fitness = await quick_eval(ablated)
        score = baseline_accuracy - ablated_fitness.primary
        credits.append(
            Credit(
                location=i,
                score=score,
                signal="ablation_delta",
                confidence=_DEFAULT_CONFIDENCE,
            )
        )

    return credits
