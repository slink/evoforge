# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Cheap mutation operators for Lean tactic sequences.

Provides four lightweight (no LLM call) mutation operators that manipulate
tactic sequences structurally: truncation, adjacent swap, window reorder,
and prefix-splicing crossover.
"""

from __future__ import annotations

import random
from typing import Literal

from evoforge.backends.lean.ir import TacticSequence, parse_tactic_sequence
from evoforge.core.mutation import MutationContext, MutationOperator
from evoforge.core.types import Credit, Individual


def _last_positive_index(credits: list[Credit]) -> int | None:
    """Return the location of the last credit with score > 0, or None."""
    last: int | None = None
    for c in credits:
        if c.score > 0:
            last = c.location
    return last


def _credit_prefix_len(credits: list[Credit]) -> int:
    """Number of steps to keep: up to (and including) the last positively-credited step.

    Returns 1 if no positive credits exist (always keep at least the first step).
    """
    idx = _last_positive_index(credits)
    if idx is None:
        return 1
    return idx + 1


class PrefixTruncation(MutationOperator):
    """Cut after the last positively-credited step."""

    @property
    def name(self) -> str:
        return "prefix_truncation"

    @property
    def cost(self) -> Literal["cheap", "llm"]:
        return "cheap"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        seq = parse_tactic_sequence(parent.genome)
        if seq is None:
            return parent.genome
        keep = _credit_prefix_len(context.credits)
        return seq.prefix(keep).serialize()


class TacticSwap(MutationOperator):
    """Swap two adjacent tactics at a random position."""

    @property
    def name(self) -> str:
        return "tactic_swap"

    @property
    def cost(self) -> Literal["cheap", "llm"]:
        return "cheap"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        seq = parse_tactic_sequence(parent.genome)
        if seq is None or len(seq.steps) < 2:
            return parent.genome
        steps = list(seq.steps)
        i = random.randrange(len(steps) - 1)
        steps[i], steps[i + 1] = steps[i + 1], steps[i]
        return TacticSequence(steps=steps).serialize()


class TacticReorder(MutationOperator):
    """Permute a small window of tactics."""

    @property
    def name(self) -> str:
        return "tactic_reorder"

    @property
    def cost(self) -> Literal["cheap", "llm"]:
        return "cheap"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        seq = parse_tactic_sequence(parent.genome)
        if seq is None or len(seq.steps) < 2:
            return parent.genome
        steps = list(seq.steps)
        start = random.randrange(len(steps))
        window_size = min(3, len(steps) - start)
        window = steps[start : start + window_size]
        random.shuffle(window)
        steps[start : start + window_size] = window
        return TacticSequence(steps=steps).serialize()


class SplicePrefixes(MutationOperator):
    """Take credit-guided prefix of parent A, append suffix from parent B."""

    @property
    def name(self) -> str:
        return "splice_prefixes"

    @property
    def cost(self) -> Literal["cheap", "llm"]:
        return "cheap"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        if context.guidance_individual is not None:
            other_genome = context.guidance_individual.genome
        else:
            return parent.genome

        seq_a = parse_tactic_sequence(parent.genome)
        seq_b = parse_tactic_sequence(other_genome)
        if seq_a is None or seq_b is None:
            return parent.genome

        keep = _credit_prefix_len(context.credits)
        prefix_steps = seq_a.steps[:keep]
        suffix_steps = seq_b.steps[keep:]
        merged = prefix_steps + suffix_steps
        if not merged:
            return parent.genome
        return TacticSequence(steps=merged).serialize()
