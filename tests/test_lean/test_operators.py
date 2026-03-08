# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for evoforge.backends.lean.operators — cheap mutation operators."""

from __future__ import annotations

from evoforge.backends.lean.ir import parse_tactic_sequence
from evoforge.backends.lean.operators import (
    PrefixTruncation,
    SplicePrefixes,
    TacticReorder,
    TacticSwap,
)
from evoforge.core.mutation import MutationContext
from evoforge.core.types import Credit, Individual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_individual(genome: str) -> Individual:
    """Create a minimal Individual for testing."""
    return Individual(
        genome=genome,
        ir=parse_tactic_sequence(genome),
        ir_hash="test-hash",
        generation=0,
    )


def _make_context(
    credits: list[Credit] | None = None,
    guidance: str = "",
    guidance_individual: Individual | None = None,
) -> MutationContext:
    """Create a minimal MutationContext for testing."""
    return MutationContext(
        generation=0,
        memory=None,
        guidance=guidance,
        temperature=0.5,
        backend=None,
        credits=credits or [],
        guidance_individual=guidance_individual,
    )


# ---------------------------------------------------------------------------
# PrefixTruncation
# ---------------------------------------------------------------------------


class TestPrefixTruncation:
    """Tests for the PrefixTruncation mutation operator."""

    def test_name_and_cost(self) -> None:
        op = PrefixTruncation()
        assert op.name == "prefix_truncation"
        assert op.cost == "cheap"

    async def test_truncates_to_last_positive_credit(self) -> None:
        """Given credits [+, +, -, -, -], truncates to first 2 steps."""
        genome = "intro x\napply h\nsimp\nexact rfl\nskip"
        parent = _make_individual(genome)
        credits = [
            Credit(location=0, score=1.0, signal="good"),
            Credit(location=1, score=0.5, signal="good"),
            Credit(location=2, score=-1.0, signal="bad"),
            Credit(location=3, score=-0.5, signal="bad"),
            Credit(location=4, score=-1.0, signal="bad"),
        ]
        ctx = _make_context(credits=credits)

        op = PrefixTruncation()
        result = await op.apply(parent, ctx)

        seq = parse_tactic_sequence(result)
        assert seq is not None
        assert len(seq.steps) == 2
        assert seq.steps[0].tactic == "intro"
        assert seq.steps[1].tactic == "apply"

    async def test_no_positive_credits_returns_first_step(self) -> None:
        """No positive credits -> returns first step only."""
        genome = "intro x\napply h\nsimp"
        parent = _make_individual(genome)
        credits = [
            Credit(location=0, score=-1.0, signal="bad"),
            Credit(location=1, score=-0.5, signal="bad"),
            Credit(location=2, score=-1.0, signal="bad"),
        ]
        ctx = _make_context(credits=credits)

        op = PrefixTruncation()
        result = await op.apply(parent, ctx)

        seq = parse_tactic_sequence(result)
        assert seq is not None
        assert len(seq.steps) == 1
        assert seq.steps[0].tactic == "intro"

    async def test_empty_credits_returns_first_step(self) -> None:
        """Empty credits list -> returns first step only."""
        genome = "intro x\napply h\nsimp"
        parent = _make_individual(genome)
        ctx = _make_context(credits=[])

        op = PrefixTruncation()
        result = await op.apply(parent, ctx)

        seq = parse_tactic_sequence(result)
        assert seq is not None
        assert len(seq.steps) == 1
        assert seq.steps[0].tactic == "intro"


# ---------------------------------------------------------------------------
# TacticSwap
# ---------------------------------------------------------------------------


class TestTacticSwap:
    """Tests for the TacticSwap mutation operator."""

    def test_name_and_cost(self) -> None:
        op = TacticSwap()
        assert op.name == "tactic_swap"
        assert op.cost == "cheap"

    async def test_produces_valid_swap(self) -> None:
        """Produces valid output with two adjacent tactics swapped."""
        genome = "intro x\napply h\nsimp\nexact rfl"
        parent = _make_individual(genome)
        ctx = _make_context()

        op = TacticSwap()
        result = await op.apply(parent, ctx)

        original = parse_tactic_sequence(genome)
        mutated = parse_tactic_sequence(result)
        assert original is not None
        assert mutated is not None
        # Same number of steps
        assert len(mutated.steps) == len(original.steps)
        # Same set of tactics (just reordered)
        orig_tactics = sorted(s.raw for s in original.steps)
        mut_tactics = sorted(s.raw for s in mutated.steps)
        assert orig_tactics == mut_tactics
        # Exactly one adjacent pair was swapped — so at most 2 positions differ
        diffs = [
            i for i in range(len(original.steps)) if original.steps[i].raw != mutated.steps[i].raw
        ]
        assert len(diffs) in (0, 2)
        if len(diffs) == 2:
            assert diffs[1] - diffs[0] == 1  # adjacent

    async def test_single_step_unchanged(self) -> None:
        """Single-step sequence -> returns unchanged."""
        genome = "intro x"
        parent = _make_individual(genome)
        ctx = _make_context()

        op = TacticSwap()
        result = await op.apply(parent, ctx)

        assert result == "intro x"


# ---------------------------------------------------------------------------
# TacticReorder
# ---------------------------------------------------------------------------


class TestTacticReorder:
    """Tests for the TacticReorder mutation operator."""

    def test_name_and_cost(self) -> None:
        op = TacticReorder()
        assert op.name == "tactic_reorder"
        assert op.cost == "cheap"

    async def test_produces_same_tactics_reordered(self) -> None:
        """Produces output with same tactics (just reordered in a window)."""
        genome = "intro x\napply h\nsimp\nexact rfl\nskip"
        parent = _make_individual(genome)
        ctx = _make_context()

        op = TacticReorder()
        result = await op.apply(parent, ctx)

        original = parse_tactic_sequence(genome)
        mutated = parse_tactic_sequence(result)
        assert original is not None
        assert mutated is not None
        # Same number of steps
        assert len(mutated.steps) == len(original.steps)
        # Same multiset of tactics
        orig_tactics = sorted(s.raw for s in original.steps)
        mut_tactics = sorted(s.raw for s in mutated.steps)
        assert orig_tactics == mut_tactics

    async def test_single_step_unchanged(self) -> None:
        """Single-step sequence -> returns unchanged."""
        genome = "intro x"
        parent = _make_individual(genome)
        ctx = _make_context()

        op = TacticReorder()
        result = await op.apply(parent, ctx)

        assert result == "intro x"


# ---------------------------------------------------------------------------
# SplicePrefixes
# ---------------------------------------------------------------------------


class TestSplicePrefixes:
    """Tests for the SplicePrefixes mutation operator."""

    def test_name_and_cost(self) -> None:
        op = SplicePrefixes()
        assert op.name == "splice_prefixes"
        assert op.cost == "cheap"

    async def test_splice_prefix_a_suffix_b(self) -> None:
        """Prefix of parent A + suffix of parent B produces correct splice."""
        genome_a = "intro x\napply h\nsimp\nexact rfl\nskip"
        genome_b = "intro y\napply g\nring\nomega\ndecide"
        parent = _make_individual(genome_a)
        other = _make_individual(genome_b)
        credits = [
            Credit(location=0, score=1.0, signal="good"),
            Credit(location=1, score=0.5, signal="good"),
            Credit(location=2, score=-1.0, signal="bad"),
            Credit(location=3, score=-0.5, signal="bad"),
            Credit(location=4, score=-1.0, signal="bad"),
        ]
        ctx = _make_context(credits=credits, guidance_individual=other)

        op = SplicePrefixes()
        result = await op.apply(parent, ctx)

        seq = parse_tactic_sequence(result)
        assert seq is not None
        # Prefix of A (2 steps with positive credits) + suffix of B (from index 2 onward)
        assert len(seq.steps) == 5
        assert seq.steps[0].tactic == "intro"
        assert seq.steps[0].args == "x"
        assert seq.steps[1].tactic == "apply"
        assert seq.steps[1].args == "h"
        # Suffix from B starts at index 2
        assert seq.steps[2].tactic == "ring"
        assert seq.steps[3].tactic == "omega"
        assert seq.steps[4].tactic == "decide"

    async def test_no_guidance_individual_returns_parent_unchanged(self) -> None:
        """No guidance_individual returns parent unchanged."""
        genome = "intro x\napply h\nsimp"
        parent = _make_individual(genome)
        ctx = _make_context(credits=[])

        op = SplicePrefixes()
        result = await op.apply(parent, ctx)

        assert result == genome

    async def test_no_positive_credits_takes_first_step_as_prefix(self) -> None:
        """No positive credits -> prefix is first step, rest from B."""
        genome_a = "intro x\napply h\nsimp"
        genome_b = "intro y\nring\nomega"
        parent = _make_individual(genome_a)
        other = _make_individual(genome_b)
        credits = [
            Credit(location=0, score=-1.0, signal="bad"),
            Credit(location=1, score=-0.5, signal="bad"),
            Credit(location=2, score=-1.0, signal="bad"),
        ]
        ctx = _make_context(credits=credits, guidance_individual=other)

        op = SplicePrefixes()
        result = await op.apply(parent, ctx)

        seq = parse_tactic_sequence(result)
        assert seq is not None
        # First step from A, then suffix from B starting at index 1
        assert len(seq.steps) == 3
        assert seq.steps[0].tactic == "intro"
        assert seq.steps[0].args == "x"
        assert seq.steps[1].tactic == "ring"
        assert seq.steps[2].tactic == "omega"
