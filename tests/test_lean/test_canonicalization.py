# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for evoforge.backends.lean.ir — TacticStep, TacticSequence, parse_tactic_sequence."""

from __future__ import annotations

import pytest

from evoforge.backends.lean.ir import TacticSequence, TacticStep, parse_tactic_sequence
from evoforge.core.ir import IRProtocol

# ---------------------------------------------------------------------------
# IRProtocol conformance
# ---------------------------------------------------------------------------


class TestIRProtocolConformance:
    def test_tactic_sequence_is_ir_protocol(self) -> None:
        """TacticSequence should satisfy the IRProtocol structural protocol."""
        seq = TacticSequence(steps=[TacticStep(tactic="intro", args="x", raw="intro x")])
        assert isinstance(seq, IRProtocol)


# ---------------------------------------------------------------------------
# Canonicalization: simp list sorting
# ---------------------------------------------------------------------------


class TestSimpSorting:
    def test_simp_list_sorted_alphabetically(self) -> None:
        """simp [b, a] should equal simp [a, b] after canonicalization."""
        seq1 = parse_tactic_sequence("simp [b, a]")
        seq2 = parse_tactic_sequence("simp [a, b]")
        assert seq1 is not None
        assert seq2 is not None
        canon1 = seq1.canonicalize()
        canon2 = seq2.canonicalize()
        assert canon1.structural_hash() == canon2.structural_hash()

    def test_simp_only_list_sorted(self) -> None:
        """simp only [c, a, b] should sort lemma names."""
        seq = parse_tactic_sequence("simp only [c, a, b]")
        assert seq is not None
        canon = seq.canonicalize()
        assert canon.steps[0].raw == "simp only [a, b, c]"

    def test_simp_without_brackets_unchanged(self) -> None:
        """Plain 'simp' tactic without brackets should pass through."""
        seq = parse_tactic_sequence("simp")
        assert seq is not None
        canon = seq.canonicalize()
        assert canon.steps[0].raw == "simp"


# ---------------------------------------------------------------------------
# Canonicalization: whitespace normalization
# ---------------------------------------------------------------------------


class TestWhitespaceNormalization:
    def test_multiple_spaces_collapsed(self) -> None:
        """'intro   x' should become 'intro x' after canonicalization."""
        seq1 = parse_tactic_sequence("intro   x")
        seq2 = parse_tactic_sequence("intro x")
        assert seq1 is not None
        assert seq2 is not None
        canon1 = seq1.canonicalize()
        canon2 = seq2.canonicalize()
        assert canon1.structural_hash() == canon2.structural_hash()

    def test_leading_trailing_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace should be removed."""
        seq = parse_tactic_sequence("  intro x  ")
        assert seq is not None
        canon = seq.canonicalize()
        assert canon.steps[0].raw == "intro x"


# ---------------------------------------------------------------------------
# Canonicalization: skip removal
# ---------------------------------------------------------------------------


class TestSkipRemoval:
    def test_skip_removed(self) -> None:
        """A sequence containing 'skip' should have it removed after canonicalization."""
        seq = parse_tactic_sequence("intro x\nskip\napply h")
        assert seq is not None
        canon = seq.canonicalize()
        assert len(canon.steps) == 2
        assert all(s.tactic != "skip" for s in canon.steps)

    def test_all_skips_removed(self) -> None:
        """Multiple consecutive skips should all be removed."""
        seq = parse_tactic_sequence("skip\nskip\nskip\nintro x")
        assert seq is not None
        canon = seq.canonicalize()
        assert len(canon.steps) == 1
        assert canon.steps[0].tactic == "intro"


# ---------------------------------------------------------------------------
# Structural hash
# ---------------------------------------------------------------------------


class TestStructuralHash:
    def test_hash_equivalence_for_semantically_equivalent(self) -> None:
        """Semantically equivalent sequences should produce the same hash."""
        seq1 = parse_tactic_sequence("intro   x\nskip\nsimp [b, a]")
        seq2 = parse_tactic_sequence("intro x\nsimp [a, b]")
        assert seq1 is not None
        assert seq2 is not None
        assert seq1.canonicalize().structural_hash() == seq2.canonicalize().structural_hash()

    def test_hash_uniqueness_for_different_sequences(self) -> None:
        """Semantically different sequences should produce different hashes."""
        seq1 = parse_tactic_sequence("intro x\napply h")
        seq2 = parse_tactic_sequence("intro y\napply h")
        assert seq1 is not None
        assert seq2 is not None
        assert seq1.canonicalize().structural_hash() != seq2.canonicalize().structural_hash()

    def test_hash_is_sha256_hex(self) -> None:
        """Structural hash should be a 64-character hex string (SHA-256)."""
        seq = parse_tactic_sequence("intro x")
        assert seq is not None
        h = seq.canonicalize().structural_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Prefix truncation
# ---------------------------------------------------------------------------


class TestPrefix:
    def test_prefix_truncation(self) -> None:
        """prefix(2) of a 4-step sequence should have 2 steps."""
        seq = parse_tactic_sequence("intro x\napply h\nsimp\nexact rfl")
        assert seq is not None
        pre = seq.prefix(2)
        assert len(pre.steps) == 2
        assert pre.steps[0].tactic == "intro"
        assert pre.steps[1].tactic == "apply"

    def test_prefix_larger_than_length(self) -> None:
        """prefix(k) where k > len should return all steps."""
        seq = parse_tactic_sequence("intro x\napply h")
        assert seq is not None
        pre = seq.prefix(10)
        assert len(pre.steps) == 2


# ---------------------------------------------------------------------------
# Serialize round-trip
# ---------------------------------------------------------------------------


class TestSerializeRoundTrip:
    def test_serialize_then_parse_same_hash(self) -> None:
        """Serializing then parsing should produce the same structural hash."""
        seq = parse_tactic_sequence("intro x\napply h\nsimp [a, b]")
        assert seq is not None
        canon = seq.canonicalize()
        serialized = canon.serialize()
        reparsed = parse_tactic_sequence(serialized)
        assert reparsed is not None
        assert reparsed.canonicalize().structural_hash() == canon.structural_hash()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_canonicalize_idempotent(self) -> None:
        """canonicalize(canonicalize(x)) should equal canonicalize(x)."""
        seq = parse_tactic_sequence("intro   x\nskip\nsimp [c, a, b]")
        assert seq is not None
        once = seq.canonicalize()
        twice = once.canonicalize()
        assert once.structural_hash() == twice.structural_hash()
        assert once.serialize() == twice.serialize()


# ---------------------------------------------------------------------------
# Empty genome
# ---------------------------------------------------------------------------


class TestParseEdgeCases:
    def test_empty_genome_returns_none(self) -> None:
        """Empty string should return None from parse_tactic_sequence."""
        assert parse_tactic_sequence("") is None

    def test_whitespace_only_returns_none(self) -> None:
        """Whitespace-only string should return None."""
        assert parse_tactic_sequence("   \n\n  ") is None

    def test_complexity_equals_step_count(self) -> None:
        """complexity() should return the number of steps."""
        seq = parse_tactic_sequence("intro x\napply h\nsimp")
        assert seq is not None
        assert seq.complexity() == 3


# ---------------------------------------------------------------------------
# TacticStep construction
# ---------------------------------------------------------------------------


class TestTacticStep:
    def test_frozen(self) -> None:
        """TacticStep should be frozen."""
        step = TacticStep(tactic="intro", args="x", raw="intro x")
        with pytest.raises(AttributeError):
            step.tactic = "apply"  # type: ignore[misc]

    def test_fields(self) -> None:
        """TacticStep should store tactic, args, raw."""
        step = TacticStep(tactic="simp", args="[a, b]", raw="simp [a, b]")
        assert step.tactic == "simp"
        assert step.args == "[a, b]"
        assert step.raw == "simp [a, b]"
