# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for Lean structural validation."""

from __future__ import annotations

from evoforge.backends.lean.ir import parse_tactic_sequence
from evoforge.backends.lean.validation import validate_structure_lean


class TestSearchTacticRejection:
    def test_exact_question_rejected(self) -> None:
        """exact? (search tactic) is rejected by structural validation."""
        seq = parse_tactic_sequence("exact?")
        assert seq is not None
        violations = validate_structure_lean(seq)
        assert any("exact?" in v for v in violations)

    def test_apply_question_rejected(self) -> None:
        """apply? (search tactic) is rejected by structural validation."""
        seq = parse_tactic_sequence("apply?")
        assert seq is not None
        violations = validate_structure_lean(seq)
        assert any("apply?" in v for v in violations)

    def test_exact_without_question_allowed(self) -> None:
        """exact (without ?) is still allowed."""
        seq = parse_tactic_sequence("exact rfl")
        assert seq is not None
        violations = validate_structure_lean(seq)
        assert violations == []

    def test_apply_without_question_allowed(self) -> None:
        """apply (without ?) is still allowed."""
        seq = parse_tactic_sequence("apply le_of_eq")
        assert seq is not None
        violations = validate_structure_lean(seq)
        assert violations == []
