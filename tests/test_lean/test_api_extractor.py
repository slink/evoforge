"""Tests for evoforge.backends.lean.api_extractor module."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from evoforge.backends.lean.api_extractor import (
    APIEntry,
    extract_api_from_file,
    extract_hypothesis_types,
)


@pytest.fixture
def sample_lean_file(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        import Mathlib

        namespace Foo

        /-- Docstring for bar. -/
        theorem bar (x : Nat) : x + 0 = x := by simp

        lemma baz (a b : Int) : a + b = b + a := by ring

        def helper (n : Nat) : Nat := n + 1

        theorem sorry_theorem (x : Nat) : x = x := by
          sorry

        end Foo

        namespace Other
        theorem other_thing : True := trivial
        end Other
    """)
    p = tmp_path / "Test.lean"
    p.write_text(content)
    return p


class TestApiEntry:
    def test_api_entry_fields(self) -> None:
        entry = APIEntry(
            name="foo",
            full_name="Bar.foo",
            signature="(x : Nat) : Nat",
            has_sorry=False,
        )
        assert entry.name == "foo"
        assert entry.full_name == "Bar.foo"
        assert entry.signature == "(x : Nat) : Nat"
        assert entry.has_sorry is False


class TestExtractApi:
    def test_extracts_declarations_in_namespace(self, sample_lean_file: Path) -> None:
        entries = extract_api_from_file(sample_lean_file, "Foo")
        names = [e.name for e in entries]
        assert "bar" in names
        assert "baz" in names
        assert "helper" in names

    def test_excludes_other_namespace(self, sample_lean_file: Path) -> None:
        entries = extract_api_from_file(sample_lean_file, "Foo")
        names = [e.name for e in entries]
        assert "other_thing" not in names

    def test_captures_signature(self, sample_lean_file: Path) -> None:
        entries = extract_api_from_file(sample_lean_file, "Foo")
        bar = next(e for e in entries if e.name == "bar")
        assert "(x : Nat)" in bar.signature
        assert "x + 0 = x" in bar.signature

    def test_marks_sorry_declarations(self, sample_lean_file: Path) -> None:
        entries = extract_api_from_file(sample_lean_file, "Foo")
        sorry_entry = next(e for e in entries if e.name == "sorry_theorem")
        assert sorry_entry.has_sorry is True

    def test_non_sorry_not_marked(self, sample_lean_file: Path) -> None:
        entries = extract_api_from_file(sample_lean_file, "Foo")
        bar = next(e for e in entries if e.name == "bar")
        assert bar.has_sorry is False

    def test_returns_empty_for_missing_namespace(self, sample_lean_file: Path) -> None:
        entries = extract_api_from_file(sample_lean_file, "NonExistent")
        assert entries == []

    def test_full_name_includes_namespace(self, sample_lean_file: Path) -> None:
        entries = extract_api_from_file(sample_lean_file, "Foo")
        bar = next(e for e in entries if e.name == "bar")
        assert bar.full_name == "Foo.bar"


class TestExtractHypothesisTypes:
    def test_extracts_type_from_hypothesis(self) -> None:
        stmt = (
            "theorem norm_le_one {φ : ℝ → ℂ} (hφ : IsPositiveDefinite φ) "
            "(h0 : φ 0 = 1) (ξ : ℝ) : ‖φ ξ‖ ≤ 1"
        )
        assert extract_hypothesis_types(stmt) == ["IsPositiveDefinite"]

    def test_skips_builtin_types(self) -> None:
        stmt = "theorem foo (n : ℕ) (x : ℝ) (c : ℂ) : n = n"
        assert extract_hypothesis_types(stmt) == []

    def test_multiple_custom_types(self) -> None:
        stmt = "theorem bar (hf : Continuous f) (hg : Measurable g) : True"
        assert extract_hypothesis_types(stmt) == ["Continuous", "Measurable"]

    def test_handles_no_hypotheses(self) -> None:
        stmt = "theorem trivial_thing : True"
        assert extract_hypothesis_types(stmt) == []
