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


class TestFindLeanFiles:
    def test_finds_file_with_namespace(self, tmp_path: Path) -> None:
        from evoforge.backends.lean.api_extractor import find_files_with_namespace

        (tmp_path / "A.lean").write_text("namespace Foo\ntheorem x : True := trivial\nend Foo\n")
        (tmp_path / "B.lean").write_text("namespace Bar\nend Bar\n")
        files = find_files_with_namespace(tmp_path, "Foo")
        assert len(files) == 1
        assert files[0].name == "A.lean"

    def test_searches_subdirectories(self, tmp_path: Path) -> None:
        from evoforge.backends.lean.api_extractor import find_files_with_namespace

        sub = tmp_path / "Sub"
        sub.mkdir()
        (sub / "C.lean").write_text("namespace Foo\nend Foo\n")
        files = find_files_with_namespace(tmp_path, "Foo")
        assert len(files) == 1

    def test_returns_empty_when_not_found(self, tmp_path: Path) -> None:
        from evoforge.backends.lean.api_extractor import find_files_with_namespace

        (tmp_path / "A.lean").write_text("namespace Bar\nend Bar\n")
        files = find_files_with_namespace(tmp_path, "Foo")
        assert files == []


class TestExtractApiForTheorem:
    def test_end_to_end(self, tmp_path: Path) -> None:
        from evoforge.backends.lean.api_extractor import extract_api_for_theorem

        lean_file = tmp_path / "Foo.lean"
        lean_file.write_text(
            "namespace IsPositiveDefinite\n"
            "theorem re_nonneg (n : ℕ) : 0 ≤ n := by omega\n"
            "theorem conj_neg (t : ℝ) : t = t := by rfl\n"
            "theorem sorry_thing : True := by\n  sorry\n"
            "end IsPositiveDefinite\n"
        )
        entries = extract_api_for_theorem(
            project_dir=tmp_path,
            theorem_statement="theorem foo (hφ : IsPositiveDefinite φ) : True",
        )
        names = [e.name for e in entries]
        assert "re_nonneg" in names
        assert "conj_neg" in names

    def test_extra_namespaces(self, tmp_path: Path) -> None:
        from evoforge.backends.lean.api_extractor import extract_api_for_theorem

        (tmp_path / "A.lean").write_text("namespace Extra\ndef helper : Nat := 0\nend Extra\n")
        entries = extract_api_for_theorem(
            project_dir=tmp_path,
            theorem_statement="theorem foo : True",
            extra_namespaces=["Extra"],
        )
        names = [e.name for e in entries]
        assert "helper" in names


class TestStartupApiExtraction:
    def test_extract_api_for_theorem_with_project(self, tmp_path: Path) -> None:
        from evoforge.backends.lean.api_extractor import extract_api_for_theorem

        lean_file = tmp_path / "Foo.lean"
        lean_file.write_text(
            textwrap.dedent("""\
            namespace IsPositiveDefinite
            theorem re_nonneg (n : ℕ) : 0 ≤ n := by omega
            theorem conj_neg (t : ℝ) : t = t := by rfl
            theorem norm_le_one (x : ℝ) : True := by
              sorry
            end IsPositiveDefinite
        """)
        )
        entries = extract_api_for_theorem(
            project_dir=tmp_path,
            theorem_statement="theorem foo (hφ : IsPositiveDefinite φ) : True",
        )
        names = [e.name for e in entries]
        assert "re_nonneg" in names
        assert "conj_neg" in names
        sorry_entries = [e for e in entries if e.has_sorry]
        assert len(sorry_entries) >= 1
