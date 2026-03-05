"""Tests for LeanBackend.format_proof()."""

from __future__ import annotations

from evoforge.backends.lean.backend import LeanBackend


class TestFormatProof:
    """LeanBackend.format_proof() produces a standalone Lean 4 proof."""

    def _make_backend(self, imports: str = "", theorem: str = "theorem foo") -> LeanBackend:
        return LeanBackend(
            theorem_statement=theorem,
            project_dir="/tmp/fake",
            imports=imports,
        )

    def test_basic_proof_structure(self) -> None:
        backend = self._make_backend(theorem="theorem norm_le_one (x : Real)")
        result = backend.format_proof("intro x\nlinarith")
        assert "theorem norm_le_one (x : Real) := by" in result
        assert "  intro x" in result
        assert "  linarith" in result
        assert result.endswith("\n")

    def test_includes_imports(self) -> None:
        backend = self._make_backend(
            imports="import Mathlib.Tactic",
            theorem="theorem foo",
        )
        result = backend.format_proof("simp")
        lines = result.split("\n")
        assert lines[0] == "import Mathlib.Tactic"
        assert lines[1] == ""
        assert "theorem foo := by" in result

    def test_no_imports(self) -> None:
        backend = self._make_backend(imports="", theorem="theorem bar")
        result = backend.format_proof("ring")
        assert result.startswith("theorem bar := by")

    def test_strips_empty_tactic_lines(self) -> None:
        backend = self._make_backend(theorem="theorem baz")
        result = backend.format_proof("intro x\n\n  \nlinarith")
        tactic_lines = [line for line in result.split("\n") if line.startswith("  ")]
        assert len(tactic_lines) == 2
        assert "  intro x" in tactic_lines
        assert "  linarith" in tactic_lines

    def test_strips_leading_trailing_whitespace_from_tactics(self) -> None:
        backend = self._make_backend(theorem="theorem qux")
        result = backend.format_proof("  simp  \n  ring  ")
        assert "  simp" in result
        assert "  ring" in result
