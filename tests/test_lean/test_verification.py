"""Tests for proof verification, configurable seeds, structured parsing, and prompts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evoforge.backends.lean.backend import _SEED_BANK, LeanBackend
from evoforge.backends.lean.evaluator import LeanDiagnostics
from evoforge.backends.lean.ir import parse_tactic_sequence
from evoforge.core.archive import Archive
from evoforge.core.types import Fitness
from tests.conftest import make_individual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(
    theorem: str = "theorem norm_le_one : forall x, norm x <= 1",
    project_dir: str = "/tmp/lean_project",
    imports: str = "",
    seeds: list[str] | None = None,
) -> LeanBackend:
    return LeanBackend(
        theorem_statement=theorem,
        project_dir=project_dir,
        imports=imports,
        seeds=seeds,
    )


# ---------------------------------------------------------------------------
# verify_proof
# ---------------------------------------------------------------------------


class TestVerifyProof:
    async def test_verify_proof_returns_false_on_nonzero_exit(self, tmp_path: Path) -> None:
        """verify_proof returns False when lake env lean fails."""
        backend = _make_backend(project_dir=str(tmp_path))
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"error: unknown identifier"

        with patch("subprocess.run", return_value=mock_result):
            result = await backend.verify_proof("ring")
        assert result is False

    async def test_verify_proof_returns_true_on_zero_exit(self, tmp_path: Path) -> None:
        """verify_proof returns True when lake env lean succeeds with clean stderr."""
        backend = _make_backend(project_dir=str(tmp_path))
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result):
            result = await backend.verify_proof("exact rfl")
        assert result is True

    async def test_verify_proof_returns_false_on_sorry_warning(self, tmp_path: Path) -> None:
        """verify_proof returns False when exit 0 but stderr mentions sorry."""
        backend = _make_backend(project_dir=str(tmp_path))
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = b"warning: declaration uses 'sorry'\n"

        with patch("subprocess.run", return_value=mock_result):
            result = await backend.verify_proof("exact norm_le_one")
        assert result is False

    async def test_verify_proof_returns_false_on_exception(self, tmp_path: Path) -> None:
        """verify_proof returns False when subprocess raises."""
        backend = _make_backend(project_dir=str(tmp_path))

        with patch("subprocess.run", side_effect=TimeoutError("timed out")):
            result = await backend.verify_proof("ring")
        assert result is False


# ---------------------------------------------------------------------------
# Engine verification gate
# ---------------------------------------------------------------------------


class TestEngineVerificationGate:
    async def test_engine_downgrades_unverified(self, archive: Archive) -> None:
        """Engine downgrades fitness=1.0 when verify_proof returns False."""
        from evoforge.core.config import (
            EvoforgeConfig,
            EvolutionConfig,
            PopulationConfig,
            SelectionConfig,
        )
        from evoforge.core.engine import EvolutionEngine
        from tests.test_core.test_engine import PerfectFitnessBackend

        backend = PerfectFitnessBackend()
        backend.verify_proof = AsyncMock(return_value=False)  # type: ignore[method-assign]

        config = EvoforgeConfig(
            population=PopulationConfig(size=5, elite_k=2),
            selection=SelectionConfig(strategy="scalar_tournament", tournament_size=2),
            evolution=EvolutionConfig(max_generations=3, log_level="DEBUG"),
        )

        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        result = await engine.run()

        # With verification failing, fitness should be downgraded from 1.0
        assert result.best_fitness < 1.0
        assert result.best_fitness == pytest.approx(0.95)

    async def test_engine_allows_verified_proof(self, archive: Archive) -> None:
        """Engine keeps fitness=1.0 when verify_proof returns True."""
        from evoforge.core.config import (
            EvoforgeConfig,
            EvolutionConfig,
            PopulationConfig,
            SelectionConfig,
        )
        from evoforge.core.engine import EvolutionEngine
        from tests.test_core.test_engine import PerfectFitnessBackend

        backend = PerfectFitnessBackend()
        backend.verify_proof = AsyncMock(return_value=True)  # type: ignore[method-assign]

        config = EvoforgeConfig(
            population=PopulationConfig(size=5, elite_k=2),
            selection=SelectionConfig(strategy="scalar_tournament", tournament_size=2),
            evolution=EvolutionConfig(max_generations=3, log_level="DEBUG"),
        )

        engine = EvolutionEngine(config=config, backend=backend, archive=archive)
        result = await engine.run()

        assert result.best_fitness >= 1.0


# ---------------------------------------------------------------------------
# Configurable seed bank
# ---------------------------------------------------------------------------


class TestConfigurableSeedBank:
    def test_default_seeds_used_when_no_config(self) -> None:
        """Without config seeds, the default seed bank is used."""
        backend = _make_backend(seeds=None)
        seeds = backend.seed_population(5)
        assert len(seeds) == 5
        assert seeds[0] == _SEED_BANK[0]

    def test_config_seeds_prepended(self) -> None:
        """Config seeds appear first in the population."""
        custom = ["my_custom_tactic", "another_tactic"]
        backend = _make_backend(seeds=custom)
        seeds = backend.seed_population(3)
        assert seeds[0] == "my_custom_tactic"
        assert seeds[1] == "another_tactic"
        # Third wraps to the default bank
        assert seeds[2] == _SEED_BANK[0]

    def test_config_seeds_cycle(self) -> None:
        """Config seeds + default seeds cycle for large populations."""
        custom = ["custom1"]
        backend = _make_backend(seeds=custom)
        seeds = backend.seed_population(100)
        assert len(seeds) == 100
        assert seeds[0] == "custom1"
        assert seeds[1] == _SEED_BANK[0]


# ---------------------------------------------------------------------------
# Structured tactic parsing
# ---------------------------------------------------------------------------


class TestStructuredTacticParse:
    def test_by_cases_with_focused_blocks(self) -> None:
        """by_cases with · branches parsed as a single block."""
        genome = "by_cases h : x = 0\n· simp [h]\n· ring"
        seq = parse_tactic_sequence(genome)
        assert seq is not None
        # Should be ONE top-level step (the by_cases block)
        assert len(seq.steps) == 1
        assert seq.steps[0].tactic == "by_cases"
        assert "· simp [h]" in seq.steps[0].raw
        assert "· ring" in seq.steps[0].raw

    def test_indented_continuation(self) -> None:
        """Indented lines grouped with the previous top-level tactic."""
        genome = "have h : True := by\n  trivial\nexact h"
        seq = parse_tactic_sequence(genome)
        assert seq is not None
        assert len(seq.steps) == 2
        assert seq.steps[0].tactic == "have"
        assert "trivial" in seq.steps[0].raw
        assert seq.steps[1].tactic == "exact"

    def test_flat_tactics_still_work(self) -> None:
        """Simple flat tactics parse as before."""
        genome = "intro x\nsimp\nring"
        seq = parse_tactic_sequence(genome)
        assert seq is not None
        assert len(seq.steps) == 3
        assert seq.steps[0].tactic == "intro"
        assert seq.steps[1].tactic == "simp"
        assert seq.steps[2].tactic == "ring"

    def test_calc_block(self) -> None:
        """calc block with indented steps parsed as one block."""
        genome = "calc _ ≤ _ := by sorry\n  _ = _ := by sorry\nsimp"
        seq = parse_tactic_sequence(genome)
        assert seq is not None
        assert len(seq.steps) == 2
        assert seq.steps[0].tactic == "calc"
        assert seq.steps[1].tactic == "simp"

    def test_serialize_roundtrip_structured(self) -> None:
        """Serializing then re-parsing preserves blocks."""
        genome = "by_cases h : x = 0\n· simp [h]\n· ring"
        seq = parse_tactic_sequence(genome)
        assert seq is not None
        serialized = seq.serialize()
        reparsed = parse_tactic_sequence(serialized)
        assert reparsed is not None
        assert len(reparsed.steps) == len(seq.steps)

    def test_empty_lines_between_blocks(self) -> None:
        """Empty lines between top-level tactics are skipped."""
        genome = "intro x\n\nsimp\n\nring"
        seq = parse_tactic_sequence(genome)
        assert seq is not None
        assert len(seq.steps) == 3

    def test_canonicalize_block(self) -> None:
        """Canonicalization works on multi-line blocks."""
        genome = "by_cases h : x = 0\n· simp  [b, a]\n· ring"
        seq = parse_tactic_sequence(genome)
        assert seq is not None
        canon = seq.canonicalize()
        assert len(canon.steps) == 1
        # simp list should be sorted inside the block
        assert "simp [a, b]" in canon.steps[0].raw


# ---------------------------------------------------------------------------
# Mutation prompt includes goal state
# ---------------------------------------------------------------------------


class TestMutationPromptGoalState:
    def test_goal_state_in_prompt(self) -> None:
        """Mutation prompt includes goal state from diagnostics."""
        backend = _make_backend()
        diag = LeanDiagnostics(
            success=False,
            goals_remaining=1,
            goal_types=["‖φ ξ‖ ≤ 1"],
            goal_contexts=["φ : ℝ → ℂ, hφ : IsPositiveDefinite φ"],
            error_type=None,
            error_message=None,
            stuck_tactic_index=None,
            stuck_tactic=None,
            steps_succeeded=0,
            metavar_count=0,
        )
        parent = make_individual(
            genome="simp",
            fitness=Fitness(primary=0.0, auxiliary={}, constraints={}, feasible=False),
        )
        parent.diagnostics = diag

        prompt = backend.format_mutation_prompt(parent, context=None)
        assert "‖φ ξ‖ ≤ 1" in prompt
        assert "IsPositiveDefinite" in prompt

    def test_no_goal_state_when_no_diagnostics(self) -> None:
        """No goal state section when diagnostics is None."""
        backend = _make_backend()
        parent = make_individual(genome="simp")
        prompt = backend.format_mutation_prompt(parent, context=None)
        assert "Current Goal State" not in prompt


# ---------------------------------------------------------------------------
# System prompt includes math context
# ---------------------------------------------------------------------------


class TestSystemPromptMathContext:
    def test_system_prompt_includes_pd_context(self) -> None:
        """System prompt has positive definite context for relevant thms."""
        backend = _make_backend(
            theorem=("theorem norm_le_one {φ : ℝ → ℂ} (hφ : IsPositiveDefinite φ) : ‖φ ξ‖ ≤ 1"),
            imports="import LeanLevy",
        )
        prompt = backend.system_prompt()
        assert "positive definite" in prompt.lower()
        assert "Hermitian" in prompt

    def test_system_prompt_includes_norm_context(self) -> None:
        """System prompt includes norm hints for norm theorems."""
        backend = _make_backend(theorem="theorem foo : ‖x‖ ≤ 1")
        prompt = backend.system_prompt()
        assert "norm" in prompt.lower()

    def test_system_prompt_includes_proof_strategies(self) -> None:
        """System prompt mentions structured proof patterns."""
        backend = _make_backend()
        prompt = backend.system_prompt()
        assert "by_cases" in prompt
        assert "calc" in prompt
        assert "suffices" in prompt
