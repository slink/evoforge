# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for proof verification, configurable seeds, structured parsing, and prompts."""

from __future__ import annotations

import asyncio
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

    async def test_verify_proof_passes_threads_flag(self, tmp_path: Path) -> None:
        """verify_proof passes -j flag to lean."""
        backend = _make_backend(project_dir=str(tmp_path))
        backend._verification_threads = 4
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            await backend.verify_proof("exact rfl")
        cmd = mock_run.call_args[0][0]
        assert "-j4" in cmd


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

        # With verification failing, all proofs marked infeasible.
        # best_fitness only considers feasible individuals, so it should be < 1.0
        assert result.best_fitness < 1.0
        pop = engine.population.get_all()
        for ind in pop:
            if ind.fitness is not None and ind.fitness.primary >= 1.0:
                assert ind.fitness.feasible is False

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


# ---------------------------------------------------------------------------
# format_proof indentation preservation
# ---------------------------------------------------------------------------


class TestFormatProof:
    def test_preserves_relative_indentation(self) -> None:
        """format_proof preserves relative indentation for block proofs."""
        backend = _make_backend(
            theorem="theorem foo : True",
            imports="import Mathlib",
        )
        genome = "by_cases h : x = 0\n  · simp [h]\n  · ring"
        result = backend.format_proof(genome)
        lines = result.strip().split("\n")
        by_cases_line = [ln for ln in lines if "by_cases" in ln][0]
        cdot_line = [ln for ln in lines if "·" in ln][0]
        by_cases_indent = len(by_cases_line) - len(by_cases_line.lstrip())
        cdot_indent = len(cdot_line) - len(cdot_line.lstrip())
        assert cdot_indent > by_cases_indent

    def test_flat_tactics_indented_uniformly(self) -> None:
        """Flat tactics all get 2-space indent under 'by'."""
        backend = _make_backend(theorem="theorem foo : True")
        genome = "intro x\nsimp\nring"
        result = backend.format_proof(genome)
        tactic_lines = [
            ln
            for ln in result.strip().split("\n")
            if ln.strip() and ":= by" not in ln and "import" not in ln
        ]
        for line in tactic_lines:
            indent = len(line) - len(line.lstrip())
            assert indent == 2, f"Expected 2-space indent, got {indent}: {line!r}"

    def test_empty_lines_skipped(self) -> None:
        """Empty lines in genome are skipped."""
        backend = _make_backend(theorem="theorem foo : True")
        genome = "intro x\n\nsimp"
        result = backend.format_proof(genome)
        assert "\n\n" not in result.split(":= by")[1]


# ---------------------------------------------------------------------------
# REPL cmd verification
# ---------------------------------------------------------------------------


class TestREPLCmdVerification:
    def test_example_statement_replaces_theorem_name(self) -> None:
        """_example_statement() replaces 'theorem <name>' with 'example'."""
        backend = _make_backend(
            theorem="theorem norm_le_one {φ : ℝ → ℂ} (hφ : IsPositiveDefinite φ) : ‖φ ξ‖ ≤ 1",
        )
        result = backend._example_statement()
        assert result.startswith("example ")
        assert "norm_le_one" not in result
        assert "IsPositiveDefinite" in result

    async def test_verify_via_repl_cmd_success(self) -> None:
        """REPL cmd verification returns True on clean success."""
        backend = _make_backend()
        mock_repl = AsyncMock()
        mock_repl.send_command = AsyncMock(return_value={"env": 2})
        backend._repl = mock_repl
        backend._repl_lock = asyncio.Lock()
        backend._import_env = 0

        ok, err = await backend._verify_via_repl_cmd("exact rfl")
        assert ok is True
        assert err is None

    async def test_verify_via_repl_cmd_failure_on_error(self) -> None:
        """REPL cmd verification returns False on error response."""
        backend = _make_backend()
        mock_repl = AsyncMock()
        mock_repl.send_command = AsyncMock(
            return_value={"message": "type mismatch", "severity": "error"}
        )
        backend._repl = mock_repl
        backend._repl_lock = asyncio.Lock()
        backend._import_env = 0

        ok, err = await backend._verify_via_repl_cmd("ring")
        assert ok is False
        assert err is not None

    async def test_verify_via_repl_cmd_failure_on_sorry(self) -> None:
        """REPL cmd verification returns False when sorries present."""
        backend = _make_backend()
        mock_repl = AsyncMock()
        mock_repl.send_command = AsyncMock(return_value={"env": 2, "sorries": [{"proofState": 1}]})
        backend._repl = mock_repl
        backend._repl_lock = asyncio.Lock()
        backend._import_env = 0

        ok, err = await backend._verify_via_repl_cmd("sorry")
        assert ok is False
        assert err is not None

    async def test_verify_via_repl_cmd_no_repl(self) -> None:
        """Returns False when REPL is None."""
        backend = _make_backend()
        backend._repl = None
        ok, err = await backend._verify_via_repl_cmd("exact rfl")
        assert ok is False
        assert err is not None

    async def test_verify_via_repl_cmd_empty_genome(self) -> None:
        """Returns False for empty genome."""
        backend = _make_backend()
        mock_repl = AsyncMock()
        backend._repl = mock_repl
        backend._import_env = 0
        ok, err = await backend._verify_via_repl_cmd("")
        assert ok is False
        assert err is not None


# ---------------------------------------------------------------------------
# _verify_via_repl_cmd error messages
# ---------------------------------------------------------------------------


class TestVerifyViaCmdError:
    async def test_returns_error_message_on_severity_error(self) -> None:
        """Severity error returns the error message."""
        backend = _make_backend()
        mock_repl = AsyncMock()
        mock_repl.send_command = AsyncMock(
            return_value={"message": "type mismatch", "severity": "error"}
        )
        backend._repl = mock_repl
        backend._repl_lock = asyncio.Lock()
        backend._import_env = 0

        ok, err = await backend._verify_via_repl_cmd("simp")
        assert ok is False
        assert err == "type mismatch"

    async def test_returns_error_message_on_message_without_env(self) -> None:
        """Message without env returns the error message."""
        backend = _make_backend()
        mock_repl = AsyncMock()
        mock_repl.send_command = AsyncMock(return_value={"message": "unknown identifier 'foo'"})
        backend._repl = mock_repl
        backend._repl_lock = asyncio.Lock()
        backend._import_env = 0

        ok, err = await backend._verify_via_repl_cmd("simp")
        assert ok is False
        assert err == "unknown identifier 'foo'"

    async def test_returns_none_on_success(self) -> None:
        """Success returns (True, None)."""
        backend = _make_backend()
        mock_repl = AsyncMock()
        mock_repl.send_command = AsyncMock(return_value={"env": 2})
        backend._repl = mock_repl
        backend._repl_lock = asyncio.Lock()
        backend._import_env = 0

        ok, err = await backend._verify_via_repl_cmd("exact rfl")
        assert ok is True
        assert err is None

    async def test_returns_sorry_message(self) -> None:
        """Sorries returns (False, 'proof contains sorry')."""
        backend = _make_backend()
        mock_repl = AsyncMock()
        mock_repl.send_command = AsyncMock(return_value={"env": 2, "sorries": [{"proofState": 1}]})
        backend._repl = mock_repl
        backend._repl_lock = asyncio.Lock()
        backend._import_env = 0

        ok, err = await backend._verify_via_repl_cmd("sorry")
        assert ok is False
        assert err == "proof contains sorry"


# ---------------------------------------------------------------------------
# Two-tier fitness (inline REPL cmd verification in evaluate)
# ---------------------------------------------------------------------------


class TestTwoTierFitness:
    async def test_repl_complete_cmd_verified_gets_1_0(self) -> None:
        """Proof that passes both step-by-step and cmd verification gets fitness=1.0."""
        backend = _make_backend(project_dir="/tmp/test")
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value=(
                Fitness(
                    primary=1.0,
                    auxiliary={
                        "proof_complete": 1.0,
                        "steps_succeeded": 1.0,
                        "goals_remaining": 0.0,
                    },
                    constraints={},
                    feasible=True,
                ),
                MagicMock(),
                MagicMock(),
            )
        )
        backend._evaluator = mock_evaluator
        backend._repl_lock = asyncio.Lock()
        backend._verify_via_repl_cmd = AsyncMock(return_value=(True, None))  # type: ignore[method-assign]

        fitness, _, _ = await backend.evaluate(MagicMock())
        assert fitness.primary == 1.0
        assert fitness.feasible is True
        assert fitness.auxiliary.get("cmd_verified") == 1.0

    async def test_repl_complete_cmd_rejected_gets_graduated_score(self) -> None:
        """Proof that passes step-by-step but fails cmd verification gets graduated fitness."""
        backend = _make_backend(project_dir="/tmp/test")
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value=(
                Fitness(
                    primary=1.0,
                    auxiliary={
                        "proof_complete": 1.0,
                        "steps_succeeded": 1.0,
                        "goals_remaining": 0.0,
                    },
                    constraints={},
                    feasible=True,
                ),
                MagicMock(),
                MagicMock(),
            )
        )
        backend._evaluator = mock_evaluator
        backend._repl_lock = asyncio.Lock()
        backend._verify_via_repl_cmd = AsyncMock(return_value=(False, "type mismatch"))  # type: ignore[method-assign]

        ir = MagicMock()
        ir.serialize = MagicMock(return_value="ring")
        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.primary == pytest.approx(0.4)
        assert fitness.feasible is False
        assert fitness.auxiliary.get("cmd_verified") == 0.0
        assert fitness.constraints.get("cmd_verification") is True

    async def test_partial_proof_unchanged(self) -> None:
        """Partial proof (fitness < 1.0) is not affected by verification."""
        backend = _make_backend(project_dir="/tmp/test")
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value=(
                Fitness(
                    primary=0.5,
                    auxiliary={
                        "proof_complete": 0.0,
                        "steps_succeeded": 1.0,
                        "goals_remaining": 1.0,
                    },
                    constraints={},
                    feasible=True,
                ),
                MagicMock(),
                MagicMock(),
            )
        )
        backend._evaluator = mock_evaluator
        backend._repl_lock = asyncio.Lock()

        fitness, _, _ = await backend.evaluate(MagicMock())
        assert fitness.primary == pytest.approx(0.5)
        assert "cmd_verified" not in fitness.auxiliary


# ---------------------------------------------------------------------------
# Graduated cmd-verification scoring
# ---------------------------------------------------------------------------


class TestGraduatedCmdScoring:
    """False-positive proofs are marked infeasible with formula-derived fitness."""

    def _make_complete_backend(self, error_msg: str, *, cmd_passes: bool = False) -> LeanBackend:
        backend = _make_backend(project_dir="/tmp/test")
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value=(
                Fitness(
                    primary=1.0,
                    auxiliary={
                        "proof_complete": 1.0,
                        "steps_succeeded": 3.0,
                        "goals_remaining": 0.0,
                        "goal_reduction": 1.0,
                    },
                    constraints={},
                    feasible=True,
                ),
                MagicMock(),
                MagicMock(),
            )
        )
        backend._evaluator = mock_evaluator
        backend._repl_lock = asyncio.Lock()
        backend._verify_via_repl_cmd = AsyncMock(  # type: ignore[method-assign]
            return_value=(cmd_passes, None if cmd_passes else error_msg),
        )
        return backend

    @pytest.mark.asyncio
    async def test_false_positive_is_infeasible(self) -> None:
        backend = self._make_complete_backend("unsolved goals\n⊢ False")
        ir = MagicMock()
        ir.serialize = MagicMock(return_value="simp")
        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.feasible is False

    @pytest.mark.asyncio
    async def test_false_positive_uses_partial_formula(self) -> None:
        """For a 1-goal theorem (goal_reduction=1 → initial_goals=1) where all
        steps succeed but proof_complete=False:
        0.4 * (3/3) + 0.6 * ((1 - max(1,1)) / max(1,1)) = 0.4 * 1.0 + 0.6 * 0.0 = 0.4
        """
        backend = self._make_complete_backend("unsolved goals\n⊢ False")
        ir = MagicMock()
        ir.serialize = MagicMock(return_value="simp")
        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.primary == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_false_positive_has_cmd_constraint(self) -> None:
        backend = self._make_complete_backend("type mismatch\nexpected Nat")
        ir = MagicMock()
        ir.serialize = MagicMock(return_value="simp")
        fitness, _, _ = await backend.evaluate(ir)
        assert "cmd_verification" in fitness.constraints
        assert fitness.constraints["cmd_verification"] is True

    @pytest.mark.asyncio
    async def test_false_positive_preserves_error_pattern(self) -> None:
        backend = self._make_complete_backend("unknown identifier 'foo'")
        ir = MagicMock()
        ir.serialize = MagicMock(return_value="simp")
        fitness, _, _ = await backend.evaluate(ir)
        assert "cmd_error_pattern" in fitness.auxiliary

    @pytest.mark.asyncio
    async def test_cmd_verified_proof_stays_feasible(self) -> None:
        backend = self._make_complete_backend("", cmd_passes=True)
        ir = MagicMock()
        ir.serialize = MagicMock(return_value="simp")
        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.feasible is True
        assert fitness.primary == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cmd error diagnostics fields
# ---------------------------------------------------------------------------


class TestCmdErrorDiagnostics:
    def test_cmd_error_fields_default_none(self) -> None:
        """New cmd error fields default to False/None."""
        diag = LeanDiagnostics(
            success=True,
            goals_remaining=0,
            goal_types=[],
            goal_contexts=[],
            error_type=None,
            error_message=None,
            stuck_tactic_index=None,
            stuck_tactic=None,
            steps_succeeded=1,
            metavar_count=0,
        )
        assert diag.cmd_verification_attempted is False
        assert diag.cmd_error_message is None

    def test_cmd_error_appears_in_summary(self) -> None:
        """Cmd error message appears in summary when present."""
        diag = LeanDiagnostics(
            success=False,
            goals_remaining=0,
            goal_types=[],
            goal_contexts=[],
            error_type=None,
            error_message=None,
            stuck_tactic_index=None,
            stuck_tactic=None,
            steps_succeeded=1,
            metavar_count=0,
            cmd_verification_attempted=True,
            cmd_error_message="type mismatch in kernel",
        )
        s = diag.summary()
        assert "Cmd verification failed: type mismatch in kernel" in s

    def test_cmd_success_no_error_in_summary(self) -> None:
        """No cmd error text when cmd succeeded (no error message)."""
        diag = LeanDiagnostics(
            success=True,
            goals_remaining=0,
            goal_types=[],
            goal_contexts=[],
            error_type=None,
            error_message=None,
            stuck_tactic_index=None,
            stuck_tactic=None,
            steps_succeeded=1,
            metavar_count=0,
            cmd_verification_attempted=True,
            cmd_error_message=None,
        )
        s = diag.summary()
        assert "Cmd verification failed" not in s


# ---------------------------------------------------------------------------
# _classify_cmd_error
# ---------------------------------------------------------------------------


class TestClassifyCmdError:
    def test_unknown_identifier(self) -> None:
        """Extracts identifier name from unknown identifier error."""
        result = LeanBackend._classify_cmd_error("unknown identifier 'sum_nonneg'")
        assert result == "unknown_identifier:sum_nonneg"

    def test_unknown_identifier_with_context(self) -> None:
        """Extracts identifier from multiline unknown identifier error."""
        msg = (
            "unknown identifier 'Mathlib.Analysis.Normed.Group.Basic.norm_nonneg'"
            "\nsome extra context"
        )
        result = LeanBackend._classify_cmd_error(msg)
        ident = "Mathlib.Analysis.Normed.Group.Basic.norm_nonneg"
        assert result == f"unknown_identifier:{ident}"

    def test_type_mismatch(self) -> None:
        """Type mismatch errors are normalized."""
        result = LeanBackend._classify_cmd_error("type mismatch\nexpected Nat got Int")
        assert result == "type_mismatch"

    def test_unsolved_goals(self) -> None:
        """Unsolved goals errors are normalized."""
        result = LeanBackend._classify_cmd_error("unsolved goals\n⊢ False")
        assert result == "unsolved_goals"

    def test_sorry(self) -> None:
        """Sorry-containing messages are normalized."""
        result = LeanBackend._classify_cmd_error("proof contains sorry")
        assert result == "sorry"

    def test_fallback(self) -> None:
        """Unknown errors get 'other:' prefix with truncation."""
        msg = (
            "some completely unknown error that is quite long"
            " and should be truncated at sixty characters boundary"
        )
        result = LeanBackend._classify_cmd_error(msg)
        assert result.startswith("other:")
        assert len(result) <= 80

    def test_lean_error_prefix_stripped_unknown_identifier(self) -> None:
        """REPL 'Lean error: ' prefix is stripped before classification."""
        result = LeanBackend._classify_cmd_error("Lean error: unknown identifier 'foo'")
        assert result == "unknown_identifier:foo"

    def test_lean_error_prefix_stripped_type_mismatch(self) -> None:
        """REPL 'Lean error: ' prefix is stripped for type mismatch."""
        result = LeanBackend._classify_cmd_error("Lean error: type mismatch\nexpected Nat")
        assert result == "type_mismatch"


# ---------------------------------------------------------------------------
# API context in mutation prompt
# ---------------------------------------------------------------------------


class TestApiInMutationPrompt:
    def test_api_context_rendered_in_mutation_prompt(self) -> None:
        from evoforge.backends.lean.api_extractor import APIEntry

        b = _make_backend()
        b._api_context = [
            APIEntry(
                name="re_nonneg",
                full_name="IsPositiveDefinite.re_nonneg",
                signature="(n : ℕ) : 0 ≤ (∑ ...).re",
            ),
        ]
        parent = make_individual(genome="simp")
        prompt = b.format_mutation_prompt(parent, context=None)
        assert "Available API" in prompt
        assert "re_nonneg" in prompt

    def test_empty_api_no_section_in_mutation_prompt(self) -> None:
        b = _make_backend()
        b._api_context = []
        parent = make_individual(genome="simp")
        prompt = b.format_mutation_prompt(parent, context=None)
        assert "Available API" not in prompt

    def test_sorry_entries_excluded_in_mutation_prompt(self) -> None:
        from evoforge.backends.lean.api_extractor import APIEntry

        b = _make_backend()
        b._api_context = [
            APIEntry(name="good", full_name="Ns.good", signature=": True", has_sorry=False),
            APIEntry(name="bad", full_name="Ns.bad", signature=": True", has_sorry=True),
        ]
        parent = make_individual(genome="simp")
        prompt = b.format_mutation_prompt(parent, context=None)
        assert "Ns.good" in prompt
        assert "Ns.bad" not in prompt


# ---------------------------------------------------------------------------
# API context in system prompt
# ---------------------------------------------------------------------------


class TestApiInSystemPrompt:
    def test_api_context_rendered_in_system_prompt(self) -> None:
        from evoforge.backends.lean.api_extractor import APIEntry

        b = _make_backend()
        b._api_context = [
            APIEntry(
                name="re_nonneg",
                full_name="IsPositiveDefinite.re_nonneg",
                signature="(n : ℕ) (x : Fin n → ℝ) (c : Fin n → ℂ) : 0 ≤ (∑ ...).re",
            ),
            APIEntry(
                name="conj_neg",
                full_name="IsPositiveDefinite.conj_neg",
                signature="(t : ℝ) : φ (-t) = starRingEnd ℂ (φ t)",
            ),
        ]
        b.system_prompt.cache_clear()
        prompt = b.system_prompt()
        assert "re_nonneg" in prompt
        assert "conj_neg" in prompt
        assert "Available API" in prompt

    def test_empty_api_context_no_section(self) -> None:
        b = _make_backend()
        b._api_context = []
        b.system_prompt.cache_clear()
        prompt = b.system_prompt()
        assert "Available API" not in prompt

    def test_sorry_entries_excluded(self) -> None:
        from evoforge.backends.lean.api_extractor import APIEntry

        b = _make_backend()
        b._api_context = [
            APIEntry(name="good", full_name="Ns.good", signature=": True", has_sorry=False),
            APIEntry(name="bad", full_name="Ns.bad", signature=": True", has_sorry=True),
        ]
        b.system_prompt.cache_clear()
        prompt = b.system_prompt()
        assert "Ns.good" in prompt
        assert "Ns.bad" not in prompt


# ---------------------------------------------------------------------------
# evaluate() threads cmd error into diagnostics
# ---------------------------------------------------------------------------


class TestEvaluateCmdErrorThreading:
    @pytest.fixture
    def backend(self) -> LeanBackend:
        b = _make_backend(imports="import LeanLevy")
        b._repl = AsyncMock()
        b._repl_lock = asyncio.Lock()
        b._import_env = 0
        return b

    async def test_false_positive_stores_cmd_error_on_diagnostics(
        self, backend: LeanBackend
    ) -> None:
        ir = parse_tactic_sequence("simp")
        complete_fitness = Fitness(
            primary=1.0,
            auxiliary={"steps_succeeded": 1.0, "goals_remaining": 0.0, "proof_complete": 1.0},
            constraints={},
            feasible=True,
        )
        diag = LeanDiagnostics(
            success=True,
            goals_remaining=0,
            goal_types=[],
            goal_contexts=[],
            error_type=None,
            error_message=None,
            stuck_tactic_index=None,
            stuck_tactic=None,
            steps_succeeded=1,
            metavar_count=0,
        )
        backend._evaluator = AsyncMock()
        backend._evaluator.evaluate = AsyncMock(return_value=(complete_fitness, diag, None))
        backend._repl.send_command = AsyncMock(
            return_value={"severity": "error", "message": "unknown identifier 'sum_nonneg'"}
        )

        fitness, returned_diag, _ = await backend.evaluate(ir)
        assert fitness.primary == pytest.approx(0.4)
        assert fitness.feasible is False
        assert fitness.constraints.get("cmd_verification") is True
        assert returned_diag.cmd_verification_attempted is True
        assert returned_diag.cmd_error_message == "unknown identifier 'sum_nonneg'"

    async def test_cmd_error_pattern_in_auxiliary(self, backend: LeanBackend) -> None:
        """Classified cmd error pattern is stored in Fitness.auxiliary."""
        ir = parse_tactic_sequence("simp")
        complete_fitness = Fitness(
            primary=1.0,
            auxiliary={"steps_succeeded": 1.0, "goals_remaining": 0.0, "proof_complete": 1.0},
            constraints={},
            feasible=True,
        )
        diag = LeanDiagnostics(
            success=True,
            goals_remaining=0,
            goal_types=[],
            goal_contexts=[],
            error_type=None,
            error_message=None,
            stuck_tactic_index=None,
            stuck_tactic=None,
            steps_succeeded=1,
            metavar_count=0,
        )
        backend._evaluator = AsyncMock()
        backend._evaluator.evaluate = AsyncMock(return_value=(complete_fitness, diag, None))
        backend._repl.send_command = AsyncMock(
            return_value={"severity": "error", "message": "unknown identifier 'sum_nonneg'"}
        )

        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.auxiliary["cmd_error_pattern"] == "unknown_identifier:sum_nonneg"

    async def test_cmd_success_sets_attempted_no_error(self, backend: LeanBackend) -> None:
        ir = parse_tactic_sequence("simp")
        complete_fitness = Fitness(
            primary=1.0,
            auxiliary={"steps_succeeded": 1.0, "goals_remaining": 0.0, "proof_complete": 1.0},
            constraints={},
            feasible=True,
        )
        diag = LeanDiagnostics(
            success=True,
            goals_remaining=0,
            goal_types=[],
            goal_contexts=[],
            error_type=None,
            error_message=None,
            stuck_tactic_index=None,
            stuck_tactic=None,
            steps_succeeded=1,
            metavar_count=0,
        )
        backend._evaluator = AsyncMock()
        backend._evaluator.evaluate = AsyncMock(return_value=(complete_fitness, diag, None))
        backend._repl.send_command = AsyncMock(return_value={"env": 42})

        fitness, returned_diag, _ = await backend.evaluate(ir)
        assert fitness.primary == 1.0
        assert returned_diag.cmd_verification_attempted is True
        assert returned_diag.cmd_error_message is None


# ---------------------------------------------------------------------------
# Engine: cmd error patterns → SearchMemory dead_ends
# ---------------------------------------------------------------------------


class TestEngineErrorToMemory:
    def test_cmd_error_pattern_flows_to_dead_ends(self) -> None:
        """Simulate the engine pattern: extract cmd_error_pattern from auxiliary, add to memory."""
        from evoforge.core.memory import SearchMemory

        memory = SearchMemory(max_dead_ends=50)
        ind = make_individual(
            genome="simp",
            fitness=Fitness(
                primary=0.4,
                auxiliary={
                    "cmd_verified": 0.0,
                    "cmd_error_pattern": "unknown_identifier:sum_nonneg",
                },
                constraints={"cmd_verification": True},
                feasible=False,
            ),
        )
        # Simulate what the engine does
        pattern = ind.fitness.auxiliary.get("cmd_error_pattern")
        if pattern and isinstance(pattern, str):
            memory.dead_ends.add(f"cmd_error:{pattern}")

        assert "cmd_error:unknown_identifier:sum_nonneg" in memory.dead_ends

    def test_no_pattern_no_dead_end(self) -> None:
        """No cmd_error_pattern means no dead end added."""
        from evoforge.core.memory import SearchMemory

        memory = SearchMemory(max_dead_ends=50)
        ind = make_individual(
            genome="simp",
            fitness=Fitness(
                primary=1.0,
                auxiliary={"cmd_verified": 1.0},
                constraints={},
                feasible=True,
            ),
        )
        pattern = ind.fitness.auxiliary.get("cmd_error_pattern")
        if pattern and isinstance(pattern, str):
            memory.dead_ends.add(f"cmd_error:{pattern}")

        assert len(memory.dead_ends) == 0

    def test_dead_ends_appear_in_prompt_section(self) -> None:
        """Verify dead ends show up in the memory prompt section."""
        from evoforge.core.memory import SearchMemory

        memory = SearchMemory(max_dead_ends=50)
        memory.dead_ends.add("cmd_error:unknown_identifier:sum_nonneg")
        section = memory.prompt_section()
        assert "sum_nonneg" in section
        assert "do not exist" in section.lower() or "do not use" in section.lower()
        assert "Dead ends" in section
