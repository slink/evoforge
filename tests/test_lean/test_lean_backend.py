"""Tests for the 5 new abstract methods on LeanBackend.

Covers: version(), eval_config_hash(), default_operator_weights(),
format_reflection_prompt(), and the corrected format_crossover_prompt() signature.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import evoforge.backends.lean.backend as backend_mod
from evoforge.backends.lean.backend import LeanBackend
from evoforge.core.types import Fitness
from tests.conftest import make_individual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(
    theorem: str = "theorem norm_le_one : forall x, norm x <= 1",
    project_dir: str = "/tmp/lean_project",
) -> LeanBackend:
    return LeanBackend(
        theorem_statement=theorem,
        project_dir=project_dir,
    )


# ---------------------------------------------------------------------------
# version()
# ---------------------------------------------------------------------------


class TestLeanBackendVersion:
    def test_returns_non_empty_string(self) -> None:
        backend = _make_backend()
        result = backend.version()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_lean(self) -> None:
        backend = _make_backend()
        result = backend.version()
        assert "lean" in result.lower()


# ---------------------------------------------------------------------------
# eval_config_hash()
# ---------------------------------------------------------------------------


class TestLeanBackendEvalConfigHash:
    def test_returns_non_empty_string(self) -> None:
        backend = _make_backend()
        result = backend.eval_config_hash()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_consistent_hash(self) -> None:
        """Same config should produce the same hash."""
        a = _make_backend()
        b = _make_backend()
        assert a.eval_config_hash() == b.eval_config_hash()

    def test_different_theorems_different_hash(self) -> None:
        a = _make_backend(theorem="theorem A : True")
        b = _make_backend(theorem="theorem B : False")
        assert a.eval_config_hash() != b.eval_config_hash()

    def test_different_project_dirs_different_hash(self) -> None:
        a = _make_backend(project_dir="/tmp/proj_a")
        b = _make_backend(project_dir="/tmp/proj_b")
        assert a.eval_config_hash() != b.eval_config_hash()

    def test_hash_length_is_16(self) -> None:
        backend = _make_backend()
        assert len(backend.eval_config_hash()) == 16


# ---------------------------------------------------------------------------
# default_operator_weights()
# ---------------------------------------------------------------------------


class TestLeanBackendDefaultOperatorWeights:
    def test_returns_dict_of_floats(self) -> None:
        backend = _make_backend()
        weights = backend.default_operator_weights()
        assert isinstance(weights, dict)
        for k, v in weights.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_has_four_operators(self) -> None:
        backend = _make_backend()
        weights = backend.default_operator_weights()
        assert len(weights) == 4

    def test_weights_sum_to_one(self) -> None:
        backend = _make_backend()
        weights = backend.default_operator_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_expected_operator_names(self) -> None:
        backend = _make_backend()
        weights = backend.default_operator_weights()
        expected = {"prefix_truncation", "tactic_swap", "tactic_reorder", "splice_prefixes"}
        assert set(weights.keys()) == expected


# ---------------------------------------------------------------------------
# format_reflection_prompt()
# ---------------------------------------------------------------------------


class TestLeanBackendFormatReflectionPrompt:
    def test_returns_non_empty_string(self) -> None:
        backend = _make_backend()
        population = [
            make_individual(
                genome="intro x\nsimp",
                fitness=Fitness(primary=0.5, auxiliary={}, constraints={}, feasible=True),
            ),
        ]
        result = backend.format_reflection_prompt(population, memory=None, generation=1)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_generation_number(self) -> None:
        backend = _make_backend()
        population = [
            make_individual(
                genome="intro x\nsimp",
                fitness=Fitness(primary=0.5, auxiliary={}, constraints={}, feasible=True),
            ),
        ]
        result = backend.format_reflection_prompt(population, memory=None, generation=42)
        assert "42" in result

    def test_includes_fitness_info(self) -> None:
        backend = _make_backend()
        population = [
            make_individual(
                genome="intro x\nsimp",
                fitness=Fitness(primary=0.8, auxiliary={}, constraints={}, feasible=True),
            ),
        ]
        result = backend.format_reflection_prompt(population, memory=None, generation=1)
        assert "0.8" in result

    def test_empty_population(self) -> None:
        backend = _make_backend()
        result = backend.format_reflection_prompt([], memory=None, generation=0)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# format_crossover_prompt() — new signature with two Individuals
# ---------------------------------------------------------------------------


class TestLeanBackendFormatCrossoverPrompt:
    def test_returns_non_empty_string(self) -> None:
        backend = _make_backend()
        parent_a = make_individual(genome="intro x\nsimp")
        parent_b = make_individual(genome="intro x\nring")
        result = backend.format_crossover_prompt(parent_a, parent_b, context=None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_both_genomes(self) -> None:
        backend = _make_backend()
        parent_a = make_individual(genome="intro x\nsimp")
        parent_b = make_individual(genome="intro x\nring")
        result = backend.format_crossover_prompt(parent_a, parent_b, context=None)
        assert "intro x\nsimp" in result
        assert "intro x\nring" in result

    def test_accepts_individual_not_string(self) -> None:
        """parent_b must be an Individual, not a raw genome string."""
        backend = _make_backend()
        parent_a = make_individual(genome="tactic_a")
        parent_b = make_individual(genome="tactic_b")
        # Should not raise — this confirms the signature accepts Individual
        result = backend.format_crossover_prompt(parent_a, parent_b, context=None)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# startup() import command ordering
# ---------------------------------------------------------------------------


class TestLeanBackendStartup:
    async def test_startup_sends_import_before_theorem(self) -> None:
        """startup() should send the import cmd first, then the theorem cmd."""
        mock_repl = MagicMock()
        mock_repl.start = AsyncMock()
        mock_repl.send_command = AsyncMock(return_value={"sorries": []})

        backend = LeanBackend(
            theorem_statement="theorem T : True",
            project_dir="/tmp/proj",
            imports="import LeanLevy",
        )
        # Inject mock REPL so we don't spawn a real process
        backend._repl = mock_repl  # type: ignore[assignment]

        with patch.object(backend_mod, "LeanREPLProcess", return_value=mock_repl):
            await backend.startup()

        # Verify send_command was called twice in the correct order
        assert mock_repl.send_command.await_count == 2
        calls = mock_repl.send_command.call_args_list
        assert calls[0] == call({"cmd": "import LeanLevy"})
        assert calls[1] == call({"cmd": "theorem T : True := by\n sorry"})

    async def test_startup_skips_import_when_empty(self) -> None:
        """startup() should not send an import cmd when imports is empty."""
        mock_repl = MagicMock()
        mock_repl.start = AsyncMock()
        mock_repl.send_command = AsyncMock(return_value={"sorries": []})

        backend = LeanBackend(
            theorem_statement="theorem T : True",
            project_dir="/tmp/proj",
            imports="",
        )

        with patch.object(backend_mod, "LeanREPLProcess", return_value=mock_repl):
            await backend.startup()

        # Only the theorem init command should be sent
        assert mock_repl.send_command.await_count == 1
        calls = mock_repl.send_command.call_args_list
        assert calls[0] == call({"cmd": "theorem T : True := by\n sorry"})
