"""Tests for evoforge.backends.lean.evaluator — stepwise Lean evaluation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evoforge.backends.lean.evaluator import (
    Goal,
    LeanDiagnostics,
    LeanEvalTrace,
    LeanStepwiseEvaluator,
    TacticStepResult,
    _compute_fitness,
)
from evoforge.backends.lean.ir import TacticSequence, TacticStep
from evoforge.core.types import Credit, Diagnostics, EvaluationTrace

# ---------------------------------------------------------------------------
# Mock REPL: returns pre-recorded responses
# ---------------------------------------------------------------------------


class MockREPLProcess:
    """Simulates a LeanREPLProcess by returning pre-recorded responses."""

    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self._alive = True

    async def start(self) -> None:
        self._alive = True

    async def send_command(self, cmd: dict[str, object]) -> dict[str, object]:
        if self._call_index >= len(self._responses):
            return {"message": "no more responses", "severity": "error"}
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp

    async def send_tactic(self, tactic: str, state: int = 0) -> dict[str, object]:
        return await self.send_command({"tactic": tactic, "proofState": state})

    async def restart(self) -> None:
        self._call_index = 0
        self._alive = True

    async def close(self) -> None:
        self._alive = False

    def is_healthy(self) -> bool:
        return self._alive


# ---------------------------------------------------------------------------
# Dataclass construction tests
# ---------------------------------------------------------------------------


class TestGoal:
    def test_frozen(self) -> None:
        g = Goal(type_str="Nat", context="x : Nat")
        with pytest.raises(AttributeError):
            g.type_str = "Bool"  # type: ignore[misc]

    def test_fields(self) -> None:
        g = Goal(type_str="Nat", context="x : Nat")
        assert g.type_str == "Nat"
        assert g.context == "x : Nat"


class TestTacticStepResult:
    def test_frozen(self) -> None:
        r = TacticStepResult(
            succeeded=True,
            goals_before=[Goal(type_str="Nat", context="")],
            goals_after=[],
        )
        with pytest.raises(AttributeError):
            r.succeeded = False  # type: ignore[misc]

    def test_successful_step(self) -> None:
        r = TacticStepResult(
            succeeded=True,
            goals_before=[Goal(type_str="Nat", context="")],
            goals_after=[],
        )
        assert r.succeeded is True
        assert r.error_type is None
        assert r.error_message is None

    def test_failed_step(self) -> None:
        r = TacticStepResult(
            succeeded=False,
            goals_before=[Goal(type_str="Nat", context="")],
            goals_after=[Goal(type_str="Nat", context="")],
            error_type="tactic_failed",
            error_message="unknown identifier 'foo'",
        )
        assert r.succeeded is False
        assert r.error_type == "tactic_failed"


class TestLeanEvalTrace:
    def test_extends_evaluation_trace(self) -> None:
        trace = LeanEvalTrace(step_results=[])
        assert isinstance(trace, EvaluationTrace)


# ---------------------------------------------------------------------------
# LeanDiagnostics tests
# ---------------------------------------------------------------------------


class TestLeanDiagnostics:
    def test_satisfies_diagnostics_protocol(self) -> None:
        diag = LeanDiagnostics(
            success=True,
            goals_remaining=0,
            goal_types=[],
            goal_contexts=[],
            error_type=None,
            error_message=None,
            stuck_tactic_index=None,
            stuck_tactic=None,
            steps_succeeded=3,
            metavar_count=0,
        )
        assert isinstance(diag, Diagnostics)

    def test_summary_success(self) -> None:
        diag = LeanDiagnostics(
            success=True,
            goals_remaining=0,
            goal_types=[],
            goal_contexts=[],
            error_type=None,
            error_message=None,
            stuck_tactic_index=None,
            stuck_tactic=None,
            steps_succeeded=3,
            metavar_count=0,
        )
        s = diag.summary(max_tokens=500)
        assert "success" in s.lower() or "complete" in s.lower()

    def test_summary_failure(self) -> None:
        diag = LeanDiagnostics(
            success=False,
            goals_remaining=1,
            goal_types=["Nat"],
            goal_contexts=["x : Nat"],
            error_type="tactic_failed",
            error_message="unknown identifier 'foo'",
            stuck_tactic_index=2,
            stuck_tactic="exact foo",
            steps_succeeded=2,
            metavar_count=0,
        )
        s = diag.summary(max_tokens=500)
        assert "error" in s.lower() or "fail" in s.lower()
        assert "foo" in s

    def test_credit_summary(self) -> None:
        diag = LeanDiagnostics(
            success=False,
            goals_remaining=1,
            goal_types=["Nat"],
            goal_contexts=[],
            error_type=None,
            error_message=None,
            stuck_tactic_index=None,
            stuck_tactic=None,
            steps_succeeded=2,
            metavar_count=0,
        )
        credits = [
            Credit(location=0, score=1.0, signal="step_success"),
            Credit(location=1, score=1.0, signal="step_success"),
            Credit(location=2, score=0.0, signal="step_failure"),
        ]
        s = diag.credit_summary(credits, max_tokens=300)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_respects_max_tokens_roughly(self) -> None:
        """summary output length should be bounded (not strictly, but reasonable)."""
        diag = LeanDiagnostics(
            success=False,
            goals_remaining=5,
            goal_types=["Nat"] * 5,
            goal_contexts=["x : Nat"] * 5,
            error_type="tactic_failed",
            error_message="a" * 2000,
            stuck_tactic_index=0,
            stuck_tactic="exact foo",
            steps_succeeded=0,
            metavar_count=3,
        )
        s = diag.summary(max_tokens=50)
        # Rough bound: each token ~4 chars, 50 tokens ~ 200 chars, allow generous margin
        assert len(s) < 1000


# ---------------------------------------------------------------------------
# Fitness computation tests
# ---------------------------------------------------------------------------


class TestFitnessComputation:
    """Test that LeanStepwiseEvaluator computes fitness correctly."""

    async def test_all_steps_succeed_proof_complete(self) -> None:
        """All tactics succeed and no goals remain -> primary=1.0, proof_complete=1.0."""
        responses: list[dict[str, object]] = [
            # Step 1: intro x -> success, 1 goal remaining
            {"proofState": 1, "goals": ["x : Nat\n|- Nat"]},
            # Step 2: exact x -> success, no goals (proof complete)
            {"proofState": 2, "goals": []},
        ]
        repl = MockREPLProcess(responses)
        evaluator = LeanStepwiseEvaluator(repl=repl)  # type: ignore[arg-type]

        seq = TacticSequence(
            steps=[
                TacticStep(tactic="intro", args="x", raw="intro x"),
                TacticStep(tactic="exact", args="x", raw="exact x"),
            ]
        )

        fitness, diag, trace = await evaluator.evaluate(seq)

        assert fitness.primary == 1.0
        assert fitness.auxiliary["proof_complete"] == 1.0
        assert fitness.auxiliary["steps_succeeded"] == 2
        assert fitness.auxiliary["goals_remaining"] == 0
        assert diag.success is True
        assert diag.goals_remaining == 0
        assert len(trace.step_results) == 2
        assert all(r.succeeded for r in trace.step_results)

    async def test_partial_success_fraction(self) -> None:
        """Some tactics succeed, then one fails -> weighted fitness."""
        responses: list[dict[str, object]] = [
            # Step 1: intro x -> success, 1 goal remaining
            {"proofState": 1, "goals": ["x : Nat\n|- Nat"]},
            # Step 2: exact foo -> failure
            {"message": "unknown identifier 'foo'", "severity": "error"},
        ]
        repl = MockREPLProcess(responses)
        evaluator = LeanStepwiseEvaluator(repl=repl)  # type: ignore[arg-type]

        seq = TacticSequence(
            steps=[
                TacticStep(tactic="intro", args="x", raw="intro x"),
                TacticStep(tactic="exact", args="foo", raw="exact foo"),
            ]
        )

        fitness, diag, trace = await evaluator.evaluate(seq)

        # 1/2 steps, 1 goal remaining out of 1 initial -> goal_reduction=0
        # primary = 0.4 * (1/2) + 0.6 * 0.0 = 0.2
        assert fitness.primary == pytest.approx(0.2)
        assert fitness.auxiliary["proof_complete"] == 0.0
        assert fitness.auxiliary["steps_succeeded"] == 1
        assert diag.success is False
        assert diag.error_message is not None
        assert "foo" in diag.error_message
        assert trace.step_results[0].succeeded is True
        assert trace.step_results[1].succeeded is False

    async def test_single_step_failure(self) -> None:
        """First tactic fails -> primary=0.0."""
        responses: list[dict[str, object]] = [
            {"message": "unknown tactic 'bogus'", "severity": "error"},
        ]
        repl = MockREPLProcess(responses)
        evaluator = LeanStepwiseEvaluator(repl=repl)  # type: ignore[arg-type]

        seq = TacticSequence(
            steps=[
                TacticStep(tactic="bogus", args="", raw="bogus"),
            ]
        )

        fitness, diag, trace = await evaluator.evaluate(seq)

        assert fitness.primary == pytest.approx(0.0)
        assert fitness.auxiliary["proof_complete"] == 0.0
        assert diag.success is False
        assert diag.stuck_tactic_index == 0
        assert diag.stuck_tactic == "bogus"

    async def test_empty_sequence(self) -> None:
        """Empty tactic sequence -> primary=0.0, not feasible."""
        repl = MockREPLProcess([])
        evaluator = LeanStepwiseEvaluator(repl=repl)  # type: ignore[arg-type]

        seq = TacticSequence(steps=[])

        fitness, diag, trace = await evaluator.evaluate(seq)

        assert fitness.primary == 0.0
        assert diag.success is False
        assert len(trace.step_results) == 0


# ---------------------------------------------------------------------------
# Diagnostics from evaluation
# ---------------------------------------------------------------------------


class TestDiagnosticsFromEvaluation:
    async def test_failed_step_diagnostics(self) -> None:
        """A failing step should populate error_type and error_message in diagnostics."""
        responses: list[dict[str, object]] = [
            {"proofState": 1, "goals": ["x : Nat\n|- Nat"]},
            {"message": "type mismatch", "severity": "error"},
        ]
        repl = MockREPLProcess(responses)
        evaluator = LeanStepwiseEvaluator(repl=repl)  # type: ignore[arg-type]

        seq = TacticSequence(
            steps=[
                TacticStep(tactic="intro", args="x", raw="intro x"),
                TacticStep(tactic="apply", args="h", raw="apply h"),
            ]
        )

        _, diag, _ = await evaluator.evaluate(seq)

        assert diag.error_type is not None
        assert diag.error_message == "type mismatch"
        assert diag.stuck_tactic_index == 1
        assert diag.stuck_tactic == "apply h"


# ---------------------------------------------------------------------------
# Prefix cache tests
# ---------------------------------------------------------------------------


class TestPrefixCache:
    async def test_prefix_cache_reuse(self) -> None:
        """Evaluating [A,B,C] then [A,B,D] should reuse cached state for [A,B]."""
        # First evaluation: A -> B -> C (all succeed)
        responses_first: list[dict[str, object]] = [
            {"proofState": 1, "goals": ["|- Nat"]},  # A
            {"proofState": 2, "goals": ["|- Bool"]},  # B
            {"proofState": 3, "goals": []},  # C (complete)
        ]
        repl1 = MockREPLProcess(responses_first)
        cache: dict[str, int] = {}
        evaluator1 = LeanStepwiseEvaluator(repl=repl1, prefix_cache=cache)  # type: ignore[arg-type]

        seq1 = TacticSequence(
            steps=[
                TacticStep(tactic="intro", args="x", raw="intro x"),
                TacticStep(tactic="apply", args="f", raw="apply f"),
                TacticStep(tactic="exact", args="rfl", raw="exact rfl"),
            ]
        )

        await evaluator1.evaluate(seq1)

        # The cache should now have entries for prefixes
        assert len(cache) > 0

        # Second evaluation: A, B, D — should reuse cache for [A,B]
        # Only need response for D (starting from state 2)
        responses_second: list[dict[str, object]] = [
            {"proofState": 4, "goals": []},  # D (complete)
        ]
        repl2 = MockREPLProcess(responses_second)
        evaluator2 = LeanStepwiseEvaluator(repl=repl2, prefix_cache=cache)  # type: ignore[arg-type]

        seq2 = TacticSequence(
            steps=[
                TacticStep(tactic="intro", args="x", raw="intro x"),
                TacticStep(tactic="apply", args="f", raw="apply f"),
                TacticStep(tactic="simp", args="", raw="simp"),
            ]
        )

        fitness2, diag2, trace2 = await evaluator2.evaluate(seq2)

        # D succeeded, so all 3 steps "succeeded" (2 from cache + 1 new)
        assert fitness2.primary == 1.0
        assert diag2.success is True
        # Only 1 REPL call was made (for step D), confirming cache reuse
        assert repl2._call_index == 1

    async def test_no_cache_on_first_step_failure(self) -> None:
        """If the first step fails, nothing should be cached."""
        responses: list[dict[str, object]] = [
            {"message": "error", "severity": "error"},
        ]
        repl = MockREPLProcess(responses)
        cache: dict[str, int] = {}
        evaluator = LeanStepwiseEvaluator(repl=repl, prefix_cache=cache)  # type: ignore[arg-type]

        seq = TacticSequence(
            steps=[
                TacticStep(tactic="bogus", args="", raw="bogus"),
            ]
        )

        await evaluator.evaluate(seq)

        # Nothing should be cached since nothing succeeded
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# Initial proof state tests
# ---------------------------------------------------------------------------


class TestInitialProofState:
    """Test that the evaluator uses initial_proof_state correctly."""

    async def test_initial_proof_state_used_for_first_tactic(self) -> None:
        """Evaluator should start from initial_proof_state when no cache hit."""
        responses: list[dict[str, object]] = [
            {"proofState": 6, "goals": []},
        ]
        repl = MockREPLProcess(responses)
        evaluator = LeanStepwiseEvaluator(
            repl=repl,  # type: ignore[arg-type]
            initial_proof_state=5,
        )

        seq = TacticSequence(steps=[TacticStep(tactic="trivial", args="", raw="trivial")])
        fitness, diag, trace = await evaluator.evaluate(seq)

        assert fitness.primary == 1.0
        assert diag.success is True
        # Only 1 REPL call (the tactic), no theorem setup call
        assert repl._call_index == 1

    async def test_cache_hit_overrides_initial_proof_state(self) -> None:
        """When prefix cache has a hit, its state is used instead of initial_proof_state."""
        responses: list[dict[str, object]] = [
            {"proofState": 10, "goals": []},
        ]
        repl = MockREPLProcess(responses)
        cache: dict[str, int] = {"intro x": 7}
        evaluator = LeanStepwiseEvaluator(
            repl=repl,  # type: ignore[arg-type]
            initial_proof_state=5,
            prefix_cache=cache,
        )

        seq = TacticSequence(
            steps=[
                TacticStep(tactic="intro", args="x", raw="intro x"),
                TacticStep(tactic="trivial", args="", raw="trivial"),
            ]
        )

        fitness, diag, trace = await evaluator.evaluate(seq)

        assert fitness.primary == 1.0
        # Only 1 call: the new tactic (prefix was cached)
        assert repl._call_index == 1

    async def test_default_initial_proof_state_is_zero(self) -> None:
        """Default initial_proof_state=0 preserves backward compatibility."""
        responses: list[dict[str, object]] = [
            {"proofState": 1, "goals": []},
        ]
        repl = MockREPLProcess(responses)
        evaluator = LeanStepwiseEvaluator(repl=repl)  # type: ignore[arg-type]

        seq = TacticSequence(steps=[TacticStep(tactic="trivial", args="", raw="trivial")])
        fitness, diag, trace = await evaluator.evaluate(seq)

        assert fitness.primary == 1.0
        assert repl._call_index == 1


# ---------------------------------------------------------------------------
# Standalone _compute_fitness tests
# ---------------------------------------------------------------------------


class TestComputeFitness:
    def test_goal_reduction_improves_fitness(self) -> None:
        """A proof that reduces goals should score higher than one that doesn't."""
        fitness_good = _compute_fitness(
            steps_succeeded=3,
            total_steps=5,
            initial_goals=3,
            goals_remaining=1,
            proof_complete=False,
        )
        fitness_bad = _compute_fitness(
            steps_succeeded=3,
            total_steps=5,
            initial_goals=3,
            goals_remaining=3,
            proof_complete=False,
        )
        assert fitness_good.primary > fitness_bad.primary

    def test_complete_proof_gets_one(self) -> None:
        """Complete proof should always get 1.0."""
        f = _compute_fitness(
            steps_succeeded=3,
            total_steps=5,
            initial_goals=1,
            goals_remaining=0,
            proof_complete=True,
        )
        assert f.primary == 1.0
        assert f.feasible is True

    def test_zero_steps_gets_zero(self) -> None:
        """No steps should give 0.0."""
        f = _compute_fitness(
            steps_succeeded=0,
            total_steps=0,
            initial_goals=1,
            goals_remaining=1,
            proof_complete=False,
        )
        assert f.primary == 0.0

    def test_formula_values(self) -> None:
        """Verify the 0.4/0.6 weighting formula."""
        f = _compute_fitness(
            steps_succeeded=2,
            total_steps=4,
            initial_goals=2,
            goals_remaining=1,
            proof_complete=False,
        )
        # step_ratio = 2/4 = 0.5, goal_reduction = (2-1)/2 = 0.5
        # primary = 0.4 * 0.5 + 0.6 * 0.5 = 0.5
        assert f.primary == pytest.approx(0.5)
        assert f.auxiliary["goal_reduction"] == 1.0  # 2 - 1 = 1 goal reduced
        assert f.auxiliary["goals_remaining"] == 1.0

    def test_auxiliary_fields(self) -> None:
        """Auxiliary dict should contain expected keys."""
        f = _compute_fitness(
            steps_succeeded=1,
            total_steps=3,
            initial_goals=1,
            goals_remaining=0,
            proof_complete=False,
        )
        assert "steps_succeeded" in f.auxiliary
        assert "goals_remaining" in f.auxiliary
        assert "goal_reduction" in f.auxiliary
        assert "proof_complete" in f.auxiliary


# ---------------------------------------------------------------------------
# Integration test (requires real Lean REPL)
# ---------------------------------------------------------------------------

_EVOFORGE_ROOT = Path(__file__).resolve().parents[2]
LEAN_PROJECT_DIR = Path(
    os.environ.get("LEAN_PROJECT_DIR", str(_EVOFORGE_ROOT.parent / "LeanLevy"))
)
REPL_BIN = LEAN_PROJECT_DIR / ".lake" / "packages" / "repl" / ".lake" / "build" / "bin" / "repl"


@pytest.mark.lean
class TestLeanIntegration:
    """These tests require a real Lean installation. Skipped in CI."""

    @pytest.mark.timeout(120)
    async def test_simple_proof(self) -> None:
        if not REPL_BIN.exists():
            pytest.skip(f"REPL binary not found at {REPL_BIN}")

        from evoforge.backends.lean.evaluator import LeanREPLProcess

        repl = LeanREPLProcess(
            project_dir=str(LEAN_PROJECT_DIR),
            repl_path=str(REPL_BIN),
        )
        try:
            await repl.start()

            # Establish proof context and capture the initial proof state
            resp = await repl.send_command({"cmd": "theorem test : 1 + 1 = 2 := by\n sorry"})
            assert "sorries" in resp, f"Expected sorries in response: {resp}"
            initial_state = resp["sorries"][0]["proofState"]  # type: ignore[index]

            evaluator = LeanStepwiseEvaluator(
                repl,
                initial_proof_state=initial_state,
            )
            seq = TacticSequence(steps=[TacticStep(tactic="decide", args="", raw="decide")])
            fitness, diag, _trace = await evaluator.evaluate(seq)

            assert fitness.primary == 1.0
            assert diag.success is True
            assert fitness.auxiliary["proof_complete"] == 1.0
        finally:
            await repl.close()
