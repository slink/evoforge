"""Tests for evoforge.backends.lean.credit and evoforge.backends.lean.validation."""

from __future__ import annotations

from dataclasses import dataclass, field

from evoforge.backends.lean.credit import assign_credit_lean
from evoforge.backends.lean.ir import TacticSequence, TacticStep
from evoforge.backends.lean.validation import validate_structure_lean

# ---------------------------------------------------------------------------
# Lightweight stubs for Goal / TacticStepResult / LeanEvalTrace
# These mirror the real evaluator types but avoid depending on agent A4a.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Goal:
    type_str: str = ""
    context: str = ""


@dataclass(frozen=True)
class StepResult:
    succeeded: bool = True
    goals_before: list[Goal] = field(default_factory=list)
    goals_after: list[Goal] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class FakeTrace:
    step_results: list[StepResult] = field(default_factory=list)


@dataclass
class FakeDiagnostics:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIAG = FakeDiagnostics()


def _goals(n: int) -> list[Goal]:
    """Return a list of *n* dummy goals."""
    return [Goal(type_str=f"g{i}") for i in range(n)]


def _make_ir(*tactic_names: str) -> TacticSequence:
    return TacticSequence(steps=[TacticStep(tactic=t, args="", raw=t) for t in tactic_names])


# ---------------------------------------------------------------------------
# Credit assignment tests
# ---------------------------------------------------------------------------


class TestAllSuccessCredits:
    """All-success traces should produce all-positive scores."""

    def test_all_positive_scores(self) -> None:
        ir = _make_ir("intro", "apply", "exact")
        trace = FakeTrace(
            step_results=[
                StepResult(succeeded=True, goals_before=_goals(2), goals_after=_goals(1)),
                StepResult(succeeded=True, goals_before=_goals(1), goals_after=_goals(0)),
                StepResult(succeeded=True, goals_before=_goals(1), goals_after=_goals(1)),
            ]
        )
        credits = assign_credit_lean(ir, _DIAG, trace)
        assert len(credits) == 3
        assert all(c.score > 0 for c in credits)
        assert all(c.confidence == 1.0 for c in credits)


class TestFailureStopsCrediting:
    """After the first failure, no more credits should be emitted."""

    def test_stops_after_failure(self) -> None:
        ir = _make_ir("intro", "bad_tactic", "exact")
        trace = FakeTrace(
            step_results=[
                StepResult(succeeded=True, goals_before=_goals(1), goals_after=_goals(0)),
                StepResult(
                    succeeded=False,
                    goals_before=_goals(1),
                    goals_after=_goals(1),
                    error_type="tactic_failure",
                    error_message="unknown identifier 'bad_tactic'",
                ),
                StepResult(succeeded=True, goals_before=_goals(1), goals_after=_goals(0)),
            ]
        )
        credits = assign_credit_lean(ir, _DIAG, trace)
        # Should have credit for step 0 (success) and step 1 (failure), but NOT step 2
        assert len(credits) == 2
        assert credits[0].score > 0
        assert credits[1].score < 0

    def test_failure_signal_contains_error(self) -> None:
        ir = _make_ir("bad")
        trace = FakeTrace(
            step_results=[
                StepResult(
                    succeeded=False,
                    goals_before=[],
                    goals_after=[],
                    error_type="type_error",
                    error_message="expected Nat, got String",
                ),
            ]
        )
        credits = assign_credit_lean(ir, _DIAG, trace)
        assert len(credits) == 1
        assert "type_error" in credits[0].signal
        assert "expected Nat" in credits[0].signal


class TestGoalReductionScoring:
    """Steps that close more goals should receive higher credit."""

    def test_higher_reduction_higher_score(self) -> None:
        ir = _make_ir("intro", "simp")
        trace = FakeTrace(
            step_results=[
                # Closes 0 goals  (1 -> 1)
                StepResult(succeeded=True, goals_before=_goals(1), goals_after=_goals(1)),
                # Closes 2 goals  (3 -> 1)
                StepResult(succeeded=True, goals_before=_goals(3), goals_after=_goals(1)),
            ]
        )
        credits = assign_credit_lean(ir, _DIAG, trace)
        assert credits[1].score > credits[0].score

    def test_reduction_signal_closed_goals(self) -> None:
        ir = _make_ir("simp")
        trace = FakeTrace(
            step_results=[
                StepResult(succeeded=True, goals_before=_goals(2), goals_after=_goals(0)),
            ]
        )
        credits = assign_credit_lean(ir, _DIAG, trace)
        assert "closed 2 goals" in credits[0].signal

    def test_no_reduction_signal_maintained(self) -> None:
        ir = _make_ir("intro")
        trace = FakeTrace(
            step_results=[
                StepResult(succeeded=True, goals_before=_goals(1), goals_after=_goals(1)),
            ]
        )
        credits = assign_credit_lean(ir, _DIAG, trace)
        assert credits[0].signal == "maintained progress"

    def test_credit_location_is_step_index(self) -> None:
        ir = _make_ir("intro", "apply")
        trace = FakeTrace(
            step_results=[
                StepResult(succeeded=True, goals_before=_goals(1), goals_after=_goals(1)),
                StepResult(succeeded=True, goals_before=_goals(1), goals_after=_goals(0)),
            ]
        )
        credits = assign_credit_lean(ir, _DIAG, trace)
        assert credits[0].location == 0
        assert credits[1].location == 1


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestSorryRejected:
    """validate_structure_lean should flag sorry tactics."""

    def test_sorry_violation(self) -> None:
        ir = TacticSequence(
            steps=[
                TacticStep(tactic="intro", args="x", raw="intro x"),
                TacticStep(tactic="sorry", args="", raw="sorry"),
            ]
        )
        violations = validate_structure_lean(ir)
        assert any("sorry" in v.lower() for v in violations)


class TestUnknownTacticRejected:
    """validate_structure_lean should flag unknown tactics."""

    def test_unknown_tactic_violation(self) -> None:
        ir = TacticSequence(
            steps=[
                TacticStep(tactic="magic_solve", args="", raw="magic_solve"),
            ]
        )
        violations = validate_structure_lean(ir)
        assert any("magic_solve" in v for v in violations)

    def test_known_tactics_pass(self) -> None:
        """All known tactics from the whitelist should pass validation."""
        known = ["intro", "apply", "exact", "simp", "ring", "omega", "linarith"]
        ir = TacticSequence(steps=[TacticStep(tactic=t, args="", raw=t) for t in known])
        violations = validate_structure_lean(ir)
        # No whitelist violations
        assert not any("unknown tactic" in v.lower() for v in violations)


class TestUnbalancedDelimiters:
    """validate_structure_lean should flag unbalanced delimiters."""

    def test_unbalanced_braces(self) -> None:
        ir = TacticSequence(
            steps=[
                TacticStep(tactic="intro", args="x {", raw="intro x {"),
                TacticStep(tactic="exact", args="h", raw="exact h"),
            ]
        )
        violations = validate_structure_lean(ir)
        assert any("balanced" in v.lower() or "delimiter" in v.lower() for v in violations)

    def test_balanced_delimiters_pass(self) -> None:
        ir = TacticSequence(
            steps=[
                TacticStep(tactic="simp", args="[a, b]", raw="simp [a, b]"),
                TacticStep(tactic="intro", args="x", raw="intro x"),
            ]
        )
        violations = validate_structure_lean(ir)
        assert not any("delimiter" in v.lower() for v in violations)


class TestMaxTacticCount:
    """validate_structure_lean should flag sequences exceeding 100 tactics."""

    def test_over_100_violation(self) -> None:
        steps = [TacticStep(tactic="intro", args="x", raw="intro x") for _ in range(101)]
        ir = TacticSequence(steps=steps)
        violations = validate_structure_lean(ir)
        assert any("100" in v or "max" in v.lower() for v in violations)

    def test_100_steps_ok(self) -> None:
        steps = [TacticStep(tactic="intro", args="x", raw="intro x") for _ in range(100)]
        ir = TacticSequence(steps=steps)
        violations = validate_structure_lean(ir)
        assert not any("100" in v or "max" in v.lower() for v in violations)


class TestValidSequencePasses:
    """A clean, valid sequence should produce zero violations."""

    def test_no_violations(self) -> None:
        ir = TacticSequence(
            steps=[
                TacticStep(tactic="intro", args="x", raw="intro x"),
                TacticStep(tactic="apply", args="h", raw="apply h"),
                TacticStep(tactic="exact", args="rfl", raw="exact rfl"),
            ]
        )
        violations = validate_structure_lean(ir)
        assert violations == []


class TestRepeatWithoutBound:
    """validate_structure_lean should flag unbounded repeat tactics."""

    def test_unbounded_repeat_violation(self) -> None:
        ir = TacticSequence(
            steps=[
                TacticStep(tactic="repeat", args="assumption", raw="repeat assumption"),
            ]
        )
        violations = validate_structure_lean(ir)
        assert any("repeat" in v.lower() for v in violations)

    def test_repeat_with_max_depth_ok(self) -> None:
        ir = TacticSequence(
            steps=[
                TacticStep(
                    tactic="repeat",
                    args="assumption maxDepth 10",
                    raw="repeat assumption maxDepth 10",
                ),
            ]
        )
        violations = validate_structure_lean(ir)
        assert not any("repeat" in v.lower() for v in violations)
