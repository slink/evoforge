"""Tests for evoforge.core.mutation – mutation operators, context, stats, and ensemble."""

from __future__ import annotations

from collections import Counter

import pytest

from evoforge.core.mutation import (
    MutationContext,
    MutationEnsemble,
    MutationOperator,
    OperatorStats,
)
from evoforge.core.types import Credit, Individual

# ---------------------------------------------------------------------------
# Mock operators for testing
# ---------------------------------------------------------------------------


class MockCheapOp(MutationOperator):
    """A cheap (non-LLM) mutation operator for testing."""

    @property
    def name(self) -> str:
        return "cheap_op"

    @property
    def cost(self) -> str:
        return "cheap"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        return parent.genome + "_cheap"


class MockLLMOp(MutationOperator):
    """An LLM-backed mutation operator for testing."""

    @property
    def name(self) -> str:
        return "llm_op"

    @property
    def cost(self) -> str:
        return "llm"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        return parent.genome + "_llm"


class AnotherCheapOp(MutationOperator):
    """A second cheap operator for testing ensemble selection."""

    @property
    def name(self) -> str:
        return "another_cheap"

    @property
    def cost(self) -> str:
        return "cheap"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        return parent.genome + "_another"


class AnotherLLMOp(MutationOperator):
    """A second LLM operator for testing ensembles with no cheap operators."""

    @property
    def name(self) -> str:
        return "another_llm"

    @property
    def cost(self) -> str:
        return "llm"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        return parent.genome + "_another_llm"


# ---------------------------------------------------------------------------
# MutationContext construction
# ---------------------------------------------------------------------------


class TestMutationContext:
    def test_construction(self) -> None:
        ctx = MutationContext(
            generation=5,
            memory=None,
            guidance="try adding a helper lemma",
            temperature=0.8,
            backend=None,
            credits=[Credit(location=0, score=1.0, signal="good")],
        )
        assert ctx.generation == 5
        assert ctx.guidance == "try adding a helper lemma"
        assert ctx.temperature == 0.8
        assert len(ctx.credits) == 1

    def test_empty_credits(self) -> None:
        ctx = MutationContext(
            generation=0,
            memory=None,
            guidance="",
            temperature=0.5,
            backend=None,
            credits=[],
        )
        assert ctx.credits == []


# ---------------------------------------------------------------------------
# OperatorStats tracking
# ---------------------------------------------------------------------------


class TestOperatorStats:
    def test_defaults(self) -> None:
        stats = OperatorStats()
        assert stats.applications == 0
        assert stats.successes == 0
        assert stats.failures == 0
        assert stats.total_fitness_delta == 0.0

    def test_success_rate_no_applications(self) -> None:
        stats = OperatorStats()
        assert stats.success_rate == 0.0

    def test_success_rate_with_data(self) -> None:
        stats = OperatorStats(applications=10, successes=3, failures=7)
        assert stats.success_rate == pytest.approx(0.3)

    def test_success_rate_all_successful(self) -> None:
        stats = OperatorStats(applications=5, successes=5, failures=0)
        assert stats.success_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MutationOperator ABC
# ---------------------------------------------------------------------------


class TestMutationOperatorABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            MutationOperator()  # type: ignore[abstract]

    async def test_mock_cheap_apply(self) -> None:
        op = MockCheapOp()
        parent = Individual(genome="hello", ir=None, ir_hash="h1", generation=0)
        ctx = MutationContext(
            generation=0,
            memory=None,
            guidance="",
            temperature=0.5,
            backend=None,
            credits=[],
        )
        result = await op.apply(parent, ctx)
        assert result == "hello_cheap"
        assert op.name == "cheap_op"
        assert op.cost == "cheap"

    async def test_mock_llm_apply(self) -> None:
        op = MockLLMOp()
        parent = Individual(genome="hello", ir=None, ir_hash="h1", generation=0)
        ctx = MutationContext(
            generation=0,
            memory=None,
            guidance="",
            temperature=0.5,
            backend=None,
            credits=[],
        )
        result = await op.apply(parent, ctx)
        assert result == "hello_llm"
        assert op.cost == "llm"


# ---------------------------------------------------------------------------
# MutationEnsemble – weight scheduling
# ---------------------------------------------------------------------------


class TestEnsembleWeightScheduling:
    def test_fixed_weights_dont_change(self) -> None:
        cheap = MockCheapOp()
        llm = MockLLMOp()
        ensemble = MutationEnsemble(operators=[cheap, llm], schedule="fixed", weights=[0.3, 0.7])
        initial = ensemble.get_weights()
        assert initial["cheap_op"] == pytest.approx(0.3)
        assert initial["llm_op"] == pytest.approx(0.7)

        # Record some stats – fixed weights should not change
        for _ in range(20):
            ensemble.update_stats("cheap_op", success=True, fitness_delta=1.0)
            ensemble.update_stats("llm_op", success=False, fitness_delta=-0.5)

        after = ensemble.get_weights()
        assert after["cheap_op"] == pytest.approx(0.3)
        assert after["llm_op"] == pytest.approx(0.7)

    def test_uniform_weights_when_none(self) -> None:
        cheap = MockCheapOp()
        llm = MockLLMOp()
        ensemble = MutationEnsemble(operators=[cheap, llm], schedule="fixed")
        weights = ensemble.get_weights()
        assert weights["cheap_op"] == pytest.approx(0.5)
        assert weights["llm_op"] == pytest.approx(0.5)

    def test_adaptive_weights_shift_toward_successful_operator(self) -> None:
        cheap = MockCheapOp()
        llm = MockLLMOp()
        ensemble = MutationEnsemble(
            operators=[cheap, llm], schedule="adaptive", weights=[0.5, 0.5]
        )

        # cheap_op always succeeds, llm_op always fails
        for _ in range(50):
            ensemble.update_stats("cheap_op", success=True, fitness_delta=1.0)
            ensemble.update_stats("llm_op", success=False, fitness_delta=-0.5)

        weights = ensemble.get_weights()
        # After adaptation, cheap_op should have higher weight
        assert weights["cheap_op"] > weights["llm_op"]


# ---------------------------------------------------------------------------
# MutationEnsemble – cheapest_operator
# ---------------------------------------------------------------------------


class TestCheapestOperator:
    def test_returns_first_cheap_operator(self) -> None:
        cheap = MockCheapOp()
        llm = MockLLMOp()
        another = AnotherCheapOp()
        ensemble = MutationEnsemble(operators=[llm, cheap, another])
        result = ensemble.cheapest_operator()
        # Should return the first cheap operator found
        assert result.cost == "cheap"

    def test_no_cheap_operators_raises(self) -> None:
        llm1 = MockLLMOp()
        llm2 = AnotherLLMOp()
        ensemble = MutationEnsemble(operators=[llm1, llm2])
        with pytest.raises(ValueError, match="[Cc]heap"):
            ensemble.cheapest_operator()


# ---------------------------------------------------------------------------
# MutationEnsemble – stats recording
# ---------------------------------------------------------------------------


class TestStatsRecording:
    def test_application_count(self) -> None:
        cheap = MockCheapOp()
        llm = MockLLMOp()
        ensemble = MutationEnsemble(operators=[cheap, llm])

        ensemble.update_stats("cheap_op", success=True, fitness_delta=0.5)
        ensemble.update_stats("cheap_op", success=True, fitness_delta=0.3)
        ensemble.update_stats("cheap_op", success=False, fitness_delta=-0.1)

        stats = ensemble.stats["cheap_op"]
        assert stats.applications == 3
        assert stats.successes == 2
        assert stats.failures == 1
        assert stats.total_fitness_delta == pytest.approx(0.7)

    def test_success_rate_tracked(self) -> None:
        cheap = MockCheapOp()
        llm = MockLLMOp()
        ensemble = MutationEnsemble(operators=[cheap, llm])

        for _ in range(4):
            ensemble.update_stats("llm_op", success=True, fitness_delta=1.0)
        ensemble.update_stats("llm_op", success=False, fitness_delta=-1.0)

        assert ensemble.stats["llm_op"].success_rate == pytest.approx(0.8)

    def test_unknown_operator_ignored(self) -> None:
        cheap = MockCheapOp()
        ensemble = MutationEnsemble(operators=[cheap])
        # Should not raise for unknown operator
        ensemble.update_stats("nonexistent_op", success=True, fitness_delta=1.0)


# ---------------------------------------------------------------------------
# MutationEnsemble – select_operator respects weights
# ---------------------------------------------------------------------------


class TestSelectOperator:
    def test_respects_weights_statistical(self) -> None:
        """With heavily skewed weights, one operator should be selected much more."""
        cheap = MockCheapOp()
        llm = MockLLMOp()
        ensemble = MutationEnsemble(operators=[cheap, llm], schedule="fixed", weights=[0.95, 0.05])

        counts: Counter[str] = Counter()
        n_trials = 1000
        for _ in range(n_trials):
            op = ensemble.select_operator()
            counts[op.name] += 1

        # cheap_op should be selected ~950 times; allow wide margin
        assert counts["cheap_op"] > 800
        assert counts["llm_op"] < 200

    def test_single_operator(self) -> None:
        cheap = MockCheapOp()
        ensemble = MutationEnsemble(operators=[cheap])
        op = ensemble.select_operator()
        assert op.name == "cheap_op"
