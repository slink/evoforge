"""Tests for per-generation LLM budget enforcement in ExecutionScheduler."""

from __future__ import annotations

from evoforge.core.scheduler import ExecutionScheduler, SchedulerConfig


class TestSchedulerConfigLLMBudget:
    """SchedulerConfig accepts llm_budget_per_gen with a sensible default."""

    def test_default_llm_budget_per_gen(self) -> None:
        cfg = SchedulerConfig()
        assert cfg.llm_budget_per_gen == 15

    def test_custom_llm_budget_per_gen(self) -> None:
        cfg = SchedulerConfig(llm_budget_per_gen=5)
        assert cfg.llm_budget_per_gen == 5


class TestGenLLMBudgetEnforcement:
    """ExecutionScheduler enforces per-generation LLM call limits."""

    def test_gen_llm_calls_starts_at_zero(self) -> None:
        sched = ExecutionScheduler(SchedulerConfig())
        assert sched.gen_llm_calls == 0

    def test_can_use_llm_true_when_under_budget(self) -> None:
        sched = ExecutionScheduler(SchedulerConfig(llm_budget_per_gen=3))
        assert sched.can_use_llm() is True

    def test_can_use_llm_false_when_at_budget(self) -> None:
        cfg = SchedulerConfig(llm_budget_per_gen=2)
        sched = ExecutionScheduler(cfg)
        sched.record_gen_llm_call()
        sched.record_gen_llm_call()
        assert sched.can_use_llm() is False

    def test_can_use_llm_false_when_over_budget(self) -> None:
        cfg = SchedulerConfig(llm_budget_per_gen=1)
        sched = ExecutionScheduler(cfg)
        sched.record_gen_llm_call()
        sched.record_gen_llm_call()
        assert sched.can_use_llm() is False

    def test_record_gen_llm_call_increments(self) -> None:
        sched = ExecutionScheduler(SchedulerConfig())
        assert sched.gen_llm_calls == 0
        sched.record_gen_llm_call()
        assert sched.gen_llm_calls == 1
        sched.record_gen_llm_call()
        assert sched.gen_llm_calls == 2

    def test_reset_generation_clears_counter(self) -> None:
        cfg = SchedulerConfig(llm_budget_per_gen=2)
        sched = ExecutionScheduler(cfg)
        sched.record_gen_llm_call()
        sched.record_gen_llm_call()
        assert sched.can_use_llm() is False
        sched.reset_generation()
        assert sched.gen_llm_calls == 0
        assert sched.can_use_llm() is True
