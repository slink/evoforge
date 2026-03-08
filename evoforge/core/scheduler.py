# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Execution scheduler with concurrency control and budget tracking."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass
class CostTracker:
    """Tracks cumulative resource usage across the evolutionary run."""

    llm_calls: int = 0
    llm_tokens: int = 0
    eval_time_seconds: float = 0.0
    wall_time_seconds: float = 0.0
    estimated_cost_usd: float = 0.0

    def record_llm_call(self, tokens: int, cost_usd: float) -> None:
        """Record a single LLM call, incrementing calls, tokens, and cost."""
        self.llm_calls += 1
        self.llm_tokens += tokens
        self.estimated_cost_usd += cost_usd

    def record_eval(self, duration: float) -> None:
        """Record an evaluation duration."""
        self.eval_time_seconds += duration

    def summary(self) -> dict[str, float]:
        """Return a dict of all tracked metrics."""
        return {
            "llm_calls": float(self.llm_calls),
            "llm_tokens": float(self.llm_tokens),
            "eval_time_seconds": self.eval_time_seconds,
            "wall_time_seconds": self.wall_time_seconds,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


@dataclass
class SchedulerConfig:
    """Configuration for the execution scheduler."""

    max_concurrent_evals: int = 4
    max_concurrent_llm: int = 2
    max_llm_calls: int = 1000
    max_cost_usd: float = 50.0
    eval_timeout_seconds: float = 60.0
    llm_budget_per_gen: int = 15


class ExecutionScheduler:
    """Manages concurrency limits and budget for evaluations and LLM calls.

    Uses asyncio semaphores for backpressure so callers naturally block
    when too many operations are in flight.
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self._config = config
        self._eval_semaphore = asyncio.Semaphore(config.max_concurrent_evals)
        self._llm_semaphore = asyncio.Semaphore(config.max_concurrent_llm)
        self._tracker = CostTracker()
        self._gen_llm_calls: int = 0

    @asynccontextmanager
    async def acquire_eval(self) -> AsyncIterator[None]:
        """Acquire a slot for an evaluation, blocking if at capacity."""
        async with self._eval_semaphore:
            yield

    @asynccontextmanager
    async def acquire_llm(self) -> AsyncIterator[None]:
        """Acquire a slot for an LLM call, blocking if at capacity."""
        async with self._llm_semaphore:
            yield

    def should_stop(self) -> bool:
        """Return True if the budget has been exceeded."""
        return (
            self._tracker.llm_calls >= self._config.max_llm_calls
            or self._tracker.estimated_cost_usd >= self._config.max_cost_usd
        )

    @property
    def gen_llm_calls(self) -> int:
        """Return the number of LLM calls made in the current generation."""
        return self._gen_llm_calls

    def can_use_llm(self) -> bool:
        """Return True if the per-generation LLM budget has not been exhausted."""
        return self._gen_llm_calls < self._config.llm_budget_per_gen

    def record_gen_llm_call(self) -> None:
        """Record an LLM call against the per-generation budget."""
        self._gen_llm_calls += 1

    def reset_generation(self) -> None:
        """Reset per-generation counters for a new generation."""
        self._gen_llm_calls = 0

    @property
    def tracker(self) -> CostTracker:
        """Return the cost tracker."""
        return self._tracker
