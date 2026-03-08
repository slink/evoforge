# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for evaluator caching, batching, timeout, and determinism."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evoforge.core.evaluator import (
    AsyncEvaluator,
    DeterministicEvaluator,
    EvaluationCache,
)
from evoforge.core.types import Fitness, Individual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fitness(primary: float = 1.0) -> Fitness:
    return Fitness(
        primary=primary,
        auxiliary={"speed": 2.0},
        constraints={"valid": True},
        feasible=True,
    )


def _make_individual(
    *,
    genome: str = "x + 1",
    ir: Any = "parsed_ir",
    ir_hash: str = "hash_abc",
    generation: int = 0,
    fitness: Fitness | None = None,
) -> Individual:
    return Individual(
        genome=genome,
        ir=ir,
        ir_hash=ir_hash,
        generation=generation,
        fitness=fitness,
    )


def _make_backend(
    fitness: Fitness | None = None,
    diagnostics: Any = None,
    trace: Any = None,
    sleep_seconds: float = 0.0,
) -> MagicMock:
    """Create a mock backend whose evaluate is an async callable."""
    fit = fitness or _make_fitness()
    diag = diagnostics or {"info": "ok"}
    tr = trace or {"steps": []}

    async def _evaluate(ir: Any, seed: int | None = None) -> tuple[Fitness, Any, Any]:
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)
        return (fit, diag, tr)

    backend = MagicMock()
    backend.evaluate = MagicMock(side_effect=_evaluate)
    return backend


def _make_archive() -> MagicMock:
    """Create a mock archive with async lookup_fitness and store_fitness."""
    archive = MagicMock()
    archive.lookup_fitness = AsyncMock(return_value=None)
    archive.store_fitness = AsyncMock(return_value=None)
    return archive


# ---------------------------------------------------------------------------
# EvaluationCache tests
# ---------------------------------------------------------------------------


class TestEvaluationCache:
    async def test_parse_cached_stores_result(self) -> None:
        cache = EvaluationCache()
        result = cache.parse_cached("genome_a", lambda g: f"parsed({g})")
        assert result == "parsed(genome_a)"
        # Second call should return cached value without calling parse_fn again
        call_count = 0

        def counting_parse(g: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"parsed({g})"

        result2 = cache.parse_cached("genome_a", counting_parse)
        assert result2 == "parsed(genome_a)"
        assert call_count == 0  # parse_fn was NOT called again

    async def test_parse_cached_caches_none(self) -> None:
        cache = EvaluationCache()
        result = cache.parse_cached("bad_genome", lambda g: None)
        assert result is None
        # Even None should be cached (using sentinel)
        call_count = 0

        def counting_parse(g: str) -> None:
            nonlocal call_count
            call_count += 1
            return None

        result2 = cache.parse_cached("bad_genome", counting_parse)
        assert result2 is None
        assert call_count == 0

    async def test_get_delegates_to_archive(self) -> None:
        archive = _make_archive()
        cached_fitness = _make_fitness(primary=99.0)
        archive.lookup_fitness = AsyncMock(return_value=cached_fitness)

        cache = EvaluationCache(archive=archive)
        result = await cache.get("ir1", "v1", "cfg1")

        assert result is not None
        assert result.primary == pytest.approx(99.0)
        archive.lookup_fitness.assert_awaited_once_with("ir1", "v1", "cfg1")

    async def test_put_delegates_to_archive(self) -> None:
        archive = _make_archive()
        cache = EvaluationCache(archive=archive)
        fitness = _make_fitness()

        await cache.put("ir1", "v1", "cfg1", fitness, '{"info": "ok"}')

        archive.store_fitness.assert_awaited_once_with(
            ir_hash="ir1",
            backend_version="v1",
            config_hash="cfg1",
            fitness=fitness,
            diagnostics_json='{"info": "ok"}',
        )


# ---------------------------------------------------------------------------
# AsyncEvaluator tests
# ---------------------------------------------------------------------------


class TestAsyncEvaluator:
    async def test_cache_hit_skips_backend(self) -> None:
        """When the cache has a fitness, the backend should NOT be called."""
        cached_fitness = _make_fitness(primary=42.0)
        archive = _make_archive()
        archive.lookup_fitness = AsyncMock(return_value=cached_fitness)

        backend = _make_backend()
        evaluator = AsyncEvaluator(
            backend=backend,
            archive=archive,
            backend_version="v1.0",
            config_hash="cfg_abc",
        )

        ind = _make_individual(ir_hash="cached_hash")
        result = await evaluator.evaluate(ind)

        assert result.fitness is not None
        assert result.fitness.primary == pytest.approx(42.0)
        # Backend.evaluate should NOT have been called
        backend.evaluate.assert_not_called()

    async def test_cache_miss_calls_backend_and_stores(self) -> None:
        """On cache miss, backend.evaluate is called and result is stored."""
        archive = _make_archive()
        archive.lookup_fitness = AsyncMock(return_value=None)

        backend = _make_backend(fitness=_make_fitness(primary=7.0))
        evaluator = AsyncEvaluator(
            backend=backend,
            archive=archive,
            backend_version="v2.0",
            config_hash="cfg_xyz",
        )

        ind = _make_individual(ir_hash="new_hash")
        result = await evaluator.evaluate(ind)

        assert result.fitness is not None
        assert result.fitness.primary == pytest.approx(7.0)
        backend.evaluate.assert_called_once()
        archive.store_fitness.assert_awaited_once()

    async def test_cache_key_includes_version_and_config(self) -> None:
        """Different backend_version or config_hash should produce cache miss."""
        cached_fitness = _make_fitness(primary=42.0)
        archive = _make_archive()

        # Return cached fitness only for v1.0 + cfg_a
        async def selective_lookup(
            ir_hash: str, backend_version: str, config_hash: str
        ) -> Fitness | None:
            if backend_version == "v1.0" and config_hash == "cfg_a":
                return cached_fitness
            return None

        archive.lookup_fitness = AsyncMock(side_effect=selective_lookup)

        backend = _make_backend(fitness=_make_fitness(primary=5.0))

        # Evaluator with v2.0 should miss
        evaluator_v2 = AsyncEvaluator(
            backend=backend,
            archive=archive,
            backend_version="v2.0",
            config_hash="cfg_a",
        )
        ind = _make_individual(ir_hash="some_hash")
        result = await evaluator_v2.evaluate(ind)
        assert result.fitness is not None
        assert result.fitness.primary == pytest.approx(5.0)
        backend.evaluate.assert_called_once()

        # Reset mock
        backend.evaluate.reset_mock()

        # Evaluator with v1.0 + cfg_a should hit
        evaluator_v1 = AsyncEvaluator(
            backend=backend,
            archive=archive,
            backend_version="v1.0",
            config_hash="cfg_a",
        )
        result2 = await evaluator_v1.evaluate(ind)
        assert result2.fitness is not None
        assert result2.fitness.primary == pytest.approx(42.0)
        backend.evaluate.assert_not_called()

    async def test_batch_eval_assigns_fitness_to_all(self) -> None:
        """evaluate_batch should assign fitness to every individual."""
        archive = _make_archive()
        backend = _make_backend(fitness=_make_fitness(primary=3.14))

        evaluator = AsyncEvaluator(
            backend=backend,
            archive=archive,
            backend_version="v1.0",
            config_hash="cfg_abc",
            max_concurrent=2,
        )

        individuals = [
            _make_individual(ir_hash=f"hash_{i}", genome=f"genome_{i}") for i in range(5)
        ]

        results = await evaluator.evaluate_batch(individuals)

        assert len(results) == 5
        for ind in results:
            assert ind.fitness is not None
            assert ind.fitness.primary == pytest.approx(3.14)

    async def test_timeout_assigns_fallback_fitness(self) -> None:
        """When backend.evaluate times out, a fallback infeasible fitness is assigned."""
        archive = _make_archive()
        # Backend sleeps for 5 seconds
        backend = _make_backend(sleep_seconds=5.0)

        evaluator = AsyncEvaluator(
            backend=backend,
            archive=archive,
            backend_version="v1.0",
            config_hash="cfg_abc",
            timeout_seconds=0.1,  # 100ms timeout
        )

        ind = _make_individual()
        result = await evaluator.evaluate(ind)

        assert result.fitness is not None
        assert result.fitness.primary == pytest.approx(0.0)
        assert result.fitness.feasible is False
        assert result.fitness.auxiliary == {}
        assert result.fitness.constraints == {}

    async def test_batch_timeout_partial(self) -> None:
        """In a batch, timed-out individuals get fallback; others succeed."""
        archive = _make_archive()

        call_count = 0

        async def _alternating_evaluate(
            ir: Any, seed: int | None = None
        ) -> tuple[Fitness, Any, Any]:
            nonlocal call_count
            current = call_count
            call_count += 1
            if current % 2 == 0:
                # Even calls timeout
                await asyncio.sleep(5.0)
            return (_make_fitness(primary=10.0), {}, {})

        backend = MagicMock()
        backend.evaluate = MagicMock(side_effect=_alternating_evaluate)

        evaluator = AsyncEvaluator(
            backend=backend,
            archive=archive,
            backend_version="v1.0",
            config_hash="cfg_abc",
            timeout_seconds=0.2,
            max_concurrent=4,
        )

        individuals = [
            _make_individual(ir_hash=f"hash_{i}", genome=f"genome_{i}") for i in range(4)
        ]

        results = await evaluator.evaluate_batch(individuals)
        assert len(results) == 4

        for ind in results:
            assert ind.fitness is not None

    async def test_serializes_dataclass_diagnostics(self) -> None:
        """Backend returning a frozen dataclass as diagnostics should serialize via asdict()."""

        @dataclass(frozen=True)
        class FakeDiagnostics:
            success: bool
            errors: list[str]
            detail: str | None

        diag = FakeDiagnostics(success=False, errors=["type error"], detail=None)
        backend = _make_backend(
            fitness=_make_fitness(primary=0.5),
            diagnostics=diag,
        )
        archive = _make_archive()

        evaluator = AsyncEvaluator(
            backend=backend,
            archive=archive,
            backend_version="v1.0",
            config_hash="cfg_abc",
        )

        ind = _make_individual(ir_hash="diag_hash")
        result = await evaluator.evaluate(ind)

        assert result.fitness is not None
        assert result.fitness.primary == pytest.approx(0.5)
        # Verify store_fitness received valid JSON string
        archive.store_fitness.assert_awaited_once()
        call_kwargs = archive.store_fitness.call_args.kwargs
        import json

        parsed = json.loads(call_kwargs["diagnostics_json"])
        assert parsed["success"] is False
        assert parsed["errors"] == ["type error"]
        assert parsed["detail"] is None


# ---------------------------------------------------------------------------
# DeterministicEvaluator tests
# ---------------------------------------------------------------------------


class TestDeterministicEvaluator:
    async def test_passes_fixed_seed(self) -> None:
        """DeterministicEvaluator should call backend.evaluate with the fixed seed."""
        backend = _make_backend()
        det_eval = DeterministicEvaluator(backend=backend, eval_seed=12345)

        fitness, diagnostics, trace = await det_eval.evaluate("some_ir")

        assert fitness.primary == pytest.approx(1.0)
        backend.evaluate.assert_called_once_with("some_ir", seed=12345)

    async def test_same_seed_produces_same_call(self) -> None:
        """Two calls with the same evaluator pass the same seed."""
        backend = _make_backend()
        det_eval = DeterministicEvaluator(backend=backend, eval_seed=999)

        await det_eval.evaluate("ir_a")
        await det_eval.evaluate("ir_b")

        calls = backend.evaluate.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["seed"] == 999
        assert calls[1].kwargs["seed"] == 999
