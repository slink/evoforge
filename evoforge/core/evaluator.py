"""Evaluator layer: caching, async batching, timeout, and determinism."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from evoforge.core.types import Fitness, Individual

# Sentinel for distinguishing "not cached" from "cached but returned None"
_SENTINEL = object()

_FALLBACK_FITNESS = Fitness(
    primary=0.0,
    auxiliary={},
    constraints={},
    feasible=False,
)


class EvaluationCache:
    """3-level evaluation cache.

    Level 1: in-memory parse cache (genome -> parsed IR or None).
    Level 2: prefix cache via Archive (async).
    Level 3: full eval cache via Archive (async) -- keyed by
             (ir_hash, backend_version, config_hash).
    """

    def __init__(self, archive: Any | None = None) -> None:
        self._archive = archive
        # L1: in-memory parse cache
        self._parse_cache: dict[str, Any] = {}

    # -- L1: synchronous parse cache ----------------------------------------

    def parse_cached(self, genome: str, parse_fn: Callable[[str], Any]) -> Any:
        """Return parsed IR from cache, or call *parse_fn* and cache it.

        Caches ``None`` results so that unparseable genomes are not
        re-parsed on every access.
        """
        cached = self._parse_cache.get(genome, _SENTINEL)
        if cached is not _SENTINEL:
            return cached
        result = parse_fn(genome)
        self._parse_cache[genome] = result
        return result

    # -- L3: full eval cache (async, archive-backed) ------------------------

    async def get(self, ir_hash: str, backend_version: str, config_hash: str) -> Fitness | None:
        """Look up a cached fitness result by composite key."""
        if self._archive is None:
            return None
        result: Fitness | None = await self._archive.lookup_fitness(
            ir_hash, backend_version, config_hash
        )
        return result

    async def put(
        self,
        ir_hash: str,
        backend_version: str,
        config_hash: str,
        fitness: Fitness,
        diagnostics_json: str,
    ) -> None:
        """Store a fitness result in the archive."""
        if self._archive is None:
            return
        await self._archive.store_fitness(
            ir_hash=ir_hash,
            backend_version=backend_version,
            config_hash=config_hash,
            fitness=fitness,
            diagnostics_json=diagnostics_json,
        )


class AsyncEvaluator:
    """Async evaluator with cache-first lookup, semaphore concurrency, and timeout."""

    def __init__(
        self,
        backend: Any,
        archive: Any,
        backend_version: str,
        config_hash: str,
        max_concurrent: int = 4,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._backend = backend
        self._cache = EvaluationCache(archive=archive)
        self._backend_version = backend_version
        self._config_hash = config_hash
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._timeout_seconds = timeout_seconds

    async def evaluate(self, individual: Individual) -> Individual:
        """Evaluate a single individual, using cache when possible.

        Returns a new Individual with *fitness* set.
        On timeout the individual receives a fallback infeasible fitness.
        """
        # 1. Check cache
        cached = await self._cache.get(
            individual.ir_hash, self._backend_version, self._config_hash
        )
        if cached is not None:
            return replace(individual, fitness=cached)

        # 2. Cache miss -- call backend with timeout
        try:
            fitness, diagnostics, _trace = await asyncio.wait_for(
                self._backend.evaluate(individual.ir),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            return replace(individual, fitness=_FALLBACK_FITNESS)

        # 3. Store in cache
        if diagnostics is None:
            diagnostics_json = "{}"
        elif dataclasses.is_dataclass(diagnostics) and not isinstance(diagnostics, type):
            diagnostics_json = json.dumps(dataclasses.asdict(diagnostics))
        else:
            diagnostics_json = json.dumps(diagnostics)
        await self._cache.put(
            individual.ir_hash,
            self._backend_version,
            self._config_hash,
            fitness,
            diagnostics_json,
        )

        return replace(individual, fitness=fitness)

    async def evaluate_batch(self, individuals: list[Individual]) -> list[Individual]:
        """Evaluate a batch of individuals with bounded concurrency."""

        async def _eval_with_sem(ind: Individual) -> Individual:
            async with self._semaphore:
                return await self.evaluate(ind)

        tasks = [asyncio.create_task(_eval_with_sem(ind)) for ind in individuals]
        results: list[Individual] = list(await asyncio.gather(*tasks))
        return results


class DeterministicEvaluator:
    """Thin wrapper that passes a fixed seed to the backend for reproducibility."""

    def __init__(self, backend: Any, eval_seed: int) -> None:
        self._backend = backend
        self._eval_seed = eval_seed

    async def evaluate(self, ir: Any) -> tuple[Fitness, Any, Any]:
        """Evaluate *ir* with a deterministic seed.

        Returns (fitness, diagnostics, trace).
        """
        result: tuple[Fitness, Any, Any] = await self._backend.evaluate(ir, seed=self._eval_seed)
        return result
