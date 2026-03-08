# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.

"""End-to-end smoke test: run the full EvolutionEngine with the CFD backend."""

from __future__ import annotations

import math

import pytest

from evoforge.backends.cfd.backend import CFDBackend
from evoforge.core.archive import Archive
from evoforge.core.config import (
    AblationConfig,
    CFDBackendConfig,
    CFDBenchmarkCase,
    EvoforgeConfig,
    EvolutionConfig,
    PopulationConfig,
    SelectionConfig,
)
from evoforge.core.engine import EvolutionEngine


def _smoke_config() -> EvoforgeConfig:
    """Minimal config for a fast 2-generation smoke test."""
    return EvoforgeConfig(
        population=PopulationConfig(size=10, elite_k=2),
        selection=SelectionConfig(strategy="scalar_tournament", tournament_size=3),
        evolution=EvolutionConfig(max_generations=2, stagnation_window=5),
        ablation=AblationConfig(
            disable_llm=True,
            disable_reflection=True,
        ),
        cfd_backend=CFDBackendConfig(
            n_cycles=2,
            grid_N=32,
            benchmark_cases=[
                CFDBenchmarkCase(name="smoke_Re394", Re=394.0, reference_fw=0.226),
            ],
        ),
    )


@pytest.mark.timeout(120)
async def test_cfd_engine_smoke() -> None:
    """Full engine loop: seed -> parse -> evaluate -> select -> mutate -> repeat."""
    config = _smoke_config()
    backend = CFDBackend(config.cfd_backend)
    archive = Archive("sqlite+aiosqlite://")
    await archive.create_tables()

    engine = EvolutionEngine(
        config=config,
        backend=backend,
        archive=archive,
        llm_client=None,
    )

    result = await engine.run()

    # Engine completed at least 2 generations
    assert result.generations_run >= 2, f"Only ran {result.generations_run} generations"

    # At least the seed population was evaluated
    assert result.total_evaluations >= 10, (
        f"Only {result.total_evaluations} evaluations (expected >= 10)"
    )

    # Best fitness is positive and finite
    assert result.best_fitness > 0.0, f"Best fitness is {result.best_fitness}"
    assert math.isfinite(result.best_fitness), f"Best fitness is not finite: {result.best_fitness}"

    # Best individual exists
    assert result.best_individual is not None, "No best individual returned"

    # Population did not collapse
    assert engine.population.size >= 5, (
        f"Population collapsed to {engine.population.size} (expected >= 5)"
    )
