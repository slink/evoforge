"""Tests for checkpoint save/load and resume functionality."""

from __future__ import annotations

from unittest.mock import patch

from evoforge.core.archive import Archive
from evoforge.core.config import (
    AblationConfig,
    EvoforgeConfig,
    EvolutionConfig,
    PopulationConfig,
    SelectionConfig,
)
from evoforge.core.engine import EvolutionEngine

# Re-use the real MockBackend from test_engine (proper Backend subclass).
from tests.test_core.test_engine import MockBackend


def _make_config(
    *,
    max_generations: int = 10,
    population_size: int = 3,
    checkpoint_every: int = 5,
    resume: bool = False,
    stagnation_window: int = 200,
) -> EvoforgeConfig:
    """Create a minimal config for checkpoint testing."""
    return EvoforgeConfig(
        population=PopulationConfig(size=population_size, elite_k=1),
        selection=SelectionConfig(strategy="scalar_tournament", tournament_size=2),
        evolution=EvolutionConfig(
            max_generations=max_generations,
            stagnation_window=stagnation_window,
            checkpoint_every=checkpoint_every,
            log_level="DEBUG",
            resume=resume,
        ),
        ablation=AblationConfig(
            disable_llm=True,
            disable_reflection=True,
        ),
    )


async def _run_engine(
    config: EvoforgeConfig,
    archive: Archive,
) -> EvolutionEngine:
    """Build and run an engine with the real MockBackend, returning it."""
    backend = MockBackend()
    engine = EvolutionEngine(
        config=config,
        backend=backend,
        archive=archive,
    )
    await engine.run()
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCheckpointSave:
    """Verify that checkpoints are written at the expected intervals."""

    async def test_checkpoint_saved_at_interval(self, archive: Archive) -> None:
        """Running 10 gens with checkpoint_every=5 stores checkpoint at gen 10."""
        config = _make_config(max_generations=10, checkpoint_every=5)
        await _run_engine(config, archive)

        latest = await archive.load_latest_checkpoint()
        assert latest is not None
        assert latest["generation"] == 10

    async def test_checkpoint_every_generation(self, archive: Archive) -> None:
        """checkpoint_every=1 should store a checkpoint every generation."""
        config = _make_config(max_generations=3, checkpoint_every=1)
        await _run_engine(config, archive)

        latest = await archive.load_latest_checkpoint()
        assert latest is not None
        assert latest["generation"] == 3

    async def test_checkpoint_contains_required_fields(self, archive: Archive) -> None:
        """Checkpoint JSON should contain all required state fields."""
        config = _make_config(max_generations=5, checkpoint_every=5)
        await _run_engine(config, archive)

        latest = await archive.load_latest_checkpoint()
        assert latest is not None

        required_keys = {
            "generation",
            "total_evaluations",
            "reflected",
            "temperature",
            "temperature_boost",
            "dedup_count",
            "total_offspring_attempted",
            "population_hashes",
            "memory",
            "ensemble",
            "cost_summary",
        }
        assert required_keys.issubset(set(latest.keys()))

    async def test_checkpoint_population_hashes_nonempty(self, archive: Archive) -> None:
        """population_hashes in checkpoint should list current pop ir_hashes."""
        config = _make_config(max_generations=5, checkpoint_every=5)
        engine = await _run_engine(config, archive)

        latest = await archive.load_latest_checkpoint()
        assert latest is not None
        assert len(latest["population_hashes"]) > 0
        # Should match the engine's population at checkpoint time
        engine_hashes = {ind.ir_hash for ind in engine.population.get_all()}
        checkpoint_hashes = set(latest["population_hashes"])
        assert checkpoint_hashes == engine_hashes


class TestCheckpointLoad:
    """Verify that _load_checkpoint restores engine state."""

    async def test_no_checkpoint_returns_none(self, archive: Archive) -> None:
        """_load_checkpoint should return None on a fresh archive."""
        config = _make_config()
        backend = MockBackend()
        engine = EvolutionEngine(config=config, backend=backend, archive=archive)

        result = await engine._load_checkpoint()
        assert result is None

    async def test_resume_restores_generation(self, archive: Archive) -> None:
        """After checkpoint at gen 5, _load_checkpoint returns start_gen=6."""
        config = _make_config(max_generations=5, checkpoint_every=5)
        await _run_engine(config, archive)

        # Create a fresh engine and load checkpoint
        config2 = _make_config(resume=True)
        backend2 = MockBackend()
        engine2 = EvolutionEngine(config=config2, backend=backend2, archive=archive)
        start_gen = await engine2._load_checkpoint()

        assert start_gen == 6

    async def test_resume_restores_evaluations(self, archive: Archive) -> None:
        """total_evaluations should be preserved after resume."""
        config = _make_config(max_generations=5, checkpoint_every=5)
        engine1 = await _run_engine(config, archive)
        original_evals = engine1._total_evaluations

        config2 = _make_config(resume=True)
        backend2 = MockBackend()
        engine2 = EvolutionEngine(config=config2, backend=backend2, archive=archive)
        await engine2._load_checkpoint()

        assert engine2._total_evaluations == original_evals

    async def test_resume_restores_temperature(self, archive: Archive) -> None:
        """Temperature and temperature_boost should be preserved."""
        config = _make_config(max_generations=5, checkpoint_every=5)
        engine1 = await _run_engine(config, archive)
        original_temp = engine1._temperature
        original_boost = engine1._temperature_boost

        config2 = _make_config(resume=True)
        backend2 = MockBackend()
        engine2 = EvolutionEngine(config=config2, backend=backend2, archive=archive)
        await engine2._load_checkpoint()

        assert engine2._temperature == original_temp
        assert engine2._temperature_boost == original_boost

    async def test_resume_restores_population(self, archive: Archive) -> None:
        """Population should be non-empty after resuming from checkpoint."""
        config = _make_config(max_generations=5, checkpoint_every=5)
        engine1 = await _run_engine(config, archive)
        original_size = engine1.population.size
        assert original_size > 0

        config2 = _make_config(resume=True)
        backend2 = MockBackend()
        engine2 = EvolutionEngine(config=config2, backend=backend2, archive=archive)
        await engine2._load_checkpoint()

        assert engine2.population.size > 0
        # Should restore the same number of individuals
        assert engine2.population.size == original_size

    async def test_resume_restores_memory(self, archive: Archive) -> None:
        """Memory state (best_fitness_history) should be preserved."""
        config = _make_config(max_generations=5, checkpoint_every=5)
        engine1 = await _run_engine(config, archive)
        original_history_len = len(engine1._memory.best_fitness_history)

        config2 = _make_config(resume=True)
        backend2 = MockBackend()
        engine2 = EvolutionEngine(config=config2, backend=backend2, archive=archive)
        await engine2._load_checkpoint()

        assert len(engine2._memory.best_fitness_history) == original_history_len

    async def test_resume_restores_dedup_count(self, archive: Archive) -> None:
        """Dedup count should be preserved after resume."""
        config = _make_config(max_generations=5, checkpoint_every=5)
        engine1 = await _run_engine(config, archive)
        original_dedup = engine1._dedup_count

        config2 = _make_config(resume=True)
        backend2 = MockBackend()
        engine2 = EvolutionEngine(config=config2, backend=backend2, archive=archive)
        await engine2._load_checkpoint()

        assert engine2._dedup_count == original_dedup


class TestResumeIntegration:
    """End-to-end resume: checkpoint then continue evolution."""

    async def test_resume_skips_seeding(self, archive: Archive) -> None:
        """When resuming, seed_population should NOT be called again."""
        config = _make_config(max_generations=5, checkpoint_every=5)
        await _run_engine(config, archive)

        # Resume with a patched backend to spy on seed_population
        config2 = _make_config(resume=True, max_generations=6)
        backend2 = MockBackend()

        with patch.object(backend2, "seed_population", wraps=backend2.seed_population) as spy:
            engine2 = EvolutionEngine(config=config2, backend=backend2, archive=archive)
            await engine2.run()
            spy.assert_not_called()

    async def test_resume_continues_from_checkpoint(self, archive: Archive) -> None:
        """Resumed engine should run starting from checkpoint gen + 1."""
        config = _make_config(max_generations=5, checkpoint_every=5)
        engine1 = await _run_engine(config, archive)
        evals_before = engine1._total_evaluations

        config2 = _make_config(resume=True, max_generations=8)
        backend2 = MockBackend()
        engine2 = EvolutionEngine(config=config2, backend=backend2, archive=archive)
        result = await engine2.run()

        # Should have run gens 6, 7, 8 (3 more generations)
        assert result.generations_run == 8
        # Should have more evaluations than the first run
        assert result.total_evaluations >= evals_before

    async def test_resume_no_checkpoint_seeds_normally(self, archive: Archive) -> None:
        """When resume=True but no checkpoint exists, engine seeds normally."""
        config = _make_config(resume=True, max_generations=3, checkpoint_every=3)
        backend = MockBackend()

        with patch.object(backend, "seed_population", wraps=backend.seed_population) as spy:
            engine = EvolutionEngine(config=config, backend=backend, archive=archive)
            result = await engine.run()
            spy.assert_called_once()
            assert result.generations_run >= 1
