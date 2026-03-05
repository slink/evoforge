"""Tests for config schema — verifies all DESIGN_v4 sections exist with correct defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from evoforge.core.config import (
    AblationConfig,
    BackendConfig,
    DiversityConfig,
    EvalConfig,
    EvoforgeConfig,
    LLMConfig,
    MemoryConfig,
    ReflectionConfig,
    RunConfig,
    SchedulerSettings,
    load_config,
)

# ---------------------------------------------------------------------------
# New config model defaults
# ---------------------------------------------------------------------------


class TestRunConfig:
    def test_defaults(self) -> None:
        cfg = RunConfig()
        assert cfg.name == ""
        assert cfg.backend == "lean"
        assert cfg.seed == 42


class TestReflectionConfig:
    def test_defaults(self) -> None:
        cfg = ReflectionConfig()
        assert cfg.interval == 10
        assert cfg.include_top_k == 5
        assert cfg.include_bottom_k == 5


class TestMemoryConfig:
    def test_defaults(self) -> None:
        cfg = MemoryConfig()
        assert cfg.max_patterns == 20
        assert cfg.max_dead_ends == 15
        assert cfg.max_constructs == 30


class TestSchedulerSettings:
    def test_defaults(self) -> None:
        cfg = SchedulerSettings()
        assert cfg.mode == "async_batch"
        assert cfg.max_llm_concurrent == 4
        assert cfg.max_eval_concurrent == 8
        assert cfg.max_pending == 16
        assert cfg.llm_budget_per_gen == 15


class TestDiversityConfig:
    def test_defaults(self) -> None:
        cfg = DiversityConfig()
        assert cfg.strategy == "map_elites"
        assert cfg.sampling == "weighted"


class TestAblationConfig:
    def test_defaults(self) -> None:
        cfg = AblationConfig()
        assert cfg.disable_llm is False
        assert cfg.disable_diagnostics is False
        assert cfg.disable_reflection is False
        assert cfg.disable_memory is False
        assert cfg.disable_cheap_operators is False
        assert cfg.disable_credit is False


# ---------------------------------------------------------------------------
# BackendConfig imports field
# ---------------------------------------------------------------------------


class TestBackendConfig:
    def test_imports_default_empty(self) -> None:
        cfg = BackendConfig()
        assert cfg.imports == ""

    def test_imports_accepts_value(self) -> None:
        cfg = BackendConfig(imports="import Foo")
        assert cfg.imports == "import Foo"

    def test_imports_round_trip_toml(self) -> None:
        """The lean_default.toml should load the imports field."""
        cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "lean_default.toml")
        expected = "import LeanLevy\nopen ProbabilityTheory ProbabilityTheory.IsPositiveDefinite"
        assert cfg.backend.imports == expected


# ---------------------------------------------------------------------------
# Expanded LLMConfig
# ---------------------------------------------------------------------------


class TestLLMConfig:
    def test_temperature_schedule_fields(self) -> None:
        cfg = LLMConfig()
        assert cfg.temperature_start == pytest.approx(1.0)
        assert cfg.temperature_end == pytest.approx(0.3)
        assert cfg.temperature_schedule == "linear"

    def test_reflection_model_contains_sonnet(self) -> None:
        cfg = LLMConfig()
        assert "sonnet" in cfg.reflection_model

    def test_max_attempts_default(self) -> None:
        cfg = LLMConfig()
        assert cfg.max_attempts == 3


# ---------------------------------------------------------------------------
# EvoforgeConfig has all new sections
# ---------------------------------------------------------------------------


class TestEvalConfig:
    def test_verification_threads_default_zero(self) -> None:
        cfg = EvalConfig()
        assert cfg.verification_threads == 0


class TestEvoforgeConfig:
    def test_has_run_section(self) -> None:
        cfg = EvoforgeConfig()
        assert isinstance(cfg.run, RunConfig)

    def test_has_reflection_section(self) -> None:
        cfg = EvoforgeConfig()
        assert isinstance(cfg.reflection, ReflectionConfig)

    def test_has_memory_section(self) -> None:
        cfg = EvoforgeConfig()
        assert isinstance(cfg.memory, MemoryConfig)

    def test_has_scheduler_section(self) -> None:
        cfg = EvoforgeConfig()
        assert isinstance(cfg.scheduler, SchedulerSettings)

    def test_has_diversity_section(self) -> None:
        cfg = EvoforgeConfig()
        assert isinstance(cfg.diversity, DiversityConfig)

    def test_has_ablation_section(self) -> None:
        cfg = EvoforgeConfig()
        assert isinstance(cfg.ablation, AblationConfig)


# ---------------------------------------------------------------------------
# TOML round-trip
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_lean_default_toml(self) -> None:
        """load_config succeeds and populates all new sections."""
        cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "lean_default.toml")
        assert isinstance(cfg, EvoforgeConfig)
        # Spot-check new sections loaded from file
        assert cfg.run.name == "levy_proof_search_001"
        assert cfg.reflection.interval == 10
        assert cfg.memory.max_patterns == 20
        assert cfg.scheduler.mode == "async_batch"
        assert cfg.diversity.strategy == "map_elites"
        assert cfg.ablation.disable_llm is False
        # Check updated existing values
        assert cfg.llm.temperature_start == pytest.approx(1.0)
        assert cfg.llm.reflection_model != ""
        assert cfg.llm.max_attempts == 3
