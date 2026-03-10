# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Pydantic configuration models and TOML loader for evoforge."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator


class RunConfig(BaseModel):
    """Top-level run identification and seed."""

    name: str = ""
    backend: str = "lean"
    seed: int = 42


class PopulationConfig(BaseModel):
    """Population sizing and elitism parameters."""

    size: int = 50
    elite_k: int = 5


class SelectionConfig(BaseModel):
    """Parent selection strategy and parameters."""

    strategy: Literal["scalar_tournament", "pareto_nsga2", "lexicase", "map_elites"] = "lexicase"
    tournament_size: int = 3
    epsilon: float = 0.0


class MutationConfig(BaseModel):
    """Mutation operator weights and schedule."""

    schedule: Literal["fixed", "phased", "adaptive"] = "adaptive"
    llm_weight: float = 0.6
    cheap_weight: float = 0.4
    crossover_weight: float = 0.2


class LLMConfig(BaseModel):
    """LLM backend configuration and budget limits."""

    provider: Literal["anthropic", "gemini", "openai"] = "anthropic"
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str | None = None
    model: str = "claude-sonnet-4-5-20250929"
    reflection_model: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.7
    temperature_start: float = 1.0
    temperature_end: float = 0.3
    temperature_schedule: Literal["linear", "fixed"] = "linear"
    max_tokens: int = 4096
    max_calls: int = 1000
    max_cost_usd: float = 50.0
    max_attempts: int = 3
    prompt_caching: bool = True
    batch_enabled: bool = False
    batch_poll_interval: float = 2.0


class EvalConfig(BaseModel):
    """Evaluation concurrency, timeout, and reproducibility settings."""

    max_concurrent: int = 4
    timeout_seconds: float = 60.0
    seed: int = 42
    verification_threads: int = 0  # 0 = auto (cpu_count // 2), >0 = exact


class BackendConfig(BaseModel):
    """Formal-verification backend settings."""

    name: str = "lean"
    theorem_statement: str = ""
    project_dir: str = ""
    repl_path: str | None = None
    imports: str = ""
    seeds: list[str] = []
    theorem_file: str | None = None
    extra_api_namespaces: list[str] = []


class EvolutionConfig(BaseModel):
    """Top-level evolution loop parameters."""

    max_generations: int = 100
    stagnation_window: int = 10
    checkpoint_every: int = 10
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    resume: bool = False
    tree_search_enabled: bool = False
    tree_search_max_nodes: int = 200
    tree_search_beam_width: int = 5
    tree_search_min_fitness: float = 0.3


class ReflectionConfig(BaseModel):
    """Periodic reflection/summarization settings."""

    interval: int = 10
    include_top_k: int = 5
    include_bottom_k: int = 5


class MemoryConfig(BaseModel):
    """Pattern and dead-end memory limits."""

    max_patterns: int = 20
    max_dead_ends: int = 15
    max_constructs: int = 30


class SchedulerSettings(BaseModel):
    """Concurrency and batching settings for the scheduler."""

    mode: Literal["async_batch"] = "async_batch"
    max_llm_concurrent: int = 4
    max_eval_concurrent: int = 8
    max_pending: int = 16
    llm_budget_per_gen: int = 15


class DiversityConfig(BaseModel):
    """Diversity-maintenance strategy."""

    strategy: Literal["map_elites"] = "map_elites"
    sampling: Literal["weighted"] = "weighted"


class AblationConfig(BaseModel):
    """Feature toggles for ablation studies."""

    disable_llm: bool = False
    disable_diagnostics: bool = False
    disable_reflection: bool = False
    disable_memory: bool = False
    disable_cheap_operators: bool = False
    disable_credit: bool = False


class CFDBenchmarkCase(BaseModel):
    """A single benchmark case for CFD evaluation."""

    name: str
    Re: float
    S: float = 0.0
    Lambda: float = 0.0
    reference_fw: float = 0.0
    reference_regime: str = ""


class CFDBackendConfig(BaseModel):
    """Configuration for the CFD turbulence closure backend."""

    solver_project_dir: str = ""
    n_cycles: int = 20
    convergence_tol: float = 0.01
    grid_N: int = 128
    grid_H: float = 5.0
    grid_gamma: float = 2.0
    Sc_t: float = 1.0
    max_complexity: int = 30
    benchmark_cases: list[CFDBenchmarkCase] = []
    seeds: list[str] = []


class EvoforgeConfig(BaseModel):
    """Root configuration for an evoforge run."""

    run: RunConfig = RunConfig()
    population: PopulationConfig = PopulationConfig()
    selection: SelectionConfig = SelectionConfig()
    mutation: MutationConfig = MutationConfig()
    llm: LLMConfig = LLMConfig()
    eval: EvalConfig = EvalConfig()
    backend: BackendConfig = BackendConfig()
    evolution: EvolutionConfig = EvolutionConfig()
    reflection: ReflectionConfig = ReflectionConfig()
    memory: MemoryConfig = MemoryConfig()
    scheduler: SchedulerSettings = SchedulerSettings()
    diversity: DiversityConfig = DiversityConfig()
    ablation: AblationConfig = AblationConfig()
    cfd_backend: CFDBackendConfig = CFDBackendConfig()

    @model_validator(mode="after")
    def _check_backend_fields(self) -> EvoforgeConfig:
        """Validate that backend-specific required fields are set.

        Only fires when the corresponding backend config has been explicitly
        populated (i.e. is not pure defaults), so that partial configs for
        tests and non-primary backends don't trigger false errors.
        """
        if self.run.backend == "lean" and self.backend.name == "lean":
            has_lean_config = self.backend.theorem_statement or self.backend.project_dir
            if has_lean_config:
                if not self.backend.theorem_statement:
                    raise ValueError("backend.theorem_statement is required for the lean backend")
                if not self.backend.project_dir:
                    raise ValueError("backend.project_dir is required for the lean backend")
        elif self.run.backend == "cfd":
            if not self.cfd_backend.solver_project_dir and self.cfd_backend.benchmark_cases:
                raise ValueError("cfd_backend.solver_project_dir is required for the cfd backend")
        return self


def load_config(path: str | Path) -> EvoforgeConfig:
    """Load an EvoforgeConfig from a TOML file.

    Args:
        path: Path to a TOML configuration file.

    Returns:
        A validated EvoforgeConfig instance.

    Raises:
        FileNotFoundError: If the TOML file does not exist.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
        pydantic.ValidationError: If values fail Pydantic validation.
    """
    with Path(path).open("rb") as f:
        raw = tomllib.load(f)
    return EvoforgeConfig.model_validate(raw)
