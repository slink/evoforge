"""Pydantic configuration models and TOML loader for evoforge."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel


class PopulationConfig(BaseModel):
    """Population sizing and elitism parameters."""

    size: int = 50
    elite_k: int = 5


class SelectionConfig(BaseModel):
    """Parent selection strategy and parameters."""

    strategy: str = "lexicase"
    tournament_size: int = 3
    epsilon: float = 0.0


class MutationConfig(BaseModel):
    """Mutation operator weights and schedule."""

    schedule: str = "adaptive"
    llm_weight: float = 0.6
    cheap_weight: float = 0.4
    crossover_weight: float = 0.2


class LLMConfig(BaseModel):
    """LLM backend configuration and budget limits."""

    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_calls: int = 1000
    max_cost_usd: float = 50.0


class EvalConfig(BaseModel):
    """Evaluation concurrency, timeout, and reproducibility settings."""

    max_concurrent: int = 4
    timeout_seconds: float = 60.0
    seed: int = 42


class BackendConfig(BaseModel):
    """Formal-verification backend settings."""

    name: str = "lean"
    theorem_statement: str = ""
    project_dir: str = ""
    repl_path: str | None = None


class EvolutionConfig(BaseModel):
    """Top-level evolution loop parameters."""

    max_generations: int = 100
    stagnation_window: int = 10
    checkpoint_every: int = 10
    log_level: str = "INFO"


class EvoforgeConfig(BaseModel):
    """Root configuration for an evoforge run."""

    population: PopulationConfig = PopulationConfig()
    selection: SelectionConfig = SelectionConfig()
    mutation: MutationConfig = MutationConfig()
    llm: LLMConfig = LLMConfig()
    eval: EvalConfig = EvalConfig()
    backend: BackendConfig = BackendConfig()
    evolution: EvolutionConfig = EvolutionConfig()


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
