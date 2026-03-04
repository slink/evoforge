#!/usr/bin/env python
"""Run an evoforge evolution experiment."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys


def main() -> None:
    """Parse arguments, build components, and run the evolutionary engine."""
    parser = argparse.ArgumentParser(description="evoforge evolutionary engine")
    parser.add_argument("--config", required=True, help="Path to TOML config file")
    parser.add_argument(
        "--max-generations", type=int, default=None, help="Override max generations"
    )
    parser.add_argument("--log-level", default=None, help="Override log level")
    args = parser.parse_args()

    # Load config
    from evoforge.core.config import load_config

    config = load_config(args.config)

    # Override max_generations if provided
    if args.max_generations is not None:
        config.evolution.max_generations = args.max_generations

    # Set up logging
    log_level = args.log_level or config.evolution.log_level
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Create backend
    from evoforge.backends.lean.backend import LeanBackend

    # LEAN_PROJECT_DIR env var overrides config
    project_dir = os.environ.get("LEAN_PROJECT_DIR", config.backend.project_dir)

    # Treat empty repl_path as None (TOML has repl_path = "")
    repl_path = config.backend.repl_path or None

    backend = LeanBackend(
        theorem_statement=config.backend.theorem_statement,
        project_dir=project_dir,
        repl_path=repl_path,
    )

    # Create archive (in-memory SQLite)
    from evoforge.core.archive import Archive

    archive = Archive("sqlite+aiosqlite://")

    # Create LLM client if API key is available
    llm_client = None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        from evoforge.llm.client import LLMClient

        llm_client = LLMClient(api_key=api_key)

    # Create engine
    from evoforge.core.engine import EvolutionEngine

    async def _run() -> None:
        await archive.create_tables()
        engine = EvolutionEngine(
            config=config,
            backend=backend,
            archive=archive,
            llm_client=llm_client,
        )
        result = await engine.run()
        print(f"Best fitness: {result.best_fitness}")
        print(f"Generations: {result.generations_run}")
        print(f"Evaluations: {result.total_evaluations}")

        sys.exit(0 if result.best_fitness >= 1.0 else 1)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
