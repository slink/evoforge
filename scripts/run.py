#!/usr/bin/env python
"""Run an evoforge evolution experiment."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path


def main() -> None:
    """Parse arguments, build components, and run the evolutionary engine."""
    parser = argparse.ArgumentParser(description="evoforge evolutionary engine")
    parser.add_argument("--config", required=True, help="Path to TOML config file")
    parser.add_argument(
        "--max-generations", type=int, default=None, help="Override max generations"
    )
    parser.add_argument("--log-level", default=None, help="Override log level")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: runs/<run-name>/)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify winning proof with lake env lean after the run",
    )
    args = parser.parse_args()

    # Load config
    from evoforge.core.config import load_config

    config = load_config(args.config)

    # Override config from CLI flags
    if args.max_generations is not None:
        config.evolution.max_generations = args.max_generations
    if args.resume:
        config.evolution.resume = True

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("runs") / (config.run.name or "default")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_level = args.log_level or config.evolution.log_level
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Suppress noisy third-party loggers that clash with the rich progress bar
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

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
        imports=config.backend.imports,
        seeds=config.backend.seeds or None,
    )

    # Create archive (file-backed SQLite in output dir)
    from evoforge.core.archive import Archive

    db_path = output_dir / "archive.db"
    archive = Archive(f"sqlite+aiosqlite:///{db_path}")

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

        if result.best_fitness >= 1.0 and result.best_individual is not None:
            print("\n--- Winning proof ---")
            print(result.best_individual.genome)

            proof_text = backend.format_proof(result.best_individual.genome)
            proof_path = output_dir / "proof.lean"
            proof_path.write_text(proof_text)
            print(f"Proof written to {proof_path}")

            if args.verify:
                import subprocess as _sp

                print("\nVerifying proof with lake env lean...")
                ret = _sp.run(
                    ["lake", "env", "lean", str(proof_path)],
                    cwd=project_dir,
                    capture_output=True,
                    timeout=120,
                )
                if ret.returncode == 0:
                    print("Proof verified by lake build!")
                else:
                    print("WARNING: Proof failed lake verification:")
                    print(ret.stderr.decode(errors="replace"))

        sys.exit(0 if result.best_fitness >= 1.0 else 1)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
