# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
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
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write DEBUG-level logs to this file (console stays at --log-level)",
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
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console handler at user-specified level
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root_logger.addHandler(console)

    # File handler at DEBUG if requested
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers that clash with the rich progress bar
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    # Create backend
    if config.run.backend == "cfd":
        from evoforge.backends.cfd.backend import CFDBackend

        backend = CFDBackend(config.cfd_backend)
    else:
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
            extra_api_namespaces=config.backend.extra_api_namespaces or None,
            verification_threads=config.eval.verification_threads,
        )

    # Create archive (file-backed SQLite in output dir)
    from evoforge.core.archive import Archive

    db_path = output_dir / "archive.db"
    archive = Archive(f"sqlite+aiosqlite:///{db_path}")

    # Create LLM client if API key is available
    llm_client = None
    api_key = os.environ.get(config.llm.api_key_env)
    if api_key:
        from evoforge.llm.client import LLMClient
        from evoforge.llm.providers import create_provider

        provider = create_provider(config)
        llm_client = LLMClient(api_key=api_key, provider=provider)
        # NOTE: Batch API is Anthropic-only. When using a non-Anthropic provider,
        # batch_enabled should be False (the BatchCollector accesses the raw
        # anthropic.AsyncAnthropic client). Per-provider batch abstraction is a
        # future extension — Gemini and OpenAI both offer batch APIs with
        # different semantics.
        if config.llm.provider != "anthropic" and config.llm.batch_enabled:
            logging.getLogger(__name__).warning(
                "Batch API is only supported with the Anthropic provider. "
                "Disabling batch mode for provider=%r.",
                config.llm.provider,
            )
            config.llm.batch_enabled = False

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

            if args.verify and config.run.backend != "cfd":
                import subprocess as _sp

                if "sorry" in result.best_individual.genome:
                    print("\nWARNING: Proof contains sorry — not a complete proof")
                else:
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
            elif args.verify and config.run.backend == "cfd":
                print("(--verify is not applicable to CFD backend)")

        sys.exit(0 if result.best_fitness >= 1.0 else 1)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
