"""EvolutionEngine — main evolutionary loop tying all components together."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

try:
    from rich.logging import RichHandler
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False

from evoforge.core.archive import Archive
from evoforge.core.config import EvoforgeConfig
from evoforge.core.evaluator import AsyncEvaluator, EvaluationCache
from evoforge.core.identity import IdentityPipeline
from evoforge.core.memory import SearchMemory
from evoforge.core.mutation import MutationContext, MutationEnsemble, MutationOperator
from evoforge.core.population import PopulationManager
from evoforge.core.scheduler import ExecutionScheduler, SchedulerConfig
from evoforge.core.selection import (
    Lexicase,
    MAPElites,
    ParetoNSGA2,
    ScalarTournament,
    SelectionStrategy,
)
from evoforge.core.types import Fitness, Individual

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of an evolutionary experiment."""

    best_individual: Individual | None
    best_fitness: float
    generations_run: int
    total_evaluations: int
    cost: dict[str, float]
    archive_size: int
    reflected: bool
    metrics: dict[str, float]


def _build_selection(config: EvoforgeConfig) -> SelectionStrategy:
    """Instantiate a selection strategy from config."""
    name = config.selection.strategy
    if name == "scalar_tournament":
        return ScalarTournament(tournament_size=config.selection.tournament_size)
    if name == "pareto_nsga2":
        return ParetoNSGA2()
    if name == "lexicase":
        return Lexicase()
    if name == "map_elites":
        # MAP-Elites requires grid dims — use a minimal fallback
        return MAPElites(grid_dims={"default": ["a", "b"]})
    msg = f"Unknown selection strategy: {name}"
    raise ValueError(msg)


class EvolutionEngine:
    """Main evolutionary loop integrating all evoforge components.

    Orchestrates seeding, identity processing, evaluation, credit assignment,
    selection, mutation, archiving, memory updates, and stagnation handling.
    """

    def __init__(
        self,
        config: EvoforgeConfig,
        backend: Any,
        archive: Archive,
        llm_client: Any = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.archive = archive
        self.llm_client = llm_client

        # Core components
        self.population = PopulationManager(max_size=config.population.size)
        self._strategy = _build_selection(config)
        self._identity = IdentityPipeline(backend)
        self._cache = EvaluationCache(archive)
        self._evaluator = AsyncEvaluator(
            backend=backend,
            archive=archive,
            backend_version=backend.version(),
            config_hash=backend.eval_config_hash(),
            max_concurrent=config.eval.max_concurrent,
            timeout_seconds=config.eval.timeout_seconds,
        )
        self._memory = SearchMemory(
            stagnation_window=config.evolution.stagnation_window,
            max_patterns=config.memory.max_patterns,
            max_dead_ends=config.memory.max_dead_ends,
        )
        self._scheduler = ExecutionScheduler(
            SchedulerConfig(
                max_concurrent_evals=config.eval.max_concurrent,
                max_concurrent_llm=config.scheduler.max_llm_concurrent,
                max_llm_calls=config.llm.max_calls,
                max_cost_usd=config.llm.max_cost_usd,
                eval_timeout_seconds=config.eval.timeout_seconds,
                llm_budget_per_gen=config.scheduler.llm_budget_per_gen,
            )
        )

        # Build mutation ensemble respecting ablation flags
        all_ops: list[Any] = []
        if not config.ablation.disable_cheap_operators:
            all_ops.extend(backend.mutation_operators())
        if llm_client is not None and not config.ablation.disable_llm:
            from evoforge.llm.operators import LLMCrossover, LLMMutate

            all_ops.append(LLMMutate(llm_client, config.llm.model, config.llm.max_tokens))
            all_ops.append(LLMCrossover(llm_client, config.llm.model, config.llm.max_tokens))

        # Fallback: if ablation disabled everything, use cheap ops anyway
        if not all_ops:
            all_ops.extend(backend.mutation_operators())

        self._ensemble = MutationEnsemble(
            operators=all_ops,
            schedule=config.mutation.schedule,
        )

        # Generator (only if LLM client provided and LLM not ablated)
        self._generator = None
        if llm_client is not None and not config.ablation.disable_llm:
            from evoforge.core.generator import ValidatedGenerator

            self._generator = ValidatedGenerator(
                backend=backend,
                llm_client=llm_client,
                model=config.llm.model,
            )

        # Tracking state
        self._total_evaluations = 0
        self._reflected = False
        self._temperature = config.llm.temperature
        self._temperature_boost: float = 0.0
        self._dedup_count: int = 0
        self._total_offspring_attempted: int = 0
        self._start_generation: int = 1
        self._verification_cache: dict[str, bool] = {}

    async def run(self) -> ExperimentResult:
        """Execute the full evolutionary loop and return results."""
        max_gen = self.config.evolution.max_generations
        ablation = self.config.ablation
        checkpoint_every = self.config.evolution.checkpoint_every

        await self.backend.startup()
        gen = 0
        resumed = False
        try:
            # --- Resume from checkpoint if requested ---
            if self.config.evolution.resume:
                start_gen = await self._load_checkpoint()
                if start_gen is not None:
                    self._start_generation = start_gen
                    resumed = True
                    logger.info(
                        "Resumed: starting at generation %d with %d individuals",
                        self._start_generation,
                        self.population.size,
                    )

            # --- Generation 0: Seed (skip if resumed) ---
            if not resumed:
                logger.info("Seeding population with %d individuals", self.config.population.size)
                seed_genomes = self.backend.seed_population(self.config.population.size)

                seed_individuals = self._process_genomes(seed_genomes, generation=0)
                if not seed_individuals:
                    logger.warning("No valid seed individuals produced; aborting")
                    return self._build_result(generations_run=0)

                evaluated_seeds = await self._evaluator.evaluate_batch(seed_individuals)
                self._total_evaluations += len(evaluated_seeds)

                credited_seeds = self._assign_credits(evaluated_seeds)
                await self._verify_perfect_individuals(credited_seeds)
                self._assign_behavior_descriptors(credited_seeds)
                self._add_to_population(credited_seeds)

                logger.info(
                    "Generation 0: pop_size=%d, best_fitness=%.4f",
                    self.population.size,
                    self._best_fitness(),
                )

            # --- Subsequent generations ---
            total_gens = max_gen + 1 - self._start_generation

            if _RICH_AVAILABLE:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Evolving"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    TextColumn("{task.fields[status]}"),
                )
            else:
                progress = None

            gen_range = range(self._start_generation, max_gen + 1)

            rich_handler: logging.Handler | None = None
            old_handlers: list[logging.Handler] = []
            if progress is not None:
                progress.start()
                task_id = progress.add_task("evolving", total=total_gens, status="")
                # Route log lines through Rich so they render above the progress bar
                rich_handler = RichHandler(
                    console=progress.console,
                    show_path=False,
                    show_time=True,
                )
                # Remove plain StreamHandlers (from basicConfig) to prevent
                # duplicate output; keep other handlers (e.g. pytest caplog).
                root = logging.getLogger()
                old_handlers = [h for h in root.handlers if type(h) is logging.StreamHandler]
                for h in old_handlers:
                    root.removeHandler(h)
                root.addHandler(rich_handler)

            try:
                for gen in gen_range:
                    # Reset per-generation scheduler state
                    self._scheduler.reset_generation()

                    # Temperature scheduling (base + decaying boost)
                    base_temp = self._compute_temperature(
                        generation=gen,
                        max_generations=max_gen,
                        start=self.config.llm.temperature_start,
                        end=self.config.llm.temperature_end,
                        schedule=self.config.llm.temperature_schedule,
                    )
                    self._temperature = base_temp + self._temperature_boost
                    self._temperature_boost *= 0.8  # decay each generation

                    # Check stopping conditions
                    if self._scheduler.should_stop():
                        logger.info("Budget exhausted at generation %d", gen)
                        break

                    pop_list = self.population.get_all()
                    if not pop_list:
                        logger.warning("Population empty at generation %d; stopping", gen)
                        break

                    # Select parents
                    k = min(self.config.population.size, len(pop_list))
                    parents = self._strategy.select(pop_list, k)

                    # Mutate (concurrently)
                    offspring_genomes: list[tuple[str, str, str]] = []
                    guidance = self._memory.prompt_section(max_tokens=200)
                    tasks: list[asyncio.Task[tuple[str, str, str] | None]] = []
                    for parent in parents:
                        operator = self._ensemble.select_operator()

                        # Skip LLM operators when per-gen budget exhausted
                        if operator.cost == "llm" and not self._scheduler.can_use_llm():
                            operator = self._ensemble.cheapest_operator()

                        context = MutationContext(
                            generation=gen,
                            memory=self._memory,
                            guidance=guidance,
                            temperature=self._temperature,
                            backend=self.backend,
                            credits=parent.credits,
                        )
                        tasks.append(
                            asyncio.create_task(self._mutate_one(parent, operator, context))
                        )
                    results = await asyncio.gather(*tasks)
                    for r in results:
                        if r is not None:
                            offspring_genomes.append(r)

                    # Identity pipeline + dedup (include known-bad verification hashes)
                    known_hashes = {ind.ir_hash for ind in pop_list}
                    known_hashes |= {h for h, v in self._verification_cache.items() if not v}
                    offspring_individuals: list[Individual] = []
                    offspring_lineage: list[tuple[str, str, str]] = []

                    for genome, parent_hash, op_name in offspring_genomes:
                        ind = self._identity.process(genome)
                        if ind is None:
                            continue
                        # Set generation
                        ind.generation = gen
                        self._total_offspring_attempted += 1
                        if self._identity.is_duplicate(ind.ir_hash, known_hashes):
                            self._dedup_count += 1
                            continue
                        known_hashes.add(ind.ir_hash)
                        offspring_individuals.append(ind)
                        offspring_lineage.append((parent_hash, ind.ir_hash, op_name))

                    if not offspring_individuals:
                        logger.debug("No novel offspring in generation %d", gen)
                        if not ablation.disable_memory:
                            self._memory.update([], gen)
                        await self._check_stagnation(gen)
                        if progress is not None:
                            progress.update(task_id, advance=1, status=self._gen_status(gen))
                        continue

                    # Evaluate offspring
                    evaluated_offspring = await self._evaluator.evaluate_batch(
                        offspring_individuals
                    )
                    self._total_evaluations += len(evaluated_offspring)

                    # Credit assignment
                    credited_offspring = self._assign_credits(evaluated_offspring)

                    # Verify fitness=1.0 proofs before survival selection
                    await self._verify_perfect_individuals(credited_offspring)

                    # Assign behavior descriptors to offspring
                    self._assign_behavior_descriptors(credited_offspring)

                    # Survive
                    pop_list = self.population.get_all()
                    survivors = self._strategy.survive(
                        pop_list, credited_offspring, self.config.population.elite_k
                    )

                    # Rebuild population from survivors
                    self.population = PopulationManager(max_size=self.config.population.size)
                    for ind in survivors:
                        self.population.add(ind)

                    # Refill if population collapsed due to dedup
                    await self._refill_population(gen)

                    # Archive offspring and lineage
                    for ind in credited_offspring:
                        await self.archive.store(ind)

                    for parent_hash, child_hash, op_name in offspring_lineage:
                        await self.archive.store_lineage(
                            parent_hash=parent_hash,
                            child_hash=child_hash,
                            operator_name=op_name,
                            generation=gen,
                        )

                    # Memory update
                    if not ablation.disable_memory:
                        self._memory.update(credited_offspring, gen)
                        # Log cmd error patterns to search memory as dead ends
                        for ind in credited_offspring:
                            if ind.fitness is not None:
                                pattern = ind.fitness.auxiliary.get("cmd_error_pattern")
                                if pattern and isinstance(pattern, str):
                                    self._memory.dead_ends.add(f"cmd_error:{pattern}")

                    # Stagnation check
                    await self._check_stagnation(gen)

                    # Periodic reflection
                    if (
                        not ablation.disable_reflection
                        and self.llm_client is not None
                        and gen % self.config.reflection.interval == 0
                    ):
                        logger.info("Periodic reflection at generation %d", gen)
                        await self._reflect(generation=gen)

                    # Checkpoint
                    if checkpoint_every > 0 and gen % checkpoint_every == 0:
                        await self._save_checkpoint(gen)

                    best_fit = self._best_fitness()
                    diversity = self.population.diversity_entropy()

                    # Early exit: proof is complete (only after verification)
                    if best_fit >= 1.0:
                        logger.info(
                            "Verified proof found at generation %d; early exit",
                            gen,
                        )
                        break

                    logger.info(
                        "Generation %d: pop_size=%d, best_fitness=%.4f, diversity=%.4f, evals=%d",
                        gen,
                        self.population.size,
                        best_fit,
                        diversity,
                        self._total_evaluations,
                    )

                    if progress is not None:
                        progress.update(
                            task_id,
                            advance=1,
                            status=self._gen_status(gen, best_fit, diversity),
                        )
            finally:
                if rich_handler is not None:
                    root = logging.getLogger()
                    root.removeHandler(rich_handler)
                    for h in old_handlers:
                        root.addHandler(h)
                if progress is not None:
                    progress.stop()
        finally:
            await self.backend.shutdown()

        return self._build_result(generations_run=min(max_gen, gen if max_gen > 0 else 0))

    # ------------------------------------------------------------------
    # Temperature scheduling
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_temperature(
        generation: int,
        max_generations: int,
        start: float,
        end: float,
        schedule: str,
    ) -> float:
        """Compute the LLM temperature for a given generation.

        Supports two schedules:
        - "linear": linearly interpolates from *start* to *end* over
          *max_generations*, clamping at *end* for generations beyond max.
        - "fixed": always returns *start* regardless of generation.

        Returns *start* when *max_generations* <= 0 (avoids division by zero).
        """
        if schedule == "fixed" or max_generations <= 0:
            return start
        t = min(generation / max_generations, 1.0)
        return start + (end - start) * t

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _mutate_one(
        self,
        parent: Individual,
        operator: MutationOperator,
        context: MutationContext,
    ) -> tuple[str, str, str] | None:
        """Apply a single mutation, returning (genome, parent_hash, op_name) or None."""
        try:
            if operator.cost == "llm":
                async with self._scheduler.acquire_llm():
                    if not self._scheduler.can_use_llm():
                        # Budget exhausted while waiting — fall back to cheap
                        op = self._ensemble.cheapest_operator()
                        new_genome = await op.apply(parent, context)
                        return (new_genome, parent.ir_hash, op.name)
                    self._scheduler.record_gen_llm_call()
                    new_genome = await operator.apply(parent, context)
            else:
                new_genome = await operator.apply(parent, context)
            return (new_genome, parent.ir_hash, operator.name)
        except Exception:
            logger.debug("Mutation failed for operator %s", operator.name, exc_info=True)
            return None

    def _process_genomes(self, genomes: list[str], generation: int) -> list[Individual]:
        """Process raw genomes through the identity pipeline."""
        results: list[Individual] = []
        seen: set[str] = set()
        for genome in genomes:
            ind = self._identity.process(genome)
            if ind is None:
                continue
            ind.generation = generation
            if ind.ir_hash in seen:
                continue
            seen.add(ind.ir_hash)
            results.append(ind)
        return results

    def _assign_credits(self, individuals: list[Individual]) -> list[Individual]:
        """Assign credit to each evaluated individual."""
        if self.config.ablation.disable_credit:
            return individuals
        for ind in individuals:
            if ind.fitness is not None and ind.ir is not None:
                try:
                    ind.credits = self.backend.assign_credit(
                        ind.ir, ind.fitness, ind.diagnostics, None
                    )
                except Exception:
                    logger.debug("Credit assignment failed", exc_info=True)
        return individuals

    def _assign_behavior_descriptors(self, individuals: list[Individual]) -> None:
        """Assign behavior descriptors to all individuals that have IR."""
        for ind in individuals:
            if ind.ir is not None and ind.behavior_descriptor is None:
                try:
                    ind.behavior_descriptor = self.backend.behavior_descriptor(
                        ind.ir, ind.diagnostics
                    )
                except Exception:
                    pass

    async def _evaluate_and_fill(
        self,
        individuals: list[Individual],
        existing_hashes: set[str],
        target: int,
    ) -> None:
        """Evaluate novel individuals and add them to the population up to target size."""
        known_bad = {h for h, v in self._verification_cache.items() if not v}
        novel = [
            ind
            for ind in individuals
            if ind.ir_hash not in existing_hashes and ind.ir_hash not in known_bad
        ]
        if not novel:
            return
        evaluated = await self._evaluator.evaluate_batch(novel)
        self._total_evaluations += len(evaluated)
        self._assign_credits(evaluated)
        await self._verify_perfect_individuals(evaluated)
        self._assign_behavior_descriptors(evaluated)
        for ind in evaluated:
            if self.population.size >= target:
                break
            if self.population.add(ind):
                existing_hashes.add(ind.ir_hash)

    async def _refill_population(self, generation: int) -> None:
        """Refill population to target size when it drops after survival selection.

        Uses fresh seeds from the backend, evaluates and deduplicates them.
        Falls back to cheap mutations on existing individuals if seeds are exhausted.
        """
        target = self.config.population.size
        if self.population.size >= target:
            return

        deficit = target - self.population.size
        logger.debug(
            "Population at %d/%d after survival — refilling %d slots",
            self.population.size,
            target,
            deficit,
        )

        existing_hashes = {ind.ir_hash for ind in self.population.get_all()}

        # Phase 1: try fresh seeds
        seed_genomes = self.backend.seed_population(deficit * 2)
        seed_individuals = self._process_genomes(seed_genomes, generation=generation)
        await self._evaluate_and_fill(seed_individuals, existing_hashes, target)

        if self.population.size >= target:
            return

        # Phase 2: cheap mutations on existing population members
        pop_list = self.population.get_all()
        if not pop_list:
            return

        guidance = self._memory.prompt_section(max_tokens=200)
        cheapest = self._ensemble.cheapest_operator()
        tasks: list[asyncio.Task[str | None]] = []

        async def _try_mutate(parent: Individual) -> str | None:
            context = MutationContext(
                generation=generation,
                memory=self._memory,
                guidance=guidance,
                temperature=self._temperature,
                backend=self.backend,
                credits=parent.credits,
            )
            try:
                return await cheapest.apply(parent, context)
            except Exception:
                return None

        for i in range(deficit * 2):
            tasks.append(asyncio.create_task(_try_mutate(pop_list[i % len(pop_list)])))

        results = await asyncio.gather(*tasks)
        mutant_genomes = [g for g in results if g is not None]
        mutant_individuals = self._process_genomes(mutant_genomes, generation=generation)
        await self._evaluate_and_fill(mutant_individuals, existing_hashes, target)

    def _add_to_population(self, individuals: list[Individual]) -> None:
        """Add individuals to the population manager, deduplicating by ir_hash."""
        for ind in individuals:
            self.population.add(ind)

    def _gen_status(
        self, gen: int, best_fit: float | None = None, diversity: float | None = None
    ) -> str:
        """Format a one-line status string for the current generation."""
        if best_fit is None:
            best_fit = self._best_fitness()
        if diversity is None:
            diversity = self.population.diversity_entropy()
        return (
            f"gen {gen} | best={best_fit:.4f}"
            f" | pop={self.population.size}"
            f" | div={diversity:.4f}"
            f" | evals={self._total_evaluations}"
        )

    def _best_fitness(self) -> float:
        """Return the best primary fitness in the population (feasible only)."""
        best_val = 0.0
        for ind in self.population.get_all():
            if ind.fitness is not None and ind.fitness.feasible:
                best_val = max(best_val, ind.fitness.primary)
        return best_val

    def _trailing_stagnation(self) -> int:
        """Count how many consecutive generations share the best fitness at the tail."""
        history = self._memory.best_fitness_history
        if not history:
            return 0
        last = history[-1]
        count = 0
        for val in reversed(history):
            if val == last:
                count += 1
            else:
                break
        return count

    async def _check_stagnation(self, generation: int) -> None:
        """Check for stagnation and trigger reflection if needed."""
        if not self._memory.is_stagnant():
            return
        logger.info(
            "Stagnation detected at generation %d — triggering reflection",
            generation,
        )
        self._reflected = True
        max_boost = 0.3
        old_boost = self._temperature_boost
        self._temperature_boost = min(self._temperature_boost + 0.2, max_boost)
        # Apply the increment to the current generation's temperature for reflection
        increment = self._temperature_boost - old_boost
        self._temperature = min(
            self._temperature + increment, self.config.llm.temperature_start + max_boost
        )
        if self.llm_client is not None and not self.config.ablation.disable_reflection:
            await self._reflect(generation)

    async def _reflect(self, generation: int) -> None:
        """Call the LLM for a reflection on the current population state."""
        try:
            # Limit population to top_k for reflection prompt
            top_k = self.config.reflection.include_top_k
            pop = self.population.best(k=top_k)
            prompt = self.backend.format_reflection_prompt(
                population=pop,
                memory=self._memory,
                generation=generation,
            )
            system = self.backend.system_prompt()
            model = self.config.llm.reflection_model
            response = await self.llm_client.async_generate(
                prompt,
                system,
                model,
                self._temperature,
                self.config.llm.max_tokens,
            )
            logger.info("Reflection response received (%d chars)", len(response.text))
        except Exception:
            logger.warning("Reflection LLM call failed", exc_info=True)

    async def _save_checkpoint(self, generation: int) -> None:
        """Serialize engine state and store a checkpoint in the archive."""
        state = {
            "generation": generation,
            "total_evaluations": self._total_evaluations,
            "reflected": self._reflected,
            "temperature": self._temperature,
            "temperature_boost": self._temperature_boost,
            "dedup_count": self._dedup_count,
            "total_offspring_attempted": self._total_offspring_attempted,
            "population_hashes": [ind.ir_hash for ind in self.population.get_all()],
            "memory": self._memory.to_dict(),
            "ensemble": self._ensemble.to_dict(),
            "cost_summary": self._scheduler.tracker.summary(),
            "verification_cache": self._verification_cache,
        }
        await self.archive.store_checkpoint(generation, state)
        logger.info("Checkpoint saved at generation %d", generation)

    async def _load_checkpoint(self) -> int | None:
        """Attempt to load the latest checkpoint.

        Returns the generation to start from, or None if no checkpoint exists.
        """
        data = await self.archive.load_latest_checkpoint()
        if data is None:
            return None

        checkpoint_gen: int = data["generation"]
        logger.info("Resuming from checkpoint at generation %d", checkpoint_gen)

        # Restore scalar state
        self._total_evaluations = data["total_evaluations"]
        self._reflected = data["reflected"]
        self._temperature = data["temperature"]
        self._temperature_boost = data["temperature_boost"]
        self._dedup_count = data["dedup_count"]
        self._total_offspring_attempted = data["total_offspring_attempted"]

        # Restore memory
        self._memory.from_dict(data["memory"])

        # Restore ensemble weights/stats
        self._ensemble.from_dict(data["ensemble"])

        # Restore verification cache
        self._verification_cache = dict(data.get("verification_cache", {}))

        # Restore population from archive by ir_hash (batch lookup)
        pop_hashes: list[str] = data["population_hashes"]
        archived = await self.archive.lookup_many(pop_hashes)
        restored_count = 0
        for ir_hash in pop_hashes:
            ind = archived.get(ir_hash)
            if ind is not None:
                # Re-process through identity pipeline to restore IR
                reparsed = self._identity.process(ind.genome)
                if reparsed is not None:
                    reparsed.fitness = ind.fitness
                    reparsed.generation = ind.generation
                    reparsed.id = ind.id
                    reparsed.behavior_descriptor = ind.behavior_descriptor
                    reparsed.mutation_source = ind.mutation_source
                    self.population.add(reparsed)
                    restored_count += 1

        logger.info(
            "Restored %d/%d individuals from checkpoint",
            restored_count,
            len(pop_hashes),
        )

        return checkpoint_gen + 1

    async def _verify_perfect_individuals(self, individuals: list[Individual]) -> None:
        """Verify fitness=1.0 individuals via backend.verify_proof().

        Only runs on individuals that passed REPL cmd verification
        (cmd_verified=1.0 in auxiliary). Uses a cache keyed by ir_hash.
        On failure, records the genome as a dead end in search memory.
        """
        for ind in individuals:
            if (
                ind.fitness is not None
                and ind.fitness.primary >= 1.0
                and float(ind.fitness.auxiliary.get("cmd_verified", 0.0)) >= 1.0
            ):
                # Check cache first
                cache_hit = ind.ir_hash in self._verification_cache
                if cache_hit:
                    verified = self._verification_cache[ind.ir_hash]
                else:
                    verified = await self.backend.verify_proof(ind.genome)
                    self._verification_cache[ind.ir_hash] = verified

                if not verified:
                    if cache_hit:
                        logger.debug(
                            "Proof failed verification (cached) — marking infeasible: %s",
                            ind.genome[:80],
                        )
                    else:
                        logger.warning(
                            "Proof failed verification — marking infeasible: %s",
                            ind.genome[:80],
                        )
                        self._memory.record_verification_failure(ind.genome)
                    ind.fitness = Fitness(
                        primary=ind.fitness.primary,
                        auxiliary=ind.fitness.auxiliary,
                        constraints=ind.fitness.constraints,
                        feasible=False,
                    )

    def _build_result(self, generations_run: int) -> ExperimentResult:
        """Build the final experiment result."""
        best = self.population.best(k=1)
        best_individual = best[0] if best else None
        best_fitness = self._best_fitness()

        metrics = {
            "cache_hit_rate": 0.0,  # Would need evaluator to track; placeholder
            "identity_dedup_rate": (
                self._dedup_count / self._total_offspring_attempted
                if self._total_offspring_attempted > 0
                else 0.0
            ),
            "stagnation_counter": float(self._trailing_stagnation()),
            "temperature_boost": self._temperature_boost,
        }

        return ExperimentResult(
            best_individual=best_individual,
            best_fitness=best_fitness,
            generations_run=generations_run,
            total_evaluations=self._total_evaluations,
            cost=self._scheduler.tracker.summary(),
            archive_size=self.population.size,
            reflected=self._reflected,
            metrics=metrics,
        )
