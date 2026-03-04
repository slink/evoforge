"""EvolutionEngine — main evolutionary loop tying all components together."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from evoforge.core.archive import Archive
from evoforge.core.config import EvoforgeConfig
from evoforge.core.evaluator import AsyncEvaluator, EvaluationCache
from evoforge.core.identity import IdentityPipeline
from evoforge.core.memory import SearchMemory
from evoforge.core.mutation import MutationContext, MutationEnsemble
from evoforge.core.population import PopulationManager
from evoforge.core.scheduler import ExecutionScheduler, SchedulerConfig
from evoforge.core.selection import (
    Lexicase,
    MAPElites,
    ParetoNSGA2,
    ScalarTournament,
    SelectionStrategy,
)
from evoforge.core.types import Individual

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
            backend_version="mock_v1",
            config_hash="cfg_test",
            max_concurrent=config.eval.max_concurrent,
            timeout_seconds=config.eval.timeout_seconds,
        )
        self._memory = SearchMemory(
            stagnation_window=config.evolution.stagnation_window,
        )
        self._scheduler = ExecutionScheduler(
            SchedulerConfig(
                max_concurrent_evals=config.eval.max_concurrent,
                max_llm_calls=config.llm.max_calls,
                max_cost_usd=config.llm.max_cost_usd,
                eval_timeout_seconds=config.eval.timeout_seconds,
            )
        )

        # Build mutation ensemble from backend (cheap) operators only when no LLM
        cheap_ops = backend.mutation_operators()
        all_ops = list(cheap_ops)
        # If LLM client is available, add LLM operators
        if llm_client is not None:
            from evoforge.llm.operators import LLMCrossover, LLMMutate

            all_ops.append(LLMMutate(llm_client, config.llm.model, config.llm.max_tokens))
            all_ops.append(LLMCrossover(llm_client, config.llm.model, config.llm.max_tokens))

        self._ensemble = MutationEnsemble(
            operators=all_ops,
            schedule=config.mutation.schedule,
        )

        # Generator (only if LLM client provided)
        self._generator = None
        if llm_client is not None:
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

    async def run(self) -> ExperimentResult:
        """Execute the full evolutionary loop and return results."""
        max_gen = self.config.evolution.max_generations

        # --- Generation 0: Seed ---
        logger.info("Seeding population with %d individuals", self.config.population.size)
        seed_genomes = self.backend.seed_population(self.config.population.size)

        seed_individuals = self._process_genomes(seed_genomes, generation=0)
        if not seed_individuals:
            logger.warning("No valid seed individuals produced; aborting")
            return self._build_result(generations_run=0)

        evaluated_seeds = await self._evaluator.evaluate_batch(seed_individuals)
        self._total_evaluations += len(evaluated_seeds)

        credited_seeds = self._assign_credits(evaluated_seeds)
        self._add_to_population(credited_seeds)

        logger.info(
            "Generation 0: pop_size=%d, best_fitness=%.4f",
            self.population.size,
            self._best_fitness(),
        )

        # --- Subsequent generations ---
        for gen in range(1, max_gen + 1):
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

            # Mutate
            offspring_genomes: list[tuple[str, str]] = []  # (genome, parent_hash)
            for parent in parents:
                operator = self._ensemble.select_operator()
                context = MutationContext(
                    generation=gen,
                    memory=self._memory,
                    guidance=self._memory.prompt_section(max_tokens=200),
                    temperature=self._temperature,
                    backend=self.backend,
                    credits=parent.credits,
                )
                try:
                    new_genome = await operator.apply(parent, context)
                    offspring_genomes.append((new_genome, parent.ir_hash))
                except Exception:
                    logger.debug("Mutation failed for operator %s", operator.name, exc_info=True)

            # Identity pipeline + dedup
            known_hashes = {ind.ir_hash for ind in self.population.get_all()}
            offspring_individuals: list[Individual] = []
            offspring_lineage: list[tuple[str, str]] = []  # (parent_hash, child_hash)

            for genome, parent_hash in offspring_genomes:
                ind = self._identity.process(genome)
                if ind is None:
                    continue
                # Set generation
                ind.generation = gen
                if self._identity.is_duplicate(ind.ir_hash, known_hashes):
                    continue
                known_hashes.add(ind.ir_hash)
                offspring_individuals.append(ind)
                offspring_lineage.append((parent_hash, ind.ir_hash))

            if not offspring_individuals:
                logger.debug("No novel offspring in generation %d", gen)
                self._memory.update([], gen)
                self._check_stagnation(gen)
                continue

            # Evaluate offspring
            evaluated_offspring = await self._evaluator.evaluate_batch(offspring_individuals)
            self._total_evaluations += len(evaluated_offspring)

            # Credit assignment
            credited_offspring = self._assign_credits(evaluated_offspring)

            # Assign behavior descriptors
            for ind in credited_offspring:
                if ind.ir is not None:
                    try:
                        ind.behavior_descriptor = self.backend.behavior_descriptor(
                            ind.ir, ind.diagnostics
                        )
                    except Exception:
                        pass

            # Survive
            pop_list = self.population.get_all()
            survivors = self._strategy.survive(
                pop_list, credited_offspring, self.config.population.elite_k
            )

            # Rebuild population from survivors
            self.population = PopulationManager(max_size=self.config.population.size)
            for ind in survivors:
                self.population.add(ind)

            # Archive offspring and lineage
            for ind in credited_offspring:
                await self.archive.store(ind)

            for parent_hash, child_hash in offspring_lineage:
                await self.archive.store_lineage(
                    parent_hash=parent_hash,
                    child_hash=child_hash,
                    operator_name="mutation",
                    generation=gen,
                )

            # Memory update
            self._memory.update(credited_offspring, gen)

            # Stagnation check
            self._check_stagnation(gen)

            logger.info(
                "Generation %d: pop_size=%d, best_fitness=%.4f, diversity=%.4f, evals=%d",
                gen,
                self.population.size,
                self._best_fitness(),
                self.population.diversity_entropy(),
                self._total_evaluations,
            )

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
        for ind in individuals:
            if ind.fitness is not None and ind.ir is not None:
                try:
                    ind.credits = self.backend.assign_credit(
                        ind.ir, ind.fitness, ind.diagnostics, None
                    )
                except Exception:
                    logger.debug("Credit assignment failed", exc_info=True)
        return individuals

    def _add_to_population(self, individuals: list[Individual]) -> None:
        """Add individuals to the population manager, deduplicating by ir_hash."""
        for ind in individuals:
            self.population.add(ind)

    def _best_fitness(self) -> float:
        """Return the best primary fitness in the population."""
        best = self.population.best(k=1)
        if best and best[0].fitness is not None:
            return best[0].fitness.primary
        return 0.0

    def _check_stagnation(self, generation: int) -> None:
        """Check for stagnation and trigger reflection if needed."""
        if self._memory.is_stagnant():
            logger.info(
                "Stagnation detected at generation %d — triggering reflection",
                generation,
            )
            self._reflected = True
            # Adjust temperature to encourage exploration
            self._temperature = min(self._temperature + 0.1, 1.5)

    def _build_result(self, generations_run: int) -> ExperimentResult:
        """Build the final experiment result."""
        best = self.population.best(k=1)
        best_individual = best[0] if best else None
        best_fitness = (
            best_individual.fitness.primary
            if (best_individual and best_individual.fitness)
            else 0.0
        )

        return ExperimentResult(
            best_individual=best_individual,
            best_fitness=best_fitness,
            generations_run=generations_run,
            total_evaluations=self._total_evaluations,
            cost=self._scheduler.tracker.summary(),
            archive_size=self.population.size,
            reflected=self._reflected,
        )
