"""Population management for the evoforge evolutionary engine."""

from __future__ import annotations

import math
from collections import Counter

from evoforge.core.types import Individual


class PopulationManager:
    """Manages a population of individuals keyed by ir_hash, with fitness-based
    ordering and diversity metrics."""

    def __init__(self, max_size: int = 100) -> None:
        self._individuals: dict[str, Individual] = {}
        self.max_size = max_size

    # -- Mutation ----------------------------------------------------------

    def add(self, individual: Individual) -> bool:
        """Add an individual to the population.

        Returns False if ir_hash already exists (duplicate rejected).
        Returns True on successful add.
        """
        if individual.ir_hash in self._individuals:
            return False
        self._individuals[individual.ir_hash] = individual
        return True

    def remove_worst(self, k: int = 1) -> list[Individual]:
        """Remove and return the *k* worst individuals by fitness.primary (lowest).

        Individuals without fitness are considered worst.  If *k* exceeds the
        population size, all individuals are removed.
        """
        sorted_inds = self._sorted_by_fitness(ascending=True)
        to_remove = sorted_inds[: min(k, len(sorted_inds))]
        for ind in to_remove:
            del self._individuals[ind.ir_hash]
        return to_remove

    # -- Queries -----------------------------------------------------------

    def get_all(self) -> list[Individual]:
        """Return all individuals as a list."""
        return list(self._individuals.values())

    def best(self, k: int = 1) -> list[Individual]:
        """Return *k* best individuals by fitness.primary (highest).

        Individuals without fitness are sorted last.  If *k* exceeds the
        population size, all individuals are returned (best-first).
        """
        sorted_inds = self._sorted_by_fitness(ascending=False)
        return sorted_inds[: min(k, len(sorted_inds))]

    @property
    def size(self) -> int:
        """Return current population size."""
        return len(self._individuals)

    def contains(self, ir_hash: str) -> bool:
        """Check if *ir_hash* is in population."""
        return ir_hash in self._individuals

    # -- Diversity ---------------------------------------------------------

    def diversity_entropy(self) -> float:
        """Compute Shannon entropy of behavior descriptors.

        If no behavior descriptors exist in the population, return 0.0.
        Counts occurrences of each unique descriptor and computes
        ``-sum(p * log(p))``.
        """
        descriptors: list[tuple[object, ...]] = [
            ind.behavior_descriptor
            for ind in self._individuals.values()
            if ind.behavior_descriptor is not None
        ]
        if not descriptors:
            return 0.0

        counts = Counter(descriptors)
        total = len(descriptors)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

    # -- Internal ----------------------------------------------------------

    def _sorted_by_fitness(self, *, ascending: bool) -> list[Individual]:
        """Return individuals sorted by fitness.primary.

        Individuals without fitness get ``-inf`` so they sort as worst.
        """

        def _key(ind: Individual) -> float:
            if ind.fitness is None:
                return float("-inf")
            return ind.fitness.primary

        return sorted(
            self._individuals.values(),
            key=_key,
            reverse=not ascending,
        )
