# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Selection strategies for the evoforge evolutionary engine."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any

from evoforge.core.types import Individual


class SelectionStrategy(ABC):
    """Abstract base class for all selection strategies."""

    @abstractmethod
    def select(self, population: list[Individual], k: int) -> list[Individual]:
        """Select k parents from the population."""
        ...

    @abstractmethod
    def survive(
        self,
        population: list[Individual],
        offspring: list[Individual],
        elite_k: int,
    ) -> list[Individual]:
        """Combine population and offspring, select survivors preserving elite_k best."""
        ...


def _primary_fitness(ind: Individual) -> float:
    """Extract primary fitness, defaulting to -inf for unscored individuals.

    Infeasible individuals (Deb 2000) always rank below all feasible ones.
    """
    if ind.fitness is None:
        return -math.inf
    if not ind.fitness.feasible:
        return -math.inf
    return ind.fitness.primary


# ---------------------------------------------------------------------------
# ScalarTournament
# ---------------------------------------------------------------------------


class ScalarTournament(SelectionStrategy):
    """Tournament selection on fitness.primary (higher is better)."""

    def __init__(self, tournament_size: int = 3) -> None:
        self.tournament_size = tournament_size

    def _tournament(self, population: list[Individual]) -> Individual:
        """Run a single tournament and return the winner."""
        contestants = random.choices(population, k=self.tournament_size)
        return max(contestants, key=_primary_fitness)

    def select(self, population: list[Individual], k: int) -> list[Individual]:
        """Select k parents via tournament selection."""
        return [self._tournament(population) for _ in range(k)]

    def survive(
        self,
        population: list[Individual],
        offspring: list[Individual],
        elite_k: int,
    ) -> list[Individual]:
        """Combine and select survivors, preserving elite_k best by primary fitness."""
        combined = population + offspring
        target_size = len(population)

        # Sort by primary fitness descending
        sorted_combined = sorted(combined, key=_primary_fitness, reverse=True)

        # Keep top elite_k as elites
        elites = sorted_combined[:elite_k]

        # Fill remaining from combined pool via tournament
        remaining = target_size - len(elites)
        fill = [self._tournament(combined) for _ in range(remaining)]

        return elites + fill


# ---------------------------------------------------------------------------
# ParetoNSGA2
# ---------------------------------------------------------------------------


def _non_dominated_sort(population: list[Individual]) -> list[list[Individual]]:
    """Non-dominated sorting: partition population into Pareto fronts."""
    n = len(population)
    if n == 0:
        return []

    # domination_count[i] = number of individuals that dominate i
    domination_count: list[int] = [0] * n
    # dominated_set[i] = set of indices that i dominates
    dominated_set: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        fi = population[i].fitness
        for j in range(i + 1, n):
            fj = population[j].fitness
            if fi is not None and fj is not None:
                if fi.dominates(fj):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif fj.dominates(fi):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

    fronts: list[list[Individual]] = []
    current_front_indices = [i for i in range(n) if domination_count[i] == 0]

    while current_front_indices:
        front = [population[i] for i in current_front_indices]
        fronts.append(front)
        next_front_indices: list[int] = []
        for i in current_front_indices:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front_indices.append(j)
        current_front_indices = next_front_indices

    return fronts


def _crowding_distance(front: list[Individual]) -> dict[str, float]:
    """Compute crowding distance for each individual in a front.

    Returns a dict mapping individual id to crowding distance.
    """
    n = len(front)
    distances: dict[str, float] = {ind.id: 0.0 for ind in front}

    if n <= 2:
        for ind in front:
            distances[ind.id] = math.inf
        return distances

    # Gather objective names
    objectives: list[str] = ["__primary__"]
    if front[0].fitness is not None:
        objectives += sorted(
            k for k, v in front[0].fitness.auxiliary.items() if isinstance(v, (int, float))
        )

    for obj in objectives:
        # Sort by this objective
        def _obj_value(ind: Individual, _obj: str = obj) -> float:
            if ind.fitness is None:
                return -math.inf
            if _obj == "__primary__":
                return ind.fitness.primary
            val = ind.fitness.auxiliary.get(_obj, 0.0)
            return val if isinstance(val, (int, float)) else 0.0

        sorted_front = sorted(front, key=_obj_value)
        obj_min = _obj_value(sorted_front[0])
        obj_max = _obj_value(sorted_front[-1])
        obj_range = obj_max - obj_min

        # Boundary individuals get infinite distance
        distances[sorted_front[0].id] = math.inf
        distances[sorted_front[-1].id] = math.inf

        if obj_range > 0:
            for i in range(1, n - 1):
                distances[sorted_front[i].id] += (
                    _obj_value(sorted_front[i + 1]) - _obj_value(sorted_front[i - 1])
                ) / obj_range

    return distances


def _nsga2_select(population: list[Individual], k: int) -> list[Individual]:
    """NSGA-II selection: non-dominated sort, fill k slots front by front."""
    fronts = _non_dominated_sort(population)
    selected: list[Individual] = []

    for front in fronts:
        if len(selected) + len(front) <= k:
            selected.extend(front)
        else:
            # Need to break this front with crowding distance
            remaining = k - len(selected)
            cd = _crowding_distance(front)
            sorted_front = sorted(front, key=lambda ind: cd[ind.id], reverse=True)
            selected.extend(sorted_front[:remaining])
            break

    return selected


class ParetoNSGA2(SelectionStrategy):
    """NSGA-II selection with non-dominated sorting and crowding distance."""

    def select(self, population: list[Individual], k: int) -> list[Individual]:
        """Select k individuals using NSGA-II."""
        return _nsga2_select(population, k)

    def survive(
        self,
        population: list[Individual],
        offspring: list[Individual],
        elite_k: int,
    ) -> list[Individual]:
        """Survive using NSGA-II on combined pop+offspring, preserving elites."""
        combined = population + offspring
        target_size = len(population)

        # Guarantee elites by primary fitness
        sorted_by_primary = sorted(combined, key=_primary_fitness, reverse=True)
        elites = sorted_by_primary[:elite_k]
        elite_ids = {ind.id for ind in elites}

        # Fill remaining via NSGA-II from combined pool (excluding already-picked elites)
        remaining_pool = [ind for ind in combined if ind.id not in elite_ids]
        remaining_needed = target_size - len(elites)

        if remaining_needed > 0 and remaining_pool:
            fill = _nsga2_select(remaining_pool, remaining_needed)
        else:
            fill = []

        return elites + fill


# ---------------------------------------------------------------------------
# Lexicase
# ---------------------------------------------------------------------------


class Lexicase(SelectionStrategy):
    """Epsilon-lexicase selection on fitness.auxiliary keys."""

    def _epsilon_lexicase_one(self, population: list[Individual]) -> Individual:
        """Run one epsilon-lexicase selection pass."""
        # Get auxiliary keys from the first individual with fitness
        aux_keys: list[str] = []
        for ind in population:
            if ind.fitness is not None and ind.fitness.auxiliary:
                aux_keys = [
                    k for k, v in ind.fitness.auxiliary.items() if isinstance(v, (int, float))
                ]
                break

        if not aux_keys:
            # Fallback: random selection
            return random.choice(population)

        cases = list(aux_keys)
        random.shuffle(cases)

        candidates = list(population)

        for case in cases:
            if len(candidates) <= 1:
                break

            # Get values for this case
            values: list[float] = []
            for ind in candidates:
                if ind.fitness is not None:
                    val = ind.fitness.auxiliary.get(case, 0.0)
                    values.append(val if isinstance(val, (int, float)) else 0.0)
                else:
                    values.append(-math.inf)

            # Compute epsilon = median absolute deviation
            epsilon = _median_absolute_deviation(values)

            best_val = max(values)
            threshold = best_val - epsilon

            # Filter candidates within epsilon of best
            new_candidates: list[Individual] = []
            for ind, val in zip(candidates, values):
                if val >= threshold:
                    new_candidates.append(ind)
            candidates = new_candidates

        return random.choice(candidates)

    def select(self, population: list[Individual], k: int) -> list[Individual]:
        """Select k individuals via epsilon-lexicase."""
        return [self._epsilon_lexicase_one(population) for _ in range(k)]

    def survive(
        self,
        population: list[Individual],
        offspring: list[Individual],
        elite_k: int,
    ) -> list[Individual]:
        """Elites by primary + fill via lexicase select."""
        combined = population + offspring
        target_size = len(population)

        sorted_by_primary = sorted(combined, key=_primary_fitness, reverse=True)
        elites = sorted_by_primary[:elite_k]
        elite_ids = {ind.id for ind in elites}

        remaining_needed = target_size - len(elites)
        # Sample with deduplication to prevent population collapse
        fill: list[Individual] = []
        seen = set(elite_ids)
        max_attempts = remaining_needed * 5
        attempts = 0
        while len(fill) < remaining_needed and attempts < max_attempts:
            candidate = self._epsilon_lexicase_one(combined)
            if candidate.ir_hash not in seen:
                seen.add(candidate.ir_hash)
                fill.append(candidate)
            attempts += 1

        return elites + fill


def _median_absolute_deviation(values: list[float]) -> float:
    """Compute the median absolute deviation of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    deviations = [abs(v - median) for v in values]
    sorted_devs = sorted(deviations)
    m = len(sorted_devs)
    if m % 2 == 1:
        mad = sorted_devs[m // 2]
    else:
        mad = (sorted_devs[m // 2 - 1] + sorted_devs[m // 2]) / 2.0
    return mad


# ---------------------------------------------------------------------------
# MAPElites
# ---------------------------------------------------------------------------


class MAPElites(SelectionStrategy):
    """MAP-Elites grid selection strategy.

    Stores the best individual per cell based on behavior_descriptor.
    """

    def __init__(self, grid_dims: dict[str, list[str]]) -> None:
        self.grid_dims = grid_dims
        # Compute total cells
        self._dim_names = list(grid_dims.keys())
        self._dim_labels = [grid_dims[name] for name in self._dim_names]
        self._total_cells = 1
        for labels in self._dim_labels:
            self._total_cells *= len(labels)

        # Grid: maps behavior descriptor tuple -> Individual
        self._grid: dict[tuple[Any, ...], Individual] = {}

    def _descriptor_to_key(self, descriptor: tuple[Any, ...]) -> tuple[Any, ...] | None:
        """Validate and normalize a behavior descriptor to a grid key.

        Returns None if the descriptor doesn't map to a valid cell.
        """
        if len(descriptor) != len(self._dim_names):
            return None
        key_parts: list[Any] = []
        for i, label in enumerate(descriptor):
            if label not in self._dim_labels[i]:
                return None
            key_parts.append(label)
        return tuple(key_parts)

    def coverage(self) -> float:
        """Fraction of grid cells that are occupied."""
        if self._total_cells == 0:
            return 0.0
        return len(self._grid) / self._total_cells

    def _insert(self, ind: Individual) -> None:
        """Insert an individual into the grid if it has a valid behavior descriptor."""
        if ind.behavior_descriptor is None:
            return
        key = self._descriptor_to_key(ind.behavior_descriptor)
        if key is None:
            return

        existing = self._grid.get(key)
        if existing is None or _primary_fitness(ind) > _primary_fitness(existing):
            self._grid[key] = ind

    def select(self, population: list[Individual], k: int) -> list[Individual]:
        """Randomly sample k individuals from occupied grid cells."""
        occupied = list(self._grid.values())
        if not occupied:
            # Fallback to population if grid is empty
            if population:
                return random.choices(population, k=k)
            return []
        return random.choices(occupied, k=k)

    def survive(
        self,
        population: list[Individual],
        offspring: list[Individual],
        elite_k: int,
    ) -> list[Individual]:
        """Insert all individuals into the grid; return grid contents."""
        for ind in population:
            self._insert(ind)
        for ind in offspring:
            self._insert(ind)

        return list(self._grid.values())
