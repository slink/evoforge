"""Tests for evoforge.core.selection – selection strategy implementations."""

from __future__ import annotations

from collections import Counter

import pytest

from evoforge.core.selection import (
    Lexicase,
    MAPElites,
    ParetoNSGA2,
    ScalarTournament,
    SelectionStrategy,
)
from evoforge.core.types import Fitness, Individual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ind(
    primary: float,
    *,
    auxiliary: dict[str, float] | None = None,
    behavior_descriptor: tuple[object, ...] | None = None,
    genome: str = "",
    generation: int = 0,
) -> Individual:
    """Create a minimal Individual with known fitness values."""
    aux = auxiliary if auxiliary is not None else {}
    return Individual(
        genome=genome or f"g_{primary}",
        ir=None,
        ir_hash=f"hash_{primary}_{id(aux)}",
        fitness=Fitness(primary=primary, auxiliary=aux, constraints={}, feasible=True),
        diagnostics=None,
        credits=[],
        lineage={},
        generation=generation,
        behavior_descriptor=behavior_descriptor,
    )


# ---------------------------------------------------------------------------
# SelectionStrategy ABC
# ---------------------------------------------------------------------------


class TestSelectionStrategyABC:
    """Verify that SelectionStrategy cannot be instantiated directly."""

    def test_abc_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            SelectionStrategy()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# ScalarTournament
# ---------------------------------------------------------------------------


class TestScalarTournament:
    """Higher fitness individuals should be selected more often."""

    def test_higher_fitness_selected_more_often(self) -> None:
        """Statistical bias test: the best individual should appear most often
        when we run many tournament selections."""
        best = _make_ind(100.0, genome="best")
        mediocre = _make_ind(50.0, genome="mediocre")
        worst = _make_ind(1.0, genome="worst")
        population = [best, mediocre, worst]

        strategy = ScalarTournament(tournament_size=2)
        selected = strategy.select(population, k=300)

        counts = Counter(ind.genome for ind in selected)
        # The best individual should be selected strictly more than the worst
        assert counts["best"] > counts["worst"]

    def test_select_returns_k_individuals(self) -> None:
        population = [_make_ind(float(i)) for i in range(10)]
        strategy = ScalarTournament(tournament_size=3)
        selected = strategy.select(population, k=5)
        assert len(selected) == 5

    def test_tournament_size_default(self) -> None:
        strategy = ScalarTournament()
        assert strategy.tournament_size == 3

    def test_survive_preserves_elites(self) -> None:
        """Elitism: the top elite_k by primary fitness must survive."""
        population = [_make_ind(float(i)) for i in range(10)]
        offspring = [_make_ind(float(i + 10)) for i in range(5)]
        strategy = ScalarTournament(tournament_size=2)

        survivors = strategy.survive(population, offspring, elite_k=3)

        # The top 3 by primary fitness are 14.0, 13.0, 12.0
        elite_primaries = {14.0, 13.0, 12.0}
        survivor_primaries = {ind.fitness.primary for ind in survivors if ind.fitness}
        assert elite_primaries.issubset(survivor_primaries)

    def test_survive_returns_population_size(self) -> None:
        population = [_make_ind(float(i)) for i in range(10)]
        offspring = [_make_ind(float(i + 10)) for i in range(5)]
        strategy = ScalarTournament(tournament_size=2)

        survivors = strategy.survive(population, offspring, elite_k=2)
        # Should return len(population) survivors
        assert len(survivors) == len(population)


# ---------------------------------------------------------------------------
# ParetoNSGA2
# ---------------------------------------------------------------------------


class TestParetoNSGA2:
    """Non-dominated individuals must appear in the first Pareto front."""

    def test_pareto_front_correctness(self) -> None:
        """Non-dominated individuals should be selected preferentially."""
        # Pareto front: high primary + low aux, or low primary + high aux
        front_a = _make_ind(10.0, auxiliary={"x": 1.0}, genome="front_a")
        front_b = _make_ind(1.0, auxiliary={"x": 10.0}, genome="front_b")
        # Dominated by both front members
        dominated = _make_ind(0.5, auxiliary={"x": 0.5}, genome="dominated")

        population = [front_a, front_b, dominated]

        strategy = ParetoNSGA2()
        selected = strategy.select(population, k=2)

        selected_genomes = {ind.genome for ind in selected}
        # Both non-dominated individuals should be selected
        assert "front_a" in selected_genomes
        assert "front_b" in selected_genomes

    def test_select_returns_k(self) -> None:
        population = [_make_ind(float(i), auxiliary={"x": float(10 - i)}) for i in range(10)]
        strategy = ParetoNSGA2()
        selected = strategy.select(population, k=5)
        assert len(selected) == 5

    def test_survive_preserves_elites(self) -> None:
        """Elitism in NSGA2 survive: top elite_k by primary must be present."""
        population = [_make_ind(float(i), auxiliary={"x": float(10 - i)}) for i in range(10)]
        offspring = [_make_ind(float(i + 10), auxiliary={"x": float(i)}) for i in range(5)]
        strategy = ParetoNSGA2()

        survivors = strategy.survive(population, offspring, elite_k=2)
        survivor_primaries = {ind.fitness.primary for ind in survivors if ind.fitness}
        assert 14.0 in survivor_primaries
        assert 13.0 in survivor_primaries


# ---------------------------------------------------------------------------
# Lexicase
# ---------------------------------------------------------------------------


class TestLexicase:
    """Epsilon-lexicase on auxiliary fitness keys."""

    def test_specialist_selection(self) -> None:
        """An individual that is the best on one auxiliary metric should be
        selectable when that metric comes first in the shuffle order."""
        # specialist_a is best on metric "speed" but bad on "accuracy"
        specialist_a = _make_ind(
            5.0, auxiliary={"speed": 100.0, "accuracy": 1.0}, genome="speed_specialist"
        )
        # specialist_b is best on "accuracy" but bad on "speed"
        specialist_b = _make_ind(
            5.0, auxiliary={"speed": 1.0, "accuracy": 100.0}, genome="accuracy_specialist"
        )
        # mediocre is average on both
        mediocre = _make_ind(5.0, auxiliary={"speed": 50.0, "accuracy": 50.0}, genome="mediocre")

        population = [specialist_a, specialist_b, mediocre]
        strategy = Lexicase()

        # Run many selections; both specialists should appear
        selected = strategy.select(population, k=200)
        genomes = Counter(ind.genome for ind in selected)

        # Both specialists should be selected at least sometimes
        assert genomes["speed_specialist"] > 0
        assert genomes["accuracy_specialist"] > 0

    def test_select_returns_k(self) -> None:
        population = [
            _make_ind(float(i), auxiliary={"a": float(i), "b": float(10 - i)}) for i in range(10)
        ]
        strategy = Lexicase()
        selected = strategy.select(population, k=7)
        assert len(selected) == 7

    def test_survive_preserves_elites(self) -> None:
        """Survive should preserve top elite_k by primary fitness."""
        population = [_make_ind(float(i), auxiliary={"a": float(i)}) for i in range(10)]
        offspring = [_make_ind(float(i + 10), auxiliary={"a": float(i)}) for i in range(5)]
        strategy = Lexicase()

        survivors = strategy.survive(population, offspring, elite_k=2)
        survivor_primaries = {ind.fitness.primary for ind in survivors if ind.fitness}
        assert 14.0 in survivor_primaries
        assert 13.0 in survivor_primaries


# ---------------------------------------------------------------------------
# MAPElites
# ---------------------------------------------------------------------------


class TestMAPElites:
    """MAP-Elites grid: cell occupancy and coverage."""

    def test_cell_occupancy_and_coverage(self) -> None:
        """Inserting individuals fills cells, coverage increases."""
        grid_dims: dict[str, list[str]] = {
            "strategy": ["a", "b"],
            "length": ["short", "long"],
        }
        strategy = MAPElites(grid_dims=grid_dims)

        assert strategy.coverage() == 0.0

        population: list[Individual] = []
        offspring = [
            _make_ind(10.0, behavior_descriptor=("a", "short"), genome="ind1"),
            _make_ind(20.0, behavior_descriptor=("b", "long"), genome="ind2"),
        ]

        strategy.survive(population, offspring, elite_k=0)

        # 2 out of 4 cells occupied
        assert strategy.coverage() == pytest.approx(0.5)

    def test_better_individual_replaces_in_cell(self) -> None:
        """A better individual should replace a worse one in the same cell."""
        grid_dims: dict[str, list[str]] = {
            "strategy": ["a"],
            "length": ["short"],
        }
        strategy = MAPElites(grid_dims=grid_dims)

        weak = _make_ind(5.0, behavior_descriptor=("a", "short"), genome="weak")
        strong = _make_ind(50.0, behavior_descriptor=("a", "short"), genome="strong")

        strategy.survive([], [weak], elite_k=0)
        strategy.survive([], [strong], elite_k=0)

        selected = strategy.select([], k=1)
        assert selected[0].genome == "strong"

    def test_select_from_occupied_cells(self) -> None:
        """Select should return k individuals from occupied cells."""
        grid_dims: dict[str, list[str]] = {
            "strategy": ["a", "b", "c"],
            "length": ["short", "long"],
        }
        strategy = MAPElites(grid_dims=grid_dims)

        inds = [
            _make_ind(float(i), behavior_descriptor=("a", "short"), genome=f"g{i}")
            for i in range(3)
        ]
        # Only the best (2.0) should remain in ("a", "short")
        for ind in inds:
            strategy.survive([], [ind], elite_k=0)

        # Also add one to a different cell
        other = _make_ind(99.0, behavior_descriptor=("b", "long"), genome="other")
        strategy.survive([], [other], elite_k=0)

        selected = strategy.select([], k=10)
        assert len(selected) == 10
        # All selected should be from occupied cells
        genomes = {ind.genome for ind in selected}
        assert genomes.issubset({"g2", "other"})  # best in each cell

    def test_coverage_full(self) -> None:
        """Coverage should be 1.0 when all cells are filled."""
        grid_dims: dict[str, list[str]] = {
            "dim": ["x", "y"],
        }
        strategy = MAPElites(grid_dims=grid_dims)

        strategy.survive(
            [],
            [
                _make_ind(1.0, behavior_descriptor=("x",)),
                _make_ind(2.0, behavior_descriptor=("y",)),
            ],
            elite_k=0,
        )

        assert strategy.coverage() == pytest.approx(1.0)

    def test_ignores_individuals_without_behavior_descriptor(self) -> None:
        """Individuals with no behavior_descriptor should be skipped."""
        grid_dims: dict[str, list[str]] = {
            "dim": ["x", "y"],
        }
        strategy = MAPElites(grid_dims=grid_dims)

        no_bd = _make_ind(100.0, behavior_descriptor=None)
        strategy.survive([], [no_bd], elite_k=0)

        assert strategy.coverage() == 0.0


# ---------------------------------------------------------------------------
# Elitism cross-strategy
# ---------------------------------------------------------------------------


class TestElitism:
    """All strategies with survive must preserve top-k by primary fitness."""

    @pytest.mark.parametrize(
        "strategy",
        [
            ScalarTournament(tournament_size=2),
            ParetoNSGA2(),
            Lexicase(),
        ],
    )
    def test_survive_always_preserves_top_k(self, strategy: SelectionStrategy) -> None:
        """Top elite_k individuals by primary fitness are always preserved."""
        population = [_make_ind(float(i)) for i in range(20)]
        offspring = [_make_ind(float(i + 20)) for i in range(10)]

        elite_k = 5
        survivors = strategy.survive(population, offspring, elite_k=elite_k)

        # The top 5 by primary are 29, 28, 27, 26, 25
        expected_elites = {29.0, 28.0, 27.0, 26.0, 25.0}
        survivor_primaries = {ind.fitness.primary for ind in survivors if ind.fitness}
        assert expected_elites.issubset(survivor_primaries)
