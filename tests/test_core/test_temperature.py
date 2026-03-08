# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for EvolutionEngine._compute_temperature static method."""

from __future__ import annotations

import pytest

from evoforge.core.engine import EvolutionEngine


class TestComputeTemperature:
    """Tests for linear temperature scheduling."""

    def test_generation_zero_returns_start(self) -> None:
        """At generation 0, temperature should equal temperature_start."""
        result = EvolutionEngine._compute_temperature(
            generation=0,
            max_generations=100,
            start=1.0,
            end=0.3,
            schedule="linear",
        )
        assert result == pytest.approx(1.0)

    def test_max_generation_returns_end(self) -> None:
        """At max_generations, temperature should equal temperature_end."""
        result = EvolutionEngine._compute_temperature(
            generation=100,
            max_generations=100,
            start=1.0,
            end=0.3,
            schedule="linear",
        )
        assert result == pytest.approx(0.3)

    def test_midpoint_returns_halfway(self) -> None:
        """At midpoint (50/100), temperature should be halfway between start and end."""
        result = EvolutionEngine._compute_temperature(
            generation=50,
            max_generations=100,
            start=1.0,
            end=0.3,
            schedule="linear",
        )
        assert result == pytest.approx(0.65)

    def test_fixed_schedule_returns_start(self) -> None:
        """With schedule='fixed', always return the start value regardless of generation."""
        for gen in [0, 25, 50, 75, 100]:
            result = EvolutionEngine._compute_temperature(
                generation=gen,
                max_generations=100,
                start=1.0,
                end=0.3,
                schedule="fixed",
            )
            assert result == pytest.approx(1.0), f"Failed at generation {gen}"

    def test_max_generations_zero_returns_start(self) -> None:
        """With max_generations=0, return start value (avoid division by zero)."""
        result = EvolutionEngine._compute_temperature(
            generation=0,
            max_generations=0,
            start=1.0,
            end=0.3,
            schedule="linear",
        )
        assert result == pytest.approx(1.0)

    def test_generation_beyond_max_clamps_to_end(self) -> None:
        """Generation exceeding max_generations should clamp to end value."""
        result = EvolutionEngine._compute_temperature(
            generation=200,
            max_generations=100,
            start=1.0,
            end=0.3,
            schedule="linear",
        )
        assert result == pytest.approx(0.3)
