# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for evoforge.core.ir — IRProtocol, BehaviorDimension, BehaviorSpaceConfig."""

from __future__ import annotations

import dataclasses

import pytest

from evoforge.core.ir import BehaviorDimension, BehaviorSpaceConfig, IRProtocol

# ---------------------------------------------------------------------------
# Helpers: concrete classes for structural subtyping tests
# ---------------------------------------------------------------------------


class _ConformingIR:
    """A class that has all four IRProtocol methods."""

    def canonicalize(self) -> _ConformingIR:
        return self

    def structural_hash(self) -> str:
        return "abc123"

    def serialize(self) -> str:
        return "{}"

    def complexity(self) -> int:
        return 1


class _NonConformingIR:
    """A class that is missing the `complexity` method."""

    def canonicalize(self) -> _NonConformingIR:
        return self

    def structural_hash(self) -> str:
        return "abc123"

    def serialize(self) -> str:
        return "{}"


# ---------------------------------------------------------------------------
# IRProtocol structural subtyping
# ---------------------------------------------------------------------------


class TestIRProtocol:
    def test_conforming_class_isinstance(self) -> None:
        """A class with all four methods should pass isinstance check."""
        obj = _ConformingIR()
        assert isinstance(obj, IRProtocol)

    def test_nonconforming_class_not_isinstance(self) -> None:
        """A class missing a method should NOT pass isinstance check."""
        obj = _NonConformingIR()
        assert not isinstance(obj, IRProtocol)


# ---------------------------------------------------------------------------
# BehaviorDimension
# ---------------------------------------------------------------------------


class TestBehaviorDimension:
    def test_construction(self) -> None:
        dim = BehaviorDimension(name="length", bins=["short", "medium", "long"])
        assert dim.name == "length"
        assert dim.bins == ["short", "medium", "long"]

    def test_frozen_immutability(self) -> None:
        dim = BehaviorDimension(name="length", bins=["short", "long"])
        with pytest.raises(dataclasses.FrozenInstanceError):
            dim.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BehaviorSpaceConfig
# ---------------------------------------------------------------------------


class TestBehaviorSpaceConfig:
    def test_construction_multiple_dimensions(self) -> None:
        d1 = BehaviorDimension(name="length", bins=["short", "long"])
        d2 = BehaviorDimension(name="tone", bins=["formal", "casual"])
        d3 = BehaviorDimension(name="complexity", bins=["low", "high"])

        config = BehaviorSpaceConfig(dimensions=(d1, d2, d3))

        assert len(config.dimensions) == 3
        assert config.dimensions[0].name == "length"
        assert config.dimensions[1].name == "tone"
        assert config.dimensions[2].name == "complexity"

    def test_frozen_immutability(self) -> None:
        d1 = BehaviorDimension(name="length", bins=["short", "long"])
        config = BehaviorSpaceConfig(dimensions=(d1,))
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.dimensions = ()  # type: ignore[misc]
