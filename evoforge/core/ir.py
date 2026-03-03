"""Intermediate-representation primitives for evoforge.

Defines the structural protocol every IR node must satisfy, plus the
behaviour-space configuration dataclasses used by MAP-Elites archives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self, runtime_checkable


@runtime_checkable
class IRProtocol(Protocol):
    """Structural protocol that every IR node must implement."""

    def canonicalize(self) -> Self: ...

    def structural_hash(self) -> str: ...

    def serialize(self) -> str: ...

    def complexity(self) -> int: ...


@dataclass(frozen=True)
class BehaviorDimension:
    """A single named axis of the behaviour space, divided into discrete bins."""

    name: str
    bins: list[str]


@dataclass(frozen=True)
class BehaviorSpaceConfig:
    """Configuration for a multi-dimensional behaviour space (MAP-Elites archive)."""

    dimensions: tuple[BehaviorDimension, ...]
