# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Identity pipeline: parse, canonicalize, hash, and deduplicate genomes."""

from __future__ import annotations

from typing import Any, Protocol

from evoforge.core.ir import IRProtocol
from evoforge.core.types import Individual


class Backend(Protocol):
    """Protocol for a backend that can parse genome strings into IR nodes."""

    def parse(self, genome: str) -> IRProtocol | None: ...


class IdentityPipeline:
    """Takes a backend and processes a genome string through:
    parse -> canonicalize -> hash, producing an Individual."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def process(self, genome: str) -> Individual | None:
        """Parse a genome via the backend, canonicalize, hash, and return an
        Individual.  Returns None if parsing fails."""
        ir: IRProtocol | None = self._backend.parse(genome)
        if ir is None:
            return None

        canonical_ir = ir.canonicalize()
        ir_hash = canonical_ir.structural_hash()

        return Individual(
            genome=genome,
            ir=canonical_ir,
            ir_hash=ir_hash,
            generation=0,
        )

    def is_duplicate(self, ir_hash: str, known_hashes: set[str]) -> bool:
        """Return True if *ir_hash* is already in the known set."""
        return ir_hash in known_hashes
