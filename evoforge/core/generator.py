# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Validated generator: LLM-backed generation with a 4-stage validation pipeline.

Calls the LLM, extracts a genome from the raw response, parses it into IR,
validates structural constraints, then canonicalizes and hashes to produce
an Individual.  Retries up to ``max_attempts`` on any stage failure.
"""

from __future__ import annotations

import logging
from typing import Any

from evoforge.core.types import Individual

logger = logging.getLogger(__name__)


class ValidatedGenerator:
    """Generate validated Individuals through a 4-stage pipeline.

    Stages:
    1. Call LLM -> extract genome from raw text
    2. Parse genome into IR
    3. Validate IR structure
    4. Canonicalize, serialize, hash -> return Individual
    """

    def __init__(
        self,
        backend: Any,
        llm_client: Any,
        model: str,
        max_attempts: int = 3,
    ) -> None:
        self.backend = backend
        self.client = llm_client
        self.model = model
        self.max_attempts = max_attempts

    def generate(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> Individual | None:
        """Run the 4-stage validation pipeline synchronously.

        Returns an Individual on success, or None after *max_attempts* failures.
        """
        for attempt in range(self.max_attempts):
            try:
                individual = self._attempt(prompt, system, temperature, max_tokens)
                if individual is not None:
                    return individual
                logger.debug("Attempt %d/%d failed validation", attempt + 1, self.max_attempts)
            except Exception:
                logger.debug(
                    "Attempt %d/%d raised an exception",
                    attempt + 1,
                    self.max_attempts,
                    exc_info=True,
                )
        return None

    async def async_generate(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> Individual | None:
        """Run the 4-stage validation pipeline asynchronously.

        Returns an Individual on success, or None after *max_attempts* failures.
        """
        for attempt in range(self.max_attempts):
            try:
                individual = await self._async_attempt(prompt, system, temperature, max_tokens)
                if individual is not None:
                    return individual
                logger.debug("Attempt %d/%d failed validation", attempt + 1, self.max_attempts)
            except Exception:
                logger.debug(
                    "Attempt %d/%d raised an exception",
                    attempt + 1,
                    self.max_attempts,
                    exc_info=True,
                )
        return None

    # ------------------------------------------------------------------
    # Internal: single-attempt pipelines
    # ------------------------------------------------------------------

    def _attempt(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> Individual | None:
        """Execute one synchronous attempt of the 4-stage pipeline."""
        # Stage 1: Call LLM and extract genome
        response = self.client.generate(
            prompt=prompt,
            system=system,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_text: str = response.text

        genome: str | None = self.backend.extract_genome(raw_text)
        if genome is None:
            return None

        return self._validate_and_build(genome)

    async def _async_attempt(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> Individual | None:
        """Execute one asynchronous attempt of the 4-stage pipeline."""
        # Stage 1: Call LLM and extract genome
        response = await self.client.async_generate(
            prompt=prompt,
            system=system,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_text: str = response.text

        genome: str | None = self.backend.extract_genome(raw_text)
        if genome is None:
            return None

        return self._validate_and_build(genome)

    def _validate_and_build(self, genome: str) -> Individual | None:
        """Stages 2-4: parse, validate, canonicalize+hash."""
        # Stage 2: Parse
        ir = self.backend.parse(genome)
        if ir is None:
            return None

        # Stage 3: Validate structure
        violations: list[str] = self.backend.validate_structure(ir)
        if violations:
            return None

        # Stage 4: Canonicalize, serialize, hash
        canonical = ir.canonicalize()
        serialized = canonical.serialize()
        ir_hash = canonical.structural_hash()

        return Individual(
            genome=serialized,
            ir=canonical,
            ir_hash=ir_hash,
            generation=0,
        )
