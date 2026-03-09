# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""LLM-powered mutation operators for the evoforge evolutionary engine.

Provides two operators that use an LLM to generate new genome candidates:
:class:`LLMMutate` rewrites a single parent, and :class:`LLMCrossover`
combines two parents guided by credit analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from evoforge.core.mutation import MutationContext, MutationOperator
from evoforge.core.types import Individual
from evoforge.llm.batch import get_batch_collector

logger = logging.getLogger(__name__)


class LLMMutate(MutationOperator):
    """Mutation operator that asks an LLM to improve a parent genome.

    The backend's :meth:`format_mutation_prompt` and :meth:`system_prompt`
    methods supply the domain-specific prompting, while this operator
    handles the LLM call and genome extraction.
    """

    def __init__(self, client: Any, model: str, max_tokens: int = 4096) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "llm_mutate"

    @property
    def cost(self) -> Literal["cheap", "llm"]:
        return "llm"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        """Produce a mutated genome by prompting an LLM."""
        prompt = context.backend.format_mutation_prompt(parent, context)
        system = context.backend.system_prompt()

        collector = get_batch_collector()
        if collector is not None:
            future = collector.register(
                prompt, system, self._model, context.temperature, self._max_tokens
            )
            response = await future
        else:
            response = await self._client.async_generate(
                prompt,
                system,
                self._model,
                context.temperature,
                self._max_tokens,
            )

        if response is None:
            logger.warning("LLMMutate: batch request failed, falling back to parent")
            return parent.genome

        genome: str | None = context.backend.extract_genome(response.text)
        if genome is not None:
            return genome

        logger.warning("LLMMutate: genome extraction failed, falling back to parent")
        return parent.genome


class LLMCrossover(MutationOperator):
    """Crossover operator that asks an LLM to combine two parent genomes.

    The second parent is expected in ``context.guidance_individual``.  When
    no guidance individual is available the operator falls back to a standard
    mutation prompt.
    """

    def __init__(self, client: Any, model: str, max_tokens: int = 4096) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "llm_crossover"

    @property
    def cost(self) -> Literal["cheap", "llm"]:
        return "llm"

    async def apply(self, parent: Individual, context: MutationContext) -> str:
        """Produce a child genome by combining two parents via LLM."""
        if context.guidance_individual is not None:
            prompt = context.backend.format_crossover_prompt(
                parent, context.guidance_individual, context
            )
        else:
            prompt = context.backend.format_mutation_prompt(parent, context)

        system = context.backend.system_prompt()

        collector = get_batch_collector()
        if collector is not None:
            future = collector.register(
                prompt, system, self._model, context.temperature, self._max_tokens
            )
            response = await future
        else:
            response = await self._client.async_generate(
                prompt,
                system,
                self._model,
                context.temperature,
                self._max_tokens,
            )

        if response is None:
            logger.warning("LLMCrossover: batch request failed, falling back to parent")
            return parent.genome

        genome: str | None = context.backend.extract_genome(response.text)
        if genome is not None:
            return genome

        logger.warning("LLMCrossover: extraction failed, falling back to parent")
        return parent.genome
