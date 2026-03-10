"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from evoforge.llm.client import LLMResponse


class LLMProvider(ABC):
    """Interface for LLM API providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Generate a response asynchronously."""

    @abstractmethod
    def generate_sync(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Generate a response synchronously."""

    @abstractmethod
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        **kwargs: Any,
    ) -> float:
        """Estimate USD cost for the given token counts and model."""
