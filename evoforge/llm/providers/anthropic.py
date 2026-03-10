"""Anthropic API provider implementation."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import anthropic
from anthropic.types import TextBlockParam

from evoforge.llm.client import LLMResponse
from evoforge.llm.providers.base import LLMProvider
from evoforge.llm.retry import compute_delay

logger = logging.getLogger(__name__)

# Pricing per million tokens: (input_cost, output_cost) in USD
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "sonnet": (3.0, 15.0),
    "haiku": (0.25, 1.25),
    "opus": (15.0, 75.0),
}

_DEFAULT_PRICING = _MODEL_PRICING["sonnet"]


def _pricing_for_model(model: str) -> tuple[float, float]:
    """Return (input_cost, output_cost) per million tokens for the given model name."""
    model_lower = model.lower()
    for key, pricing in _MODEL_PRICING.items():
        if key in model_lower:
            return pricing
    return _DEFAULT_PRICING


class AnthropicProvider(LLMProvider):
    """LLM provider backed by the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 6,
        base_delay: float = 2.0,
        max_delay: float = 120.0,
        prompt_caching: bool = True,
    ) -> None:
        self._api_key = api_key
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._prompt_caching = prompt_caching
        self._sync_client: anthropic.Anthropic | None = None
        self._async_client: anthropic.AsyncAnthropic | None = None

    def _format_system(self, system: str) -> str | list[TextBlockParam]:
        """Format system prompt, optionally adding cache_control for prompt caching."""
        if not self._prompt_caching:
            return system
        return [
            TextBlockParam(
                type="text",
                text=system,
                cache_control={"type": "ephemeral"},
            )
        ]

    @staticmethod
    def _extract_cache_tokens(usage: Any) -> tuple[int, int]:
        """Extract cache token counts from API usage, defaulting to 0."""
        cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", None) or 0
        return cache_read, cache_creation

    def get_sync_client(self) -> anthropic.Anthropic:
        """Return (lazily created) synchronous Anthropic client."""
        if self._sync_client is None:
            self._sync_client = anthropic.Anthropic(api_key=self._api_key)
        return self._sync_client

    def get_async_client(self) -> anthropic.AsyncAnthropic:
        """Return (lazily created) asynchronous Anthropic client."""
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._async_client

    async def generate(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call the Anthropic API asynchronously with exponential-backoff retry."""
        client = self.get_async_client()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=self._format_system(system),
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text  # type: ignore[union-attr]
                cache_read, cache_creation = self._extract_cache_tokens(response.usage)
                return LLMResponse(
                    text=text,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    model=model,
                    cache_read_tokens=cache_read,
                    cache_creation_tokens=cache_creation,
                )
            except (anthropic.RateLimitError, anthropic.APIError) as exc:
                last_exc = exc
                delay = compute_delay(attempt, self._base_delay, self._max_delay)
                logger.warning(
                    "Anthropic API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        msg = f"Anthropic call failed after {self._max_retries} retries"
        raise RuntimeError(msg) from last_exc

    def generate_sync(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call the Anthropic API synchronously with exponential-backoff retry."""
        client = self.get_sync_client()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=self._format_system(system),
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text  # type: ignore[union-attr]
                cache_read, cache_creation = self._extract_cache_tokens(response.usage)
                return LLMResponse(
                    text=text,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    model=model,
                    cache_read_tokens=cache_read,
                    cache_creation_tokens=cache_creation,
                )
            except (anthropic.RateLimitError, anthropic.APIError) as exc:
                last_exc = exc
                delay = compute_delay(attempt, self._base_delay, self._max_delay)
                logger.warning(
                    "Anthropic API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)

        msg = f"Anthropic call failed after {self._max_retries} retries"
        raise RuntimeError(msg) from last_exc

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        **kwargs: Any,
    ) -> float:
        """Estimate USD cost for the given token counts and model.

        Supports optional kwargs:
            cache_read_tokens: charged at 10% of input rate
            cache_creation_tokens: charged at 125% of input rate
        """
        cache_read_tokens: int = int(kwargs.get("cache_read_tokens", 0))
        cache_creation_tokens: int = int(kwargs.get("cache_creation_tokens", 0))
        input_rate, output_rate = _pricing_for_model(model)
        return float(
            (
                input_tokens * input_rate
                + output_tokens * output_rate
                + cache_read_tokens * input_rate * 0.1
                + cache_creation_tokens * input_rate * 1.25
            )
            / 1_000_000
        )
