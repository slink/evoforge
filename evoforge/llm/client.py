# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""LLM client with retry logic and cost estimation."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass

import anthropic

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


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


class LLMClient:
    """Thin wrapper around the Anthropic API with retry and cost estimation."""

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 6,
        base_delay: float = 2.0,
        max_delay: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sync_client: anthropic.Anthropic | None = None
        self._async_client: anthropic.AsyncAnthropic | None = None

    def _compute_delay(self, attempt: int) -> float:
        """Compute retry delay with exponential backoff, jitter, and cap."""
        delay = self._base_delay * (2**attempt)
        jitter = random.uniform(0, self._base_delay)
        return float(min(delay + jitter, self._max_delay))

    def _get_sync_client(self) -> anthropic.Anthropic:
        if self._sync_client is None:
            self._sync_client = anthropic.Anthropic(api_key=self._api_key)
        return self._sync_client

    def _get_async_client(self) -> anthropic.AsyncAnthropic:
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._async_client

    def generate(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call the Anthropic API synchronously with exponential-backoff retry."""
        client = self._get_sync_client()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text  # type: ignore[union-attr]
                return LLMResponse(
                    text=text,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    model=model,
                )
            except (anthropic.RateLimitError, anthropic.APIError) as exc:
                last_exc = exc
                delay = self._compute_delay(attempt)
                logger.warning(
                    "LLM API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)

        msg = f"LLM call failed after {self._max_retries} retries"
        raise RuntimeError(msg) from last_exc

    async def async_generate(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call the Anthropic API asynchronously with exponential-backoff retry."""
        client = self._get_async_client()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text  # type: ignore[union-attr]
                return LLMResponse(
                    text=text,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    model=model,
                )
            except (anthropic.RateLimitError, anthropic.APIError) as exc:
                last_exc = exc
                delay = self._compute_delay(attempt)
                logger.warning(
                    "LLM API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        msg = f"LLM call failed after {self._max_retries} retries"
        raise RuntimeError(msg) from last_exc

    @staticmethod
    def estimate_cost(
        input_tokens: int,
        output_tokens: int,
        model: str,
        *,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> float:
        """Estimate USD cost for the given token counts and model."""
        input_rate, output_rate = _pricing_for_model(model)
        return (
            input_tokens * input_rate
            + output_tokens * output_rate
            + cache_read_tokens * input_rate * 0.1
            + cache_creation_tokens * input_rate * 1.25
        ) / 1_000_000
