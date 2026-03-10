"""OpenAI-compatible LLM provider (supports OpenAI, Ollama, vLLM, Groq)."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

from evoforge.llm.client import LLMResponse
from evoforge.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# Lazy import sentinel
openai: Any = None

# Pricing per million tokens: (input_cost, output_cost) in USD
# Sorted by key length descending so longest match wins.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4o": (2.50, 10.0),
}

_DEFAULT_PRICING: tuple[float, float] = (2.50, 10.0)  # gpt-4o


def _ensure_openai() -> Any:
    """Lazily import openai, raising a clear error if not installed."""
    global openai  # noqa: PLW0603
    if openai is None:
        try:
            import openai as _openai

            openai = _openai
        except ImportError as exc:
            msg = "openai package is required for OpenAIProvider. Install it with: uv add openai"
            raise ImportError(msg) from exc
    return openai


def _pricing_for_model(model: str) -> tuple[float, float]:
    """Return (input_cost, output_cost) per million tokens, longest key match first."""
    model_lower = model.lower()
    for key in sorted(_MODEL_PRICING, key=len, reverse=True):
        if key in model_lower:
            return _MODEL_PRICING[key]
    return _DEFAULT_PRICING


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI-compatible APIs.

    Supports OpenAI, Ollama (base_url=http://localhost:11434/v1),
    vLLM, Groq, and any other OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 6,
        base_delay: float = 2.0,
        max_delay: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sync_client: Any = None
        self._async_client: Any = None

    def _compute_delay(self, attempt: int) -> float:
        """Compute retry delay with exponential backoff, jitter, and cap."""
        delay = self._base_delay * (2**attempt)
        jitter = random.uniform(0, self._base_delay)
        return float(min(delay + jitter, self._max_delay))

    def _get_sync_client(self) -> Any:
        if self._sync_client is None:
            oai = _ensure_openai()
            self._sync_client = oai.OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._sync_client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            oai = _ensure_openai()
            self._async_client = oai.AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._async_client

    @staticmethod
    def _build_messages(system: str, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

    @staticmethod
    def _parse_response(response: Any, model: str) -> LLMResponse:
        return LLMResponse(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=model,
        )

    async def generate(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response asynchronously with retry."""
        _ensure_openai()
        client = self._get_async_client()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=self._build_messages(system, prompt),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return self._parse_response(response, model)
            except (openai.RateLimitError, openai.APIError) as exc:
                last_exc = exc
                delay = self._compute_delay(attempt)
                logger.warning(
                    "OpenAI API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        msg = f"OpenAI call failed after {self._max_retries} retries"
        raise RuntimeError(msg) from last_exc

    def generate_sync(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response synchronously with retry."""
        _ensure_openai()
        client = self._get_sync_client()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=self._build_messages(system, prompt),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return self._parse_response(response, model)
            except (openai.RateLimitError, openai.APIError) as exc:
                last_exc = exc
                delay = self._compute_delay(attempt)
                logger.warning(
                    "OpenAI API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)

        msg = f"OpenAI call failed after {self._max_retries} retries"
        raise RuntimeError(msg) from last_exc

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        **kwargs: Any,
    ) -> float:
        """Estimate USD cost for the given token counts and model."""
        input_rate, output_rate = _pricing_for_model(model)
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
