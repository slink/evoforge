"""Gemini LLM provider using the google-genai SDK."""

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
genai: Any = None
_types: Any = None


def _ensure_genai() -> None:
    """Lazily import google-genai SDK, raising a helpful error if missing."""
    global genai, _types  # noqa: PLW0603
    if genai is not None:
        return
    try:
        from google import genai as _genai
        from google.genai import types

        genai = _genai
        _types = types
    except ImportError as exc:
        msg = (
            "google-genai is required for the Gemini provider. "
            "Install it with: uv add google-genai"
        )
        raise ImportError(msg) from exc


# Pricing per million tokens: (input_cost, output_cost) in USD
# Ordered longest-key-first so "2.5-flash-lite" matches before "2.5-flash".
_MODEL_PRICING: list[tuple[str, tuple[float, float]]] = [
    ("gemini-2.5-flash-lite", (0.10, 0.40)),
    ("gemini-3.1-pro", (2.0, 12.0)),
    ("gemini-3-flash", (0.50, 3.0)),
    ("gemini-2.5-flash", (0.30, 2.50)),
    ("gemini-2.5-pro", (1.25, 10.0)),
]

_DEFAULT_PRICING: tuple[float, float] = (0.50, 3.0)  # gemini-3-flash


def _pricing_for_model(model: str) -> tuple[float, float]:
    """Return (input_cost, output_cost) per million tokens, longest key match first."""
    model_lower = model.lower()
    for key, pricing in sorted(_MODEL_PRICING, key=lambda kv: len(kv[0]), reverse=True):
        if key in model_lower:
            return pricing
    return _DEFAULT_PRICING


class GeminiProvider(LLMProvider):
    """LLM provider backed by the Google Gemini API (google-genai SDK)."""

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 6,
        base_delay: float = 2.0,
        max_delay: float = 120.0,
    ) -> None:
        _ensure_genai()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._client: Any = genai.Client(api_key=api_key)

    def _compute_delay(self, attempt: int) -> float:
        """Compute retry delay with exponential backoff, jitter, and cap."""
        delay = self._base_delay * (2**attempt)
        jitter = random.uniform(0, self._base_delay)
        return float(min(delay + jitter, self._max_delay))

    def _build_config(self, system: str, temperature: float, max_tokens: int) -> Any:
        """Build a GenerateContentConfig."""
        return genai.types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    @staticmethod
    def _parse_response(response: Any, model: str) -> LLMResponse:
        """Convert a Gemini response to LLMResponse."""
        usage = response.usage_metadata
        cached = getattr(usage, "cached_content_token_count", 0) or 0
        return LLMResponse(
            text=response.text,
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count,
            model=model,
            cache_read_tokens=cached,
        )

    async def generate(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response asynchronously via the Gemini API."""
        config = self._build_config(system, temperature, max_tokens)
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = await self._client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                return self._parse_response(response, model)
            except Exception as exc:
                last_exc = exc
                delay = self._compute_delay(attempt)
                logger.warning(
                    "Gemini API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        msg = f"Gemini call failed after {self._max_retries} retries"
        raise RuntimeError(msg) from last_exc

    def generate_sync(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response synchronously via the Gemini API."""
        config = self._build_config(system, temperature, max_tokens)
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                return self._parse_response(response, model)
            except Exception as exc:
                last_exc = exc
                delay = self._compute_delay(attempt)
                logger.warning(
                    "Gemini API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)

        msg = f"Gemini call failed after {self._max_retries} retries"
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
