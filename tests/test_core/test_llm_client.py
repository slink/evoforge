# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for LLM client retry logic and cost estimation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from evoforge.llm.client import LLMClient


class TestRetryWithJitter:
    async def test_retry_respects_max_retries(self) -> None:
        """Client retries up to max_retries times before raising."""
        client = LLMClient(api_key="test", max_retries=3, base_delay=0.01)

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            error = anthropic.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body={"error": {"message": "rate limited"}},
            )
            mock_instance.messages.create = AsyncMock(side_effect=error)
            mock_cls.return_value = mock_instance

            with pytest.raises(RuntimeError, match="after 3 retries"):
                await client.async_generate("test", "sys", "haiku", 0.7)

            assert mock_instance.messages.create.call_count == 3

    async def test_delay_capped_at_max(self) -> None:
        """Delay should never exceed max_delay."""
        client = LLMClient(api_key="test", max_retries=10, base_delay=1.0, max_delay=5.0)
        for attempt in range(10):
            delay = client._compute_delay(attempt)
            assert delay <= 5.0

    async def test_success_on_retry(self) -> None:
        """Client succeeds if a retry works after initial failure."""
        client = LLMClient(api_key="test", max_retries=3, base_delay=0.01)

        error = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={"error": {"message": "rate limited"}},
        )
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="result")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(side_effect=[error, mock_response])
            mock_cls.return_value = mock_instance

            with patch("evoforge.llm.client.asyncio.sleep", new_callable=AsyncMock):
                result = await client.async_generate("test", "sys", "haiku", 0.7)

        assert result.text == "result"

    def test_compute_delay_exponential(self) -> None:
        """Delay grows exponentially (before jitter)."""
        client = LLMClient(base_delay=1.0, max_delay=1000.0)
        delays = [client._compute_delay(i) for i in range(5)]
        # Each base is 1, 2, 4, 8, 16 plus jitter [0, 1)
        # So each should be >= base * 2^attempt
        for i, d in enumerate(delays):
            assert d >= 1.0 * (2**i), f"Delay {d} too small for attempt {i}"


class TestCostEstimation:
    def test_haiku_pricing(self) -> None:
        cost = LLMClient.estimate_cost(1_000_000, 1_000_000, "claude-haiku-4-5-20251001")
        assert cost == pytest.approx(0.25 + 1.25)

    def test_unknown_model_uses_sonnet_default(self) -> None:
        cost = LLMClient.estimate_cost(1_000_000, 1_000_000, "unknown-model")
        assert cost == pytest.approx(3.0 + 15.0)
