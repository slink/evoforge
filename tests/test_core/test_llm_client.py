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
        from evoforge.llm.retry import compute_delay

        for attempt in range(10):
            delay = compute_delay(attempt, base_delay=1.0, max_delay=5.0)
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
        from evoforge.llm.retry import compute_delay

        delays = [compute_delay(i, base_delay=1.0, max_delay=1000.0) for i in range(5)]
        # Each base is 1, 2, 4, 8, 16 plus jitter [0, 1)
        # So each should be >= base * 2^attempt
        for i, d in enumerate(delays):
            assert d >= 1.0 * (2**i), f"Delay {d} too small for attempt {i}"


class TestCostEstimation:
    def test_haiku_pricing(self) -> None:
        cost = LLMClient().estimate_cost(1_000_000, 1_000_000, "claude-haiku-4-5-20251001")
        assert cost == pytest.approx(0.25 + 1.25)

    def test_unknown_model_uses_sonnet_default(self) -> None:
        cost = LLMClient().estimate_cost(1_000_000, 1_000_000, "unknown-model")
        assert cost == pytest.approx(3.0 + 15.0)


class TestCostEstimationWithCache:
    def test_cache_read_tokens_at_10_percent(self) -> None:
        cost = LLMClient().estimate_cost(
            input_tokens=0,
            output_tokens=0,
            model="claude-sonnet-4-5-20250929",
            cache_read_tokens=1_000_000,
            cache_creation_tokens=0,
        )
        assert cost == pytest.approx(0.30)

    def test_cache_creation_tokens_at_125_percent(self) -> None:
        cost = LLMClient().estimate_cost(
            input_tokens=0,
            output_tokens=0,
            model="claude-sonnet-4-5-20250929",
            cache_read_tokens=0,
            cache_creation_tokens=1_000_000,
        )
        assert cost == pytest.approx(3.75)

    def test_mixed_cache_and_regular_tokens(self) -> None:
        cost = LLMClient().estimate_cost(
            input_tokens=100_000,
            output_tokens=50_000,
            model="claude-sonnet-4-5-20250929",
            cache_read_tokens=500_000,
            cache_creation_tokens=200_000,
        )
        assert cost == pytest.approx(0.30 + 0.75 + 0.15 + 0.75)

    def test_existing_cost_estimation_unchanged(self) -> None:
        cost = LLMClient().estimate_cost(1_000_000, 1_000_000, "claude-haiku-4-5-20251001")
        assert cost == pytest.approx(0.25 + 1.25)


class TestLLMResponseCacheFields:
    def test_default_cache_fields_are_zero(self) -> None:
        from evoforge.llm.client import LLMResponse

        r = LLMResponse(text="hi", input_tokens=10, output_tokens=5, model="test")
        assert r.cache_read_tokens == 0
        assert r.cache_creation_tokens == 0

    def test_cache_fields_can_be_set(self) -> None:
        from evoforge.llm.client import LLMResponse

        r = LLMResponse(
            text="hi",
            input_tokens=10,
            output_tokens=5,
            model="test",
            cache_read_tokens=100,
            cache_creation_tokens=50,
        )
        assert r.cache_read_tokens == 100
        assert r.cache_creation_tokens == 50


class TestPromptCaching:
    async def test_async_generate_sends_cache_control_when_enabled(self) -> None:
        client = LLMClient(api_key="test", prompt_caching=True)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="result")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = 100
        mock_response.usage.cache_creation_input_tokens = 0

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_instance

            result = await client.async_generate("prompt", "system text", "haiku", 0.7)

            call_kwargs = mock_instance.messages.create.call_args[1]
            assert call_kwargs["system"] == [
                {
                    "type": "text",
                    "text": "system text",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            assert result.cache_read_tokens == 100
            assert result.cache_creation_tokens == 0

    async def test_async_generate_no_cache_control_when_disabled(self) -> None:
        client = LLMClient(api_key="test", prompt_caching=False)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="result")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = None
        mock_response.usage.cache_creation_input_tokens = None

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_instance

            result = await client.async_generate("prompt", "system text", "haiku", 0.7)

            call_kwargs = mock_instance.messages.create.call_args[1]
            assert call_kwargs["system"] == "system text"
            assert result.cache_read_tokens == 0
            assert result.cache_creation_tokens == 0

    def test_sync_generate_sends_cache_control_when_enabled(self) -> None:
        client = LLMClient(api_key="test", prompt_caching=True)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="result")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = 50
        mock_response.usage.cache_creation_input_tokens = 200

        with patch("anthropic.Anthropic") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.messages.create = MagicMock(return_value=mock_response)
            mock_cls.return_value = mock_instance

            result = client.generate("prompt", "system text", "haiku", 0.7)

            call_kwargs = mock_instance.messages.create.call_args[1]
            assert call_kwargs["system"] == [
                {
                    "type": "text",
                    "text": "system text",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            assert result.cache_read_tokens == 50
            assert result.cache_creation_tokens == 200

    async def test_default_prompt_caching_is_true(self) -> None:
        client = LLMClient(api_key="test")
        assert client._prompt_caching is True


class TestLLMConfigCacheFields:
    def test_prompt_caching_defaults_true(self) -> None:
        from evoforge.core.config import LLMConfig

        cfg = LLMConfig()
        assert cfg.prompt_caching is True

    def test_prompt_caching_can_be_disabled(self) -> None:
        from evoforge.core.config import LLMConfig

        cfg = LLMConfig(prompt_caching=False)
        assert cfg.prompt_caching is False
