"""Tests for the AnthropicProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from evoforge.llm.providers.anthropic import AnthropicProvider
from evoforge.llm.providers.base import LLMResponse


class TestAsyncGenerate:
    async def test_generate_returns_llm_response(self) -> None:
        """Async generate returns an LLMResponse with correct fields."""
        provider = AnthropicProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="hello world")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 20
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_instance

            result = await provider.generate("prompt", "system", "claude-sonnet-4-5-20250929", 0.7)

        assert isinstance(result, LLMResponse)
        assert result.text == "hello world"
        assert result.input_tokens == 100
        assert result.output_tokens == 20
        assert result.model == "claude-sonnet-4-5-20250929"


class TestPromptCaching:
    async def test_cache_control_added_when_enabled(self) -> None:
        """When prompt_caching=True, system prompt gets cache_control."""
        provider = AnthropicProvider(api_key="test-key", prompt_caching=True)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="result")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = 100
        mock_response.usage.cache_creation_input_tokens = 50

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_instance

            result = await provider.generate("prompt", "system text", "haiku", 0.7)

            call_kwargs = mock_instance.messages.create.call_args[1]
            assert call_kwargs["system"] == [
                {
                    "type": "text",
                    "text": "system text",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            assert result.cache_read_tokens == 100
            assert result.cache_creation_tokens == 50

    async def test_no_cache_control_when_disabled(self) -> None:
        """When prompt_caching=False, system prompt is plain string."""
        provider = AnthropicProvider(api_key="test-key", prompt_caching=False)

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

            await provider.generate("prompt", "system text", "haiku", 0.7)

            call_kwargs = mock_instance.messages.create.call_args[1]
            assert call_kwargs["system"] == "system text"


class TestSyncGenerate:
    def test_sync_generate_returns_llm_response(self) -> None:
        """Sync generate returns an LLMResponse with correct fields."""
        provider = AnthropicProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="sync result")]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 10
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0

        with patch("anthropic.Anthropic") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.messages.create = MagicMock(return_value=mock_response)
            mock_cls.return_value = mock_instance

            result = provider.generate_sync("prompt", "system", "claude-haiku-4-5-20251001", 0.5)

        assert isinstance(result, LLMResponse)
        assert result.text == "sync result"
        assert result.input_tokens == 50
        assert result.output_tokens == 10


class TestEstimateCost:
    def test_sonnet_pricing(self) -> None:
        """Sonnet pricing: 3.0/M input, 15.0/M output."""
        provider = AnthropicProvider(api_key="test-key")
        cost = provider.estimate_cost(1_000_000, 1_000_000, "claude-sonnet-4-5-20250929")
        assert cost == pytest.approx(3.0 + 15.0)

    def test_cost_with_cache_tokens(self) -> None:
        """Cache read at 10%, cache creation at 125% of input rate."""
        provider = AnthropicProvider(api_key="test-key")
        cost = provider.estimate_cost(
            input_tokens=100_000,
            output_tokens=50_000,
            model="claude-sonnet-4-5-20250929",
            cache_read_tokens=500_000,
            cache_creation_tokens=200_000,
        )
        # input: 100k * 3.0/1M = 0.30
        # output: 50k * 15.0/1M = 0.75
        # cache_read: 500k * 3.0/1M * 0.1 = 0.15
        # cache_creation: 200k * 3.0/1M * 1.25 = 0.75
        assert cost == pytest.approx(0.30 + 0.75 + 0.15 + 0.75)


class TestRetryOnRateLimit:
    async def test_retries_on_rate_limit_then_succeeds(self) -> None:
        """Provider retries on RateLimitError and succeeds on next attempt."""
        provider = AnthropicProvider(api_key="test-key", max_retries=3, base_delay=0.01)

        error = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={"error": {"message": "rate limited"}},
        )
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="retried")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(side_effect=[error, mock_response])
            mock_cls.return_value = mock_instance

            with patch(
                "evoforge.llm.providers.anthropic.asyncio.sleep",
                new_callable=AsyncMock,
            ):
                result = await provider.generate("prompt", "system", "haiku", 0.7)

        assert result.text == "retried"
        assert mock_instance.messages.create.call_count == 2

    async def test_raises_after_max_retries(self) -> None:
        """Provider raises RuntimeError after exhausting retries."""
        provider = AnthropicProvider(api_key="test-key", max_retries=2, base_delay=0.01)

        error = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={"error": {"message": "rate limited"}},
        )

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(side_effect=error)
            mock_cls.return_value = mock_instance

            with patch(
                "evoforge.llm.providers.anthropic.asyncio.sleep",
                new_callable=AsyncMock,
            ):
                with pytest.raises(RuntimeError, match="after 2 retries"):
                    await provider.generate("prompt", "system", "haiku", 0.7)
