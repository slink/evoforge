"""Tests for OpenAI-compatible LLM provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evoforge.llm.client import LLMResponse
from evoforge.llm.providers.openai_compat import OpenAIProvider


class TestOpenAIProviderGenerate:
    """Test async generate."""

    async def test_generate_returns_llm_response(self) -> None:
        """Async generate returns an LLMResponse with correct fields."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello world"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        with patch("evoforge.llm.providers.openai_compat.openai") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.AsyncOpenAI.return_value = mock_client

            result = await provider.generate(
                prompt="Say hello",
                system="You are helpful",
                model="gpt-4o",
                temperature=0.7,
            )

        assert isinstance(result, LLMResponse)
        assert result.text == "Hello world"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.model == "gpt-4o"

    async def test_system_prompt_sent_as_system_message(self) -> None:
        """System prompt is sent as a system role message."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 1

        with patch("evoforge.llm.providers.openai_compat.openai") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.AsyncOpenAI.return_value = mock_client

            await provider.generate(
                prompt="Hi",
                system="Be concise",
                model="gpt-4o",
                temperature=0.5,
            )

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            messages = call_kwargs["messages"]
            assert messages[0] == {"role": "system", "content": "Be concise"}
            assert messages[1] == {"role": "user", "content": "Hi"}


class TestOpenAIProviderSync:
    """Test sync generate."""

    def test_sync_generate_works(self) -> None:
        """Sync generate returns an LLMResponse."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Sync response"
        mock_response.usage.prompt_tokens = 8
        mock_response.usage.completion_tokens = 3

        with patch("evoforge.llm.providers.openai_compat.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            result = provider.generate_sync(
                prompt="Hello",
                system="System",
                model="gpt-4o-mini",
                temperature=0.3,
            )

        assert isinstance(result, LLMResponse)
        assert result.text == "Sync response"
        assert result.input_tokens == 8
        assert result.output_tokens == 3


class TestOpenAIProviderCost:
    """Test cost estimation."""

    def test_estimate_cost_gpt4o(self) -> None:
        """Estimate cost for gpt-4o model."""
        provider = OpenAIProvider(api_key="test-key")
        # gpt-4o: input=2.50, output=10.0 per million tokens
        cost = provider.estimate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="gpt-4o",
        )
        assert cost == pytest.approx(12.5)

    def test_estimate_cost_gpt4o_mini(self) -> None:
        """Estimate cost for gpt-4o-mini (longest match first)."""
        provider = OpenAIProvider(api_key="test-key")
        # gpt-4o-mini: input=0.15, output=0.60 per million tokens
        cost = provider.estimate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="gpt-4o-mini",
        )
        assert cost == pytest.approx(0.75)

    def test_estimate_cost_unknown_defaults_to_gpt4o(self) -> None:
        """Unknown model defaults to gpt-4o pricing."""
        provider = OpenAIProvider(api_key="test-key")
        cost = provider.estimate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="unknown-model",
        )
        assert cost == pytest.approx(12.5)


class TestOpenAIProviderBaseURL:
    """Test base_url passthrough."""

    async def test_base_url_passed_to_client(self) -> None:
        """base_url is passed to the OpenAI client constructor."""
        provider = OpenAIProvider(api_key="test-key", base_url="http://localhost:11434/v1")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "local"
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1

        with patch("evoforge.llm.providers.openai_compat.openai") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.AsyncOpenAI.return_value = mock_client

            await provider.generate(
                prompt="Hi",
                system="Sys",
                model="llama3",
                temperature=0.5,
            )

            mock_openai.AsyncOpenAI.assert_called_once_with(
                api_key="test-key", base_url="http://localhost:11434/v1"
            )
