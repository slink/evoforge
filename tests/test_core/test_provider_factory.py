"""Tests for create_provider factory and LLMClient delegation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evoforge.core.config import LLMConfig
from evoforge.llm.client import LLMClient, LLMResponse


class TestCreateProvider:
    def test_creates_anthropic_provider(self) -> None:
        from evoforge.llm.providers import create_provider
        from evoforge.llm.providers.anthropic import AnthropicProvider

        config = LLMConfig(provider="anthropic", api_key_env="ANTHROPIC_API_KEY")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            p = create_provider(config)
        assert isinstance(p, AnthropicProvider)

    def test_creates_anthropic_with_caching_flag(self) -> None:
        from evoforge.llm.providers import create_provider

        config = LLMConfig(provider="anthropic", prompt_caching=False)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            p = create_provider(config)
        assert p._prompt_caching is False  # type: ignore[attr-defined]

    def test_unknown_provider_raises(self) -> None:
        from evoforge.llm.providers import create_provider

        config = LLMConfig(provider="anthropic")
        # Monkey-patch to bypass Pydantic validation
        object.__setattr__(config, "provider", "unknown")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_provider(config)


class TestLLMConfigProviderFields:
    def test_default_provider_is_anthropic(self) -> None:
        config = LLMConfig()
        assert config.provider == "anthropic"
        assert config.api_key_env == "ANTHROPIC_API_KEY"
        assert config.base_url is None

    def test_gemini_config(self) -> None:
        config = LLMConfig(
            provider="gemini",
            api_key_env="GOOGLE_API_KEY",
            model="gemini-2.5-flash",
        )
        assert config.provider == "gemini"
        assert config.api_key_env == "GOOGLE_API_KEY"

    def test_openai_config_with_base_url(self) -> None:
        config = LLMConfig(
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            base_url="http://localhost:11434/v1",
        )
        assert config.provider == "openai"
        assert config.base_url == "http://localhost:11434/v1"


class TestLLMClientDelegation:
    def test_sync_generate_delegates_to_provider(self) -> None:
        mock_provider = MagicMock()
        expected = LLMResponse(text="delegated", input_tokens=10, output_tokens=5, model="test")
        mock_provider.generate_sync.return_value = expected

        client = LLMClient(provider=mock_provider)
        result = client.generate("prompt", "system", "test", 0.7)

        assert result is expected
        mock_provider.generate_sync.assert_called_once_with(
            prompt="prompt",
            system="system",
            model="test",
            temperature=0.7,
            max_tokens=4096,
        )

    async def test_async_generate_delegates_to_provider(self) -> None:
        mock_provider = AsyncMock()
        expected = LLMResponse(
            text="async delegated", input_tokens=10, output_tokens=5, model="test"
        )
        mock_provider.generate.return_value = expected

        client = LLMClient(provider=mock_provider)
        result = await client.async_generate("prompt", "system", "test", 0.7)

        assert result is expected
        mock_provider.generate.assert_called_once_with(
            prompt="prompt",
            system="system",
            model="test",
            temperature=0.7,
            max_tokens=4096,
        )

    def test_estimate_cost_delegates_to_provider(self) -> None:
        mock_provider = MagicMock()
        mock_provider.estimate_cost.return_value = 0.42

        client = LLMClient(provider=mock_provider)
        result = client.estimate_cost(
            input_tokens=1000,
            output_tokens=500,
            model="test",
            cache_read_tokens=100,
            cache_creation_tokens=50,
        )

        assert result == 0.42
        mock_provider.estimate_cost.assert_called_once_with(
            input_tokens=1000,
            output_tokens=500,
            model="test",
            cache_read_tokens=100,
            cache_creation_tokens=50,
        )

    def test_no_provider_falls_back_to_anthropic(self) -> None:
        """Without a provider, LLMClient uses its built-in Anthropic code."""
        client = LLMClient(api_key="test")
        assert client._provider is None
        # estimate_cost still works via the built-in path
        cost = client.estimate_cost(1_000_000, 1_000_000, "claude-haiku-4-5-20251001")
        assert cost == pytest.approx(0.25 + 1.25)
