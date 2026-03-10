"""LLM provider implementations."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from evoforge.llm.providers.base import LLMProvider

if TYPE_CHECKING:
    from evoforge.core.config import LLMConfig

__all__ = ["LLMProvider", "create_provider"]


def create_provider(config: LLMConfig) -> LLMProvider:
    """Create an LLM provider from configuration.

    Reads the API key from the environment variable named by config.api_key_env.
    """
    api_key = os.environ.get(config.api_key_env)
    match config.provider:
        case "anthropic":
            from evoforge.llm.providers.anthropic import AnthropicProvider

            return AnthropicProvider(api_key=api_key, prompt_caching=config.prompt_caching)
        case "gemini":
            from evoforge.llm.providers.gemini import GeminiProvider

            return GeminiProvider(api_key=api_key)
        case "openai":
            from evoforge.llm.providers.openai_compat import OpenAIProvider

            return OpenAIProvider(api_key=api_key, base_url=config.base_url)
        case _:
            msg = f"Unknown LLM provider: {config.provider!r}"
            raise ValueError(msg)
