"""Tests for LLM providers."""

from __future__ import annotations

import pytest

from evoforge.llm.providers.base import LLMProvider


class TestLLMProviderABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_subclass_must_implement_generate(self) -> None:
        class Incomplete(LLMProvider):
            def generate_sync(
                self,
                prompt: str,
                system: str,
                model: str,
                temperature: float,
                max_tokens: int,
            ) -> None:
                pass

            def estimate_cost(
                self,
                input_tokens: int,
                output_tokens: int,
                model: str,
                **kwargs: object,
            ) -> float:
                return 0.0

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]
