"""Tests for the Gemini LLM provider."""

from __future__ import annotations

from collections.abc import Generator
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from evoforge.llm.client import LLMResponse

if TYPE_CHECKING:
    from evoforge.llm.providers.gemini import GeminiProvider


def _make_mock_response(
    text: str = "hello",
    prompt_tokens: int = 10,
    candidate_tokens: int = 5,
    cached_tokens: int = 0,
) -> SimpleNamespace:
    """Create a mock Gemini API response."""
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=candidate_tokens,
        cached_content_token_count=cached_tokens,
    )
    return SimpleNamespace(text=text, usage_metadata=usage)


@pytest.fixture()
def mock_genai() -> MagicMock:
    """Provide a mock google.genai module."""
    mock = MagicMock()
    # types.GenerateContentConfig should act as a passthrough constructor
    mock.types.GenerateContentConfig = lambda **kw: SimpleNamespace(**kw)
    return mock


@pytest.fixture()
def provider(mock_genai: MagicMock) -> Generator[GeminiProvider, None, None]:
    """Create a GeminiProvider with mocked SDK."""
    import evoforge.llm.providers.gemini as gemini_mod

    # Patch module-level genai so _ensure_genai sees it as already imported
    with (
        patch.object(gemini_mod, "genai", mock_genai),
        patch.object(gemini_mod, "_types", mock_genai.types),
    ):
        p = gemini_mod.GeminiProvider(api_key="test-key", max_retries=1)
        p._client = mock_genai.Client.return_value
        yield p


@pytest.mark.asyncio()
async def test_async_generate_returns_llm_response(
    provider: GeminiProvider, mock_genai: MagicMock
) -> None:
    """Async generate should return LLMResponse with correct token mapping."""
    mock_resp = _make_mock_response(
        text="result", prompt_tokens=20, candidate_tokens=8, cached_tokens=3
    )

    async def fake_generate(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return mock_resp

    provider._client.aio.models.generate_content = fake_generate

    resp = await provider.generate(
        prompt="test prompt",
        system="test system",
        model="gemini-2.5-flash",
        temperature=0.7,
        max_tokens=1024,
    )

    assert isinstance(resp, LLMResponse)
    assert resp.text == "result"
    assert resp.input_tokens == 20
    assert resp.output_tokens == 8
    assert resp.model == "gemini-2.5-flash"
    assert resp.cache_read_tokens == 3


@pytest.mark.asyncio()
async def test_system_instruction_passed_in_config(
    provider: GeminiProvider, mock_genai: MagicMock
) -> None:
    """System instruction should be passed via GenerateContentConfig."""
    mock_resp = _make_mock_response()
    call_kwargs: dict[str, Any] = {}

    async def fake_generate(*args: Any, **kwargs: Any) -> SimpleNamespace:
        call_kwargs.update(kwargs)
        return mock_resp

    provider._client.aio.models.generate_content = fake_generate

    await provider.generate(
        prompt="hello",
        system="be helpful",
        model="gemini-2.5-flash",
        temperature=0.5,
    )

    config = call_kwargs["config"]
    assert config.system_instruction == "be helpful"
    assert config.temperature == 0.5
    assert config.max_output_tokens == 4096


def test_sync_generate_works(provider: GeminiProvider, mock_genai: MagicMock) -> None:
    """Sync generate should return LLMResponse."""
    mock_resp = _make_mock_response(text="sync result", prompt_tokens=15, candidate_tokens=6)
    provider._client.models.generate_content.return_value = mock_resp

    resp = provider.generate_sync(
        prompt="sync prompt",
        system="sync system",
        model="gemini-3-flash",
        temperature=0.3,
    )

    assert isinstance(resp, LLMResponse)
    assert resp.text == "sync result"
    assert resp.input_tokens == 15
    assert resp.output_tokens == 6
    assert resp.model == "gemini-3-flash"


def test_estimate_cost_flash_model(provider: GeminiProvider) -> None:
    """Estimate cost should use flash pricing."""
    # gemini-2.5-flash: input=0.30, output=2.50 per million
    cost = provider.estimate_cost(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        model="gemini-2.5-flash",
    )
    assert cost == pytest.approx(0.30 + 2.50)


def test_estimate_cost_unknown_model_defaults_to_flash(
    provider: GeminiProvider,
) -> None:
    """Unknown model should default to gemini-3-flash pricing."""
    # gemini-3-flash: input=0.50, output=3.0 per million
    cost = provider.estimate_cost(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        model="gemini-unknown-model",
    )
    assert cost == pytest.approx(0.50 + 3.0)
