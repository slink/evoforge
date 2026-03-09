# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for BatchCollector async context manager."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evoforge.llm.batch import BatchCollector, get_batch_collector
from evoforge.llm.client import LLMClient, LLMResponse


def _make_succeeded_result(custom_id: str, text: str, model: str = "test-model") -> MagicMock:
    """Build a mock batch result with result.type == 'succeeded'."""
    result = MagicMock()
    result.custom_id = custom_id
    result.result.type = "succeeded"
    msg = result.result.message
    msg.content = [MagicMock(text=text)]
    msg.usage.input_tokens = 10
    msg.usage.output_tokens = 5
    msg.usage.cache_read_input_tokens = 2
    msg.usage.cache_creation_input_tokens = 1
    msg.model = model
    return result


def _make_errored_result(custom_id: str) -> MagicMock:
    """Build a mock batch result with result.type == 'errored'."""
    result = MagicMock()
    result.custom_id = custom_id
    result.result.type = "errored"
    return result


class _AsyncIterList:
    """Wrap a list as an async iterable for mocking batch results."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __aiter__(self) -> _AsyncIterList:
        self._index = 0
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


def _make_batch_mock(processing_status: str = "ended") -> MagicMock:
    """Build a mock batch object."""
    batch = MagicMock()
    batch.id = "batch_test_123"
    batch.processing_status = processing_status
    return batch


def _make_client_with_mock() -> tuple[LLMClient, MagicMock]:
    """Create an LLMClient with a mocked async client."""
    client = LLMClient(api_key="test-key", prompt_caching=False)
    mock_async = AsyncMock()
    client._async_client = mock_async
    return client, mock_async


class TestBatchCollectorCollectsRequests:
    """Register 2 requests, mock batch create/retrieve/results, verify futures resolve."""

    @pytest.mark.asyncio
    async def test_two_requests_resolve(self) -> None:
        client, mock_async = _make_client_with_mock()

        # Batch create returns an already-ended batch
        batch = _make_batch_mock("ended")
        mock_async.messages.batches.create = AsyncMock(return_value=batch)

        # Results: two succeeded
        r0 = _make_succeeded_result("req-0", "response zero")
        r1 = _make_succeeded_result("req-1", "response one")
        mock_async.messages.batches.results = AsyncMock(return_value=_AsyncIterList([r0, r1]))

        async with BatchCollector(client) as collector:
            f0 = collector.register("prompt0", "system", "test-model", 0.7, 1024)
            f1 = collector.register("prompt1", "system", "test-model", 0.5, 2048)

        # Futures should be resolved
        result0 = f0.result()
        result1 = f1.result()

        assert isinstance(result0, LLMResponse)
        assert result0.text == "response zero"
        assert result0.input_tokens == 10
        assert result0.output_tokens == 5
        assert result0.cache_read_tokens == 2
        assert result0.cache_creation_tokens == 1

        assert isinstance(result1, LLMResponse)
        assert result1.text == "response one"

        # Verify batch create was called with 2 requests
        mock_async.messages.batches.create.assert_awaited_once()
        call_kwargs = mock_async.messages.batches.create.call_args
        assert len(call_kwargs.kwargs["requests"]) == 2


class TestBatchCollectorEmpty:
    """Enter/exit context with no registrations -> create never called."""

    @pytest.mark.asyncio
    async def test_no_requests_no_create(self) -> None:
        client, mock_async = _make_client_with_mock()

        async with BatchCollector(client):
            pass

        mock_async.messages.batches.create.assert_not_awaited()


class TestBatchCollectorFallback:
    """Mock create to raise -> verify async_generate called as fallback."""

    @pytest.mark.asyncio
    async def test_fallback_on_create_failure(self) -> None:
        client, mock_async = _make_client_with_mock()

        # Batch create raises
        mock_async.messages.batches.create = AsyncMock(side_effect=RuntimeError("API down"))

        # Mock async_generate for fallback
        fallback_response = LLMResponse(
            text="fallback", input_tokens=5, output_tokens=3, model="test-model"
        )
        with patch.object(client, "async_generate", new=AsyncMock(return_value=fallback_response)):
            async with BatchCollector(client) as collector:
                f0 = collector.register("prompt0", "system", "test-model", 0.7, 1024)
                f1 = collector.register("prompt1", "system", "test-model", 0.5, 2048)

            assert f0.result() == fallback_response
            assert f1.result() == fallback_response
            assert client.async_generate.call_count == 2  # type: ignore[union-attr]


class TestBatchCollectorPerRequestError:
    """One request errored -> that future resolves to None, other succeeds."""

    @pytest.mark.asyncio
    async def test_errored_request_resolves_none(self) -> None:
        client, mock_async = _make_client_with_mock()

        batch = _make_batch_mock("ended")
        mock_async.messages.batches.create = AsyncMock(return_value=batch)

        r0 = _make_succeeded_result("req-0", "good response")
        r1 = _make_errored_result("req-1")
        mock_async.messages.batches.results = AsyncMock(return_value=_AsyncIterList([r0, r1]))

        async with BatchCollector(client) as collector:
            f0 = collector.register("prompt0", "system", "test-model", 0.7, 1024)
            f1 = collector.register("prompt1", "system", "test-model", 0.5, 2048)

        assert isinstance(f0.result(), LLMResponse)
        assert f0.result().text == "good response"
        assert f1.result() is None


class TestGetBatchCollector:
    """Returns None outside context, collector inside, None after exit."""

    @pytest.mark.asyncio
    async def test_context_var_lifecycle(self) -> None:
        client, mock_async = _make_client_with_mock()

        assert get_batch_collector() is None

        async with BatchCollector(client) as collector:
            assert get_batch_collector() is collector

        assert get_batch_collector() is None


class TestBatchConfigFields:
    """Verify LLMConfig batch fields have correct defaults and are settable."""

    def test_batch_defaults(self) -> None:
        from evoforge.core.config import LLMConfig

        cfg = LLMConfig()
        assert cfg.batch_enabled is False
        assert cfg.batch_poll_interval == 2.0

    def test_batch_can_be_enabled(self) -> None:
        from evoforge.core.config import LLMConfig

        cfg = LLMConfig(batch_enabled=True, batch_poll_interval=5.0)
        assert cfg.batch_enabled is True
        assert cfg.batch_poll_interval == 5.0


class TestTacticGeneratorBatchAware:
    @pytest.mark.asyncio
    async def test_registers_with_batch_when_active(self) -> None:
        import asyncio

        from evoforge.backends.lean.tactic_generator import LLMTacticGenerator
        from tests.conftest import FakeLLMResponse

        client = MagicMock()
        client.async_generate = AsyncMock()

        mock_collector = MagicMock()
        future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        future.set_result(FakeLLMResponse(text="1. simp\n2. ring\n3. linarith"))
        mock_collector.register = MagicMock(return_value=future)

        gen = LLMTacticGenerator(client, "test-model", "system prompt")

        with patch(
            "evoforge.backends.lean.tactic_generator.get_batch_collector",
            return_value=mock_collector,
        ):
            tactics = await gen.suggest_tactics("⊢ x = x", [], 3)

        mock_collector.register.assert_called_once()
        client.async_generate.assert_not_called()
        assert len(tactics) > 0

    @pytest.mark.asyncio
    async def test_direct_call_when_no_batch(self) -> None:
        from evoforge.backends.lean.tactic_generator import LLMTacticGenerator
        from tests.conftest import FakeLLMResponse

        client = MagicMock()
        client.async_generate = AsyncMock(return_value=FakeLLMResponse(text="1. simp\n2. ring"))

        gen = LLMTacticGenerator(client, "test-model", "system prompt")

        with patch(
            "evoforge.backends.lean.tactic_generator.get_batch_collector",
            return_value=None,
        ):
            tactics = await gen.suggest_tactics("⊢ x = x", [], 3)

        client.async_generate.assert_called_once()
        assert len(tactics) > 0

    @pytest.mark.asyncio
    async def test_batch_returns_none_returns_empty(self) -> None:
        import asyncio

        from evoforge.backends.lean.tactic_generator import LLMTacticGenerator

        client = MagicMock()
        mock_collector = MagicMock()
        future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        future.set_result(None)
        mock_collector.register = MagicMock(return_value=future)

        gen = LLMTacticGenerator(client, "test-model", "system prompt")

        with patch(
            "evoforge.backends.lean.tactic_generator.get_batch_collector",
            return_value=mock_collector,
        ):
            tactics = await gen.suggest_tactics("⊢ x = x", [], 3)

        assert tactics == []
