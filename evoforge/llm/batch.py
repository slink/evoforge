# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Batch API support for collecting and submitting LLM requests as a Message Batch."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evoforge.llm.client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

_active_collector: ContextVar[BatchCollector | None] = ContextVar(
    "_active_collector", default=None
)


def get_batch_collector() -> BatchCollector | None:
    """Return the active BatchCollector, or None if not inside a batch context."""
    return _active_collector.get()


class BatchCollector:
    """Async context manager that collects LLM requests and submits them as a batch."""

    def __init__(self, client: LLMClient, poll_interval: float = 2.0) -> None:
        self._client = client
        self._poll_interval = poll_interval
        self._requests: list[tuple[str, str, str, float, int]] = []
        self._futures: list[asyncio.Future[LLMResponse | None]] = []
        self._token: Any = None

    def register(
        self, prompt: str, system: str, model: str, temperature: float, max_tokens: int
    ) -> asyncio.Future[LLMResponse | None]:
        """Register a request and return a future resolved with the result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[LLMResponse | None] = loop.create_future()
        self._requests.append((prompt, system, model, temperature, max_tokens))
        self._futures.append(future)
        return future

    async def __aenter__(self) -> BatchCollector:
        self._token = _active_collector.set(self)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        _active_collector.reset(self._token)
        if not self._requests:
            return
        try:
            await self._submit_and_resolve()
        except Exception:
            logger.warning(
                "Batch submission failed, falling back to individual calls",
                exc_info=True,
            )
            await self._fallback_individual()

    async def _submit_and_resolve(self) -> None:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        from evoforge.llm.client import LLMResponse

        async_client = self._client._get_async_client()

        batch_requests = []
        for i, (prompt, system, model, temperature, max_tokens) in enumerate(self._requests):
            system_param = self._client._format_system(system)
            batch_requests.append(
                Request(
                    custom_id=f"req-{i}",
                    params=MessageCreateParamsNonStreaming(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_param,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                )
            )

        batch = await async_client.messages.batches.create(requests=batch_requests)
        batch_id = batch.id

        while batch.processing_status != "ended":
            await asyncio.sleep(self._poll_interval)
            batch = await async_client.messages.batches.retrieve(batch_id)

        results_by_id: dict[str, LLMResponse | None] = {}
        async for result in await async_client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                msg = result.result.message
                text = msg.content[0].text  # type: ignore[union-attr]
                cache_read = getattr(msg.usage, "cache_read_input_tokens", None) or 0
                cache_creation = getattr(msg.usage, "cache_creation_input_tokens", None) or 0
                results_by_id[result.custom_id] = LLMResponse(
                    text=text,
                    input_tokens=msg.usage.input_tokens,
                    output_tokens=msg.usage.output_tokens,
                    model=msg.model,
                    cache_read_tokens=cache_read,
                    cache_creation_tokens=cache_creation,
                )
            else:
                logger.warning("Batch request %s: %s", result.custom_id, result.result.type)
                results_by_id[result.custom_id] = None

        for i, future in enumerate(self._futures):
            req_id = f"req-{i}"
            future.set_result(results_by_id.get(req_id))

    async def _fallback_individual(self) -> None:
        for i, (prompt, system, model, temperature, max_tokens) in enumerate(self._requests):
            try:
                result = await self._client.async_generate(
                    prompt, system, model, temperature, max_tokens
                )
                self._futures[i].set_result(result)
            except Exception:
                logger.warning("Fallback call %d failed", i, exc_info=True)
                if not self._futures[i].done():
                    self._futures[i].set_result(None)
