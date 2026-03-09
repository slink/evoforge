# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Batch API support for collecting and submitting LLM requests as a Message Batch."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from evoforge.llm.client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

_active_collector: ContextVar[BatchCollector | None] = ContextVar(
    "_active_collector", default=None
)

_DEFAULT_MAX_WAIT: float = 1800.0  # 30 minutes


def get_batch_collector() -> BatchCollector | None:
    """Return the active BatchCollector, or None if not inside a batch context."""
    return _active_collector.get()


async def batch_aware_generate(
    client: LLMClient,
    prompt: str,
    system: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMResponse | None:
    """Generate via batch collector if active, otherwise via direct async call.

    Returns None if the batch request failed; raises on direct-call failure.
    """
    collector = get_batch_collector()
    if collector is not None:
        return await collector.register(prompt, system, model, temperature, max_tokens)
    return await client.async_generate(prompt, system, model, temperature, max_tokens)


class _BatchRequest(NamedTuple):
    prompt: str
    system: str
    model: str
    temperature: float
    max_tokens: int


class BatchCollector:
    """Async context manager that collects LLM requests and submits them as a batch."""

    def __init__(
        self,
        client: LLMClient,
        poll_interval: float = 2.0,
        max_wait: float = _DEFAULT_MAX_WAIT,
    ) -> None:
        self._client = client
        self._poll_interval = poll_interval
        self._max_wait = max_wait
        self._requests: list[_BatchRequest] = []
        self._futures: list[asyncio.Future[LLMResponse | None]] = []
        self._token: Any = None

    def register(
        self, prompt: str, system: str, model: str, temperature: float, max_tokens: int
    ) -> asyncio.Future[LLMResponse | None]:
        """Register a request and return a future resolved with the result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[LLMResponse | None] = loop.create_future()
        self._requests.append(_BatchRequest(prompt, system, model, temperature, max_tokens))
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

        from evoforge.llm.client import LLMClient, LLMResponse

        async_client = self._client.get_async_client()

        batch_requests = []
        for i, req in enumerate(self._requests):
            system_param = self._client.format_system(req.system)
            batch_requests.append(
                Request(
                    custom_id=f"req-{i}",
                    params=MessageCreateParamsNonStreaming(
                        model=req.model,
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                        system=system_param,
                        messages=[{"role": "user", "content": req.prompt}],
                    ),
                )
            )

        batch = await async_client.messages.batches.create(requests=batch_requests)
        batch_id = batch.id

        elapsed = 0.0
        while batch.processing_status != "ended":
            if elapsed >= self._max_wait:
                msg = f"Batch {batch_id} not ended after {elapsed:.0f}s"
                raise TimeoutError(msg)
            await asyncio.sleep(self._poll_interval)
            elapsed += self._poll_interval
            batch = await async_client.messages.batches.retrieve(batch_id)

        results_by_id: dict[str, LLMResponse | None] = {}
        async for result in await async_client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                msg_obj = result.result.message
                text = msg_obj.content[0].text  # type: ignore[union-attr]
                cache_read, cache_creation = LLMClient.extract_cache_tokens(msg_obj.usage)
                results_by_id[result.custom_id] = LLMResponse(
                    text=text,
                    input_tokens=msg_obj.usage.input_tokens,
                    output_tokens=msg_obj.usage.output_tokens,
                    model=msg_obj.model,
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
        async def _do_one(i: int, req: _BatchRequest) -> None:
            try:
                result = await self._client.async_generate(
                    req.prompt, req.system, req.model, req.temperature, req.max_tokens
                )
                self._futures[i].set_result(result)
            except Exception:
                logger.warning("Fallback call %d failed", i, exc_info=True)
                if not self._futures[i].done():
                    self._futures[i].set_result(None)

        await asyncio.gather(*(_do_one(i, req) for i, req in enumerate(self._requests)))
