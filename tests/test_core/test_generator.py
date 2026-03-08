# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for evoforge.core.generator — ValidatedGenerator."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pytest

from evoforge.core.generator import ValidatedGenerator
from evoforge.core.types import Individual

# ---------------------------------------------------------------------------
# Helpers: mock IR, backend, and LLM client
# ---------------------------------------------------------------------------


class _MockIR:
    """A mock IR node implementing the IRProtocol interface."""

    def __init__(self, canonical_form: str) -> None:
        self._canonical = canonical_form

    def canonicalize(self) -> _MockIR:
        return _MockIR(self._canonical)

    def structural_hash(self) -> str:
        return hashlib.sha256(self._canonical.encode()).hexdigest()

    def serialize(self) -> str:
        return self._canonical

    def complexity(self) -> int:
        return len(self._canonical)


@dataclass
class _LLMResponse:
    """Mock LLM response with a text attribute."""

    text: str


class _MockLLMClient:
    """Mock LLM client with generate() and async_generate() methods."""

    def __init__(self, responses: list[_LLMResponse | Exception]) -> None:
        self._responses = list(responses)
        self._call_index = 0

    def generate(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> _LLMResponse:
        if self._call_index >= len(self._responses):
            raise RuntimeError("No more mock responses available")
        resp = self._responses[self._call_index]
        self._call_index += 1
        if isinstance(resp, Exception):
            raise resp
        return resp

    async def async_generate(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> _LLMResponse:
        if self._call_index >= len(self._responses):
            raise RuntimeError("No more mock responses available")
        resp = self._responses[self._call_index]
        self._call_index += 1
        if isinstance(resp, Exception):
            raise resp
        return resp


class _MockBackend:
    """Mock backend with extract_genome, parse, and validate_structure."""

    def __init__(
        self,
        extract_map: dict[str, str | None] | None = None,
        parse_map: dict[str, _MockIR | None] | None = None,
        violations_map: dict[str, list[str]] | None = None,
    ) -> None:
        self._extract_map = extract_map or {}
        self._parse_map = parse_map or {}
        self._violations_map = violations_map or {}

    def extract_genome(self, raw_text: str) -> str | None:
        return self._extract_map.get(raw_text)

    def parse(self, genome: str) -> _MockIR | None:
        return self._parse_map.get(genome)

    def validate_structure(self, ir: Any) -> list[str]:
        if hasattr(ir, "_canonical"):
            return self._violations_map.get(ir._canonical, [])
        return []


# ---------------------------------------------------------------------------
# Helpers to build common fixtures
# ---------------------------------------------------------------------------


def _make_happy_path(
    raw_text: str = "raw output",
    genome: str = "extracted genome",
    canonical: str = "canonical_form",
) -> tuple[_MockBackend, _MockLLMClient]:
    """Create a backend + LLM client that succeed on the first attempt."""
    ir = _MockIR(canonical)
    backend = _MockBackend(
        extract_map={raw_text: genome},
        parse_map={genome: ir},
        violations_map={},  # empty = no violations
    )
    client = _MockLLMClient([_LLMResponse(text=raw_text)])
    return backend, client


# ---------------------------------------------------------------------------
# ValidatedGenerator.generate() — synchronous
# ---------------------------------------------------------------------------


class TestGenerateHappyPath:
    """Valid output should produce an Individual with the correct ir_hash."""

    def test_valid_output_returns_individual(self) -> None:
        backend, client = _make_happy_path()
        gen = ValidatedGenerator(backend=backend, llm_client=client, model="test-model")

        result = gen.generate(prompt="write code", system="system", temperature=0.7)

        assert result is not None
        assert isinstance(result, Individual)

    def test_individual_has_correct_ir_hash(self) -> None:
        backend, client = _make_happy_path(canonical="my_canon")
        gen = ValidatedGenerator(backend=backend, llm_client=client, model="test-model")

        result = gen.generate(prompt="write code", system="system", temperature=0.7)

        assert result is not None
        expected_hash = hashlib.sha256(b"my_canon").hexdigest()
        assert result.ir_hash == expected_hash

    def test_individual_genome_is_serialized_canonical(self) -> None:
        backend, client = _make_happy_path(canonical="my_canon")
        gen = ValidatedGenerator(backend=backend, llm_client=client, model="test-model")

        result = gen.generate(prompt="write code", system="system", temperature=0.7)

        assert result is not None
        assert result.genome == "my_canon"

    def test_individual_ir_is_set(self) -> None:
        backend, client = _make_happy_path()
        gen = ValidatedGenerator(backend=backend, llm_client=client, model="test-model")

        result = gen.generate(prompt="write code", system="system", temperature=0.7)

        assert result is not None
        assert result.ir is not None


# ---------------------------------------------------------------------------
# Extraction failure → retry → None after max_attempts
# ---------------------------------------------------------------------------


class TestExtractionFailure:
    def test_none_extraction_retries_and_returns_none(self) -> None:
        """When extract_genome always returns None, generate exhausts retries."""
        # Backend that never extracts anything
        backend = _MockBackend(extract_map={})
        client = _MockLLMClient(
            [
                _LLMResponse(text="bad1"),
                _LLMResponse(text="bad2"),
                _LLMResponse(text="bad3"),
            ]
        )
        gen = ValidatedGenerator(
            backend=backend, llm_client=client, model="test-model", max_attempts=3
        )

        result = gen.generate(prompt="p", system="s", temperature=0.5)
        assert result is None

    def test_extraction_succeeds_on_retry(self) -> None:
        """If extraction fails once then succeeds, we get an Individual."""
        ir = _MockIR("ok")
        backend = _MockBackend(
            extract_map={"bad": None, "good": "genome"},
            parse_map={"genome": ir},
            violations_map={},
        )
        client = _MockLLMClient(
            [
                _LLMResponse(text="bad"),
                _LLMResponse(text="good"),
            ]
        )
        gen = ValidatedGenerator(
            backend=backend, llm_client=client, model="test-model", max_attempts=3
        )

        result = gen.generate(prompt="p", system="s", temperature=0.5)
        assert result is not None
        assert isinstance(result, Individual)


# ---------------------------------------------------------------------------
# Parse failure → retry
# ---------------------------------------------------------------------------


class TestParseFailure:
    def test_parse_failure_retries_and_returns_none(self) -> None:
        """When parse always returns None, generate exhausts retries."""
        backend = _MockBackend(
            extract_map={"raw": "genome"},
            parse_map={},  # genome not in map → returns None
        )
        client = _MockLLMClient(
            [
                _LLMResponse(text="raw"),
                _LLMResponse(text="raw"),
                _LLMResponse(text="raw"),
            ]
        )
        gen = ValidatedGenerator(
            backend=backend, llm_client=client, model="test-model", max_attempts=3
        )

        result = gen.generate(prompt="p", system="s", temperature=0.5)
        assert result is None


# ---------------------------------------------------------------------------
# Structural violations → retry
# ---------------------------------------------------------------------------


class TestStructuralViolations:
    def test_violations_cause_retry_and_none(self) -> None:
        """When validate_structure always returns violations, returns None."""
        ir = _MockIR("bad_struct")
        backend = _MockBackend(
            extract_map={"raw": "genome"},
            parse_map={"genome": ir},
            violations_map={"bad_struct": ["missing field X"]},
        )
        client = _MockLLMClient(
            [
                _LLMResponse(text="raw"),
                _LLMResponse(text="raw"),
                _LLMResponse(text="raw"),
            ]
        )
        gen = ValidatedGenerator(
            backend=backend, llm_client=client, model="test-model", max_attempts=3
        )

        result = gen.generate(prompt="p", system="s", temperature=0.5)
        assert result is None

    def test_violations_then_success(self) -> None:
        """If first attempt has violations but second is clean, succeed."""
        bad_ir = _MockIR("bad_struct")
        good_ir = _MockIR("good_struct")
        backend = _MockBackend(
            extract_map={"raw_bad": "genome_bad", "raw_good": "genome_good"},
            parse_map={"genome_bad": bad_ir, "genome_good": good_ir},
            violations_map={"bad_struct": ["violation!"]},  # good_struct has none
        )
        client = _MockLLMClient(
            [
                _LLMResponse(text="raw_bad"),
                _LLMResponse(text="raw_good"),
            ]
        )
        gen = ValidatedGenerator(
            backend=backend, llm_client=client, model="test-model", max_attempts=3
        )

        result = gen.generate(prompt="p", system="s", temperature=0.5)
        assert result is not None
        assert isinstance(result, Individual)


# ---------------------------------------------------------------------------
# API errors caught and retried
# ---------------------------------------------------------------------------


class TestApiErrors:
    def test_api_error_retried_then_none(self) -> None:
        """If the LLM client always raises, generate returns None."""
        backend, _ = _make_happy_path()
        client = _MockLLMClient(
            [
                RuntimeError("API timeout"),
                RuntimeError("API timeout"),
                RuntimeError("API timeout"),
            ]
        )
        gen = ValidatedGenerator(
            backend=backend, llm_client=client, model="test-model", max_attempts=3
        )

        result = gen.generate(prompt="p", system="s", temperature=0.5)
        assert result is None

    def test_api_error_then_success(self) -> None:
        """If the first call raises but the second succeeds, we get a result."""
        ir = _MockIR("canon")
        backend = _MockBackend(
            extract_map={"good_raw": "genome"},
            parse_map={"genome": ir},
            violations_map={},
        )
        client = _MockLLMClient(
            [
                RuntimeError("API timeout"),
                _LLMResponse(text="good_raw"),
            ]
        )
        gen = ValidatedGenerator(
            backend=backend, llm_client=client, model="test-model", max_attempts=3
        )

        result = gen.generate(prompt="p", system="s", temperature=0.5)
        assert result is not None
        assert isinstance(result, Individual)


# ---------------------------------------------------------------------------
# Async variant
# ---------------------------------------------------------------------------


class TestAsyncGenerate:
    @pytest.mark.asyncio
    async def test_async_valid_output_returns_individual(self) -> None:
        backend, client = _make_happy_path()
        gen = ValidatedGenerator(backend=backend, llm_client=client, model="test-model")

        result = await gen.async_generate(prompt="write code", system="system", temperature=0.7)

        assert result is not None
        assert isinstance(result, Individual)

    @pytest.mark.asyncio
    async def test_async_extraction_failure_returns_none(self) -> None:
        backend = _MockBackend(extract_map={})
        client = _MockLLMClient(
            [
                _LLMResponse(text="bad1"),
                _LLMResponse(text="bad2"),
                _LLMResponse(text="bad3"),
            ]
        )
        gen = ValidatedGenerator(
            backend=backend, llm_client=client, model="test-model", max_attempts=3
        )

        result = await gen.async_generate(prompt="p", system="s", temperature=0.5)
        assert result is None

    @pytest.mark.asyncio
    async def test_async_api_error_then_success(self) -> None:
        ir = _MockIR("canon")
        backend = _MockBackend(
            extract_map={"good_raw": "genome"},
            parse_map={"genome": ir},
            violations_map={},
        )
        client = _MockLLMClient(
            [
                RuntimeError("API timeout"),
                _LLMResponse(text="good_raw"),
            ]
        )
        gen = ValidatedGenerator(
            backend=backend, llm_client=client, model="test-model", max_attempts=3
        )

        result = await gen.async_generate(prompt="p", system="s", temperature=0.5)
        assert result is not None
        assert isinstance(result, Individual)

    @pytest.mark.asyncio
    async def test_async_individual_has_correct_hash(self) -> None:
        backend, client = _make_happy_path(canonical="async_canon")
        gen = ValidatedGenerator(backend=backend, llm_client=client, model="test-model")

        result = await gen.async_generate(prompt="write code", system="system", temperature=0.7)

        assert result is not None
        expected_hash = hashlib.sha256(b"async_canon").hexdigest()
        assert result.ir_hash == expected_hash


# ---------------------------------------------------------------------------
# max_attempts configuration
# ---------------------------------------------------------------------------


class TestMaxAttempts:
    def test_default_max_attempts_is_three(self) -> None:
        backend, client = _make_happy_path()
        gen = ValidatedGenerator(backend=backend, llm_client=client, model="test-model")
        assert gen.max_attempts == 3

    def test_custom_max_attempts(self) -> None:
        backend, client = _make_happy_path()
        gen = ValidatedGenerator(
            backend=backend, llm_client=client, model="test-model", max_attempts=5
        )
        assert gen.max_attempts == 5

    def test_single_attempt_returns_none_on_failure(self) -> None:
        backend = _MockBackend(extract_map={})
        client = _MockLLMClient([_LLMResponse(text="bad")])
        gen = ValidatedGenerator(
            backend=backend, llm_client=client, model="test-model", max_attempts=1
        )

        result = gen.generate(prompt="p", system="s", temperature=0.5)
        assert result is None
