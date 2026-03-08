# REPL Inline Verification, Two-Tier Fitness, and Rate Limit Resilience

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the search stagnation caused by REPL false positives (best_fitness stuck at 0.0) and prevent rate-limit crashes.

**Architecture:** Add fast REPL `cmd`-based inline verification after step-by-step evaluation detects proof completion. False positives get fitness 0.9 instead of -inf, preserving gradient signal. Keep `lake env lean` as the final gate. Improve LLM client retry with jitter and longer backoff.

**Tech Stack:** Python 3.11, asyncio, Anthropic SDK, Lean 4 REPL protocol

---

## Background

### The REPL False Positive Problem

The step-by-step REPL `tactic` command is less strict than full kernel elaboration. Tactics like `ring` and `norm_num` can report "no goals" interactively for theorems they cannot actually prove. The current flow:

1. REPL says proof complete (fitness=1.0)
2. `lake env lean` verification rejects it (~8s per check)
3. Engine marks it `feasible=False` (fitness effectively -inf)
4. **Every** fitness=1.0 proof is a false positive -> best_fitness=0.0 forever

### The `format_proof` Indentation Bug

`format_proof` does `.strip()` on every line, destroying relative indentation for block-structured proofs (e.g., continuations under `·` bullets). Even a correct nested proof would fail verification.

### Rate Limiting

Current retry: 3 attempts, 1s/2s/4s exponential backoff. Too aggressive for 10k tokens/min limits that need 30-60s waits.

---

## Task 1: Fix `format_proof` Indentation Preservation

**Files:**
- Modify: `evoforge/backends/lean/backend.py:403-413`
- Test: `tests/test_lean/test_verification.py`

**Step 1: Write the failing test**

Add to `tests/test_lean/test_verification.py` in a new `TestFormatProof` class:

```python
class TestFormatProof:
    def test_preserves_relative_indentation(self) -> None:
        """format_proof preserves relative indentation for block proofs."""
        backend = _make_backend(
            theorem="theorem foo : True",
            imports="import Mathlib",
        )
        genome = "by_cases h : x = 0\n  · simp [h]\n  · ring"
        result = backend.format_proof(genome)
        lines = result.strip().split("\n")
        # The · lines should be MORE indented than by_cases
        by_cases_line = [l for l in lines if "by_cases" in l][0]
        cdot_line = [l for l in lines if "·" in l][0]
        by_cases_indent = len(by_cases_line) - len(by_cases_line.lstrip())
        cdot_indent = len(cdot_line) - len(cdot_line.lstrip())
        assert cdot_indent > by_cases_indent

    def test_flat_tactics_indented_uniformly(self) -> None:
        """Flat tactics all get 2-space indent under 'by'."""
        backend = _make_backend(theorem="theorem foo : True")
        genome = "intro x\nsimp\nring"
        result = backend.format_proof(genome)
        tactic_lines = [l for l in result.strip().split("\n") if l.strip() and ":= by" not in l and "import" not in l]
        for line in tactic_lines:
            indent = len(line) - len(line.lstrip())
            assert indent == 2, f"Expected 2-space indent, got {indent}: {line!r}"

    def test_empty_lines_skipped(self) -> None:
        """Empty lines in genome are skipped."""
        backend = _make_backend(theorem="theorem foo : True")
        genome = "intro x\n\nsimp"
        result = backend.format_proof(genome)
        assert "\n\n" not in result.split(":= by")[1]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lean/test_verification.py::TestFormatProof -v`
Expected: FAIL on `test_preserves_relative_indentation` (`.strip()` destroys indentation)

**Step 3: Write minimal implementation**

Replace `format_proof` in `evoforge/backends/lean/backend.py:403-413`:

```python
def format_proof(self, genome: str) -> str:
    """Wrap tactic genome into a complete, standalone Lean 4 proof."""
    lines: list[str] = []
    if self._imports:
        lines.append(self._imports)
        lines.append("")
    lines.append(f"{self.theorem_statement} := by")
    tactic_lines = [l for l in genome.split("\n") if l.strip()]
    if tactic_lines:
        min_indent = min(len(l) - len(l.lstrip()) for l in tactic_lines)
        for l in tactic_lines:
            lines.append("  " + l[min_indent:])
    return "\n".join(lines) + "\n"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lean/test_verification.py::TestFormatProof -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `uv run pytest -x -v`
Expected: All pass (format_proof is only used in `verify_proof` and tests)

**Step 6: Commit**

```bash
git add evoforge/backends/lean/backend.py tests/test_lean/test_verification.py
git commit -m "Fix format_proof to preserve relative indentation for block proofs"
```

---

## Task 2: Store Import Env and Add REPL Cmd Verification

**Files:**
- Modify: `evoforge/backends/lean/backend.py:93-166` (constructor + startup + new method)
- Test: `tests/test_lean/test_verification.py`

**Step 1: Write the failing tests**

Add to `tests/test_lean/test_verification.py`:

```python
class TestREPLCmdVerification:
    def test_example_statement_replaces_theorem_name(self) -> None:
        """_example_statement() replaces 'theorem <name>' with 'example'."""
        backend = _make_backend(
            theorem="theorem norm_le_one {φ : ℝ → ℂ} (hφ : IsPositiveDefinite φ) : ‖φ ξ‖ ≤ 1",
        )
        result = backend._example_statement()
        assert result.startswith("example ")
        assert "norm_le_one" not in result
        assert "IsPositiveDefinite" in result

    async def test_verify_via_repl_cmd_success(self) -> None:
        """REPL cmd verification returns True on clean success."""
        backend = _make_backend()
        # Mock REPL that returns success (no errors, no sorries)
        mock_repl = AsyncMock()
        mock_repl.send_command = AsyncMock(return_value={"env": 2})
        backend._repl = mock_repl
        backend._repl_lock = asyncio.Lock()
        backend._import_env = 0

        result = await backend._verify_via_repl_cmd("exact rfl")
        assert result is True

    async def test_verify_via_repl_cmd_failure_on_error(self) -> None:
        """REPL cmd verification returns False on error response."""
        backend = _make_backend()
        mock_repl = AsyncMock()
        mock_repl.send_command = AsyncMock(
            return_value={"message": "type mismatch", "severity": "error"}
        )
        backend._repl = mock_repl
        backend._repl_lock = asyncio.Lock()
        backend._import_env = 0

        result = await backend._verify_via_repl_cmd("ring")
        assert result is False

    async def test_verify_via_repl_cmd_failure_on_sorry(self) -> None:
        """REPL cmd verification returns False when sorries present."""
        backend = _make_backend()
        mock_repl = AsyncMock()
        mock_repl.send_command = AsyncMock(
            return_value={"env": 2, "sorries": [{"proofState": 1}]}
        )
        backend._repl = mock_repl
        backend._repl_lock = asyncio.Lock()
        backend._import_env = 0

        result = await backend._verify_via_repl_cmd("sorry")
        assert result is False
```

Add `import asyncio` and update the `from unittest.mock` import to include `AsyncMock` at the top of the test file.

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lean/test_verification.py::TestREPLCmdVerification -v`
Expected: FAIL (`_example_statement` and `_verify_via_repl_cmd` don't exist)

**Step 3: Write minimal implementation**

In `evoforge/backends/lean/backend.py`:

Add `import re` to the imports at the top.

Add `self._import_env: int | None = None` to `__init__` (after line 108).

Update `startup()` to store the import env:

```python
async def startup(self) -> None:
    """Start the Lean REPL and initialize the stepwise evaluator."""
    self._repl = LeanREPLProcess(self.project_dir, self.repl_path)
    await self._repl.start()

    theorem_cmd = f"{self.theorem_statement} := by\n sorry"
    init_cmd: dict[str, object] = {"cmd": theorem_cmd}
    if self._imports:
        resp = await self._repl.send_command({"cmd": self._imports})
        self._import_env = int(resp.get("env", 0))
        init_cmd["env"] = self._import_env
    else:
        self._import_env = None

    resp = await self._repl.send_command(init_cmd)
    initial_proof_state = 0
    sorries = resp.get("sorries", [])
    if isinstance(sorries, list) and sorries and "proofState" in sorries[0]:
        initial_proof_state = int(sorries[0]["proofState"])
    else:
        logger.warning("REPL init response missing 'sorries': %s", resp)

    self._evaluator = LeanStepwiseEvaluator(
        self._repl,
        initial_proof_state=initial_proof_state,
        prefix_cache=self._prefix_cache,
    )
    logger.info("Lean REPL started and evaluator initialized")
```

Add two new methods to `LeanBackend`:

```python
def _example_statement(self) -> str:
    """Convert 'theorem <name> ...' to 'example ...' for REPL cmd verification."""
    return re.sub(r"^theorem\s+\S+", "example", self.theorem_statement)

async def _verify_via_repl_cmd(self, genome: str) -> bool:
    """Verify a proof by sending it as a complete cmd to the REPL.

    Uses ``example`` instead of ``theorem`` to avoid naming conflicts.
    Returns True only if the REPL accepts the proof without errors or sorries.
    """
    if self._repl is None:
        return False

    # Build the example command preserving indentation
    stmt = self._example_statement()
    tactic_lines = [l for l in genome.split("\n") if l.strip()]
    if not tactic_lines:
        return False
    min_indent = min(len(l) - len(l.lstrip()) for l in tactic_lines)
    body = "\n".join("  " + l[min_indent:] for l in tactic_lines)
    cmd_text = f"{stmt} := by\n{body}"

    cmd: dict[str, object] = {"cmd": cmd_text}
    if self._import_env is not None:
        cmd["env"] = self._import_env

    try:
        resp = await self._repl.send_command(cmd)
    except Exception:
        logger.debug("REPL cmd verification raised an exception", exc_info=True)
        return False

    # Check for errors
    if "severity" in resp and resp["severity"] == "error":
        return False
    if "message" in resp and "env" not in resp:
        return False

    # Check for sorries
    sorries = resp.get("sorries", [])
    if sorries:
        return False

    return True
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lean/test_verification.py::TestREPLCmdVerification -v`
Expected: PASS

**Step 5: Commit**

```bash
git add evoforge/backends/lean/backend.py tests/test_lean/test_verification.py
git commit -m "Add REPL cmd verification and store import env during startup"
```

---

## Task 3: Two-Tier Fitness — Inline Verification in Backend.evaluate()

**Files:**
- Modify: `evoforge/backends/lean/backend.py:160-166` (evaluate method)
- Modify: `evoforge/core/engine.py:772-805` (adjust verification gate)
- Test: `tests/test_lean/test_verification.py`
- Test: `tests/test_lean/test_stepwise.py`

**Step 1: Write the failing tests**

Add to `tests/test_lean/test_verification.py`:

```python
class TestTwoTierFitness:
    async def test_repl_complete_cmd_verified_gets_1_0(self) -> None:
        """Proof that passes both step-by-step and cmd verification gets fitness=1.0."""
        backend = _make_backend(project_dir="/tmp/test")
        # Mock evaluator returns proof_complete
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(return_value=(
            Fitness(primary=1.0, auxiliary={"proof_complete": 1.0, "steps_succeeded": 1.0, "goals_remaining": 0.0}, constraints={}, feasible=True),
            MagicMock(),
            MagicMock(),
        ))
        backend._evaluator = mock_evaluator
        backend._repl_lock = asyncio.Lock()
        # Mock REPL cmd verification succeeds
        backend._verify_via_repl_cmd = AsyncMock(return_value=True)

        fitness, _, _ = await backend.evaluate(MagicMock())
        assert fitness.primary == 1.0
        assert fitness.feasible is True

    async def test_repl_complete_cmd_rejected_gets_0_9(self) -> None:
        """Proof that passes step-by-step but fails cmd verification gets fitness=0.9."""
        backend = _make_backend(project_dir="/tmp/test")
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(return_value=(
            Fitness(primary=1.0, auxiliary={"proof_complete": 1.0, "steps_succeeded": 1.0, "goals_remaining": 0.0}, constraints={}, feasible=True),
            MagicMock(),
            MagicMock(),
        ))
        backend._evaluator = mock_evaluator
        backend._repl_lock = asyncio.Lock()
        # Mock REPL cmd verification fails
        backend._verify_via_repl_cmd = AsyncMock(return_value=False)

        ir = MagicMock()
        ir.serialize = MagicMock(return_value="ring")
        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.primary == pytest.approx(0.9)
        assert fitness.feasible is True  # NOT -inf!

    async def test_partial_proof_unchanged(self) -> None:
        """Partial proof (fitness < 1.0) is not affected by verification."""
        backend = _make_backend(project_dir="/tmp/test")
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(return_value=(
            Fitness(primary=0.5, auxiliary={"proof_complete": 0.0, "steps_succeeded": 1.0, "goals_remaining": 1.0}, constraints={}, feasible=True),
            MagicMock(),
            MagicMock(),
        ))
        backend._evaluator = mock_evaluator
        backend._repl_lock = asyncio.Lock()

        fitness, _, _ = await backend.evaluate(MagicMock())
        assert fitness.primary == pytest.approx(0.5)
```

Add `import pytest` to the test file imports if not already present.

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lean/test_verification.py::TestTwoTierFitness -v`
Expected: FAIL (evaluate doesn't do inline verification yet)

**Step 3: Write minimal implementation**

Update `evaluate` in `evoforge/backends/lean/backend.py:160-166`:

```python
async def evaluate(self, ir: Any, seed: int | None = None) -> tuple[Fitness, Any, Any]:
    """Evaluate a tactic sequence via the stepwise evaluator.

    If step-by-step evaluation reports proof complete (fitness=1.0),
    performs inline REPL cmd verification. Verified proofs keep 1.0;
    false positives are downgraded to 0.9 (still feasible, preserving
    gradient signal for the evolutionary search).
    """
    if self._evaluator is None:
        raise RuntimeError("LeanBackend.startup() must be called before evaluate()")
    async with self._repl_lock:
        fitness, diagnostics, trace = await self._evaluator.evaluate(ir)

        # Inline verification for proof-complete results
        if fitness.primary >= 1.0 and fitness.auxiliary.get("proof_complete", 0.0) >= 1.0:
            genome = ir.serialize() if hasattr(ir, "serialize") else str(ir)
            cmd_verified = await self._verify_via_repl_cmd(genome)
            if not cmd_verified:
                logger.info(
                    "REPL step-by-step said complete but cmd verification failed — "
                    "downgrading to 0.9: %s",
                    genome[:80],
                )
                fitness = Fitness(
                    primary=0.9,
                    auxiliary={
                        **fitness.auxiliary,
                        "proof_complete": 0.0,
                        "cmd_verified": 0.0,
                    },
                    constraints=fitness.constraints,
                    feasible=True,
                )
            else:
                fitness = Fitness(
                    primary=fitness.primary,
                    auxiliary={**fitness.auxiliary, "cmd_verified": 1.0},
                    constraints=fitness.constraints,
                    feasible=fitness.feasible,
                )

    return fitness, diagnostics, trace
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lean/test_verification.py::TestTwoTierFitness -v`
Expected: PASS

**Step 5: Update engine verification gate**

In `evoforge/core/engine.py:772-805`, update `_verify_perfect_individuals` to only run `lake env lean` on cmd-verified proofs (fitness=1.0 with `cmd_verified=1.0`):

```python
async def _verify_perfect_individuals(self, individuals: list[Individual]) -> None:
    """Verify fitness=1.0 individuals via backend.verify_proof().

    Only runs on individuals that passed REPL cmd verification
    (cmd_verified=1.0 in auxiliary). Uses a cache keyed by ir_hash.
    On failure, records the genome as a dead end in search memory.
    """
    for ind in individuals:
        if (
            ind.fitness is not None
            and ind.fitness.primary >= 1.0
            and ind.fitness.auxiliary.get("cmd_verified", 0.0) >= 1.0
        ):
            cache_hit = ind.ir_hash in self._verification_cache
            if cache_hit:
                verified = self._verification_cache[ind.ir_hash]
            else:
                verified = await self.backend.verify_proof(ind.genome)
                self._verification_cache[ind.ir_hash] = verified

            if not verified:
                if cache_hit:
                    logger.debug(
                        "Proof failed verification (cached) — marking infeasible: %s",
                        ind.genome[:80],
                    )
                else:
                    logger.warning(
                        "Proof failed verification — marking infeasible: %s",
                        ind.genome[:80],
                    )
                    self._memory.record_verification_failure(ind.genome)
                ind.fitness = Fitness(
                    primary=ind.fitness.primary,
                    auxiliary=ind.fitness.auxiliary,
                    constraints=ind.fitness.constraints,
                    feasible=False,
                )
```

**Step 6: Run full test suite**

Run: `uv run pytest -x -v`
Expected: All pass. Existing verification gate tests may need minor updates if they check for `cmd_verified` in auxiliary.

**Step 7: Commit**

```bash
git add evoforge/backends/lean/backend.py evoforge/core/engine.py tests/test_lean/test_verification.py
git commit -m "Add two-tier fitness with inline REPL cmd verification"
```

---

## Task 4: Rate Limit Resilience

**Files:**
- Modify: `evoforge/llm/client.py`
- Create: `tests/test_core/test_llm_client.py`

**Step 1: Write the failing tests**

Create `tests/test_core/test_llm_client.py`:

```python
"""Tests for LLM client retry logic and cost estimation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import anthropic

from evoforge.llm.client import LLMClient, LLMResponse, _pricing_for_model


class TestRetryWithJitter:
    async def test_retry_respects_max_retries(self) -> None:
        """Client retries up to max_retries times before raising."""
        client = LLMClient(api_key="test", max_retries=3, base_delay=0.01)
        call_count = 0

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            error = anthropic.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body={"error": {"message": "rate limited"}},
            )
            mock_instance.messages.create = AsyncMock(side_effect=error)
            mock_cls.return_value = mock_instance

            with pytest.raises(RuntimeError, match="after 3 retries"):
                await client.async_generate("test", "sys", "haiku", 0.7)

            assert mock_instance.messages.create.call_count == 3

    async def test_retry_delay_increases_exponentially(self) -> None:
        """Each retry waits longer than the previous one."""
        delays: list[float] = []
        original_sleep = asyncio.sleep

        async def capture_sleep(delay: float) -> None:
            delays.append(delay)

        client = LLMClient(api_key="test", max_retries=4, base_delay=0.01)

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            error = anthropic.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body={"error": {"message": "rate limited"}},
            )
            mock_instance.messages.create = AsyncMock(side_effect=error)
            mock_cls.return_value = mock_instance

            with patch("evoforge.llm.client.asyncio.sleep", side_effect=capture_sleep):
                with pytest.raises(RuntimeError):
                    await client.async_generate("test", "sys", "haiku", 0.7)

        # Each delay should be >= the previous (exponential + jitter)
        assert len(delays) == 4
        # Base delay grows: 0.01, 0.02, 0.04, 0.08 (before jitter)
        # With jitter added, each should be > base
        for d in delays:
            assert d >= 0.01  # at least base_delay

    async def test_delay_capped_at_max(self) -> None:
        """Delay should never exceed max_delay."""
        delays: list[float] = []

        async def capture_sleep(delay: float) -> None:
            delays.append(delay)

        client = LLMClient(
            api_key="test", max_retries=10, base_delay=1.0, max_delay=5.0
        )

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            error = anthropic.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body={"error": {"message": "rate limited"}},
            )
            mock_instance.messages.create = AsyncMock(side_effect=error)
            mock_cls.return_value = mock_instance

            with patch("evoforge.llm.client.asyncio.sleep", side_effect=capture_sleep):
                with pytest.raises(RuntimeError):
                    await client.async_generate("test", "sys", "haiku", 0.7)

        for d in delays:
            assert d <= 5.0

    async def test_success_on_retry(self) -> None:
        """Client succeeds if a retry works after initial failure."""
        client = LLMClient(api_key="test", max_retries=3, base_delay=0.01)

        error = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={"error": {"message": "rate limited"}},
        )
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="result")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.messages.create = AsyncMock(
                side_effect=[error, mock_response]
            )
            mock_cls.return_value = mock_instance

            with patch("evoforge.llm.client.asyncio.sleep", new_callable=AsyncMock):
                result = await client.async_generate("test", "sys", "haiku", 0.7)

        assert result.text == "result"


class TestCostEstimation:
    def test_haiku_pricing(self) -> None:
        cost = LLMClient.estimate_cost(1_000_000, 1_000_000, "claude-haiku-4-5-20251001")
        assert cost == pytest.approx(0.25 + 1.25)

    def test_unknown_model_uses_sonnet_default(self) -> None:
        cost = LLMClient.estimate_cost(1_000_000, 1_000_000, "unknown-model")
        assert cost == pytest.approx(3.0 + 15.0)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_llm_client.py -v`
Expected: FAIL (missing `max_delay` parameter, no jitter)

**Step 3: Write minimal implementation**

Replace `evoforge/llm/client.py`:

```python
"""LLM client with retry logic and cost estimation."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)

# Pricing per million tokens: (input_cost, output_cost) in USD
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "sonnet": (3.0, 15.0),
    "haiku": (0.25, 1.25),
    "opus": (15.0, 75.0),
}

_DEFAULT_PRICING = _MODEL_PRICING["sonnet"]


def _pricing_for_model(model: str) -> tuple[float, float]:
    """Return (input_cost, output_cost) per million tokens for the given model name."""
    model_lower = model.lower()
    for key, pricing in _MODEL_PRICING.items():
        if key in model_lower:
            return pricing
    return _DEFAULT_PRICING


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMClient:
    """Thin wrapper around the Anthropic API with retry and cost estimation."""

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 6,
        base_delay: float = 2.0,
        max_delay: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

    def _compute_delay(self, attempt: int) -> float:
        """Compute retry delay with exponential backoff, jitter, and cap."""
        delay = self._base_delay * (2**attempt)
        jitter = random.uniform(0, self._base_delay)
        return min(delay + jitter, self._max_delay)

    def generate(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call the Anthropic API synchronously with exponential-backoff retry."""
        client = anthropic.Anthropic(api_key=self._api_key)
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text  # type: ignore[union-attr]
                return LLMResponse(
                    text=text,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    model=model,
                )
            except (anthropic.RateLimitError, anthropic.APIError) as exc:
                last_exc = exc
                delay = self._compute_delay(attempt)
                logger.warning(
                    "LLM API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)

        msg = f"LLM call failed after {self._max_retries} retries"
        raise RuntimeError(msg) from last_exc

    async def async_generate(
        self,
        prompt: str,
        system: str,
        model: str,
        temperature: float,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call the Anthropic API asynchronously with exponential-backoff retry."""
        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text  # type: ignore[union-attr]
                return LLMResponse(
                    text=text,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    model=model,
                )
            except (anthropic.RateLimitError, anthropic.APIError) as exc:
                last_exc = exc
                delay = self._compute_delay(attempt)
                logger.warning(
                    "LLM API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        msg = f"LLM call failed after {self._max_retries} retries"
        raise RuntimeError(msg) from last_exc

    @staticmethod
    def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate USD cost for the given token counts and model."""
        input_rate, output_rate = _pricing_for_model(model)
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_core/test_llm_client.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `uv run pytest -x -v`
Expected: All pass (default parameter change from 3->6 retries, 1.0->2.0 base_delay is backward compatible)

**Step 6: Commit**

```bash
git add evoforge/llm/client.py tests/test_core/test_llm_client.py
git commit -m "Improve rate limit retry with jitter, higher limits, and delay cap"
```

---

## Task 5: Quality Gate and Final Verification

**Files:**
- No new code — verification only

**Step 1: Run full quality gate**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v
```

Expected: All pass, 0 errors.

**Step 2: Manual smoke test (optional)**

```bash
bash scripts/run_with_keychain.sh --max-generations 5 --output-dir /tmp/evotest_verify --verify
```

Expected:
- REPL false positives downgraded to 0.9 (not -inf)
- `best_fitness` should show values > 0.0
- No crashes from rate limiting
- `cmd_verified` appears in logs for proof-complete individuals

**Step 3: Update memory file**

Update MEMORY.md with the new verification architecture.

---

## Dependency Graph

```
Task 1 (format_proof fix)  ──┐
                              ├── Task 3 (two-tier fitness) ── Task 5 (quality gate)
Task 2 (REPL cmd verify)  ───┘                                     │
                                                                    │
Task 4 (rate limit retry) ─────────────────────────────────────────┘
```

Tasks 1, 2, and 4 are independent and can run in parallel.
Task 3 depends on Tasks 1 and 2.
Task 5 depends on all previous tasks.

## Graceful Degradation Note

The existing `_mutate_one` exception handler (engine.py:483-485) already catches `RuntimeError` from exhausted retries and returns `None`, causing the mutation to silently fall back to the next parent. With the improved retry logic (6 attempts, up to 120s max delay), rate limit exhaustion should be extremely rare. When it does happen, that individual simply gets no LLM mutation for that generation — the search continues with cheap operators only.
