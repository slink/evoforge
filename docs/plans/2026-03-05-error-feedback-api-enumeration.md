# Error Feedback & API Enumeration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the 0.9→1.0 fitness gap by (1) feeding REPL cmd-verification errors back to the LLM at both individual and population level, and (2) extracting available API from Lean source files and injecting it into the system prompt.

**Architecture:** Two independent features. Feature 1 threads cmd-verification error messages from `_verify_via_repl_cmd()` through `LeanDiagnostics` (individual) and `SearchMemory.dead_ends` (population). Feature 2 adds a new `api_extractor.py` module that parses `.lean` files for declarations, called once during `startup()`, results injected into `system_prompt.j2`. Informed by Goedel-Prover-V2 (arXiv:2508.03613) and APRIL (arXiv:2602.02990).

**Tech Stack:** Python 3.11, pytest-asyncio, Pydantic, Jinja2, regex for Lean parsing.

---

## Group A: Cmd Error Capture (Feature 1 — no file overlap with Group B)

### Task 1: Add cmd error fields to LeanDiagnostics

**Files:**
- Modify: `evoforge/backends/lean/evaluator.py:67-109`
- Test: `tests/test_lean/test_verification.py`

**Step 1: Write the failing test**

```python
# tests/test_lean/test_verification.py — add at end of file

class TestCmdErrorDiagnostics:
    """Tests for cmd verification error fields on LeanDiagnostics."""

    def test_cmd_error_fields_default_none(self):
        diag = LeanDiagnostics(
            success=True,
            goals_remaining=0,
            goal_types=[],
            goal_contexts=[],
            error_type=None,
            error_message=None,
            stuck_tactic_index=None,
            stuck_tactic=None,
            steps_succeeded=3,
            metavar_count=0,
        )
        assert diag.cmd_error_message is None
        assert diag.cmd_verification_attempted is False

    def test_cmd_error_appears_in_summary(self):
        diag = LeanDiagnostics(
            success=True,
            goals_remaining=0,
            goal_types=[],
            goal_contexts=[],
            error_type=None,
            error_message=None,
            stuck_tactic_index=None,
            stuck_tactic=None,
            steps_succeeded=3,
            metavar_count=0,
            cmd_verification_attempted=True,
            cmd_error_message="unknown identifier 'sum_nonneg'",
        )
        summary = diag.summary()
        assert "cmd verification failed" in summary.lower()
        assert "unknown identifier" in summary

    def test_cmd_success_no_error_in_summary(self):
        diag = LeanDiagnostics(
            success=True,
            goals_remaining=0,
            goal_types=[],
            goal_contexts=[],
            error_type=None,
            error_message=None,
            stuck_tactic_index=None,
            stuck_tactic=None,
            steps_succeeded=3,
            metavar_count=0,
            cmd_verification_attempted=True,
            cmd_error_message=None,
        )
        summary = diag.summary()
        assert "cmd verification failed" not in summary.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lean/test_verification.py::TestCmdErrorDiagnostics -v`
Expected: FAIL — `LeanDiagnostics` doesn't accept `cmd_error_message` or `cmd_verification_attempted` kwargs.

**Step 3: Write minimal implementation**

In `evoforge/backends/lean/evaluator.py`, add two fields to the `LeanDiagnostics` dataclass (after line 83):

```python
@dataclass
class LeanDiagnostics:
    # ... existing fields (lines 74-83) ...
    steps_succeeded: int
    metavar_count: int
    cmd_verification_attempted: bool = False
    cmd_error_message: str | None = None
```

Update `summary()` method (after line 103, before the text join):

```python
    def summary(self, max_tokens: int = 500) -> str:
        parts: list[str] = []
        # ... existing logic (lines 89-102) ...

        if self.cmd_verification_attempted and self.cmd_error_message:
            parts.append(f"Cmd verification failed: {self.cmd_error_message}")

        text = " ".join(parts)
        # ... rest unchanged ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lean/test_verification.py::TestCmdErrorDiagnostics -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add evoforge/backends/lean/evaluator.py tests/test_lean/test_verification.py
git commit -m "Add cmd_error_message and cmd_verification_attempted to LeanDiagnostics"
```

---

### Task 2: Return error message from `_verify_via_repl_cmd()`

**Files:**
- Modify: `evoforge/backends/lean/backend.py:470-507`
- Test: `tests/test_lean/test_verification.py`

**Step 1: Write the failing test**

```python
# tests/test_lean/test_verification.py — add to existing test class or new class

class TestVerifyViaCmdError:
    """Tests for _verify_via_repl_cmd returning error messages."""

    @pytest.fixture
    def backend(self):
        b = _make_backend(imports="import LeanLevy")
        b._repl = AsyncMock()
        b._import_env = 0
        return b

    @pytest.mark.asyncio
    async def test_returns_error_message_on_severity_error(self, backend):
        backend._repl.send_command = AsyncMock(
            return_value={"severity": "error", "message": "unknown identifier 'sum_nonneg'"}
        )
        ok, err = await backend._verify_via_repl_cmd("simp")
        assert ok is False
        assert err == "unknown identifier 'sum_nonneg'"

    @pytest.mark.asyncio
    async def test_returns_error_message_on_message_without_env(self, backend):
        backend._repl.send_command = AsyncMock(
            return_value={"message": "type mismatch\nhas type Nat\nexpected Real"}
        )
        ok, err = await backend._verify_via_repl_cmd("simp")
        assert ok is False
        assert err == "type mismatch\nhas type Nat\nexpected Real"

    @pytest.mark.asyncio
    async def test_returns_none_on_success(self, backend):
        backend._repl.send_command = AsyncMock(return_value={"env": 42})
        ok, err = await backend._verify_via_repl_cmd("simp")
        assert ok is True
        assert err is None

    @pytest.mark.asyncio
    async def test_returns_sorry_message(self, backend):
        backend._repl.send_command = AsyncMock(
            return_value={"env": 42, "sorries": [{"proofState": 1}]}
        )
        ok, err = await backend._verify_via_repl_cmd("simp")
        assert ok is False
        assert "sorry" in err.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lean/test_verification.py::TestVerifyViaCmdError -v`
Expected: FAIL — `_verify_via_repl_cmd` returns `bool`, not `tuple`.

**Step 3: Write minimal implementation**

Replace `_verify_via_repl_cmd` in `evoforge/backends/lean/backend.py:470-507`:

```python
    async def _verify_via_repl_cmd(self, genome: str) -> tuple[bool, str | None]:
        """Verify a proof by sending it as a complete cmd to the REPL.

        Uses ``example`` instead of ``theorem`` to avoid naming conflicts.
        Returns (True, None) on success, or (False, error_message) on failure.
        """
        if self._repl is None:
            return False, "REPL not started"

        stmt = self._example_statement()
        body = _reindent_tactics(genome)
        if not body:
            return False, "empty proof body"
        cmd_text = f"{stmt} := by\n{body}"

        cmd: dict[str, object] = {"cmd": cmd_text}
        if self._import_env is not None:
            cmd["env"] = self._import_env

        try:
            resp = await self._repl.send_command(cmd)
        except Exception:
            logger.debug("REPL cmd verification raised an exception", exc_info=True)
            return False, "REPL exception"

        if "severity" in resp and resp["severity"] == "error":
            return False, str(resp.get("message", "unknown error"))
        if "message" in resp and "env" not in resp:
            return False, str(resp.get("message", "unknown error"))

        sorries = resp.get("sorries", [])
        if sorries:
            return False, "proof contains sorry"

        return True, None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lean/test_verification.py::TestVerifyViaCmdError -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add evoforge/backends/lean/backend.py tests/test_lean/test_verification.py
git commit -m "Return error message from _verify_via_repl_cmd instead of bare bool"
```

---

### Task 3: Thread cmd error into evaluate() and diagnostics

**Files:**
- Modify: `evoforge/backends/lean/backend.py:174-216`
- Test: `tests/test_lean/test_verification.py`

**Step 1: Write the failing test**

```python
# tests/test_lean/test_verification.py

class TestEvaluateCmdErrorThreading:
    """Tests that evaluate() threads cmd errors into diagnostics."""

    @pytest.fixture
    def backend(self):
        b = _make_backend(imports="import LeanLevy")
        b._repl = AsyncMock()
        b._repl_lock = asyncio.Lock()
        b._import_env = 0
        return b

    @pytest.mark.asyncio
    async def test_false_positive_stores_cmd_error_on_diagnostics(self, backend):
        """When cmd verification fails, diagnostics should contain the error."""
        ir = parse_tactic_sequence("simp")
        complete_fitness = Fitness(
            primary=1.0,
            auxiliary={"steps_succeeded": 1.0, "goals_remaining": 0.0, "proof_complete": 1.0},
            constraints={},
            feasible=True,
        )
        diag = LeanDiagnostics(
            success=True, goals_remaining=0, goal_types=[], goal_contexts=[],
            error_type=None, error_message=None, stuck_tactic_index=None,
            stuck_tactic=None, steps_succeeded=1, metavar_count=0,
        )
        backend._evaluator = AsyncMock()
        backend._evaluator.evaluate = AsyncMock(return_value=(complete_fitness, diag, None))
        backend._repl.send_command = AsyncMock(
            return_value={"severity": "error", "message": "unknown identifier 'sum_nonneg'"}
        )

        fitness, returned_diag, _ = await backend.evaluate(ir)
        assert fitness.primary == 0.9
        assert returned_diag.cmd_verification_attempted is True
        assert returned_diag.cmd_error_message == "unknown identifier 'sum_nonneg'"

    @pytest.mark.asyncio
    async def test_cmd_success_sets_attempted_no_error(self, backend):
        """When cmd verification passes, diagnostics should reflect success."""
        ir = parse_tactic_sequence("simp")
        complete_fitness = Fitness(
            primary=1.0,
            auxiliary={"steps_succeeded": 1.0, "goals_remaining": 0.0, "proof_complete": 1.0},
            constraints={},
            feasible=True,
        )
        diag = LeanDiagnostics(
            success=True, goals_remaining=0, goal_types=[], goal_contexts=[],
            error_type=None, error_message=None, stuck_tactic_index=None,
            stuck_tactic=None, steps_succeeded=1, metavar_count=0,
        )
        backend._evaluator = AsyncMock()
        backend._evaluator.evaluate = AsyncMock(return_value=(complete_fitness, diag, None))
        backend._repl.send_command = AsyncMock(return_value={"env": 42})

        fitness, returned_diag, _ = await backend.evaluate(ir)
        assert fitness.primary == 1.0
        assert returned_diag.cmd_verification_attempted is True
        assert returned_diag.cmd_error_message is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lean/test_verification.py::TestEvaluateCmdErrorThreading -v`
Expected: FAIL — `evaluate()` doesn't set `cmd_error_message` on diagnostics.

**Step 3: Write minimal implementation**

Update `evaluate()` in `evoforge/backends/lean/backend.py:174-216`:

```python
    async def evaluate(self, ir: Any, seed: int | None = None) -> tuple[Fitness, Any, Any]:
        if self._evaluator is None:
            raise RuntimeError("LeanBackend.startup() must be called before evaluate()")
        async with self._repl_lock:
            fitness, diagnostics, trace = await self._evaluator.evaluate(ir)

            if fitness.primary >= 1.0 and fitness.auxiliary.get("proof_complete", 0.0) >= 1.0:
                genome = ir.serialize()
                cmd_verified, cmd_error = await self._verify_via_repl_cmd(genome)
                diagnostics.cmd_verification_attempted = True
                if not cmd_verified:
                    diagnostics.cmd_error_message = cmd_error
                    logger.info(
                        "REPL step-by-step said complete but cmd verification failed — "
                        "downgrading to %.1f: %s",
                        _FALSE_POSITIVE_FITNESS,
                        genome[:80],
                    )
                    fitness = Fitness(
                        primary=_FALSE_POSITIVE_FITNESS,
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

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lean/test_verification.py::TestEvaluateCmdErrorThreading -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add evoforge/backends/lean/backend.py tests/test_lean/test_verification.py
git commit -m "Thread cmd verification errors into LeanDiagnostics from evaluate()"
```

---

### Task 4: Classify cmd errors and log to SearchMemory

**Files:**
- Modify: `evoforge/backends/lean/backend.py` (add `_classify_cmd_error` static method)
- Modify: `evoforge/backends/lean/backend.py:174-216` (call classify + log in evaluate)
- Test: `tests/test_lean/test_verification.py`

**Step 1: Write the failing test**

```python
# tests/test_lean/test_verification.py

from evoforge.backends.lean.backend import LeanBackend

class TestClassifyCmdError:
    """Tests for _classify_cmd_error normalization."""

    def test_unknown_identifier(self):
        result = LeanBackend._classify_cmd_error("unknown identifier 'sum_nonneg'")
        assert result == "unknown_identifier:sum_nonneg"

    def test_unknown_identifier_with_context(self):
        result = LeanBackend._classify_cmd_error(
            "unknown identifier 'IsPositiveDefinite.sum_nonneg'\nsome context"
        )
        assert result == "unknown_identifier:IsPositiveDefinite.sum_nonneg"

    def test_type_mismatch(self):
        result = LeanBackend._classify_cmd_error(
            "type mismatch\n  has type Nat\n  expected Real"
        )
        assert result == "type_mismatch"

    def test_unsolved_goals(self):
        result = LeanBackend._classify_cmd_error("unsolved goals\ncase ...\n⊢ ...")
        assert result == "unsolved_goals"

    def test_sorry(self):
        result = LeanBackend._classify_cmd_error("proof contains sorry")
        assert result == "sorry"

    def test_fallback(self):
        result = LeanBackend._classify_cmd_error("some weird error nobody expected")
        assert result.startswith("other:")
        assert len(result) <= 80  # truncated
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lean/test_verification.py::TestClassifyCmdError -v`
Expected: FAIL — `_classify_cmd_error` doesn't exist.

**Step 3: Write minimal implementation**

Add to `evoforge/backends/lean/backend.py` (after `_extract_goal_state`, before `extract_genome`):

```python
    @staticmethod
    def _classify_cmd_error(error_msg: str) -> str:
        """Normalize a REPL cmd error into a short category string for SearchMemory."""
        msg = error_msg.strip()
        if msg.startswith("unknown identifier"):
            # Extract the identifier name from single quotes
            match = re.search(r"'([^']+)'", msg)
            name = match.group(1) if match else "?"
            return f"unknown_identifier:{name}"
        if msg.startswith("type mismatch"):
            return "type_mismatch"
        if msg.startswith("unsolved goals"):
            return "unsolved_goals"
        if "sorry" in msg.lower():
            return "sorry"
        # Fallback: truncated first line
        first_line = msg.split("\n", 1)[0]
        return f"other:{first_line[:60]}"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lean/test_verification.py::TestClassifyCmdError -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add evoforge/backends/lean/backend.py tests/test_lean/test_verification.py
git commit -m "Add _classify_cmd_error for normalizing REPL errors into SearchMemory patterns"
```

---

### Task 5: Wire error classification into evaluate() for population-level logging

**Files:**
- Modify: `evoforge/backends/lean/backend.py:174-216` (evaluate method)
- Test: `tests/test_lean/test_verification.py`

**Note:** The engine already calls `self._memory.update(credited_offspring, gen)` after evaluation (engine.py:376). But `SearchMemory.update()` only logs dead ends when fitness < 0.1. Cmd errors at fitness=0.9 need a different path. We'll store the classified error pattern on `Fitness.auxiliary` so the engine can access it later. Alternatively, we expose a method on the backend for the engine to call. The simplest approach: add the classified error to `Fitness.auxiliary["cmd_error_pattern"]` so the engine can pick it up.

**Step 1: Write the failing test**

```python
# tests/test_lean/test_verification.py

class TestEvaluateCmdErrorPattern:
    """Tests that evaluate() stores classified error pattern in fitness auxiliary."""

    @pytest.fixture
    def backend(self):
        b = _make_backend(imports="import LeanLevy")
        b._repl = AsyncMock()
        b._repl_lock = asyncio.Lock()
        b._import_env = 0
        return b

    @pytest.mark.asyncio
    async def test_cmd_error_pattern_in_auxiliary(self, backend):
        ir = parse_tactic_sequence("simp")
        complete_fitness = Fitness(
            primary=1.0,
            auxiliary={"steps_succeeded": 1.0, "goals_remaining": 0.0, "proof_complete": 1.0},
            constraints={},
            feasible=True,
        )
        diag = LeanDiagnostics(
            success=True, goals_remaining=0, goal_types=[], goal_contexts=[],
            error_type=None, error_message=None, stuck_tactic_index=None,
            stuck_tactic=None, steps_succeeded=1, metavar_count=0,
        )
        backend._evaluator = AsyncMock()
        backend._evaluator.evaluate = AsyncMock(return_value=(complete_fitness, diag, None))
        backend._repl.send_command = AsyncMock(
            return_value={"severity": "error", "message": "unknown identifier 'sum_nonneg'"}
        )

        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.auxiliary["cmd_error_pattern"] == "unknown_identifier:sum_nonneg"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lean/test_verification.py::TestEvaluateCmdErrorPattern -v`
Expected: FAIL — no `cmd_error_pattern` in auxiliary.

**Step 3: Write minimal implementation**

In `evaluate()`, when cmd verification fails, add the classified pattern:

```python
                if not cmd_verified:
                    diagnostics.cmd_error_message = cmd_error
                    error_pattern = self._classify_cmd_error(cmd_error or "unknown")
                    logger.info(...)
                    fitness = Fitness(
                        primary=_FALSE_POSITIVE_FITNESS,
                        auxiliary={
                            **fitness.auxiliary,
                            "proof_complete": 0.0,
                            "cmd_verified": 0.0,
                            "cmd_error_pattern": error_pattern,
                        },
                        constraints=fitness.constraints,
                        feasible=True,
                    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lean/test_verification.py::TestEvaluateCmdErrorPattern -v`
Expected: PASS

**Step 5: Commit**

```bash
git add evoforge/backends/lean/backend.py tests/test_lean/test_verification.py
git commit -m "Store classified cmd error pattern in Fitness.auxiliary for population-level tracking"
```

---

### Task 6: Engine wiring — log cmd error patterns to SearchMemory

**Files:**
- Modify: `evoforge/core/engine.py` (~line 376, after memory update)
- Test: `tests/test_engine.py` (or `tests/test_lean/test_verification.py`)

**Step 1: Write the failing test**

```python
# tests/test_lean/test_verification.py

from evoforge.core.memory import SearchMemory

class TestEngineErrorToMemory:
    """Test that cmd error patterns flow into SearchMemory dead_ends."""

    def test_cmd_error_pattern_added_to_dead_ends(self):
        """Simulate what the engine does: check auxiliary for cmd_error_pattern."""
        memory = SearchMemory(max_dead_ends=50)
        # Simulate an individual with cmd_error_pattern in auxiliary
        ind = make_individual(
            genome="simp",
            fitness=Fitness(
                primary=0.9,
                auxiliary={
                    "cmd_verified": 0.0,
                    "cmd_error_pattern": "unknown_identifier:sum_nonneg",
                },
                constraints={},
                feasible=True,
            ),
        )
        # The engine would do this after evaluation:
        pattern = ind.fitness.auxiliary.get("cmd_error_pattern")
        if pattern:
            memory.dead_ends.add(f"cmd_error:{pattern}")

        assert "cmd_error:unknown_identifier:sum_nonneg" in memory.dead_ends
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_lean/test_verification.py::TestEngineErrorToMemory -v`
Expected: PASS (this test documents the pattern; the actual wiring is straightforward).

**Step 3: Write minimal implementation**

In `evoforge/core/engine.py`, after the memory update block (after line 376), add:

```python
                    # Log cmd error patterns to search memory as dead ends
                    if not ablation.disable_memory:
                        for ind in credited_offspring:
                            if ind.fitness is not None:
                                pattern = ind.fitness.auxiliary.get("cmd_error_pattern")
                                if pattern:
                                    self._memory.dead_ends.add(f"cmd_error:{pattern}")
```

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -x -v`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add evoforge/core/engine.py tests/test_lean/test_verification.py
git commit -m "Wire cmd error patterns from evaluate() into SearchMemory dead_ends"
```

---

### Task 7: Fix _derive_math_context hallucinated API reference

**Files:**
- Modify: `evoforge/backends/lean/backend.py:263-287`

**Context:** `_derive_math_context()` at line 285 references `IsPositiveDefinite.sum_nonneg` — a hallucinated name that doesn't exist. This actively teaches the LLM wrong API names. Fix it now.

**Step 1: Fix the hallucinated reference**

In `evoforge/backends/lean/backend.py:281-286`, change:

```python
        if "leanlevy" in self._imports.lower():
            parts.append(
                "Available from the LeanLevy library: "
                "IsPositiveDefinite.conj_neg (Hermitian symmetry), "
                "IsPositiveDefinite.re_nonneg (PD form has nonneg real part), "
                "IsPositiveDefinite.apply_zero_nonneg (φ(0).re ≥ 0), "
                "IsPositiveDefinite.apply_zero_im (φ(0).im = 0)."
            )
```

**Step 2: Run quality gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v`
Expected: All pass.

**Step 3: Commit**

```bash
git add evoforge/backends/lean/backend.py
git commit -m "Fix hallucinated API reference in _derive_math_context (sum_nonneg → re_nonneg)"
```

---

## Group B: API Enumeration (Feature 2 — no file overlap with Group A except system_prompt.j2)

**Prerequisite:** Group A Task 7 must complete first (it modifies `_derive_math_context` which this group replaces with richer API context).

### Task 8: Create api_extractor module with Lean source parser

**Files:**
- Create: `evoforge/backends/lean/api_extractor.py`
- Test: `tests/test_lean/test_api_extractor.py`

**Step 1: Write the failing test**

```python
# tests/test_lean/test_api_extractor.py
"""Tests for Lean source file API extraction."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from evoforge.backends.lean.api_extractor import APIEntry, extract_api_from_file


@pytest.fixture
def sample_lean_file(tmp_path: Path) -> Path:
    """Create a minimal .lean file with a namespace and declarations."""
    content = textwrap.dedent("""\
        import Mathlib

        namespace Foo

        /-- Docstring for bar. -/
        theorem bar (x : Nat) : x + 0 = x := by simp

        lemma baz (a b : Int) : a + b = b + a := by ring

        def helper (n : Nat) : Nat := n + 1

        noncomputable instance : Inhabited Nat := ⟨0⟩

        theorem sorry_theorem (x : Nat) : x = x := by
          sorry

        end Foo

        namespace Other

        theorem other_thing : True := trivial

        end Other
    """)
    p = tmp_path / "Test.lean"
    p.write_text(content)
    return p


class TestExtractApi:
    def test_extracts_declarations_in_namespace(self, sample_lean_file: Path):
        entries = extract_api_from_file(sample_lean_file, "Foo")
        names = [e.name for e in entries]
        assert "bar" in names
        assert "baz" in names
        assert "helper" in names

    def test_excludes_other_namespace(self, sample_lean_file: Path):
        entries = extract_api_from_file(sample_lean_file, "Foo")
        names = [e.name for e in entries]
        assert "other_thing" not in names

    def test_captures_signature(self, sample_lean_file: Path):
        entries = extract_api_from_file(sample_lean_file, "Foo")
        bar = next(e for e in entries if e.name == "bar")
        assert "(x : Nat)" in bar.signature
        assert "x + 0 = x" in bar.signature

    def test_marks_sorry_declarations(self, sample_lean_file: Path):
        entries = extract_api_from_file(sample_lean_file, "Foo")
        sorry_entry = next(e for e in entries if e.name == "sorry_theorem")
        assert sorry_entry.has_sorry is True

    def test_non_sorry_not_marked(self, sample_lean_file: Path):
        entries = extract_api_from_file(sample_lean_file, "Foo")
        bar = next(e for e in entries if e.name == "bar")
        assert bar.has_sorry is False

    def test_returns_empty_for_missing_namespace(self, sample_lean_file: Path):
        entries = extract_api_from_file(sample_lean_file, "NonExistent")
        assert entries == []

    def test_full_name_includes_namespace(self, sample_lean_file: Path):
        entries = extract_api_from_file(sample_lean_file, "Foo")
        bar = next(e for e in entries if e.name == "bar")
        assert bar.full_name == "Foo.bar"


class TestExtractApiEntry:
    def test_api_entry_fields(self):
        entry = APIEntry(
            name="foo",
            full_name="Ns.foo",
            signature="(x : Nat) : Nat",
            has_sorry=False,
        )
        assert entry.name == "foo"
        assert entry.full_name == "Ns.foo"
        assert entry.signature == "(x : Nat) : Nat"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lean/test_api_extractor.py -v`
Expected: FAIL — module doesn't exist.

**Step 3: Write minimal implementation**

```python
# evoforge/backends/lean/api_extractor.py
"""Extract available API declarations from Lean 4 source files.

Parses .lean files for theorem/lemma/def declarations within a given
namespace, returning structured APIEntry objects for prompt injection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class APIEntry:
    """A single API declaration extracted from a Lean source file."""

    name: str
    full_name: str
    signature: str
    has_sorry: bool = False


# Matches: theorem|lemma|def|noncomputable def <name> ...
_DECL_RE = re.compile(
    r"^(?:noncomputable\s+)?(?:protected\s+)?(?:private\s+)?"
    r"(theorem|lemma|def)\s+(\w+)\s*(.*)",
    re.DOTALL,
)


def extract_api_from_file(
    file_path: Path,
    namespace: str,
) -> list[APIEntry]:
    """Extract API declarations from a Lean file within a specific namespace.

    Args:
        file_path: Path to the .lean source file.
        namespace: The namespace to extract from (e.g., "IsPositiveDefinite").

    Returns:
        List of APIEntry objects for declarations found in the namespace.
    """
    text = file_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    entries: list[APIEntry] = []
    in_target_ns = False
    ns_depth = 0  # track nested namespaces

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track namespace entry/exit
        if stripped.startswith("namespace "):
            ns_name = stripped.split(None, 1)[1].strip()
            if in_target_ns:
                ns_depth += 1
            elif ns_name == namespace:
                in_target_ns = True
                ns_depth = 0
        elif stripped.startswith("end "):
            end_name = stripped.split(None, 1)[1].strip()
            if in_target_ns:
                if ns_depth > 0:
                    ns_depth -= 1
                elif end_name == namespace:
                    in_target_ns = False

        if not in_target_ns:
            continue

        # Try to match a declaration
        m = _DECL_RE.match(stripped)
        if m is None:
            continue

        _kind, name, rest = m.group(1), m.group(2), m.group(3)

        # Extract signature: everything up to ':= by' or ':='
        sig = _extract_signature(rest, lines, i)

        # Check if the proof body contains sorry
        has_sorry = _check_sorry(rest, lines, i)

        entries.append(APIEntry(
            name=name,
            full_name=f"{namespace}.{name}",
            signature=sig,
            has_sorry=has_sorry,
        ))

    return entries


def _extract_signature(rest: str, lines: list[str], start_line: int) -> str:
    """Extract the type signature from declaration text.

    Collects lines until ':= by', ':=', or 'where' is found.
    """
    # Accumulate the full declaration text (may span multiple lines)
    acc = rest
    for j in range(start_line + 1, min(start_line + 20, len(lines))):
        if ":=" in acc or "where" in acc.strip():
            break
        acc += "\n" + lines[j]

    # Split at ':= by' or ':='
    for sep in [":= by", ":="]:
        if sep in acc:
            sig = acc.split(sep, 1)[0].strip()
            return sig

    return acc.strip()


def _check_sorry(rest: str, lines: list[str], start_line: int) -> bool:
    """Check if the proof body of a declaration contains sorry."""
    # Look at the declaration line and the next few lines
    acc = rest
    for j in range(start_line + 1, min(start_line + 30, len(lines))):
        next_line = lines[j].strip()
        acc += "\n" + next_line
        # Stop at next declaration or namespace boundary
        if next_line.startswith(("theorem ", "lemma ", "def ", "namespace ", "end ")):
            break
    return "sorry" in acc
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lean/test_api_extractor.py -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add evoforge/backends/lean/api_extractor.py tests/test_lean/test_api_extractor.py
git commit -m "Add api_extractor module to parse Lean source files for available API"
```

---

### Task 9: Auto-derive namespaces from theorem statement

**Files:**
- Modify: `evoforge/backends/lean/api_extractor.py`
- Test: `tests/test_lean/test_api_extractor.py`

**Step 1: Write the failing test**

```python
# tests/test_lean/test_api_extractor.py

from evoforge.backends.lean.api_extractor import extract_hypothesis_types


class TestExtractHypothesisTypes:
    def test_extracts_type_from_hypothesis(self):
        stmt = "theorem norm_le_one {φ : ℝ → ℂ} (hφ : IsPositiveDefinite φ) (h0 : φ 0 = 1) (ξ : ℝ) : ‖φ ξ‖ ≤ 1"
        types = extract_hypothesis_types(stmt)
        assert "IsPositiveDefinite" in types

    def test_skips_builtin_types(self):
        stmt = "theorem foo (n : ℕ) (x : ℝ) (c : ℂ) : n = n"
        types = extract_hypothesis_types(stmt)
        # ℕ, ℝ, ℂ are builtins — should not be returned
        assert "ℕ" not in types
        assert "ℝ" not in types

    def test_multiple_custom_types(self):
        stmt = "theorem bar (hf : Continuous f) (hg : Measurable g) : True"
        types = extract_hypothesis_types(stmt)
        assert "Continuous" in types
        assert "Measurable" in types

    def test_handles_no_hypotheses(self):
        stmt = "theorem trivial_thing : True"
        types = extract_hypothesis_types(stmt)
        assert types == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lean/test_api_extractor.py::TestExtractHypothesisTypes -v`
Expected: FAIL — `extract_hypothesis_types` doesn't exist.

**Step 3: Write minimal implementation**

Add to `evoforge/backends/lean/api_extractor.py`:

```python
# Common Lean/Mathlib types that don't need API enumeration
_BUILTIN_TYPES = frozenset({
    "ℕ", "ℤ", "ℝ", "ℂ", "Nat", "Int", "Float", "Bool", "Prop", "Type",
    "String", "Fin", "List", "Option", "Unit", "True", "False",
})

# Matches (name : Type ...) in theorem signatures
_HYPO_RE = re.compile(r"\(\w+\s*:\s*(\w+)")


def extract_hypothesis_types(theorem_statement: str) -> list[str]:
    """Extract non-builtin type names from hypothesis annotations in a theorem statement.

    Args:
        theorem_statement: The full theorem statement string.

    Returns:
        Deduplicated list of custom type names found in hypotheses.
    """
    matches = _HYPO_RE.findall(theorem_statement)
    seen: set[str] = set()
    result: list[str] = []
    for name in matches:
        if name not in _BUILTIN_TYPES and name not in seen:
            seen.add(name)
            result.append(name)
    return result
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lean/test_api_extractor.py::TestExtractHypothesisTypes -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add evoforge/backends/lean/api_extractor.py tests/test_lean/test_api_extractor.py
git commit -m "Add extract_hypothesis_types to auto-derive relevant namespaces from theorem statement"
```

---

### Task 10: Add config fields and wire API extraction into startup()

**Files:**
- Modify: `evoforge/core/config.py:67-75`
- Modify: `evoforge/backends/lean/backend.py` (startup, system_prompt)
- Modify: `configs/lean_default.toml`
- Test: `tests/test_lean/test_api_extractor.py`

**Step 1: Write the failing test**

```python
# tests/test_lean/test_api_extractor.py

import textwrap

class TestFindLeanFiles:
    """Test finding .lean files containing a namespace."""

    def test_finds_file_with_namespace(self, tmp_path: Path):
        from evoforge.backends.lean.api_extractor import find_files_with_namespace

        (tmp_path / "A.lean").write_text("namespace Foo\ntheorem x : True := trivial\nend Foo\n")
        (tmp_path / "B.lean").write_text("namespace Bar\nend Bar\n")
        files = find_files_with_namespace(tmp_path, "Foo")
        assert len(files) == 1
        assert files[0].name == "A.lean"

    def test_searches_subdirectories(self, tmp_path: Path):
        from evoforge.backends.lean.api_extractor import find_files_with_namespace

        sub = tmp_path / "Sub"
        sub.mkdir()
        (sub / "C.lean").write_text("namespace Foo\nend Foo\n")
        files = find_files_with_namespace(tmp_path, "Foo")
        assert len(files) == 1

    def test_returns_empty_when_not_found(self, tmp_path: Path):
        from evoforge.backends.lean.api_extractor import find_files_with_namespace

        (tmp_path / "A.lean").write_text("namespace Bar\nend Bar\n")
        files = find_files_with_namespace(tmp_path, "Foo")
        assert files == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lean/test_api_extractor.py::TestFindLeanFiles -v`
Expected: FAIL — `find_files_with_namespace` doesn't exist.

**Step 3: Write minimal implementation**

Add to `evoforge/backends/lean/api_extractor.py`:

```python
def find_files_with_namespace(project_dir: Path, namespace: str) -> list[Path]:
    """Find .lean files under project_dir that contain `namespace <name>`."""
    pattern = f"namespace {namespace}"
    results: list[Path] = []
    for lean_file in project_dir.rglob("*.lean"):
        try:
            text = lean_file.read_text(encoding="utf-8")
            if pattern in text:
                results.append(lean_file)
        except (OSError, UnicodeDecodeError):
            continue
    return sorted(results)


def extract_api_for_theorem(
    project_dir: Path,
    theorem_statement: str,
    extra_namespaces: list[str] | None = None,
) -> list[APIEntry]:
    """Extract all relevant API for a theorem, auto-deriving namespaces.

    Combines auto-derived hypothesis types with explicit extra namespaces,
    then searches the project for matching .lean files and extracts declarations.

    Args:
        project_dir: Root of the Lean project.
        theorem_statement: The full theorem statement string.
        extra_namespaces: Additional namespaces to enumerate.

    Returns:
        Combined list of APIEntry objects from all relevant namespaces.
    """
    namespaces = extract_hypothesis_types(theorem_statement)
    if extra_namespaces:
        for ns in extra_namespaces:
            if ns not in namespaces:
                namespaces.append(ns)

    all_entries: list[APIEntry] = []
    for ns in namespaces:
        files = find_files_with_namespace(project_dir, ns)
        for f in files:
            entries = extract_api_from_file(f, ns)
            all_entries.extend(entries)

    return all_entries
```

Add config fields to `evoforge/core/config.py` in `BackendConfig`:

```python
class BackendConfig(BaseModel):
    """Formal-verification backend settings."""
    name: str = "lean"
    theorem_statement: str = ""
    project_dir: str = ""
    repl_path: str | None = None
    imports: str = ""
    seeds: list[str] = []
    theorem_file: str | None = None
    extra_api_namespaces: list[str] = []
```

Update `configs/lean_default.toml` — add after the `seeds` array:

```toml
theorem_file = "LeanLevy/Fourier/PositiveDefinite.lean"
extra_api_namespaces = []
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lean/test_api_extractor.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add evoforge/backends/lean/api_extractor.py evoforge/core/config.py configs/lean_default.toml tests/test_lean/test_api_extractor.py
git commit -m "Add find_files_with_namespace, extract_api_for_theorem, and config fields"
```

---

### Task 11: Wire API context into startup() and system_prompt.j2

**Files:**
- Modify: `evoforge/backends/lean/backend.py` (startup, system_prompt, __init__)
- Modify: `evoforge/backends/lean/templates/system_prompt.j2`
- Test: `tests/test_lean/test_verification.py`

**Step 1: Write the failing test**

```python
# tests/test_lean/test_verification.py

class TestApiInSystemPrompt:
    """Tests that API context appears in the system prompt."""

    def test_api_context_rendered_in_system_prompt(self):
        from evoforge.backends.lean.api_extractor import APIEntry

        b = _make_backend()
        b._api_context = [
            APIEntry(name="re_nonneg", full_name="IsPositiveDefinite.re_nonneg",
                     signature="(n : ℕ) (x : Fin n → ℝ) (c : Fin n → ℂ) : 0 ≤ (∑ ...).re"),
            APIEntry(name="conj_neg", full_name="IsPositiveDefinite.conj_neg",
                     signature="(t : ℝ) : φ (-t) = starRingEnd ℂ (φ t)"),
        ]
        # Clear the lru_cache so it re-renders
        b.system_prompt.cache_clear()
        prompt = b.system_prompt()
        assert "re_nonneg" in prompt
        assert "conj_neg" in prompt
        assert "Available API" in prompt

    def test_empty_api_context_no_section(self):
        b = _make_backend()
        b._api_context = []
        b.system_prompt.cache_clear()
        prompt = b.system_prompt()
        assert "Available API" not in prompt
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lean/test_verification.py::TestApiInSystemPrompt -v`
Expected: FAIL — `_api_context` not recognized, template doesn't have the section.

**Step 3: Write minimal implementation**

Add `_api_context` to `LeanBackend.__init__()`:

```python
    def __init__(self, ...) -> None:
        # ... existing init ...
        self._api_context: list[Any] = []  # populated in startup()
```

Update `system_prompt()` to pass api_context:

```python
    @functools.lru_cache(maxsize=1)
    def system_prompt(self) -> str:
        template = self._jinja_env.get_template("system_prompt.j2")
        math_context = self._derive_math_context()
        return template.render(
            theorem_statement=self.theorem_statement,
            math_context=math_context,
            api_context=self._api_context,
        )
```

Update `system_prompt.j2` — add after the `## Mathematical Context` block:

```jinja2
{% if api_context %}

## Available API
The following lemmas, theorems, and definitions are available. Use ONLY these names — do not invent identifiers:
{% for entry in api_context %}
{% if not entry.has_sorry %}- `{{ entry.full_name }}` {{ entry.signature }}
{% endif %}{% endfor %}
{% endif %}
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lean/test_verification.py::TestApiInSystemPrompt -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add evoforge/backends/lean/backend.py evoforge/backends/lean/templates/system_prompt.j2 tests/test_lean/test_verification.py
git commit -m "Wire API context from api_extractor into system prompt template"
```

---

### Task 12: Call API extraction during startup()

**Files:**
- Modify: `evoforge/backends/lean/backend.py` (startup, __init__)
- Test: integration test

**Step 1: Write the failing test**

```python
# tests/test_lean/test_api_extractor.py

import textwrap

class TestStartupApiExtraction:
    """Test that LeanBackend.startup() would extract API from project files."""

    def test_extract_api_for_theorem_integration(self, tmp_path: Path):
        """End-to-end: create a .lean file, extract API for a matching theorem."""
        from evoforge.backends.lean.api_extractor import extract_api_for_theorem

        lean_file = tmp_path / "Foo.lean"
        lean_file.write_text(textwrap.dedent("""\
            namespace IsPositiveDefinite

            theorem re_nonneg (n : ℕ) : 0 ≤ n := by omega

            theorem conj_neg (t : ℝ) : t = t := by rfl

            theorem norm_le_one (x : ℝ) : True := by
              sorry

            end IsPositiveDefinite
        """))

        entries = extract_api_for_theorem(
            project_dir=tmp_path,
            theorem_statement="theorem foo (hφ : IsPositiveDefinite φ) : True",
        )
        names = [e.name for e in entries]
        assert "re_nonneg" in names
        assert "conj_neg" in names
        # sorry theorems are included but marked
        sorry_entries = [e for e in entries if e.has_sorry]
        assert len(sorry_entries) >= 1
```

**Step 2: Run test to verify it passes** (should pass with existing code from Task 10)

Run: `uv run pytest tests/test_lean/test_api_extractor.py::TestStartupApiExtraction -v`
Expected: PASS

**Step 3: Wire startup() to call extract_api_for_theorem**

In `LeanBackend.__init__()`, accept optional new params:

```python
    def __init__(
        self,
        theorem_statement: str,
        project_dir: str,
        repl_path: str | None = None,
        imports: str = "",
        seeds: list[str] | None = None,
        extra_api_namespaces: list[str] | None = None,
    ) -> None:
        # ... existing init ...
        self._extra_api_namespaces = extra_api_namespaces or []
        self._api_context: list[Any] = []
```

In `startup()`, after REPL initialization (after line 164):

```python
        # Extract available API from Lean source files
        from evoforge.backends.lean.api_extractor import extract_api_for_theorem
        try:
            self._api_context = extract_api_for_theorem(
                project_dir=Path(self.project_dir),
                theorem_statement=self.theorem_statement,
                extra_namespaces=self._extra_api_namespaces,
            )
            if self._api_context:
                logger.info(
                    "Extracted %d API entries from Lean source files",
                    len(self._api_context),
                )
                self.system_prompt.cache_clear()  # re-render with API context
        except Exception:
            logger.warning("Failed to extract API from Lean sources", exc_info=True)
```

Update `scripts/run.py` to pass `extra_api_namespaces` from config to backend constructor (check current wiring).

**Step 4: Run full quality gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add evoforge/backends/lean/backend.py tests/test_lean/test_api_extractor.py
git commit -m "Call extract_api_for_theorem during startup() to populate API context"
```

---

### Task 13: Wire extra_api_namespaces through run.py

**Files:**
- Modify: `scripts/run.py` (pass config.backend.extra_api_namespaces to LeanBackend)

**Step 1: Check current run.py wiring**

Read `scripts/run.py` to see how `LeanBackend` is constructed. Add `extra_api_namespaces=config.backend.extra_api_namespaces` to the constructor call.

**Step 2: Run full quality gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v`
Expected: All pass.

**Step 3: Commit**

```bash
git add scripts/run.py
git commit -m "Pass extra_api_namespaces from config to LeanBackend constructor"
```

---

## Group C: Final Integration & Quality Gate

### Task 14: Run full quality gate and verify end-to-end

**Step 1: Run quality gate**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v
```

Expected: All 434+ tests pass, no lint/type errors.

**Step 2: Fix any issues found**

Address any ruff, mypy, or test failures.

**Step 3: Verify existing verification tests still pass**

Run: `uv run pytest tests/test_lean/test_verification.py -v`
Expected: All existing tests pass (no regressions).

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "Fix lint/type issues from error feedback and API enumeration features"
```

---

## Execution Dependencies

```
Group A (Tasks 1-7): Sequential, no file overlap with Group B except Task 7
Group B (Tasks 8-13): Sequential internally, depends on Task 7 for system_prompt.j2
Group C (Task 14): Depends on both A and B

Parallelizable:
- Tasks 1-6 (Group A core) can run in parallel with Tasks 8-10 (Group B core)
- Task 7 (fix hallucinated API) should run before Task 11 (system_prompt.j2 changes)
- Task 11-13 must run after Task 7
```
