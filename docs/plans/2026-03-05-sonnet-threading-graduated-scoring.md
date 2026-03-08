# Sonnet Model, Lake Threading, Graduated Cmd Scoring

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve proof search quality and speed via three changes: upgrade mutation model to Sonnet, add `-j` threading to lake verification, and replace flat false-positive fitness with graduated scoring by error type.

**Architecture:** Three independent changes to `backend.py` and `config.py`. The graduated scoring replaces the single `_FALSE_POSITIVE_FITNESS = 0.9` constant with a lookup map keyed by the error category already produced by `_classify_cmd_error()`. Lake threading adds `-j<N>` to the `subprocess.run` call. Model change is a default swap in two files.

**Tech Stack:** Python, Pydantic config, TOML, pytest, asyncio

---

### Task 1: Graduated cmd-verification scoring — tests

**Files:**
- Modify: `tests/test_lean/test_verification.py`

**Step 1: Write failing tests for graduated scoring**

Add a new test class after `TestTwoTierFitness` (line ~597):

```python
class TestGraduatedCmdScoring:
    """Different cmd error types produce different fitness scores."""

    async def test_unsolved_goals_gets_0_85(self) -> None:
        backend = _make_backend(project_dir="/tmp/test")
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value=(
                Fitness(
                    primary=1.0,
                    auxiliary={"proof_complete": 1.0, "steps_succeeded": 1.0, "goals_remaining": 0.0},
                    constraints={},
                    feasible=True,
                ),
                MagicMock(),
                MagicMock(),
            )
        )
        backend._evaluator = mock_evaluator
        backend._repl_lock = asyncio.Lock()
        backend._verify_via_repl_cmd = AsyncMock(return_value=(False, "unsolved goals\n⊢ False"))

        ir = MagicMock()
        ir.serialize = MagicMock(return_value="simp")
        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.primary == pytest.approx(0.85)
        assert fitness.feasible is True

    async def test_type_mismatch_gets_0_75(self) -> None:
        backend = _make_backend(project_dir="/tmp/test")
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value=(
                Fitness(
                    primary=1.0,
                    auxiliary={"proof_complete": 1.0, "steps_succeeded": 1.0, "goals_remaining": 0.0},
                    constraints={},
                    feasible=True,
                ),
                MagicMock(),
                MagicMock(),
            )
        )
        backend._evaluator = mock_evaluator
        backend._repl_lock = asyncio.Lock()
        backend._verify_via_repl_cmd = AsyncMock(return_value=(False, "type mismatch\nexpected Nat"))

        ir = MagicMock()
        ir.serialize = MagicMock(return_value="ring")
        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.primary == pytest.approx(0.75)

    async def test_unknown_identifier_gets_0_60(self) -> None:
        backend = _make_backend(project_dir="/tmp/test")
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value=(
                Fitness(
                    primary=1.0,
                    auxiliary={"proof_complete": 1.0, "steps_succeeded": 1.0, "goals_remaining": 0.0},
                    constraints={},
                    feasible=True,
                ),
                MagicMock(),
                MagicMock(),
            )
        )
        backend._evaluator = mock_evaluator
        backend._repl_lock = asyncio.Lock()
        backend._verify_via_repl_cmd = AsyncMock(return_value=(False, "unknown identifier 'foo'"))

        ir = MagicMock()
        ir.serialize = MagicMock(return_value="exact foo")
        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.primary == pytest.approx(0.60)

    async def test_sorry_gets_0_50(self) -> None:
        backend = _make_backend(project_dir="/tmp/test")
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value=(
                Fitness(
                    primary=1.0,
                    auxiliary={"proof_complete": 1.0, "steps_succeeded": 1.0, "goals_remaining": 0.0},
                    constraints={},
                    feasible=True,
                ),
                MagicMock(),
                MagicMock(),
            )
        )
        backend._evaluator = mock_evaluator
        backend._repl_lock = asyncio.Lock()
        backend._verify_via_repl_cmd = AsyncMock(return_value=(False, "proof contains sorry"))

        ir = MagicMock()
        ir.serialize = MagicMock(return_value="sorry")
        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.primary == pytest.approx(0.50)

    async def test_other_error_gets_0_70(self) -> None:
        backend = _make_backend(project_dir="/tmp/test")
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value=(
                Fitness(
                    primary=1.0,
                    auxiliary={"proof_complete": 1.0, "steps_succeeded": 1.0, "goals_remaining": 0.0},
                    constraints={},
                    feasible=True,
                ),
                MagicMock(),
                MagicMock(),
            )
        )
        backend._evaluator = mock_evaluator
        backend._repl_lock = asyncio.Lock()
        backend._verify_via_repl_cmd = AsyncMock(return_value=(False, "some weird error"))

        ir = MagicMock()
        ir.serialize = MagicMock(return_value="weird")
        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.primary == pytest.approx(0.70)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lean/test_verification.py::TestGraduatedCmdScoring -v`
Expected: FAIL — all 5 tests get 0.9 instead of the graduated values

---

### Task 2: Graduated cmd-verification scoring — implementation

**Files:**
- Modify: `evoforge/backends/lean/backend.py:88-91` and `backend.py:218-235`

**Step 1: Replace flat constant with scoring map**

At `backend.py:88-91`, replace:

```python
_FALSE_POSITIVE_FITNESS = 0.9
```

with:

```python
# Graduated fitness for false-positive proofs by cmd error category.
# Higher = closer to a valid proof, giving selection real gradient signal.
_CMD_ERROR_FITNESS: dict[str, float] = {
    "unsolved_goals": 0.85,   # proof structure right, just incomplete
    "type_mismatch": 0.75,    # close but wrong types
    "other": 0.70,            # unknown failure
    "unknown_identifier": 0.60,  # hallucinated API
    "sorry": 0.50,            # gave up explicitly
}
_DEFAULT_FALSE_POSITIVE_FITNESS = 0.70  # fallback for unmatched patterns
```

**Step 2: Add helper to look up fitness from error pattern**

Add after the map:

```python
def _false_positive_fitness(error_pattern: str) -> float:
    """Look up graduated fitness score from a classified error pattern."""
    # error_pattern is like "unknown_identifier:foo" or "type_mismatch"
    category = error_pattern.split(":")[0]
    return _CMD_ERROR_FITNESS.get(category, _DEFAULT_FALSE_POSITIVE_FITNESS)
```

**Step 3: Update evaluate() to use graduated scoring**

In `evaluate()` around line 218-226, replace:

```python
                    logger.info(
                        "REPL step-by-step said complete but cmd verification failed — "
                        "downgrading to %.1f: %s",
                        _FALSE_POSITIVE_FITNESS,
                        genome[:80],
                    )
                    fitness = Fitness(
                        primary=_FALSE_POSITIVE_FITNESS,
```

with:

```python
                    fp_fitness = _false_positive_fitness(error_pattern)
                    logger.info(
                        "REPL step-by-step said complete but cmd verification failed — "
                        "downgrading to %.2f (%s): %s",
                        fp_fitness,
                        error_pattern,
                        genome[:80],
                    )
                    fitness = Fitness(
                        primary=fp_fitness,
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lean/test_verification.py::TestGraduatedCmdScoring -v`
Expected: PASS — all 5

**Step 5: Fix existing test that expects flat 0.9**

The test `TestTwoTierFitness::test_repl_complete_cmd_rejected_gets_0_9` returns error `"type mismatch"` which now scores 0.75. Update the assertion:

In `test_verification.py` line 567, change:
```python
        assert fitness.primary == pytest.approx(0.9)
```
to:
```python
        assert fitness.primary == pytest.approx(0.75)
```

And in `TestEvaluateCmdErrorThreading::test_false_positive_stores_cmd_error_on_diagnostics` line 800, the error is `"unknown identifier 'sum_nonneg'"` which now scores 0.60. Change:
```python
        assert fitness.primary == 0.9
```
to:
```python
        assert fitness.primary == pytest.approx(0.60)
```

**Step 6: Run full verification test suite**

Run: `uv run pytest tests/test_lean/test_verification.py -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add evoforge/backends/lean/backend.py tests/test_lean/test_verification.py
git commit -m "Graduate cmd-verification fitness by error category instead of flat 0.9"
```

---

### Task 3: Lake verification threading — tests

**Files:**
- Modify: `tests/test_core/test_config.py`
- Modify: `tests/test_lean/test_verification.py`

**Step 1: Write failing test for EvalConfig.verification_threads**

In `test_config.py`, add after the existing `TestLLMConfig` class:

```python
from evoforge.core.config import EvalConfig

class TestEvalConfig:
    def test_verification_threads_default_zero(self) -> None:
        cfg = EvalConfig()
        assert cfg.verification_threads == 0
```

**Step 2: Write failing test for -j flag in verify_proof**

In `test_verification.py`, add to `TestVerifyProof`:

```python
    async def test_verify_proof_passes_threads_flag(self, tmp_path: Path) -> None:
        """verify_proof passes -j flag to lean when verification_threads > 0."""
        backend = _make_backend(project_dir=str(tmp_path))
        backend._verification_threads = 4
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            await backend.verify_proof("exact rfl")
        cmd = mock_run.call_args[0][0]
        assert "-j4" in cmd

    async def test_verify_proof_no_threads_flag_when_zero(self, tmp_path: Path) -> None:
        """verify_proof omits -j flag when verification_threads is 0."""
        backend = _make_backend(project_dir=str(tmp_path))
        backend._verification_threads = 0
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = b""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            await backend.verify_proof("exact rfl")
        cmd = mock_run.call_args[0][0]
        assert not any(arg.startswith("-j") for arg in cmd)
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_config.py::TestEvalConfig tests/test_lean/test_verification.py::TestVerifyProof::test_verify_proof_passes_threads_flag tests/test_lean/test_verification.py::TestVerifyProof::test_verify_proof_no_threads_flag_when_zero -v`
Expected: FAIL

---

### Task 4: Lake verification threading — implementation

**Files:**
- Modify: `evoforge/core/config.py:59-64`
- Modify: `evoforge/backends/lean/backend.py` (constructor + verify_proof)

**Step 1: Add verification_threads to EvalConfig**

In `config.py`, `EvalConfig` class (line 59), add:

```python
class EvalConfig(BaseModel):
    """Evaluation concurrency, timeout, and reproducibility settings."""

    max_concurrent: int = 4
    timeout_seconds: float = 60.0
    seed: int = 42
    verification_threads: int = 0  # 0 = auto (cpu_count // 2), >0 = exact
```

**Step 2: Wire threading into LeanBackend constructor**

In `backend.py`, add `import os` at top if not present. In `LeanBackend.__init__` store the resolved thread count:

```python
        self._verification_threads = verification_threads
```

Add `verification_threads: int = 0` parameter to `__init__`. Resolve 0 → auto:

```python
        if verification_threads == 0:
            self._verification_threads = max(1, (os.cpu_count() or 2) // 2)
        else:
            self._verification_threads = verification_threads
```

**Step 3: Pass -j flag in verify_proof**

In `verify_proof()` at line 590-591, change:

```python
                    ["lake", "env", "lean", str(temp_path)],
```

to:

```python
                    ["lake", "env", "lean", f"-j{self._verification_threads}", str(temp_path)],
```

**Step 4: Wire config → backend in run.py**

In `scripts/run.py`, pass the config value when constructing the backend:

```python
verification_threads=config.eval.verification_threads
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_core/test_config.py::TestEvalConfig tests/test_lean/test_verification.py::TestVerifyProof -v`
Expected: PASS

**Step 6: Commit**

```bash
git add evoforge/core/config.py evoforge/backends/lean/backend.py scripts/run.py tests/test_core/test_config.py tests/test_lean/test_verification.py
git commit -m "Add -j threading flag to lake verification for parallel elaboration"
```

---

### Task 5: Switch default model to Sonnet

**Files:**
- Modify: `evoforge/core/config.py:47`
- Modify: `configs/lean_default.toml:22`

**Step 1: Update LLMConfig default**

In `config.py` line 47, change:

```python
    model: str = "claude-haiku-4-5-20251001"
```

to:

```python
    model: str = "claude-sonnet-4-5-20250929"
```

**Step 2: Update TOML config**

In `lean_default.toml` line 22, change:

```toml
model = "claude-haiku-4-5-20251001"
```

to:

```toml
model = "claude-sonnet-4-5-20250929"
```

**Step 3: Run config tests**

Run: `uv run pytest tests/test_core/test_config.py -v`
Expected: PASS (no test asserts the exact model string)

**Step 4: Commit**

```bash
git add evoforge/core/config.py configs/lean_default.toml
git commit -m "Switch default mutation model from Haiku to Sonnet for higher proof quality"
```

---

### Task 6: Quality gate

**Step 1: Run full quality suite**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v
```

Expected: ALL PASS, 0 errors

**Step 2: Fix any issues found**

If mypy or ruff flag anything, fix and re-run.
