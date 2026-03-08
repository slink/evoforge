# Stagnation Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix four interacting issues that cause evolutionary stagnation at fitness 0.7.

**Architecture:** Four independent correctness/bug fixes — no new hyperparameters. Fix 3 (API extraction) and Fix 1 (false-positive feasibility) are independent. Fix 4 (dead-end aggregation) depends on Fix 3. Fix 2 (tree search logging) is independent.

**Tech Stack:** Python 3.11, pytest-asyncio, mypy strict

**Quality gate:** `uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v`

---

### Task 1: API namespace tail-matching — tests

**Files:**
- Modify: `tests/test_lean/test_api_extractor.py`

**Step 1: Write failing tests for nested namespace extraction**

Add a new test class after `TestExtractApi` (line 96):

```python
class TestNestedNamespaceExtraction:
    """Test extraction from nested namespace blocks (e.g. ProbabilityTheory.IsPositiveDefinite)."""

    @pytest.fixture
    def nested_lean_file(self, tmp_path: Path) -> Path:
        content = textwrap.dedent("""\
            import Mathlib

            namespace ProbabilityTheory

            def IsPositiveDefinite (φ : ℝ → ℂ) : Prop := True

            namespace IsPositiveDefinite

            theorem re_nonneg (hφ : IsPositiveDefinite φ) (n : ℕ) : 0 ≤ n := by omega

            lemma conj_neg (hφ : IsPositiveDefinite φ) (t : ℝ) : t = t := by rfl

            theorem sorry_thing : True := by
              sorry

            end IsPositiveDefinite

            end ProbabilityTheory
        """)
        p = tmp_path / "Nested.lean"
        p.write_text(content)
        return p

    def test_unqualified_name_matches_nested(self, nested_lean_file: Path) -> None:
        """Searching for 'IsPositiveDefinite' finds decls inside 'ProbabilityTheory.IsPositiveDefinite'."""
        entries = extract_api_from_file(nested_lean_file, "IsPositiveDefinite")
        names = [e.name for e in entries]
        assert "re_nonneg" in names
        assert "conj_neg" in names

    def test_fully_qualified_also_works(self, nested_lean_file: Path) -> None:
        """Fully qualified 'ProbabilityTheory.IsPositiveDefinite' still matches."""
        entries = extract_api_from_file(nested_lean_file, "ProbabilityTheory.IsPositiveDefinite")
        names = [e.name for e in entries]
        assert "re_nonneg" in names

    def test_full_name_uses_full_stack(self, nested_lean_file: Path) -> None:
        """full_name should use the complete namespace stack, not just the search term."""
        entries = extract_api_from_file(nested_lean_file, "IsPositiveDefinite")
        entry = next(e for e in entries if e.name == "re_nonneg")
        assert entry.full_name == "ProbabilityTheory.IsPositiveDefinite.re_nonneg"

    def test_does_not_match_parent_namespace(self, nested_lean_file: Path) -> None:
        """Searching for 'IsPositiveDefinite' should NOT match decls directly in ProbabilityTheory."""
        entries = extract_api_from_file(nested_lean_file, "IsPositiveDefinite")
        names = [e.name for e in entries]
        # IsPositiveDefinite def is in ProbabilityTheory, not in IsPositiveDefinite
        assert "IsPositiveDefinite" not in names
```

Add a test for `find_files_with_namespace` with nested namespaces in `TestFindLeanFiles`:

```python
    def test_finds_file_with_nested_namespace(self, tmp_path: Path) -> None:
        from evoforge.backends.lean.api_extractor import find_files_with_namespace

        content = "namespace Outer\nnamespace Inner\nend Inner\nend Outer\n"
        (tmp_path / "A.lean").write_text(content)
        files = find_files_with_namespace(tmp_path, "Inner")
        assert len(files) == 1

    def test_finds_file_with_fully_qualified_namespace(self, tmp_path: Path) -> None:
        from evoforge.backends.lean.api_extractor import find_files_with_namespace

        content = "namespace Outer\nnamespace Inner\nend Inner\nend Outer\n"
        (tmp_path / "A.lean").write_text(content)
        files = find_files_with_namespace(tmp_path, "Outer.Inner")
        assert len(files) == 1
```

Also update the end-to-end test in `TestExtractApiForTheorem` to use nested namespaces:

```python
    def test_end_to_end_nested(self, tmp_path: Path) -> None:
        from evoforge.backends.lean.api_extractor import extract_api_for_theorem

        lean_file = tmp_path / "PD.lean"
        lean_file.write_text(
            "namespace ProbabilityTheory\n"
            "namespace IsPositiveDefinite\n"
            "theorem re_nonneg (n : ℕ) : 0 ≤ n := by omega\n"
            "end IsPositiveDefinite\n"
            "end ProbabilityTheory\n"
        )
        entries = extract_api_for_theorem(
            project_dir=tmp_path,
            theorem_statement="theorem foo (hφ : IsPositiveDefinite φ) : True",
        )
        names = [e.name for e in entries]
        assert "re_nonneg" in names
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lean/test_api_extractor.py -v -x`
Expected: FAIL — `test_unqualified_name_matches_nested` returns empty list

---

### Task 2: API namespace tail-matching — implementation

**Files:**
- Modify: `evoforge/backends/lean/api_extractor.py:79-90,119-166`

**Step 3: Fix `find_files_with_namespace` for nested namespaces**

Replace the function (lines 79-90):

```python
def find_files_with_namespace(project_dir: Path, namespace: str) -> list[Path]:
    """Find .lean files under project_dir that contain a matching namespace.

    Matches both exact ``namespace <name>`` lines and nested namespaces
    where the target is the last component (e.g., searching for ``Inner``
    matches a file containing ``namespace Inner`` inside ``namespace Outer``).
    For fully-qualified names like ``Outer.Inner``, searches for each
    component as a separate ``namespace`` line.
    """
    # For "Outer.Inner", require all components as separate namespace lines.
    # For unqualified "Inner", just require "namespace Inner" anywhere.
    parts = namespace.split(".")
    results: list[Path] = []
    for lean_file in project_dir.rglob("*.lean"):
        try:
            text = lean_file.read_text(encoding="utf-8")
            if all(f"namespace {part}" in text for part in parts):
                results.append(lean_file)
        except (OSError, UnicodeDecodeError):
            continue
    return sorted(results)
```

**Step 4: Fix `extract_api_from_file` for tail-matching**

Replace the namespace check at line 149 and update `full_name` to use the full stack:

```python
def extract_api_from_file(file_path: Path, namespace: str) -> list[APIEntry]:
    """Extract declarations from *file_path* that live in *namespace*.

    Tracks ``namespace`` / ``end`` blocks (including nesting) and returns
    declarations whose enclosing namespace matches *namespace*.  Supports
    tail-matching: searching for ``"Inner"`` matches a stack of
    ``["Outer", "Inner"]``.
    """
    lines = file_path.read_text().splitlines()
    ns_stack: list[str] = []
    entries: list[APIEntry] = []
    target_parts = namespace.split(".")

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Track namespace open.
        ns_match = _NAMESPACE_RE.match(stripped)
        if ns_match:
            ns_stack.append(ns_match.group(1))
            i += 1
            continue

        # Track namespace close.
        end_match = _END_RE.match(stripped)
        if end_match and ns_stack:
            ns_stack.pop()
            i += 1
            continue

        # Check for declaration when namespace stack tail matches target.
        if _ns_tail_matches(ns_stack, target_parts):
            decl_match = _DECL_RE.match(stripped)
            if decl_match:
                name = decl_match.group(2)
                full_ns = ".".join(ns_stack)
                sig = _extract_signature(decl_match.group(3), lines, i)
                sorry = _check_sorry(lines, i)
                entries.append(
                    APIEntry(
                        name=name,
                        full_name=f"{full_ns}.{name}",
                        signature=sig,
                        has_sorry=sorry,
                    )
                )

        i += 1

    return entries


def _ns_tail_matches(ns_stack: list[str], target_parts: list[str]) -> bool:
    """Check if the namespace stack ends with the target parts.

    Examples:
        _ns_tail_matches(["ProbabilityTheory", "IsPositiveDefinite"], ["IsPositiveDefinite"]) -> True
        _ns_tail_matches(["ProbabilityTheory", "IsPositiveDefinite"], ["ProbabilityTheory", "IsPositiveDefinite"]) -> True
        _ns_tail_matches(["ProbabilityTheory"], ["IsPositiveDefinite"]) -> False
    """
    if len(target_parts) > len(ns_stack):
        return False
    return ns_stack[-len(target_parts):] == target_parts
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_lean/test_api_extractor.py -v -x`
Expected: ALL PASS

**Step 6: Run full quality gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v`
Expected: PASS

**Step 7: Commit**

```
git add evoforge/backends/lean/api_extractor.py tests/test_lean/test_api_extractor.py
git commit -m "Fix API namespace extraction to support nested namespaces

extract_api_from_file() now uses tail-matching: searching for
'IsPositiveDefinite' matches declarations inside
'ProbabilityTheory.IsPositiveDefinite'. find_files_with_namespace()
updated to find files with nested namespace blocks."
```

---

### Task 3: Dead-end aggregation — tests

**Files:**
- Modify: `tests/test_core/test_memory.py`

**Step 8: Write failing tests for format_dead_ends**

Add after the existing `TestPromptSection` class:

```python
class TestFormatDeadEnds:
    def test_groups_unknown_identifiers(self) -> None:
        memory = SearchMemory(max_dead_ends=50)
        memory.dead_ends.add("cmd_error:unknown_identifier:sum_nonneg")
        memory.dead_ends.add("cmd_error:unknown_identifier:norm_sq_le")
        memory.dead_ends.add("cmd_error:unknown_identifier:matrix_two_by_two")
        result = memory.format_dead_ends()
        assert "do not exist" in result.lower() or "do not use" in result.lower()
        assert "sum_nonneg" in result
        assert "norm_sq_le" in result
        assert "matrix_two_by_two" in result

    def test_groups_other_categories(self) -> None:
        memory = SearchMemory(max_dead_ends=50)
        memory.dead_ends.add("cmd_error:type_mismatch")
        memory.dead_ends.add("cmd_error:unsolved_goals")
        result = memory.format_dead_ends()
        assert "type_mismatch" in result or "type mismatch" in result
        assert "unsolved_goals" in result or "unsolved goals" in result

    def test_non_cmd_dead_ends_passed_through(self) -> None:
        memory = SearchMemory(max_dead_ends=50)
        memory.dead_ends.add("avoid using nlinarith alone")
        result = memory.format_dead_ends()
        assert "avoid using nlinarith alone" in result

    def test_empty_dead_ends(self) -> None:
        memory = SearchMemory(max_dead_ends=50)
        result = memory.format_dead_ends()
        assert result == ""

    def test_prompt_section_uses_formatted_dead_ends(self) -> None:
        memory = SearchMemory(max_dead_ends=50)
        memory.dead_ends.add("cmd_error:unknown_identifier:fake_lemma")
        section = memory.prompt_section()
        # Should use grouped format, not raw "cmd_error:unknown_identifier:fake_lemma"
        assert "fake_lemma" in section
        assert "do not exist" in section.lower() or "do not use" in section.lower()
```

**Step 9: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_memory.py::TestFormatDeadEnds -v -x`
Expected: FAIL — `format_dead_ends` not found

---

### Task 4: Dead-end aggregation — implementation

**Files:**
- Modify: `evoforge/core/memory.py:134-185`

**Step 10: Add `format_dead_ends` method and update `prompt_section`**

Add `format_dead_ends()` method to `SearchMemory` (after `prompt_section`, around line 186):

```python
    def format_dead_ends(self) -> str:
        """Format dead ends grouped by category for LLM consumption.

        Groups ``cmd_error:unknown_identifier:name`` entries into a single
        line listing all hallucinated identifiers.  Other cmd_error categories
        and free-form dead ends are listed individually.
        """
        if not self.dead_ends:
            return ""

        unknown_ids: list[str] = []
        other_cmd: list[str] = []
        freeform: list[str] = []

        for d in sorted(self.dead_ends):
            if d.startswith("cmd_error:unknown_identifier:"):
                ident = d.split(":", 2)[2]
                unknown_ids.append(ident)
            elif d.startswith("cmd_error:"):
                category = d.split(":", 1)[1]
                other_cmd.append(category)
            else:
                freeform.append(d)

        lines: list[str] = []

        if unknown_ids:
            names = ", ".join(f"`{n}`" for n in sorted(unknown_ids))
            lines.append(f"These identifiers do not exist — do not use them: {names}")

        if other_cmd:
            for cat in sorted(set(other_cmd)):
                lines.append(f"Proofs with {cat.replace('_', ' ')} errors failed verification")

        if freeform:
            for d in freeform[:10]:
                lines.append(f"Avoid: {d}")

        return "\n".join(lines)
```

Update `prompt_section()` to use `format_dead_ends()` — replace lines 155-160:

```python
        # Dead ends (grouped by category)
        if self.dead_ends:
            formatted = self.format_dead_ends()
            if formatted:
                parts.append("Dead ends (avoid these):\n" + formatted)
```

**Step 11: Run tests to verify they pass**

Run: `uv run pytest tests/test_core/test_memory.py -v -x`
Expected: ALL PASS

**Step 12: Run full quality gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v`
Expected: PASS

**Step 13: Commit**

```
git add evoforge/core/memory.py tests/test_core/test_memory.py
git commit -m "Group dead ends by category for actionable LLM prompts

Add format_dead_ends() that groups unknown_identifier errors into a
single 'do not use these' line instead of listing each hallucinated
name separately. Other categories and freeform dead ends listed
individually."
```

---

### Task 5: False-positive feasibility — tests

**Files:**
- Modify: `tests/test_lean/test_verification.py`

**Step 14: Write failing tests for corrected false-positive scoring**

Update tests in `TestGraduatedFitness` (line 644). The existing tests assert specific fitness values (0.85, 0.75, etc.) — these need to change. Replace the class with tests for the new behavior:

```python
class TestGraduatedFitness:
    """False positives should be infeasible with formula-derived fitness."""

    @staticmethod
    def _make_complete_backend(error_msg: str) -> LeanBackend:
        # Same helper as before — creates a backend whose evaluator returns
        # fitness=1.0/proof_complete=1.0 but cmd verification fails
        backend = LeanBackend.__new__(LeanBackend)
        backend.theorem_statement = "theorem t (x : ℕ) : x = x"
        backend._prefix_cache = {}
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate = AsyncMock(
            return_value=(
                Fitness(
                    primary=1.0,
                    auxiliary={
                        "proof_complete": 1.0,
                        "steps_succeeded": 3.0,
                        "goals_remaining": 0.0,
                        "goal_reduction": 1.0,
                    },
                    constraints={},
                    feasible=True,
                ),
                MagicMock(),
                MagicMock(),
            )
        )
        backend._evaluator = mock_evaluator
        backend._repl_lock = asyncio.Lock()
        backend._verify_via_repl_cmd = AsyncMock(return_value=(False, error_msg))
        return backend

    @pytest.mark.asyncio
    async def test_false_positive_is_infeasible(self) -> None:
        backend = self._make_complete_backend("unsolved goals\n⊢ False")
        ir = MagicMock()
        ir.serialize = MagicMock(return_value="simp")
        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.feasible is False

    @pytest.mark.asyncio
    async def test_false_positive_uses_partial_formula(self) -> None:
        backend = self._make_complete_backend("unsolved goals\n⊢ False")
        ir = MagicMock()
        ir.serialize = MagicMock(return_value="simp")
        fitness, _, _ = await backend.evaluate(ir)
        # 1-goal theorem: step_ratio=1.0, goal_reduction=0.0
        # 0.4 * 1.0 + 0.6 * 0.0 = 0.4
        assert fitness.primary == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_false_positive_has_cmd_constraint(self) -> None:
        backend = self._make_complete_backend("unknown identifier 'foo'")
        ir = MagicMock()
        ir.serialize = MagicMock(return_value="simp")
        fitness, _, _ = await backend.evaluate(ir)
        assert "cmd_verification" in fitness.constraints

    @pytest.mark.asyncio
    async def test_false_positive_preserves_error_pattern(self) -> None:
        backend = self._make_complete_backend("unknown identifier 'foo'")
        ir = MagicMock()
        ir.serialize = MagicMock(return_value="simp")
        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.auxiliary.get("cmd_error_pattern") == "unknown_identifier:foo"

    @pytest.mark.asyncio
    async def test_cmd_verified_proof_stays_feasible(self) -> None:
        backend = LeanBackend.__new__(LeanBackend)
        backend.theorem_statement = "theorem t (x : ℕ) : x = x"
        backend._prefix_cache = {}
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
        backend._verify_via_repl_cmd = AsyncMock(return_value=(True, None))
        ir = MagicMock()
        ir.serialize = MagicMock(return_value="simp")
        fitness, _, _ = await backend.evaluate(ir)
        assert fitness.primary == 1.0
        assert fitness.feasible is True
```

Also update `TestEngineErrorToMemory.test_cmd_error_pattern_flows_to_dead_ends` (line 1008) to expect `feasible=False`:

```python
        ind = make_individual(
            genome="simp",
            fitness=Fitness(
                primary=0.4,  # was 0.9
                auxiliary={
                    "cmd_verified": 0.0,
                    "cmd_error_pattern": "unknown_identifier:sum_nonneg",
                },
                constraints={"cmd_verification": 1.0},  # new
                feasible=False,  # was True
            ),
        )
```

**Step 15: Run tests to verify they fail**

Run: `uv run pytest tests/test_lean/test_verification.py::TestGraduatedFitness -v -x`
Expected: FAIL — feasible is True, fitness is 0.85

---

### Task 6: False-positive feasibility — implementation

**Files:**
- Modify: `evoforge/backends/lean/backend.py:89-104,215-264`

**Step 16: Remove `_CMD_ERROR_FITNESS` and `_false_positive_fitness`, rewrite false-positive branch**

Delete lines 89-104 (`_CMD_ERROR_FITNESS`, `_DEFAULT_FALSE_POSITIVE_FITNESS`, `_false_positive_fitness`).

Replace the false-positive branch in `evaluate()` (lines 234-255). The key change: use `_compute_fitness` with corrected inputs instead of the lookup table.

```python
                if not cmd_verified:
                    diagnostics.cmd_error_message = cmd_error
                    error_pattern = self._classify_cmd_error(cmd_error or "unknown")
                    # Recompute fitness treating the proof as incomplete.
                    # We trust steps succeeded individually but not the claim
                    # that all goals are closed (Deb 2000 constraint handling).
                    steps = int(fitness.auxiliary.get("steps_succeeded", 0))
                    total = max(steps, 1)
                    initial_goals = max(
                        1, int(fitness.auxiliary.get("goal_reduction", 0))
                        + int(fitness.auxiliary.get("goals_remaining", 0))
                    )
                    fp_fitness = _compute_fitness(
                        steps_succeeded=steps,
                        total_steps=total,
                        initial_goals=initial_goals,
                        goals_remaining=max(1, initial_goals),
                        proof_complete=False,
                    )
                    logger.info(
                        "REPL step-by-step said complete but cmd verification failed — "
                        "downgrading to %.2f (%s): %s",
                        fp_fitness.primary,
                        error_pattern,
                        genome[:80],
                    )
                    fitness = Fitness(
                        primary=fp_fitness.primary,
                        auxiliary={
                            **fitness.auxiliary,
                            "proof_complete": 0.0,
                            "cmd_verified": 0.0,
                            "cmd_error_pattern": error_pattern,
                        },
                        constraints={"cmd_verification": 1.0},
                        feasible=False,
                    )
```

Add the import at the top of backend.py:

```python
from evoforge.backends.lean.evaluator import _compute_fitness
```

**Step 17: Run tests to verify they pass**

Run: `uv run pytest tests/test_lean/test_verification.py -v -x`
Expected: ALL PASS

**Step 18: Run full quality gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v`
Expected: PASS

**Step 19: Commit**

```
git add evoforge/backends/lean/backend.py tests/test_lean/test_verification.py
git commit -m "Mark false-positive proofs infeasible with formula-derived fitness

Remove arbitrary _CMD_ERROR_FITNESS lookup table. False positives now
use the same partial-proof formula with corrected inputs (proof_complete
=False, goals_remaining>=1). Feasible=False per Deb (2000) constraint
handling — they failed verification and are not valid proofs."
```

---

### Task 7: Tree search logging — tests and implementation

**Files:**
- Modify: `evoforge/core/engine.py:713-786`
- Modify: `evoforge/backends/lean/tactic_generator.py:43-58`

**Step 20: Add logging to `_try_tree_search` early returns**

```python
    async def _try_tree_search(self, generation: int) -> None:
        """If enabled, run tree search on the best partial proof."""
        if not self.config.evolution.tree_search_enabled:
            logger.debug("Tree search: disabled in config")
            return
        if self.llm_client is None:
            logger.debug("Tree search: no LLM client available")
            return

        best_list = self.population.best(k=1)
        if not best_list:
            logger.debug("Tree search: empty population")
            return

        best = best_list[0]
        if best.fitness is None:
            logger.debug("Tree search: best individual has no fitness")
            return

        min_fit = self.config.evolution.tree_search_min_fitness
        if best.fitness.primary < min_fit or best.fitness.primary >= 1.0:
            logger.debug(
                "Tree search: best fitness %.3f outside range [%.3f, 1.0)",
                best.fitness.primary,
                min_fit,
            )
            return

        # Extract the successful tactic prefix from credits
        prefix_tactics: list[str] = []
        if best.ir is not None and hasattr(best.ir, "steps"):
            for i, step in enumerate(best.ir.steps):
                credited = any(c.location == i and c.score > 0 for c in best.credits)
                if credited:
                    prefix_tactics.append(step.raw)
                else:
                    break

        if not prefix_tactics:
            logger.debug(
                "Tree search: no credited prefix tactics in best individual (genome=%s)",
                best.genome[:60],
            )
            return

        # ... rest of method unchanged
```

**Step 21: Add logging to `LLMTacticGenerator.suggest_tactics`**

```python
    async def suggest_tactics(self, goal_state: str, proof_so_far: list[str], n: int) -> list[str]:
        """Query the LLM for *n* candidate tactics given the current proof state."""
        template = self._jinja.get_template("tactic_suggest_prompt.j2")
        prompt = template.render(
            goal_state=goal_state,
            proof_so_far="\n".join(proof_so_far) if proof_so_far else "(empty)",
            n=n,
        )
        logger.debug("Requesting %d tactics for goal: %s", n, goal_state[:80])
        try:
            response = await self._client.async_generate(
                prompt,
                self._system_prompt,
                self._model,
                self._temperature,
                self._max_tokens,
            )
        except Exception:
            logger.warning("Tactic generation LLM call failed", exc_info=True)
            return []
        tactics = self._parse_tactics(response.text, n)
        logger.debug("Parsed %d tactics from LLM response", len(tactics))
        return tactics
```

**Step 22: Run full quality gate**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v`
Expected: PASS

**Step 23: Commit**

```
git add evoforge/core/engine.py evoforge/backends/lean/tactic_generator.py
git commit -m "Add diagnostic logging to tree search and tactic generator

Log at every early-return point in _try_tree_search() so silent
failures are visible. Add try/except and logging to
LLMTacticGenerator.suggest_tactics() which previously had zero
instrumentation."
```

---

### Task 8: Final integration verification

**Step 24: Run full quality gate one final time**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v`
Expected: ALL PASS, 519+ tests (new tests added)

**Step 25: Verify no regressions in existing test patterns**

Run: `uv run pytest tests/test_lean/test_verification.py tests/test_lean/test_api_extractor.py tests/test_core/test_memory.py -v`
Expected: ALL PASS
