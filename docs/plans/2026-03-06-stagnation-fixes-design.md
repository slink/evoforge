# Stagnation Fixes Design

**Date:** 2026-03-06
**Problem:** 50-generation run plateaus at fitness 0.7 from generation 6 onward. Population converges on false-positive proofs that fool the REPL step-by-step but fail cmd/lake verification.

## Root Cause Analysis

Four interacting issues create a stagnation trap:

1. **False positives dominate** — proofs that pass REPL step-by-step but fail cmd verification are scored 0.50–0.85 with `feasible=True`. They outrank genuine partial proofs and crowd out exploration.
2. **Tree search is silent** — config enables it, preconditions appear met, but no tree search log output in 50 generations. Likely failing at prefix extraction or inside the tactic generator (which has zero logging).
3. **API extraction is broken** — namespace mismatch: extractor looks for `ns_stack == ["IsPositiveDefinite"]` but declarations live at `["ProbabilityTheory", "IsPositiveDefinite"]`. Result: `api_context = []`. The LLM gets no API list and hallucrates lemma names.
4. **Dead-end suppression is weak** — dead ends stored per-lemma (`cmd_error:unknown_identifier:sum_nonneg`). Each hallucinated name is a separate entry. The LLM sees cryptic strings and invents different names next time.

These interact: empty API context forces hallucination → hallucinated proofs become false positives → false positives dominate population → dead ends can't keep up → stagnation.

## Fix 1: False-Positive Feasibility (Correctness Fix)

**Grounding:** Deb (2000) constraint handling rules from NSGA-II.

**Current behavior:**
- False positives: `feasible=True`, fitness from `_CMD_ERROR_FITNESS` lookup table (0.50–0.85)
- Partial proofs: `feasible=False`, fitness from formula `0.4 * step_ratio + 0.6 * goal_reduction`

**Problem:** False positives are not feasible proofs — they fail verification. Marking them `feasible=True` violates Deb's constraint rules and makes selection strongly prefer them over genuine partial proofs.

**Fix:**
- Mark false positives `feasible=False`
- Recompute fitness using the existing partial-proof formula with `proof_complete=False` and `goals_remaining = max(1, initial_goals)` (we trust steps succeeded individually, but don't trust the claim that all goals are closed)
- Store error type in `fitness.constraints["cmd_verification"]` for NSGA-II constraint handling
- Remove `_CMD_ERROR_FITNESS` lookup table and `_false_positive_fitness()` function

**Result for 1-goal theorem:** `0.4 * 1.0 + 0.6 * 0.0 = 0.40` — competitive with partial proofs, not dominating them. No new hyperparameters.

**Files:**
- `evoforge/backends/lean/backend.py` — evaluate() false-positive branch
- `evoforge/backends/lean/evaluator.py` — `_compute_fitness()` (no change needed, just called with corrected inputs)

## Fix 2: Tree Search Logging + Bug Fix

**Current behavior:** `_try_tree_search()` has 7 early-return points, most silent. `LLMTacticGenerator` has zero logging. No way to know why tree search isn't running.

**Fix:**
- Add `logger.debug()` at every early-return in `_try_tree_search()`
- Add `logger.info()` for tree search launch and result
- Add logging to `LLMTacticGenerator.suggest_tactics()` — log call count, response parsing, failures
- After instrumentation, identify and fix the actual blocking condition

**Files:**
- `evoforge/core/engine.py` — `_try_tree_search()`
- `evoforge/backends/lean/tactic_generator.py` — `suggest_tactics()`

## Fix 3: API Namespace Extraction (Bug Fix)

**Current behavior:**
- `extract_hypothesis_types()` produces unqualified names: `["IsPositiveDefinite"]`
- `extract_api_from_file()` requires exact match: `".".join(ns_stack) == namespace`
- Declarations at `["ProbabilityTheory", "IsPositiveDefinite"]` don't match `"IsPositiveDefinite"`
- Result: `api_context = []`, LLM gets no API list

**Fix:** Change `extract_api_from_file()` to match against the **tail** of the namespace stack:
- `"IsPositiveDefinite"` matches `["ProbabilityTheory", "IsPositiveDefinite"]` because the stack ends with `"IsPositiveDefinite"`
- Fully-qualified names like `"ProbabilityTheory.IsPositiveDefinite"` still match exactly
- `find_files_with_namespace()` needs the same tail-matching fix for nested namespace blocks

**Result:** The LLM gets an explicit API list with real lemma names and signatures. The "use ONLY these names" instruction in the template becomes actionable.

**Files:**
- `evoforge/backends/lean/api_extractor.py` — `extract_api_from_file()`, `find_files_with_namespace()`

## Fix 4: Dead-End Aggregation (Prompt Improvement)

**Current behavior:**
- Dead ends stored granularly: `"cmd_error:unknown_identifier:sum_nonneg"`, `"cmd_error:unknown_identifier:norm_sq_le"`, etc.
- Top 10 shown to LLM, alphabetically sorted
- LLM sees cryptic prefixed strings and invents different names next time

**Fix:**
- Add `format_dead_ends()` method to `SearchMemory` that groups dead ends by category
- For `unknown_identifier`: "These identifiers do not exist — do not use them: `sum_nonneg`, `norm_sq_le`, `matrix_two_by_two`"
- For `unsolved_goals`, `type_mismatch`, etc.: "These proof patterns failed verification — try a different approach: [truncated proof text]"
- Use `format_dead_ends()` in `prompt_section()` instead of raw set dump
- Cross-reference against extracted API: any identifier NOT in the API list gets flagged as "does not exist"

**Files:**
- `evoforge/core/memory.py` — `SearchMemory.format_dead_ends()`, `prompt_section()`

## Implementation Order

1. **Fix 3 (API extraction)** — unblocks Fix 4 and reduces hallucination at the source
2. **Fix 4 (dead-end aggregation)** — makes dead ends actionable for the LLM
3. **Fix 1 (false-positive feasibility)** — removes the deceptive attractor
4. **Fix 2 (tree search)** — logging first, then fix the actual bug

Fixes 1 and 3 are independent and can be parallelized. Fix 4 depends on Fix 3 for API cross-referencing. Fix 2 is independent but lower priority since the other fixes may change what tree search operates on.

## Verification

After all fixes, re-run the same 50-generation experiment and compare:
- Best fitness trajectory (should see gradual improvement, not immediate plateau)
- Diversity over time (should stay higher)
- Tree search activity (should see log messages)
- False positive count (should be lower, and they shouldn't dominate)
- API context in prompts (should list real lemma names)
