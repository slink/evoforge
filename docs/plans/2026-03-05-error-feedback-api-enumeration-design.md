# Error Feedback & API Enumeration Design

**Date**: 2026-03-05
**Status**: Approved

## Motivation

Evoforge reaches 0.9 fitness (REPL step-by-step passes, cmd verification fails) but
can't close to 1.0. Analysis of the 0.9 proofs shows they have the correct mathematical
structure but hallucinate Lean API names (e.g., `hφ.sum_nonneg` instead of `hφ n x c`).

Two problems:
1. The LLM never sees *why* cmd verification failed — errors are discarded.
2. The LLM doesn't know what API names actually exist — it guesses.

## Literature

This design is informed by state-of-the-art in LLM-guided theorem proving:

- **Goedel-Prover-V2** ([arXiv:2508.03613](https://arxiv.org/abs/2508.03613)): Verifier-in-the-loop
  that feeds specific Lean compiler errors back into the model for targeted segment repair.
  Two rounds of self-correction with modest token overhead.
- **APRIL** ([arXiv:2602.02990](https://arxiv.org/pdf/2602.02990)): 260K supervised tuples of
  (erroneous proof, compiler diagnostics) → corrected proof. Shows diagnostic-conditioned
  repair as a powerful training signal.
- **Code Repair with LLMs** ([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/d5c56ec4f69c9a473089b16000d3f8cd-Paper-Conference.pdf)):
  Repair gives exploration-exploitation tradeoff.
- **APOLLO** ([arXiv:2505.05758](https://arxiv.org/html/2505.05758v1)): Separate LLM modules for
  proof synthesis and error-conditioned correction.

Key takeaway: error feedback should operate at **both** the individual level (specific error
for this proof) and the population level (common error patterns as dead ends), matching
Goedel-Prover's per-attempt feedback plus population-level pattern avoidance.

## Feature 1: Cmd Error Feedback

### Error Capture

`_verify_via_repl_cmd()` changes from `-> bool` to `-> tuple[bool, str | None]`.
The REPL response contains `"message"` and `"severity"` keys with the actual error;
currently discarded, now returned.

### Individual-Level Storage

Two new fields on `LeanDiagnostics`:
- `cmd_error_message: str | None` — raw REPL error text
- `cmd_verification_attempted: bool` — distinguishes "never tried" from "tried and passed"

`LeanDiagnostics.summary()` appends the cmd error when present. The mutation prompt
already renders `{{ diagnostics }}`, so no template changes needed.

### Population-Level Patterns

In `LeanBackend.evaluate()`, when cmd verification fails, a new method
`_classify_cmd_error(error_msg) -> str` normalizes errors into categories:
- `"unknown_identifier:<name>"` — hallucinated API
- `"type_mismatch"` — wrong types
- `"unsolved_goals"` — proof incomplete at kernel level
- `"other:<truncated_msg>"` — fallback

These are logged as dead-end patterns in `SearchMemory` via `add_dead_end()`.
Deduplication is automatic (SearchMemory deduplicates by pattern string).

### Data Flow

```
REPL cmd response: {"message": "unknown identifier 'sum_nonneg'", "severity": "error"}
    |
    v
_verify_via_repl_cmd() -> (False, "unknown identifier 'sum_nonneg'")
    |
    v
evaluate():
  - diagnostics.cmd_error_message = "unknown identifier 'sum_nonneg'"
  - memory.add_dead_end("unknown_identifier:sum_nonneg")
    |
    v
Individual.diagnostics.summary() includes:
  "Cmd verification failed: unknown identifier 'sum_nonneg'"
    |
    v
mutation_prompt.j2 renders {{ diagnostics }} — LLM sees the error
SearchMemory.dead_ends includes the pattern — LLM sees population errors
```

## Feature 2: API Enumeration

### Namespace Discovery

At startup, parse the theorem statement to extract hypothesis types.
For `norm_le_one (hφ : IsPositiveDefinite φ) ...`, extract `IsPositiveDefinite`.

TOML config allows overrides via `extra_api_namespaces: list[str]` for types
not in the signature (e.g., `Complex`).

### Source File Parsing

New module `evoforge/backends/lean/api_extractor.py`:

1. Takes a namespace name and the Lean project directory
2. Finds `.lean` files containing `namespace <Name>`
3. Extracts `theorem`, `lemma`, `def` declarations within that namespace
4. Returns `list[APIEntry(name, full_name, signature)]`

Line-based parser, not a full Lean grammar. Looks for declaration keywords inside
`namespace X ... end X` blocks. Extracts signature up to `:= by` or `:=`.

### Prompt Injection

Extracted API rendered in `system_prompt.j2` as:

```
## Available API for IsPositiveDefinite
- re_nonneg (n : ℕ) (x : Fin n → ℝ) (c : Fin n → ℂ) : 0 ≤ (∑ ...).re
- conj_neg (t : ℝ) : φ (-t) = starRingEnd ℂ (φ t)
- apply_zero_nonneg : 0 ≤ (φ 0).re
- apply_zero_im : (φ 0).im = 0
- pdMatrix_posSemidef (m : ℕ) (x : Fin m → ℝ) : (...).PosSemidef
- mul (hψ : IsPositiveDefinite ψ) : IsPositiveDefinite (fun x => φ x * ψ x)
- closure_pointwise ... : IsPositiveDefinite φ
- of_charFun (μ : ProbabilityMeasure ℝ) : IsPositiveDefinite (...)
```

### Timing

Runs once during `LeanBackend.startup()`. Cached on `self._api_context`.
No per-generation cost.

### Generalization

Config specifies `theorem_file` (relative to `lean_project_dir`). The extractor
works on any `.lean` file. When targeting different sorries, change the config.

## Config Changes

New fields in `BackendConfig`:
- `theorem_file: str | None = None` — path to `.lean` file with the target sorry
- `extra_api_namespaces: list[str] = []` — additional namespaces to enumerate

Example in `lean_default.toml`:
```toml
theorem_file = "LeanLevy/Fourier/PositiveDefinite.lean"
extra_api_namespaces = ["Complex"]
```

## Files Changed

| File | Change |
|------|--------|
| `evoforge/backends/lean/backend.py` | `_verify_via_repl_cmd()` returns error, `evaluate()` threads errors, `startup()` calls extractor, prompts pass API context |
| `evoforge/backends/lean/evaluator.py` | `LeanDiagnostics` gets cmd error fields, `summary()` updated |
| `evoforge/backends/lean/api_extractor.py` | **New module** — parses `.lean` files for declarations |
| `evoforge/backends/lean/templates/system_prompt.j2` | New `## Available API` section |
| `evoforge/core/config.py` | `BackendConfig` gets `theorem_file` + `extra_api_namespaces` |
| `configs/lean_default.toml` | Add `theorem_file` path |

## Files NOT Changed

- `Engine` — diagnostics flow through `Individual`, memory through `SearchMemory`
- `MutationContext` — individual diagnostics accessed via parent `Individual`
- `mutation_prompt.j2` — already renders `{{ diagnostics }}`
- Core types (`Fitness`, `Individual`) — unchanged

## Testing

- Unit: `_classify_cmd_error()` with sample REPL error messages
- Unit: `api_extractor.py` parsing a sample `.lean` file
- Integration: cmd error appears in `LeanDiagnostics.summary()`
- Integration: API entries show up in rendered system prompt
