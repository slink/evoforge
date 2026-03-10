# evoforge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An evolutionary engine for formally-grounded symbolic expressions. It uses LLMs to suggest mutations and formal verification systems to evaluate fitness. Two backends: Lean 4 theorem proving (on hold) and CFD turbulence closure optimization (active).

An LLM-guided evolutionary engine for formally-grounded symbolic expressions. It maintains a population of candidates, evolves them using both cheap syntactic operators and LLM-guided mutations, and uses formal evaluation systems to measure fitness.

The first backend targeted Lean 4 theorem proving to empirically test whether evolution could compete with tree search (the standard approach in the literature). It confirmed that theorem proving's discrete fitness landscape is a poor fit for evolutionary methods — see [the post-mortem](docs/post-mortem-naive-llm-search.md) for the analysis. The experiment validated the engine architecture end-to-end and clarified what makes a domain evolvable.

The architecture is designed for domains with **continuous fitness landscapes**. The active backend targets CFD turbulence closure optimization — evolving damping functions where small mutations produce small fitness changes and partial improvements are meaningful.

## How it works

```mermaid
flowchart LR
    A[Select Parents] --> B[Mutate]
    B --> C[Identity Pipeline]
    C --> D[Evaluate]
    D --> E[Assign Credit]
    E --> F[Survive]
    F --> G[Archive + Memory]
    G -->|next generation| A
```

Each generation:

1. **Select** parents from the population (tournament, NSGA-II, lexicase, or MAP-Elites)
2. **Mutate** using a weighted ensemble of cheap syntactic operators and LLM-guided operators
3. **Deduplicate** via an identity pipeline (canonicalize, hash, reject duplicates)
4. **Evaluate** against the formal backend — for Lean, this means stepping through each tactic in the REPL and measuring how many goals are closed
5. **Assign credit** per-step so partial proofs get meaningful fitness scores
6. **Survive** — selection pressure culls the population
7. **Archive** results to SQLite; update search memory (working patterns, dead ends)

Periodically, the engine also:
- Asks the LLM to **reflect** on what's working and what isn't, feeding insights back into search memory
- Runs **best-first tree search** from promising partial proofs, expanding one tactic at a time
- **Checkpoints** all state for crash recovery

## Architecture

```mermaid
graph TD
    Engine[Evolution Engine]
    Engine --> Pop[Population Manager]
    Engine --> Sel[Selection<br/>tournament · NSGA-II · lexicase · MAP-Elites]
    Engine --> Mut[Mutation Ensemble<br/>4 cheap ops + 2 LLM ops<br/>adaptive weights]
    Engine --> Eval[Evaluator<br/>3-level cache]
    Engine --> Mem[Search Memory<br/>patterns + dead ends]
    Engine --> Arc[Archive<br/>SQLite]
    Engine --> Sched[Scheduler<br/>async batching + budget limits]
    Mut --> Backend[Backend ABC]
    Mut --> LLM[LLM Client<br/>Anthropic · Gemini · OpenAI]
    Eval --> Backend
    Backend -.-> Lean[Lean Backend<br/>REPL · stepwise credit · verification]
    Backend -.-> CFD[CFD Backend<br/>SymPy IR · solver adapter · ablation credit]
```

### Key components

- **Engine** (`evoforge/core/engine.py`) — the main loop. Wires everything together, manages lifecycle, handles checkpointing and resume.
- **Backend ABC** (`evoforge/backends/base.py`) — interface that any formal system must implement: seed population, evaluate, format prompts, assign credit.
- **Lean backend** (`evoforge/backends/lean/`) — talks to Lean 4 via a REPL subprocess over a pty. Evaluates proofs step-by-step, runs two-tier verification (REPL cmd check, then `lake env lean` for full kernel verification), extracts available API from source files. Currently on hold.
- **CFD backend** (`evoforge/backends/cfd/`) — evolves turbulence closure functions (SymPy expressions) for RANS solvers. Evaluates by monkey-patching the damping function into a 1D flow solver and comparing predicted friction factors against Jensen et al. (1989) experimental data. Uses ablation-based credit assignment and three cheap operators (constant perturbation, subtree mutation, term add/remove).
- **Mutation ensemble** (`evoforge/core/mutation.py`) — cheap operators (swap steps, truncate, splice prefixes, reorder) plus LLM-guided mutation and crossover. Weights adapt based on which operators are producing fitness improvements.
- **Selection** (`evoforge/core/selection.py`) — four strategies. Lexicase is the default and tends to maintain more diversity than tournament.
- **Search memory** (`evoforge/core/memory.py`) — tracks patterns that led to fitness gains and dead ends to avoid. Fed into LLM prompts so the model learns from the population's history.
- **Tree search** (`evoforge/backends/lean/tree_search.py`) — best-first search over REPL proof states. Used as a refinement step on promising partial proofs found by evolution.
- **LLM client** (`evoforge/llm/client.py`) — multi-provider LLM wrapper (Anthropic, Gemini, OpenAI-compatible) with exponential backoff, prompt caching (90% input cost reduction on repeated system prompts), budget tracking, and graceful degradation (if calls fail, cheap operators fill in). Provider is selected per-run via TOML config.
- **Batch collector** (`evoforge/llm/batch.py`) — optional Message Batch API integration that collects per-generation LLM requests into a single batch for 50% cost savings (stacks with prompt caching for up to 95% savings on cached input tokens). Falls back to individual calls on failure.

### Proof verification

Verification is three-tiered, from fast to authoritative:

1. **REPL step-by-step** (~ms per step) — walks through each tactic, measures goal closure. This is the primary fitness signal.
2. **REPL cmd** (~100ms) — sends the complete proof as a single `example` command. Catches false positives where individual tactics succeed interactively but the assembled proof is rejected.
3. **`lake env lean`** (~8s) — full kernel elaboration. Only run on proofs that pass tier 2. This is the final gate.

## Project structure

```
evoforge/
  core/       — engine, types, config, selection, mutation, archive,
                evaluator, memory, population, scheduler, identity
  backends/   — pluggable backends
    lean/     — Lean 4 REPL integration, credit assignment, verification,
                tree search, tactic generation, API extraction (on hold)
    cfd/      — CFD turbulence closure optimization: SymPy IR, solver adapter,
                ablation credit, expression mutation operators
  llm/        — multi-provider LLM client (Anthropic, Gemini, OpenAI),
                mutation/crossover operators, Jinja2 prompt templates,
                batch API collector
tests/        — 711 tests, strict mypy, ruff
configs/      — TOML experiment configs
scripts/      — CLI entry point (run.py)
```

## Quick start

```bash
# Install dependencies
uv sync --dev

# Run tests (no Lean installation needed — REPL tests are mocked)
uv run pytest -x -v

# Lint + type check
uv run ruff check . && uv run mypy evoforge/
```

## Running with Lean

Requires a Lean 4 project with the [REPL](https://github.com/leanprover-community/repl) package built. The default config targets [LeanLevy](https://github.com/slink/LeanLevy) as a sibling directory.

```bash
# Build REPL in LeanLevy (sibling directory)
cd ../LeanLevy
lake update && lake build repl
cd ../evoforge

# Run evolution (project dir comes from backend.project_dir in config)
uv run python scripts/run.py --config configs/lean_default.toml --max-generations 50

# Override project dir with env var
LEAN_PROJECT_DIR=/path/to/LeanLevy uv run python scripts/run.py --config configs/lean_default.toml

# With output directory and final verification
uv run python scripts/run.py --config configs/lean_default.toml \
  --max-generations 50 --output-dir runs/exp1 --verify

# Resume from checkpoint after crash or interruption
uv run python scripts/run.py --config configs/lean_default.toml \
  --output-dir runs/exp1 --resume
```

Requires an API key for your chosen LLM provider (`ANTHROPIC_API_KEY` by default, configurable via `llm.api_key_env` in TOML). Without it, the engine falls back to cheap operators only.

## Configuration

Experiments are configured via TOML files. See `configs/lean_default.toml` for a complete example. Key sections:

| Section | Controls |
|---------|----------|
| `[population]` | Size, elite count |
| `[selection]` | Strategy (lexicase, tournament, pareto, map_elites), parameters |
| `[mutation]` | LLM vs cheap operator weights, crossover weight |
| `[llm]` | Provider (anthropic/gemini/openai), model, temperature schedule, token/cost budgets, prompt caching, batch API |
| `[evolution]` | Max generations, stagnation window, tree search settings, checkpointing |
| `[backend]` | Theorem statement, project dir, imports, seed proofs |
| `[ablation]` | Flags to disable individual components for experiments |

## Dependencies

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for package management
- anthropic, pydantic, jinja2, sqlalchemy, aiosqlite, numpy, sympy, rich
- Optional: `google-genai` (Gemini provider), `openai` (OpenAI/Ollama/vLLM provider)
- For the Lean backend: a Lean 4 project with the REPL package built
- For the CFD backend: fluidflow (sibling project) or a compatible RANS solver

## Status

Research software. The core evolutionary engine, LLM integration, and both backends are complete and well-tested. The Lean backend served as an empirical validation that theorem proving's discrete fitness landscape is not evolvable (confirming the literature's convergence on tree search), while proving out the architecture for the pivot to CFD.

**Current focus:** CFD turbulence closure backend, where the continuous fitness landscape is a natural fit for evolutionary optimization.

**Lean backend status:** Functional but on hold. The engine finds partial proofs and "complete" proofs that pass REPL verification, but none survive full kernel elaboration — a result consistent with the literature's preference for tree search over evolution in this domain.

Known limitations:
- Two backends (Lean 4 on hold, CFD active)
- Tree search helps but is limited by the quality of tactic suggestions
- `greenlet` pinned to 3.1.0 due to a macOS compiler crash on newer versions
