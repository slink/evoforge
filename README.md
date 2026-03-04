# evoforge

Evolutionary engine for formally-grounded symbolic expressions.
Uses LLMs for mutation, formal systems for fitness.

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

## Architecture

```mermaid
graph TD
    Engine[Evolution Engine]
    Engine --> Pop[Population]
    Engine --> Sel[Selection<br/>tournament · pareto · lexicase · map-elites]
    Engine --> Mut[Mutation Ensemble<br/>cheap ops + LLM ops]
    Engine --> Eval[Evaluator<br/>3-level cache]
    Engine --> Mem[Search Memory]
    Engine --> Arc[Archive<br/>SQLite]
    Engine --> Sched[Scheduler<br/>async + budget]
    Mut --> Backend[Backend ABC]
    Mut --> LLM[LLM Client<br/>Anthropic]
    Eval --> Backend
    Backend -.-> Lean[Lean Backend<br/>REPL · credit · validation]
    Backend -.-> Future[future backends...]
```

## Quick start

```bash
uv sync --dev
uv run pytest -x -v
```

## Project structure

```
evoforge/
  core/       — engine, types, selection, mutation, archive, evaluator, memory, scheduler
  backends/   — pluggable domain backends (lean, ...)
  llm/        — LLM client and operators
tests/
configs/
scripts/
```

## Status

- Core framework: implemented
- Lean 4 backend: implemented
- 337 tests, strict mypy, ruff

## Running with Lean

Requires a sibling [LeanLevy](https://github.com/slink/LeanLevy) project with the REPL built:

```bash
# Build REPL in LeanLevy (sibling directory)
cd ../LeanLevy
lake update && lake build repl
cd ../evoforge

# Run (project dir comes from backend.project_dir in the config file)
uv run python scripts/run.py --config configs/lean_default.toml --max-generations 3

# Or override with env var
LEAN_PROJECT_DIR=/other/path/to/LeanLevy uv run python scripts/run.py --config configs/lean_default.toml --max-generations 3
```
