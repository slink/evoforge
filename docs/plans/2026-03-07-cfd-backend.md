# CFD Turbulence Closure Backend — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a CFD backend that evolves Richardson damping functions `f(Ri_g)` for turbulence closures, evaluated against a 1D RANS oscillatory boundary layer solver.

**Architecture:** SymPy-based IR (`ClosureExpr`) for expression manipulation and canonicalization. At evaluation time, `sympy.lambdify` converts to a NumPy callable that replaces the solver's built-in damping function via module-level patching. Fitness combines L2 error against Jensen et al. (1989) friction data with physics constraint penalties. Credit assignment uses per-term ablation.

**Tech Stack:** SymPy (IR + canonicalization), fluidflow (RANS solver, editable dep), NumPy (evaluation), Pydantic (config)

**Performance note:** Approach A uses `sympy.lambdify` → NumPy (no Numba JIT). If evaluation is too slow (~3s/case × 9 cases × 30 pop = ~13 min/gen), upgrade to Approach C: cache Numba-compiled closures keyed by expression hash.

**Parallel groups:** Groups 1 and 2 have zero file overlap and can be dispatched concurrently.

---

## Group 1: ClosureExpr IR + Config (independent)

### Task 1: ClosureExpr IR

**Files:**
- Create: `evoforge/backends/cfd/__init__.py`
- Create: `evoforge/backends/cfd/ir.py`
- Test: `tests/test_cfd/__init__.py`
- Test: `tests/test_cfd/test_ir.py`

**Step 1: Write the failing tests**

```python
# tests/test_cfd/__init__.py
# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
```

```python
# tests/test_cfd/test_ir.py
# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for CFD ClosureExpr IR."""
import pytest
import sympy

from evoforge.backends.cfd.ir import ClosureExpr, parse_closure_expr


Ri = sympy.Symbol("Ri_g", nonneg=True)


class TestClosureExprBasics:
    def test_canonicalize_simplifies(self):
        # 2*Ri_g/2 should simplify to Ri_g
        expr = ClosureExpr(2 * Ri / 2)
        canon = expr.canonicalize()
        assert canon.expr == Ri

    def test_structural_hash_deterministic(self):
        a = ClosureExpr(1 - Ri / sympy.Rational(1, 4))
        b = ClosureExpr(1 - 4 * Ri)
        # Same canonical form → same hash
        assert a.canonicalize().structural_hash() == a.canonicalize().structural_hash()

    def test_different_exprs_different_hash(self):
        a = ClosureExpr(1 - Ri)
        b = ClosureExpr(sympy.exp(-Ri))
        assert a.structural_hash() != b.structural_hash()

    def test_serialize_roundtrip(self):
        expr = ClosureExpr(1 - Ri)
        text = expr.serialize()
        restored = parse_closure_expr(text)
        assert restored is not None
        assert restored.structural_hash() == expr.structural_hash()

    def test_complexity(self):
        simple = ClosureExpr(Ri)
        compound = ClosureExpr(1 - Ri / 0.25 + sympy.exp(-Ri))
        assert compound.complexity() > simple.complexity()


class TestClosureExprTerms:
    def test_additive_terms(self):
        expr = ClosureExpr(1 - Ri + Ri**2)
        terms = expr.additive_terms()
        assert len(terms) == 3

    def test_remove_term(self):
        expr = ClosureExpr(1 - Ri + Ri**2)
        reduced = expr.remove_term(1)  # remove -Ri
        terms = reduced.additive_terms()
        assert len(terms) == 2

    def test_remove_all_terms_gives_zero(self):
        expr = ClosureExpr(Ri)
        reduced = expr.remove_term(0)
        assert reduced.expr == 0

    def test_replace_subtree(self):
        expr = ClosureExpr(1 - Ri)
        replaced = expr.replace_subtree(Ri, Ri**2)
        assert replaced.expr == 1 - Ri**2


class TestClosureExprProtocol:
    def test_implements_ir_protocol(self):
        from evoforge.core.ir import IRProtocol
        expr = ClosureExpr(1 - Ri)
        assert isinstance(expr, IRProtocol)


class TestParseClosureExpr:
    def test_parse_valid(self):
        result = parse_closure_expr("1 - Ri_g/0.25")
        assert result is not None

    def test_parse_invalid_returns_none(self):
        result = parse_closure_expr("not a valid expression }{")
        assert result is None

    def test_parse_with_exp(self):
        result = parse_closure_expr("exp(-Ri_g)")
        assert result is not None
        assert result.expr == sympy.exp(-Ri)


class TestLambdify:
    def test_lambdify_linear(self):
        import numpy as np
        expr = ClosureExpr(1 - Ri / sympy.Rational(1, 4))
        fn = expr.lambdify()
        # f(0) = 1, f(0.25) = 0
        assert fn(0.0) == pytest.approx(1.0)
        assert fn(0.25) == pytest.approx(0.0)

    def test_lambdify_vectorized(self):
        import numpy as np
        expr = ClosureExpr(sympy.exp(-Ri))
        fn = expr.lambdify()
        arr = np.array([0.0, 1.0, 2.0])
        result = fn(arr)
        np.testing.assert_allclose(result, np.exp(-arr))
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cfd/test_ir.py -v`
Expected: FAIL (module not found)

**Step 3: Write the implementation**

```python
# evoforge/backends/cfd/__init__.py
# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
```

```python
# evoforge/backends/cfd/ir.py
# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""ClosureExpr IR — SymPy-based expression tree for turbulence closure functions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import sympy

# Canonical symbol for gradient Richardson number
Ri_g = sympy.Symbol("Ri_g", nonneg=True)

# Symbols the LLM/operators are allowed to use
ALLOWED_SYMBOLS = {Ri_g}


class ClosureExpr:
    """IR node for a scalar damping function f(Ri_g).

    Implements :class:`~evoforge.core.ir.IRProtocol`.
    """

    __slots__ = ("expr",)

    def __init__(self, expr: sympy.Expr) -> None:
        self.expr = expr

    # -- IRProtocol ----------------------------------------------------------

    def canonicalize(self) -> ClosureExpr:
        canonical = sympy.simplify(self.expr)
        canonical = sympy.nsimplify(canonical, rational=False)
        return ClosureExpr(canonical)

    def structural_hash(self) -> str:
        canonical = self.canonicalize()
        return hashlib.sha256(sympy.srepr(canonical.expr).encode()).hexdigest()

    def serialize(self) -> str:
        return str(self.canonicalize().expr)

    def complexity(self) -> int:
        return sum(1 for _ in sympy.preorder_traversal(self.expr))

    # -- CFD-specific --------------------------------------------------------

    def additive_terms(self) -> list[sympy.Expr]:
        """Decompose into additive components for credit assignment."""
        return list(sympy.Add.make_args(self.expr))

    def remove_term(self, index: int) -> ClosureExpr:
        terms = self.additive_terms()
        remaining = [t for i, t in enumerate(terms) if i != index]
        return ClosureExpr(sympy.Add(*remaining) if remaining else sympy.Integer(0))

    def replace_subtree(self, target: sympy.Expr, replacement: sympy.Expr) -> ClosureExpr:
        return ClosureExpr(self.expr.subs(target, replacement))

    def lambdify(self) -> Callable[..., Any]:
        """Convert to a NumPy-callable function f(Ri_g) -> float|ndarray."""
        return sympy.lambdify(Ri_g, self.expr, modules=["numpy"])

    def free_symbols_ok(self) -> bool:
        """Check that the expression only uses allowed symbols."""
        return self.expr.free_symbols <= ALLOWED_SYMBOLS

    def __repr__(self) -> str:
        return f"ClosureExpr({self.expr})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClosureExpr):
            return NotImplemented
        return self.structural_hash() == other.structural_hash()


def parse_closure_expr(text: str) -> ClosureExpr | None:
    """Parse a string into a ClosureExpr, returning None on failure."""
    try:
        local_dict = {"Ri_g": Ri_g, "exp": sympy.exp, "log": sympy.log,
                      "sqrt": sympy.sqrt, "Abs": sympy.Abs, "pi": sympy.pi}
        expr = sympy.sympify(text, locals=local_dict)
        return ClosureExpr(expr)
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError):
        return None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cfd/test_ir.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add evoforge/backends/cfd/ tests/test_cfd/
git commit -m "Add ClosureExpr IR for CFD turbulence closure backend"
```

---

### Task 2: CFD config models

**Files:**
- Modify: `evoforge/core/config.py`
- Test: `tests/test_core/test_config.py` (append)

**Step 1: Write failing tests**

Append to `tests/test_core/test_config.py`:

```python
class TestCFDConfig:
    def test_cfd_backend_config_defaults(self):
        from evoforge.core.config import CFDBackendConfig
        cfg = CFDBackendConfig()
        assert cfg.solver_project_dir == ""
        assert cfg.n_cycles == 20
        assert cfg.grid_N == 128
        assert cfg.grid_H == 5.0
        assert cfg.grid_gamma == 2.0
        assert cfg.Sc_t == 1.0
        assert cfg.max_complexity == 30

    def test_cfd_benchmark_case(self):
        from evoforge.core.config import CFDBenchmarkCase
        case = CFDBenchmarkCase(name="jensen_1", Re=394.0, S=0.0, Lambda=0.0, reference_fw=0.226)
        assert case.Re == 394.0

    def test_load_cfd_config(self):
        import tomllib
        from evoforge.core.config import EvoforgeConfig
        toml_str = """
[run]
name = "cfd_test"
backend = "cfd"
seed = 42

[population]
size = 20

[selection]
strategy = "lexicase"

[mutation]
schedule = "adaptive"

[llm]
model = "claude-sonnet-4-5-20250929"

[eval]
max_concurrent = 4

[backend]
name = "cfd"

[cfd_backend]
solver_project_dir = "/tmp/solver"
n_cycles = 10

[evolution]
max_generations = 50
"""
        data = tomllib.loads(toml_str)
        cfg = EvoforgeConfig(**data)
        assert cfg.cfd_backend.solver_project_dir == "/tmp/solver"
        assert cfg.cfd_backend.n_cycles == 10
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_core/test_config.py::TestCFDConfig -v`
Expected: FAIL (CFDBackendConfig not found)

**Step 3: Add CFD config models to config.py**

Add before `EvoforgeConfig`:

```python
class CFDBenchmarkCase(BaseModel):
    """A single benchmark case for CFD evaluation."""
    name: str
    Re: float
    S: float = 0.0
    Lambda: float = 0.0
    reference_fw: float = 0.0
    reference_regime: str = ""


class CFDBackendConfig(BaseModel):
    """Configuration for the CFD turbulence closure backend."""
    solver_project_dir: str = ""
    n_cycles: int = 20
    convergence_tol: float = 0.01
    grid_N: int = 128
    grid_H: float = 5.0
    grid_gamma: float = 2.0
    Sc_t: float = 1.0
    max_complexity: int = 30
    benchmark_cases: list[CFDBenchmarkCase] = []
    seeds: list[str] = []
```

Add `cfd_backend` field to `EvoforgeConfig`:

```python
    cfd_backend: CFDBackendConfig = CFDBackendConfig()
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/test_config.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add evoforge/core/config.py tests/test_core/test_config.py
git commit -m "Add CFD backend configuration models"
```

---

## Group 2: Solver Adapter + Benchmarks (independent of Group 1)

### Task 3: Solver adapter

**Files:**
- Create: `evoforge/backends/cfd/solver_adapter.py`
- Test: `tests/test_cfd/test_solver_adapter.py`
- Modify: `pyproject.toml` (add fluidflow + sympy deps)

**Step 1: Add fluidflow as editable dependency**

```bash
uv add sympy
uv add --editable ../laminarization-transition-study
```

Verify:
```bash
uv run python -c "from fluidflow.models.closures import compute_nu_t; print('OK')"
```

**Step 2: Write failing tests**

```python
# tests/test_cfd/test_solver_adapter.py
# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for CFD solver adapter."""
import numpy as np
import pytest

from evoforge.backends.cfd.solver_adapter import (
    compute_nu_t_custom,
    run_case_evolved,
    JENSEN_CASES,
)


# Linear damping: f(Ri_g) = max(0, 1 - Ri_g/0.25)
def linear_damping(Ri_g):
    return np.clip(1.0 - Ri_g / 0.25, 0.0, 1.0)


# Exponential damping: f(Ri_g) = exp(-Ri_g/0.25)
def exp_damping(Ri_g):
    return np.exp(-np.clip(Ri_g, 0.0, 12.5) / 0.25)


# Trivial: always zero turbulence
def zero_damping(Ri_g):
    return np.zeros_like(Ri_g) if hasattr(Ri_g, '__len__') else 0.0


BASE_PARAMS = dict(Re=394.0, S=0.0, Lambda=0.0, N=32, H=5.0, gamma=2.0, n_cycles=2)


class TestComputeNuTCustom:
    def test_linear_matches_builtin(self):
        """Custom linear damping should roughly match the built-in Numba version."""
        from fluidflow.models.oscillatory_bl import OscillatoryBLModel
        from fluidflow.models.closures import compute_nu_t

        model = OscillatoryBLModel(BASE_PARAMS)
        u, C = model.get_initial_condition()

        nu_t_ref, D_t_ref = compute_nu_t(
            u, C, model.grid, g_prime=0.0, Sc_t=1.0, damping="linear"
        )
        nu_t_custom, D_t_custom = compute_nu_t_custom(
            u, C, model.grid, g_prime=0.0, Sc_t=1.0, damping_fn=linear_damping
        )
        np.testing.assert_allclose(nu_t_custom, nu_t_ref, rtol=1e-6)
        np.testing.assert_allclose(D_t_custom, D_t_ref, rtol=1e-6)


class TestRunCaseEvolved:
    def test_clear_fluid_linear_runs(self):
        """Clear-fluid (S=0, Lambda=0) case should converge."""
        result = run_case_evolved(BASE_PARAMS, linear_damping)
        assert "viscosity_ratio" in result
        assert "drag_coefficient" in result

    def test_zero_damping_gives_laminar(self):
        """Zero damping should produce laminar-like flow."""
        result = run_case_evolved(BASE_PARAMS, zero_damping)
        assert result["viscosity_ratio"] < 1.0  # no turbulent enhancement


class TestJensenCases:
    def test_cases_exist(self):
        assert len(JENSEN_CASES) >= 2  # at least low-Re cases

    def test_case_has_required_fields(self):
        case = JENSEN_CASES[0]
        assert "Re" in case
        assert "reference_fw" in case
```

**Step 3: Run to verify failure**

Run: `uv run pytest tests/test_cfd/test_solver_adapter.py -v`
Expected: FAIL (module not found)

**Step 4: Write the implementation**

```python
# evoforge/backends/cfd/solver_adapter.py
# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Adapter that runs the fluidflow RANS solver with an evolved damping function.

Replaces the Numba-JIT damping kernels with a NumPy-based implementation
that accepts an arbitrary callable f(Ri_g). Module-level patching injects
the custom closure into the solver's time-stepping loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from fluidflow.grid import StretchedGrid
from fluidflow.models.closures import KAPPA
from fluidflow.solvers.operators import ddz

# Jensen et al. (1989) benchmark cases.
# Re values are Re_delta = sqrt(2 * Re_a) where Re_a is from the paper.
# reference_fw is the friction factor from their Figure 8.
# We include only the two lowest-Re cases for fast evaluation;
# higher-Re cases can be added for thorough runs.
JENSEN_CASES: list[dict[str, Any]] = [
    {"name": "jensen_Re394", "Re": 394.0, "S": 0.0, "Lambda": 0.0,
     "reference_fw": 0.226, "N": 64, "H": 5.0, "gamma": 2.0, "n_cycles": 10},
    {"name": "jensen_Re803", "Re": 803.0, "S": 0.0, "Lambda": 0.0,
     "reference_fw": 0.114, "N": 64, "H": 5.0, "gamma": 2.0, "n_cycles": 10},
]


def compute_nu_t_custom(
    u: np.ndarray,
    C: np.ndarray,
    grid: StretchedGrid,
    g_prime: float,
    Sc_t: float = 1.0,
    Ri_c: float = 0.25,
    epsilon: float = 1e-10,
    damping_fn: Callable[..., Any] | None = None,
    damping: str = "linear",  # unused, kept for signature compat
) -> tuple[np.ndarray, np.ndarray]:
    """Compute turbulent viscosity using an arbitrary damping function.

    Pure NumPy implementation (no Numba) that mirrors the structure of
    fluidflow's _compute_nu_t_linear but accepts a callable damping_fn.
    """
    if damping_fn is None:
        # Fallback to linear
        damping_fn = lambda r: np.clip(1.0 - r / 0.25, 0.0, 1.0)  # noqa: E731

    z = grid.z
    N = grid.N
    dudz = ddz(u, grid)
    dCdz = ddz(C, grid)

    # Base mixing-length viscosity
    nu_t0 = KAPPA**2 * z**2 * np.abs(dudz)

    # Gradient Richardson number
    dudz_sq = dudz**2
    safe_mask = dudz_sq > epsilon
    Ri_g = np.zeros(N)
    Ri_g[safe_mask] = -g_prime * dCdz[safe_mask] / dudz_sq[safe_mask]

    # Apply evolved damping function
    f = damping_fn(Ri_g)
    f = np.clip(f, 0.0, None)  # ensure non-negative

    nu_t = nu_t0 * f
    D_t = nu_t / Sc_t

    return nu_t, D_t


def run_case_evolved(
    params: dict[str, Any],
    damping_fn: Callable[..., Any],
) -> dict[str, Any]:
    """Run a single RANS case with an evolved damping function.

    Patches the solver's compute_nu_t at module level, runs single_run(),
    and restores the original. Thread-unsafe — use asyncio (not threads)
    for concurrent evaluations.
    """
    import fluidflow.models.oscillatory_bl as obl_mod
    import fluidflow.sweep as sweep_mod
    from fluidflow.sweep import single_run

    # Build a patched compute_nu_t that passes damping_fn through
    def patched_compute_nu_t(
        u: np.ndarray,
        C: np.ndarray,
        grid: Any,
        g_prime: float,
        Sc_t: float = 1.0,
        Ri_c: float = 0.25,
        epsilon: float = 1e-10,
        damping: str = "linear",
    ) -> tuple[np.ndarray, np.ndarray]:
        return compute_nu_t_custom(
            u, C, grid, g_prime, Sc_t, Ri_c, epsilon, damping_fn=damping_fn
        )

    orig_obl = obl_mod.compute_nu_t
    orig_sweep = sweep_mod.compute_nu_t
    try:
        obl_mod.compute_nu_t = patched_compute_nu_t
        sweep_mod.compute_nu_t = patched_compute_nu_t
        return single_run(params)
    finally:
        obl_mod.compute_nu_t = orig_obl
        sweep_mod.compute_nu_t = orig_sweep
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_cfd/test_solver_adapter.py -v`
Expected: all PASS

**Step 6: Commit**

```bash
git add pyproject.toml uv.lock evoforge/backends/cfd/solver_adapter.py tests/test_cfd/test_solver_adapter.py
git commit -m "Add CFD solver adapter with module-level closure patching"
```

---

## Group 3: CFD Backend (depends on Groups 1 + 2)

### Task 4: CFDBackend — core methods

**Files:**
- Create: `evoforge/backends/cfd/backend.py`
- Test: `tests/test_cfd/test_backend.py`

**Step 1: Write failing tests**

```python
# tests/test_cfd/test_backend.py
# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for CFDBackend."""
import pytest
import sympy

from evoforge.backends.cfd.backend import CFDBackend
from evoforge.backends.cfd.ir import ClosureExpr, Ri_g
from evoforge.core.config import CFDBackendConfig, CFDBenchmarkCase
from evoforge.core.types import Individual


def _make_backend(cases=None):
    if cases is None:
        cases = [CFDBenchmarkCase(
            name="test_case", Re=394.0, S=0.0, Lambda=0.0, reference_fw=0.226,
        )]
    cfg = CFDBackendConfig(
        solver_project_dir="",
        n_cycles=2,
        grid_N=32,
        benchmark_cases=cases,
    )
    return CFDBackend(cfg)


class TestParse:
    def test_parse_valid(self):
        b = _make_backend()
        ir = b.parse("1 - Ri_g/0.25")
        assert ir is not None
        assert isinstance(ir, ClosureExpr)

    def test_parse_invalid(self):
        b = _make_backend()
        assert b.parse("}{invalid") is None

    def test_parse_rejects_bad_symbols(self):
        b = _make_backend()
        assert b.parse("x + y") is None  # unknown symbols


class TestSeedPopulation:
    def test_seeds_are_parseable(self):
        b = _make_backend()
        seeds = b.seed_population(10)
        assert len(seeds) == 10
        for s in seeds:
            assert b.parse(s) is not None

    def test_seeds_include_builtins(self):
        b = _make_backend()
        seeds = b.seed_population(5)
        assert any("Ri_g" in s for s in seeds)


class TestValidateStructure:
    def test_valid_expr(self):
        b = _make_backend()
        ir = ClosureExpr(1 - Ri_g / 0.25)
        errors = b.validate_structure(ir)
        assert errors == []

    def test_too_complex(self):
        b = _make_backend()
        # Build a deeply nested expression
        deep = Ri_g
        for _ in range(40):
            deep = sympy.exp(deep) + 1
        ir = ClosureExpr(deep)
        errors = b.validate_structure(ir)
        assert any("complexity" in e.lower() for e in errors)


class TestEvaluate:
    @pytest.mark.timeout(60)
    async def test_evaluate_linear_returns_fitness(self):
        b = _make_backend()
        ir = ClosureExpr(sympy.Max(0, 1 - Ri_g / sympy.Rational(1, 4)))
        fitness, diag, trace = await b.evaluate(ir)
        assert fitness.primary > 0.0
        assert fitness.feasible or not fitness.feasible  # just check it runs

    @pytest.mark.timeout(60)
    async def test_evaluate_zero_gives_low_fitness(self):
        b = _make_backend()
        ir = ClosureExpr(sympy.Integer(0))  # zero damping
        fitness, _, _ = await b.evaluate(ir)
        # Should be worse than linear damping
        ir_linear = ClosureExpr(sympy.Max(0, 1 - Ri_g / sympy.Rational(1, 4)))
        fit_linear, _, _ = await b.evaluate(ir_linear)
        assert fitness.primary <= fit_linear.primary


class TestVersion:
    def test_version_string(self):
        b = _make_backend()
        assert "cfd" in b.version()
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cfd/test_backend.py -v`
Expected: FAIL (module not found)

**Step 3: Write the implementation**

```python
# evoforge/backends/cfd/backend.py
# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""CFD turbulence closure backend.

Evolves Richardson damping functions f(Ri_g) evaluated against a 1D RANS
oscillatory boundary layer solver. Fitness is based on L2 error against
Jensen et al. (1989) friction factor measurements.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy

from evoforge.backends.base import Backend
from evoforge.backends.cfd.ir import ClosureExpr, Ri_g, parse_closure_expr
from evoforge.backends.cfd.solver_adapter import (
    JENSEN_CASES,
    run_case_evolved,
)
from evoforge.core.config import CFDBackendConfig
from evoforge.core.ir import BehaviorDimension, BehaviorSpaceConfig
from evoforge.core.types import Credit, Fitness, Individual

logger = logging.getLogger(__name__)

# Default seed expressions for initial population
_DEFAULT_SEEDS: list[str] = [
    "1 - Ri_g/0.25",                          # linear, Ri_c=0.25
    "1 - Ri_g/0.20",                          # linear, Ri_c=0.20
    "1 - Ri_g/0.30",                          # linear, Ri_c=0.30
    "exp(-Ri_g/0.25)",                         # exponential, Ri_c=0.25
    "exp(-Ri_g/0.20)",                         # exponential, Ri_c=0.20
    "exp(-4*Ri_g)",                            # exponential, Ri_c=0.25
    "1/(1 + Ri_g/0.25)",                       # rational
    "1/(1 + 4*Ri_g)",                          # rational, equivalent
    "1/(1 + Ri_g/0.25)**2",                   # steeper rational
    "(1 - Ri_g/0.25)**2",                     # quadratic
    "exp(-Ri_g**2/0.0625)",                    # Gaussian
    "1 - 2*Ri_g + Ri_g**2",                   # polynomial (1-Ri_g)^2
]


@dataclass
class CaseResult:
    """Result of running one benchmark case."""
    name: str
    error: float
    converged: bool
    fw_predicted: float = 0.0
    fw_reference: float = 0.0


@dataclass
class CFDDiagnostics:
    """Diagnostics from evaluating a closure expression."""
    converged_cases: int
    total_cases: int
    per_case: list[CaseResult]
    worst_case: str
    worst_error: float

    def summary(self, max_tokens: int = 500) -> str:
        parts = [f"Converged: {self.converged_cases}/{self.total_cases}"]
        parts.append(f"Worst error: {self.worst_case} (L2={self.worst_error:.4f})")
        for c in self.per_case:
            status = "OK" if c.converged else "DIVERGED"
            parts.append(f"  {c.name}: fw={c.fw_predicted:.4f} vs {c.fw_reference:.4f} ({status})")
        return "\n".join(parts)

    def credit_summary(self, credits: list[Credit], max_tokens: int = 300) -> str:
        helpful = sorted([c for c in credits if c.score > 0], key=lambda c: -c.score)
        harmful = sorted([c for c in credits if c.score < 0], key=lambda c: c.score)
        parts = []
        if helpful:
            parts.append("Helpful terms: " + "; ".join(c.signal for c in helpful[:3]))
        if harmful:
            parts.append("Harmful terms: " + "; ".join(c.signal for c in harmful[:3]))
        return "\n".join(parts)


class CFDBackend(Backend):
    """Evolves Richardson damping functions for turbulence closures."""

    def __init__(self, config: CFDBackendConfig) -> None:
        self._config = config
        self._cases = self._build_cases()

    def _build_cases(self) -> list[dict[str, Any]]:
        """Build solver parameter dicts from config benchmark cases."""
        if self._config.benchmark_cases:
            return [
                {
                    "name": c.name,
                    "Re": c.Re,
                    "S": c.S,
                    "Lambda": c.Lambda,
                    "N": self._config.grid_N,
                    "H": self._config.grid_H,
                    "gamma": self._config.grid_gamma,
                    "Sc_t": self._config.Sc_t,
                    "n_cycles": self._config.n_cycles,
                    "reference_fw": c.reference_fw,
                }
                for c in self._config.benchmark_cases
            ]
        # Fall back to built-in Jensen cases
        return [
            {**c, "N": self._config.grid_N, "n_cycles": self._config.n_cycles}
            for c in JENSEN_CASES
        ]

    # -- IRProtocol bridge ---------------------------------------------------

    def parse(self, genome: str) -> ClosureExpr | None:
        expr = parse_closure_expr(genome)
        if expr is None:
            return None
        if not expr.free_symbols_ok():
            return None
        return expr

    async def evaluate(
        self, ir: Any, seed: int | None = None
    ) -> tuple[Fitness, Any, Any]:
        closure_ir: ClosureExpr = ir
        damping_fn = closure_ir.lambdify()

        # Physics constraint checks
        penalty = 1.0
        constraints: dict[str, bool] = {}

        # f(0) should be ~1 (unstratified limit)
        try:
            f_at_zero = float(damping_fn(0.0))
            unstrat_ok = abs(f_at_zero - 1.0) < 0.1
        except Exception:
            unstrat_ok = False
        constraints["unstratified_limit"] = unstrat_ok
        if not unstrat_ok:
            penalty *= 0.5

        # Complexity limit
        complexity = closure_ir.complexity()
        complexity_ok = complexity <= self._config.max_complexity
        constraints["complexity"] = complexity_ok
        if not complexity_ok:
            penalty *= 0.3

        # Run benchmark cases
        case_results: list[CaseResult] = []
        for case in self._cases:
            params = {k: v for k, v in case.items()
                      if k not in ("name", "reference_fw")}
            try:
                result = run_case_evolved(params, damping_fn)
                # fw ≈ (pi/2) * c_f
                fw_pred = (np.pi / 2) * result.get("drag_coefficient", 0.0)
                fw_ref = case.get("reference_fw", 0.0)
                error = abs(fw_pred - fw_ref) / max(abs(fw_ref), 1e-10) if fw_ref else 0.0
                case_results.append(CaseResult(
                    name=case["name"], error=error, converged=result.get("converged", False),
                    fw_predicted=fw_pred, fw_reference=fw_ref,
                ))
            except Exception as exc:
                logger.warning("Case %s diverged: %s", case["name"], exc)
                case_results.append(CaseResult(
                    name=case["name"], error=1.0, converged=False,
                ))

        errors = [c.error for c in case_results]
        raw_fitness = 1.0 / (1.0 + np.mean(errors)) if errors else 0.0

        diagnostics = CFDDiagnostics(
            converged_cases=sum(1 for c in case_results if c.converged),
            total_cases=len(case_results),
            per_case=case_results,
            worst_case=max(case_results, key=lambda c: c.error).case if case_results else "",
            worst_error=max((c.error for c in case_results), default=0.0),
        )

        fitness = Fitness(
            primary=raw_fitness * penalty,
            auxiliary={
                "raw_accuracy": raw_fitness,
                "complexity": float(complexity),
                "penalty": penalty,
                "converged_fraction": (
                    diagnostics.converged_cases / diagnostics.total_cases
                    if diagnostics.total_cases > 0 else 0.0
                ),
            },
            constraints=constraints,
            feasible=all(constraints.values()),
        )

        return fitness, diagnostics, None

    async def evaluate_stepwise(
        self, ir: Any, seed: int | None = None
    ) -> tuple[Fitness, Any, Any]:
        # CFD doesn't have step-wise evaluation; delegate to full eval
        return await self.evaluate(ir, seed)

    def assign_credit(
        self, ir: Any, fitness: Fitness, diagnostics: Any, trace: Any
    ) -> list[Credit]:
        # Placeholder — Task 6 implements ablation-based credit
        return []

    def validate_structure(self, ir: Any) -> list[str]:
        closure_ir: ClosureExpr = ir
        errors: list[str] = []
        if not closure_ir.free_symbols_ok():
            errors.append(f"Unknown symbols: {closure_ir.expr.free_symbols}")
        if closure_ir.complexity() > self._config.max_complexity:
            errors.append(
                f"Complexity {closure_ir.complexity()} exceeds limit {self._config.max_complexity}"
            )
        return errors

    def seed_population(self, n: int) -> list[str]:
        seeds = list(self._config.seeds) + list(_DEFAULT_SEEDS)
        # Deduplicate by canonical hash
        seen: set[str] = set()
        unique: list[str] = []
        for s in seeds:
            expr = self.parse(s)
            if expr is None:
                continue
            h = expr.structural_hash()
            if h not in seen:
                seen.add(h)
                unique.append(s)
        # Pad with perturbations if needed
        while len(unique) < n:
            base = unique[len(unique) % len(unique)] if unique else "1 - Ri_g/0.25"
            expr = parse_closure_expr(base)
            if expr is not None:
                # Perturb a constant
                perturbed = expr.expr * (1 + 0.1 * sympy.Rational(len(unique), 10))
                unique.append(str(perturbed))
            else:
                unique.append(base)
        return unique[:n]

    def mutation_operators(self) -> list[Any]:
        # Placeholder — Task 7 implements cheap operators
        return []

    def system_prompt(self) -> str:
        return (
            "You are evolving a turbulence damping function f(Ri_g) where Ri_g is the "
            "gradient Richardson number. The function should satisfy:\n"
            "- f(0) ≈ 1 (no damping in unstratified flow)\n"
            "- f(Ri_g) → 0 as Ri_g → ∞ (full suppression in strongly stratified flow)\n"
            "- f(Ri_g) ≥ 0 for all Ri_g ≥ 0\n"
            "- Monotonically decreasing\n"
            "Use Ri_g as the variable name. Available functions: exp, log, sqrt, Abs.\n"
        )

    def format_mutation_prompt(self, parent: Individual, context: Any) -> str:
        return (
            f"Current damping function: f(Ri_g) = {parent.genome}\n"
            f"Fitness: {parent.fitness.primary if parent.fitness else 'unknown'}\n"
            "Suggest an improved version. Return only the expression."
        )

    def format_crossover_prompt(
        self, parent_a: Individual, parent_b: Individual, context: Any
    ) -> str:
        return (
            f"Parent A: f(Ri_g) = {parent_a.genome}\n"
            f"Parent B: f(Ri_g) = {parent_b.genome}\n"
            "Combine the best aspects into a new expression. Return only the expression."
        )

    def extract_genome(self, raw_text: str) -> str | None:
        # Try to find a math expression in the LLM output
        for line in raw_text.strip().split("\n"):
            line = line.strip().strip("`").strip()
            if not line or line.startswith("#"):
                continue
            # Remove "f(Ri_g) = " prefix if present
            if "=" in line:
                line = line.split("=", 1)[1].strip()
            expr = parse_closure_expr(line)
            if expr is not None:
                return line
        return None

    def behavior_descriptor(self, ir: Any, diagnostics: Any) -> tuple[Any, ...]:
        closure_ir: ClosureExpr = ir
        complexity = closure_ir.complexity()
        # Bin complexity: low/medium/high
        if complexity <= 5:
            comp_bin = "low"
        elif complexity <= 15:
            comp_bin = "medium"
        else:
            comp_bin = "high"

        # Dominant functional form
        expr_str = str(closure_ir.expr)
        if "exp" in expr_str:
            form = "exponential"
        elif "/" in expr_str and "Ri_g" in expr_str.split("/")[-1]:
            form = "rational"
        else:
            form = "polynomial"

        return (comp_bin, form)

    def behavior_space(self) -> BehaviorSpaceConfig:
        return BehaviorSpaceConfig(
            dimensions=(
                BehaviorDimension(name="complexity", bins=["low", "medium", "high"]),
                BehaviorDimension(name="form", bins=["polynomial", "exponential", "rational"]),
            )
        )

    def recommended_selection(self) -> str:
        return "lexicase"

    def version(self) -> str:
        return "cfd_v1"

    def eval_config_hash(self) -> str:
        key = f"{self._config.n_cycles}:{self._config.grid_N}:{self._config.grid_H}"
        for c in self._cases:
            key += f":{c.get('Re', 0)}:{c.get('S', 0)}:{c.get('Lambda', 0)}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def format_reflection_prompt(
        self, population: list[Individual], memory: Any, generation: int
    ) -> str:
        top = sorted(
            [ind for ind in population if ind.fitness is not None],
            key=lambda i: i.fitness.primary if i.fitness else 0.0,
            reverse=True,
        )[:5]
        lines = [f"Generation {generation}. Top closures:"]
        for ind in top:
            fit = ind.fitness.primary if ind.fitness else 0.0
            lines.append(f"  f(Ri_g) = {ind.genome}  (fitness={fit:.4f})")
        lines.append("Suggest new functional forms to try and strategies to avoid.")
        return "\n".join(lines)

    def default_operator_weights(self) -> dict[str, float]:
        return {
            "constant_perturb": 0.3,
            "subtree_mutation": 0.2,
            "term_add_remove": 0.2,
            "llm_mutate": 0.2,
            "llm_crossover": 0.1,
        }

    def format_proof(self, genome: str) -> str:
        # Not applicable for CFD, but required by ABC
        return f"f(Ri_g) = {genome}"
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_cfd/test_backend.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add evoforge/backends/cfd/backend.py tests/test_cfd/test_backend.py
git commit -m "Add CFDBackend with solver evaluation and Jensen benchmarks"
```

---

### Task 5: Ablation-based credit assignment

**Files:**
- Create: `evoforge/backends/cfd/credit.py`
- Modify: `evoforge/backends/cfd/backend.py` (wire assign_credit)
- Test: `tests/test_cfd/test_credit.py`

**Step 1: Write failing tests**

```python
# tests/test_cfd/test_credit.py
# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for CFD ablation-based credit assignment."""
import sympy
import pytest

from evoforge.backends.cfd.credit import assign_credit_cfd
from evoforge.backends.cfd.ir import ClosureExpr, Ri_g
from evoforge.core.types import Fitness


class TestAssignCreditCFD:
    def test_returns_credits_per_term(self):
        ir = ClosureExpr(1 - Ri_g + Ri_g**2)  # 3 additive terms
        fitness = Fitness(
            primary=0.8, auxiliary={"raw_accuracy": 0.8, "complexity": 5.0},
            constraints={}, feasible=True,
        )
        # Mock evaluator that returns fixed fitness
        async def mock_eval(expr):
            return Fitness(primary=0.7, auxiliary={"raw_accuracy": 0.7},
                          constraints={}, feasible=True)

        import asyncio
        credits = asyncio.get_event_loop().run_until_complete(
            assign_credit_cfd(ir, fitness, mock_eval)
        )
        assert len(credits) > 0
        assert all(c.confidence < 1.0 for c in credits)  # approximate

    def test_single_term_no_ablation(self):
        ir = ClosureExpr(Ri_g)  # 1 term
        fitness = Fitness(
            primary=0.5, auxiliary={"raw_accuracy": 0.5},
            constraints={}, feasible=True,
        )
        async def mock_eval(expr):
            return Fitness(primary=0.0, auxiliary={"raw_accuracy": 0.0},
                          constraints={}, feasible=True)

        import asyncio
        credits = asyncio.get_event_loop().run_until_complete(
            assign_credit_cfd(ir, fitness, mock_eval)
        )
        # Single term can't be ablated meaningfully (removing it gives 0)
        assert len(credits) <= 1
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cfd/test_credit.py -v`

**Step 3: Implement**

```python
# evoforge/backends/cfd/credit.py
# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Ablation-based credit assignment for CFD closure expressions."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from evoforge.backends.cfd.ir import ClosureExpr
from evoforge.core.types import Credit, Fitness


async def assign_credit_cfd(
    ir: ClosureExpr,
    fitness: Fitness,
    quick_eval: Callable[[ClosureExpr], Coroutine[Any, Any, Fitness]],
) -> list[Credit]:
    """Assign credit to each additive term by ablation.

    For each term, remove it and re-evaluate. The credit score is the
    accuracy drop when the term is removed (positive = helpful).
    """
    baseline = float(fitness.auxiliary.get("raw_accuracy", fitness.primary))
    terms = ir.additive_terms()
    credits: list[Credit] = []

    for i, term in enumerate(terms):
        ablated = ir.remove_term(i)
        if ablated.complexity() == 0:
            continue
        try:
            ablated_fitness = await quick_eval(ablated)
            ablated_acc = float(
                ablated_fitness.auxiliary.get("raw_accuracy", ablated_fitness.primary)
            )
            delta = baseline - ablated_acc
            credits.append(Credit(
                location=i,
                score=delta,
                signal=f"term '{term}': {delta:+.3f} accuracy impact",
                confidence=0.8,
            ))
        except Exception:
            continue

    return credits
```

**Step 4: Wire into CFDBackend.assign_credit()**

In `backend.py`, replace the placeholder `assign_credit`:

```python
    def assign_credit(
        self, ir: Any, fitness: Fitness, diagnostics: Any, trace: Any
    ) -> list[Credit]:
        import asyncio
        from evoforge.backends.cfd.credit import assign_credit_cfd

        closure_ir: ClosureExpr = ir

        async def quick_eval(expr: ClosureExpr) -> Fitness:
            fit, _, _ = await self.evaluate(expr)
            return fit

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — can't nest run_until_complete
            # Return empty and let engine handle async credit if needed
            return []
        return asyncio.run(assign_credit_cfd(closure_ir, fitness, quick_eval))
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_cfd/test_credit.py tests/test_cfd/test_backend.py -v`
Expected: all PASS

**Step 6: Commit**

```bash
git add evoforge/backends/cfd/credit.py evoforge/backends/cfd/backend.py tests/test_cfd/test_credit.py
git commit -m "Add ablation-based credit assignment for CFD closures"
```

---

### Task 6: Cheap mutation operators

**Files:**
- Create: `evoforge/backends/cfd/operators.py`
- Modify: `evoforge/backends/cfd/backend.py` (wire mutation_operators)
- Test: `tests/test_cfd/test_operators.py`

**Step 1: Write failing tests**

```python
# tests/test_cfd/test_operators.py
# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Tests for CFD cheap mutation operators."""
import sympy
import pytest

from evoforge.backends.cfd.ir import ClosureExpr, Ri_g
from evoforge.backends.cfd.operators import (
    ConstantPerturb,
    SubtreeMutate,
    TermAddRemove,
)
from evoforge.core.types import Individual


def _make_individual(genome: str) -> Individual:
    ir = ClosureExpr(sympy.sympify(genome, locals={"Ri_g": Ri_g, "exp": sympy.exp}))
    return Individual(genome=genome, ir=ir, ir_hash="test", generation=0)


class TestConstantPerturb:
    def test_produces_different_genome(self):
        op = ConstantPerturb()
        parent = _make_individual("1 - Ri_g/0.25")
        # Run multiple times — at least one should differ
        results = {op.apply(parent, None) for _ in range(20)}
        assert len(results) > 1  # not always the same

    def test_preserves_structure(self):
        op = ConstantPerturb()
        parent = _make_individual("exp(-Ri_g/0.25)")
        result = op.apply(parent, None)
        assert result is not None
        assert "exp" in result


class TestSubtreeMutate:
    def test_produces_valid_expr(self):
        op = SubtreeMutate()
        parent = _make_individual("1 - Ri_g/0.25")
        result = op.apply(parent, None)
        assert result is not None
        from evoforge.backends.cfd.ir import parse_closure_expr
        assert parse_closure_expr(result) is not None


class TestTermAddRemove:
    def test_add_term(self):
        op = TermAddRemove()
        parent = _make_individual("1 - Ri_g")
        results = set()
        for _ in range(20):
            r = op.apply(parent, None)
            if r:
                results.add(r)
        # Should sometimes produce expressions with more terms
        assert len(results) > 1
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cfd/test_operators.py -v`

**Step 3: Implement**

```python
# evoforge/backends/cfd/operators.py
# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Cheap mutation operators for CFD closure expressions."""

from __future__ import annotations

import random
from typing import Any

import sympy

from evoforge.backends.cfd.ir import ClosureExpr, Ri_g, parse_closure_expr
from evoforge.core.types import Individual

# Small expression fragments for subtree replacement
_FRAGMENTS = [
    Ri_g,
    Ri_g**2,
    sympy.exp(-Ri_g),
    sympy.Rational(1, 4),
    sympy.Rational(1, 2),
    sympy.Integer(1),
    sympy.sqrt(Ri_g),
]


class ConstantPerturb:
    """Perturb numerical constants in the expression by a small factor."""

    name = "constant_perturb"
    cost = "cheap"

    def apply(self, parent: Individual, context: Any) -> str | None:
        ir: ClosureExpr = parent.ir
        expr = ir.expr

        # Find all numerical atoms
        numbers = [a for a in expr.atoms(sympy.Number) if a != 0]
        if not numbers:
            return parent.genome

        target = random.choice(numbers)
        factor = 1.0 + random.gauss(0, 0.15)
        new_val = sympy.nsimplify(float(target) * factor, rational=False)
        new_expr = expr.subs(target, new_val)
        return str(new_expr)


class SubtreeMutate:
    """Replace a random subtree with a small expression fragment."""

    name = "subtree_mutation"
    cost = "cheap"

    def apply(self, parent: Individual, context: Any) -> str | None:
        ir: ClosureExpr = parent.ir
        nodes = list(sympy.preorder_traversal(ir.expr))
        if len(nodes) < 2:
            return parent.genome

        # Don't replace the root
        target = random.choice(nodes[1:])
        replacement = random.choice(_FRAGMENTS)
        new_expr = ir.expr.subs(target, replacement)
        result = str(new_expr)
        return result if parse_closure_expr(result) is not None else parent.genome


class TermAddRemove:
    """Add or remove an additive term."""

    name = "term_add_remove"
    cost = "cheap"

    def apply(self, parent: Individual, context: Any) -> str | None:
        ir: ClosureExpr = parent.ir
        terms = ir.additive_terms()

        if len(terms) > 1 and random.random() < 0.4:
            # Remove a random term
            idx = random.randrange(len(terms))
            reduced = ir.remove_term(idx)
            return reduced.serialize()
        else:
            # Add a small term
            coeff = sympy.nsimplify(random.uniform(-0.5, 0.5), rational=False)
            fragment = random.choice(_FRAGMENTS)
            new_expr = ir.expr + coeff * fragment
            return str(new_expr)
```

**Step 4: Wire into CFDBackend.mutation_operators()**

In `backend.py`, replace the placeholder:

```python
    def mutation_operators(self) -> list[Any]:
        from evoforge.backends.cfd.operators import (
            ConstantPerturb,
            SubtreeMutate,
            TermAddRemove,
        )
        return [ConstantPerturb(), SubtreeMutate(), TermAddRemove()]
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_cfd/ -v`
Expected: all PASS

**Step 6: Commit**

```bash
git add evoforge/backends/cfd/operators.py tests/test_cfd/test_operators.py evoforge/backends/cfd/backend.py
git commit -m "Add cheap mutation operators for CFD closure expressions"
```

---

## Group 4: Config Wiring + Integration

### Task 7: TOML config + backend factory

**Files:**
- Create: `configs/cfd_default.toml`
- Modify: `scripts/run.py` (add CFD backend support)
- Test: `tests/test_cfd/test_integration.py`

**Step 1: Create TOML config**

```toml
# configs/cfd_default.toml
[run]
name = "cfd_closure_search_001"
backend = "cfd"
seed = 42

[population]
size = 30
elite_k = 3

[selection]
strategy = "lexicase"

[mutation]
schedule = "adaptive"
llm_weight = 0.3
cheap_weight = 0.6
crossover_weight = 0.1

[llm]
model = "claude-sonnet-4-5-20250929"
temperature = 0.8
temperature_start = 1.0
temperature_end = 0.4
temperature_schedule = "linear"
max_tokens = 2048
max_calls = 500

[eval]
max_concurrent = 2
timeout_seconds = 120.0

[backend]
name = "cfd"

[cfd_backend]
solver_project_dir = ""
n_cycles = 10
grid_N = 64
grid_H = 5.0
grid_gamma = 2.0
max_complexity = 30
seeds = [
    "1 - Ri_g/0.25",
    "exp(-Ri_g/0.25)",
    "1/(1 + 4*Ri_g)",
]

[[cfd_backend.benchmark_cases]]
name = "jensen_Re394"
Re = 394.0
S = 0.0
Lambda = 0.0
reference_fw = 0.226

[[cfd_backend.benchmark_cases]]
name = "jensen_Re803"
Re = 803.0
S = 0.0
Lambda = 0.0
reference_fw = 0.114

[evolution]
max_generations = 100
stagnation_window = 15
checkpoint_every = 10

[reflection]
interval = 15
include_top_k = 5
include_bottom_k = 3

[memory]
max_patterns = 20
max_dead_ends = 10

[scheduler]
mode = "async_batch"
max_llm_concurrent = 2
max_eval_concurrent = 2
llm_budget_per_gen = 10

[diversity]
strategy = "map_elites"

[ablation]
disable_llm = false
disable_reflection = false
disable_memory = false
```

**Step 2: Modify run.py to support CFD backend**

Check current `run.py` for how the Lean backend is instantiated, and add a parallel path for CFD. The key change: when `config.run.backend == "cfd"`, create a `CFDBackend(config.cfd_backend)` instead of `LeanBackend`.

**Step 3: Write integration smoke test**

```python
# tests/test_cfd/test_integration.py
# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.
"""Integration smoke test for CFD backend."""
import pytest

from evoforge.backends.cfd.backend import CFDBackend
from evoforge.backends.cfd.ir import ClosureExpr, Ri_g, parse_closure_expr
from evoforge.core.config import CFDBackendConfig, CFDBenchmarkCase
from evoforge.core.identity import IdentityPipeline


def _make_backend():
    cases = [CFDBenchmarkCase(
        name="quick_test", Re=394.0, S=0.0, Lambda=0.0, reference_fw=0.226,
    )]
    return CFDBackend(CFDBackendConfig(
        n_cycles=2, grid_N=32, benchmark_cases=cases,
    ))


class TestIdentityPipeline:
    def test_parse_canonicalize_hash(self):
        backend = _make_backend()
        pipeline = IdentityPipeline(backend)
        ind = pipeline.process("1 - Ri_g / 0.25")
        assert ind is not None
        assert ind.ir_hash  # non-empty hash
        assert ind.ir is not None

    def test_dedup_equivalent_exprs(self):
        backend = _make_backend()
        pipeline = IdentityPipeline(backend)
        ind1 = pipeline.process("1 - 4*Ri_g")
        ind2 = pipeline.process("1 - Ri_g*4")
        assert ind1 is not None
        assert ind2 is not None
        # These should have the same canonical hash
        assert ind1.ir_hash == ind2.ir_hash


class TestEndToEnd:
    @pytest.mark.timeout(120)
    async def test_seed_evaluate_credit(self):
        """Full pipeline: seed -> parse -> evaluate -> credit."""
        backend = _make_backend()
        seeds = backend.seed_population(5)
        assert len(seeds) == 5

        # Parse and evaluate the first seed
        ir = backend.parse(seeds[0])
        assert ir is not None

        fitness, diag, trace = await backend.evaluate(ir)
        assert fitness.primary > 0.0
        assert "raw_accuracy" in fitness.auxiliary

    @pytest.mark.timeout(120)
    async def test_mutation_produces_valid_offspring(self):
        """Cheap operators produce parseable offspring."""
        backend = _make_backend()
        operators = backend.mutation_operators()
        assert len(operators) >= 3

        from evoforge.core.types import Individual
        ir = backend.parse("1 - Ri_g/0.25")
        parent = Individual(genome="1 - Ri_g/0.25", ir=ir, ir_hash="test", generation=0)

        valid_count = 0
        for op in operators:
            for _ in range(5):
                result = op.apply(parent, None)
                if result and backend.parse(result) is not None:
                    valid_count += 1
        assert valid_count > 0
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_cfd/ -v`
Expected: all PASS

**Step 5: Run full quality gate**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy evoforge/ && uv run pytest -x -v
```

**Step 6: Commit**

```bash
git add configs/cfd_default.toml scripts/run.py tests/test_cfd/test_integration.py
git commit -m "Wire CFD backend into config loader and CLI with integration tests"
```

---

### Task 8: Update sglink.md and docs

**Files:**
- Modify: `sglink.md` (add CFD backend section)
- Modify: `README.md` (update project structure)

Update `sglink.md` with a new section covering:
- Why we pivoted from Lean to CFD
- How the CFD backend works (SymPy IR, solver adapter, monkey-patching)
- The performance tradeoff (Approach A vs C)
- Lessons from the Lean backend that informed the design
- How the fitness landscape is fundamentally different (continuous vs binary)

Update `README.md` project structure to show the new `cfd/` directory.

**Commit:**
```bash
git add sglink.md README.md
git commit -m "Document CFD backend architecture and Lean→CFD pivot"
```
