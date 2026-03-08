# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.

"""CFD turbulence closure backend for evoforge.

Implements :class:`CFDBackend`, the concrete :class:`Backend` subclass for
evolving turbulence damping functions f(Ri_g) against benchmark RANS cases.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evoforge.backends.base import Backend
from evoforge.backends.cfd.credit import assign_credit_cfd
from evoforge.backends.cfd.ir import ClosureExpr, Ri_g, parse_closure_expr
from evoforge.backends.cfd.solver_adapter import run_case_evolved
from evoforge.core.config import CFDBackendConfig
from evoforge.core.ir import BehaviorDimension, BehaviorSpaceConfig
from evoforge.core.types import Credit, Fitness, Individual

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostics and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    """Result of evaluating a single benchmark case."""

    name: str
    predicted_fw: float
    reference_fw: float
    relative_error: float
    converged: bool


@dataclass
class CFDDiagnostics:
    """Diagnostics from a CFD evaluation run."""

    case_results: list[CaseResult] = field(default_factory=list)
    mean_error: float = 0.0
    complexity: int = 0
    physics_ok: bool = True
    physics_notes: list[str] = field(default_factory=list)

    def summary(self, max_tokens: int = 500) -> str:
        """Summarize diagnostics as a string."""
        lines = [f"mean_error={self.mean_error:.4f}, complexity={self.complexity}"]
        for cr in self.case_results:
            lines.append(
                f"  {cr.name}: pred={cr.predicted_fw:.4f} ref={cr.reference_fw:.4f} "
                f"err={cr.relative_error:.4f} conv={cr.converged}"
            )
        if self.physics_notes:
            lines.append("physics: " + "; ".join(self.physics_notes))
        text = "\n".join(lines)
        return text[:max_tokens]

    def credit_summary(self, credits: list[Credit], max_tokens: int = 500) -> str:
        """Summarize credit assignments (placeholder)."""
        return self.summary(max_tokens)


# ---------------------------------------------------------------------------
# Seed bank — starter damping functions
# ---------------------------------------------------------------------------

_SEED_BANK: list[str] = [
    # Linear forms with different Ri_c values
    "1 - Ri_g/0.25",
    "1 - Ri_g/0.20",
    "1 - Ri_g/0.30",
    "1 - 4*Ri_g",
    "1 - 5*Ri_g",
    # Exponential decay forms
    "exp(-Ri_g/0.25)",
    "exp(-4*Ri_g)",
    "exp(-5*Ri_g)",
    "exp(-Ri_g**2/0.1)",
    # Rational forms
    "1/(1 + Ri_g/0.25)",
    "1/(1 + 4*Ri_g)",
    "1/(1 + 5*Ri_g)",
    "1/(1 + Ri_g/0.25)**2",
    # Power-law forms
    "(1 + Ri_g)**(-1)",
    "(1 + 4*Ri_g)**(-0.5)",
    # Composite forms
    "(1 - Ri_g/0.25) * exp(-Ri_g)",
    "exp(-Ri_g) / (1 + Ri_g)",
]


# ---------------------------------------------------------------------------
# CFDBackend
# ---------------------------------------------------------------------------


class CFDBackend(Backend):
    """Backend for evolving turbulence damping functions f(Ri_g)."""

    def __init__(self, config: CFDBackendConfig) -> None:
        self.config = config

    # -- Parsing -------------------------------------------------------------

    def parse(self, genome: str) -> ClosureExpr | None:
        """Parse a genome string into a ClosureExpr IR node."""
        ir = parse_closure_expr(genome)
        if ir is not None and not ir.free_symbols_ok():
            return None
        return ir

    # -- Evaluation ----------------------------------------------------------

    async def evaluate(self, ir: Any, seed: int | None = None) -> tuple[Fitness, Any, Any]:
        """Evaluate a ClosureExpr against benchmark cases."""
        assert isinstance(ir, ClosureExpr)
        diag = CFDDiagnostics()
        diag.complexity = ir.complexity()

        # Physics constraint: f(0) should be approximately 1
        try:
            fn = ir.lambdify()
            f0 = fn(0.0)
        except Exception:
            diag.physics_ok = False
            diag.physics_notes.append("lambdify or f(0) evaluation failed")
            fitness = Fitness(
                primary=0.0,
                auxiliary={"mean_error": 1.0, "complexity": float(diag.complexity)},
                constraints={"physics_ok": False},
                feasible=False,
            )
            return fitness, diag, None

        if not (math.isfinite(f0) and abs(f0 - 1.0) < 0.1):
            diag.physics_ok = False
            diag.physics_notes.append(f"f(0)={f0:.4f}, expected ~1.0")

        # Complexity check
        if diag.complexity > self.config.max_complexity:
            diag.physics_ok = False
            diag.physics_notes.append(
                f"complexity {diag.complexity} > limit {self.config.max_complexity}"
            )

        # Penalty multiplier for physics violations
        penalty = 1.0 if diag.physics_ok else 0.5

        # Build a NumPy-safe damping function
        def damping_fn(
            ri: np.typing.NDArray[np.floating[Any]],
        ) -> np.typing.NDArray[np.floating[Any]]:
            # lambdify may return a scalar for constant expressions
            result = fn(ri)  # type: ignore[arg-type]
            return np.broadcast_to(np.asarray(result, dtype=float), ri.shape).copy()

        # Run each benchmark case
        cases = self.config.benchmark_cases
        if not cases:
            # No benchmark cases configured — return physics-only fitness
            fitness = Fitness(
                primary=0.5 * penalty,
                auxiliary={"mean_error": 0.0, "complexity": float(diag.complexity)},
                constraints={"physics_ok": diag.physics_ok},
                feasible=diag.physics_ok,
            )
            return fitness, diag, None

        errors: list[float] = []
        for case in cases:
            params: dict[str, Any] = {
                "Re": case.Re,
                "S": case.S,
                "Lambda": case.Lambda,
                "N": self.config.grid_N,
                "H": self.config.grid_H,
                "gamma": self.config.grid_gamma,
                "n_cycles": self.config.n_cycles,
                "Sc_t": self.config.Sc_t,
            }
            try:
                result = run_case_evolved(params, damping_fn)
                # fw ~ (pi/2) * drag_coefficient
                drag_coeff = float(result.get("drag_coefficient", 0.0))
                predicted_fw = (math.pi / 2.0) * drag_coeff
                ref_fw = case.reference_fw
                if ref_fw > 0:
                    rel_error = abs(predicted_fw - ref_fw) / ref_fw
                else:
                    rel_error = abs(predicted_fw)
                cr = CaseResult(
                    name=case.name,
                    predicted_fw=predicted_fw,
                    reference_fw=ref_fw,
                    relative_error=rel_error,
                    converged=bool(result.get("converged", False)),
                )
                errors.append(rel_error)
            except Exception as exc:
                logger.warning("Case %s failed: %s", case.name, exc)
                cr = CaseResult(
                    name=case.name,
                    predicted_fw=0.0,
                    reference_fw=case.reference_fw,
                    relative_error=1.0,
                    converged=False,
                )
                errors.append(1.0)
            diag.case_results.append(cr)

        mean_error = sum(errors) / len(errors) if errors else 1.0
        diag.mean_error = mean_error

        primary = (1.0 / (1.0 + mean_error)) * penalty
        fitness = Fitness(
            primary=primary,
            auxiliary={"mean_error": mean_error, "complexity": float(diag.complexity)},
            constraints={"physics_ok": diag.physics_ok},
            feasible=diag.physics_ok,
        )
        return fitness, diag, None

    async def evaluate_stepwise(
        self, ir: Any, seed: int | None = None
    ) -> tuple[Fitness, Any, Any]:
        """Delegates to evaluate (no step-wise distinction for CFD)."""
        return await self.evaluate(ir, seed)

    # -- Credit assignment ---------------------------------------------------

    def assign_credit(
        self, ir: Any, fitness: Fitness, diagnostics: Any, trace: Any
    ) -> list[Credit]:
        """Ablation-based credit assignment for additive terms.

        Since :meth:`evaluate` is async but this method is sync, we check
        whether an event loop is already running.  If so we return an empty
        list (the engine can call the async helper directly); otherwise we
        use ``asyncio.run()``.
        """
        assert isinstance(ir, ClosureExpr)

        async def _quick_eval(ablated: ClosureExpr) -> Fitness:
            fit, _diag, _trace = await self.evaluate(ablated)
            return fit

        try:
            asyncio.get_running_loop()
            # Already inside an async context — cannot nest asyncio.run().
            return []
        except RuntimeError:
            # No running loop — safe to use asyncio.run().
            return asyncio.run(assign_credit_cfd(ir, fitness, _quick_eval))

    # -- Validation ----------------------------------------------------------

    def validate_structure(self, ir: Any) -> list[str]:
        """Check structural validity of a ClosureExpr."""
        errors: list[str] = []
        if not isinstance(ir, ClosureExpr):
            errors.append("IR is not a ClosureExpr")
            return errors
        if not ir.free_symbols_ok():
            errors.append("Expression contains symbols other than Ri_g")
        if ir.complexity() > self.config.max_complexity:
            errors.append(
                f"Complexity {ir.complexity()} exceeds limit {self.config.max_complexity}"
            )
        return errors

    # -- Seeding -------------------------------------------------------------

    def seed_population(self, n: int) -> list[str]:
        """Generate n seed genomes from the built-in bank + config seeds."""
        seeds = list(self.config.seeds) + [s for s in _SEED_BANK if s not in self.config.seeds]
        if len(seeds) >= n:
            return seeds[:n]
        # Pad with perturbations of existing seeds
        result = list(seeds)
        rng = random.Random(42)
        while len(result) < n:
            base = rng.choice(seeds)
            ir = parse_closure_expr(base)
            if ir is None:
                continue
            # Random coefficient perturbation
            factor = rng.uniform(0.8, 1.2)
            perturbed = ir.expr * factor
            perturbed_str = str(perturbed)
            if perturbed_str not in result:
                result.append(perturbed_str)
        return result[:n]

    # -- Mutation operators --------------------------------------------------

    def mutation_operators(self) -> list[Any]:
        """Return cheap mutation operators for closure expressions."""
        from evoforge.backends.cfd.operators import (
            ConstantPerturb,
            SubtreeMutate,
            TermAddRemove,
        )

        return [ConstantPerturb(), SubtreeMutate(), TermAddRemove()]

    # -- LLM prompts ---------------------------------------------------------

    def system_prompt(self) -> str:
        """System prompt describing what f(Ri_g) should look like."""
        return (
            "You are an expert in turbulence modeling for oscillatory boundary layers.\n"
            "You are evolving a damping function f(Ri_g) for the gradient Richardson number\n"
            "that modifies the turbulent viscosity: nu_t = nu_t0 * f(Ri_g).\n\n"
            "Physical constraints:\n"
            "- f(0) = 1 (neutral stratification = no damping)\n"
            "- f(Ri_g) >= 0 for all Ri_g >= 0\n"
            "- f should decrease as Ri_g increases (stable stratification suppresses turbulence)\n"
            "- The expression should use only the symbol Ri_g\n"
            "- Keep expressions simple (< 30 nodes in the expression tree)\n\n"
            "Common functional forms include:\n"
            "- Linear: 1 - Ri_g/Ri_c (with critical Ri_c ~ 0.2-0.3)\n"
            "- Exponential: exp(-alpha * Ri_g)\n"
            "- Rational: 1/(1 + beta * Ri_g)\n"
            "- Power-law: (1 + gamma * Ri_g)**(-n)\n\n"
            "Output a single mathematical expression using Ri_g as the variable.\n"
            "Use standard math notation: exp(), **, /, *, +, -\n"
        )

    def format_mutation_prompt(self, parent: Individual, context: Any) -> str:
        """Format a mutation prompt showing the parent genome and fitness."""
        lines = [
            f"Current damping function: f(Ri_g) = {parent.genome}",
        ]
        if parent.fitness is not None:
            lines.append(f"Fitness: {parent.fitness.primary:.4f}")
            if "mean_error" in parent.fitness.auxiliary:
                lines.append(f"Mean relative error: {parent.fitness.auxiliary['mean_error']}")
        lines.append(
            "\nPropose an improved f(Ri_g) expression. "
            "Output only the mathematical expression, no explanation."
        )
        return "\n".join(lines)

    def format_crossover_prompt(
        self, parent_a: Individual, parent_b: Individual, context: Any
    ) -> str:
        """Format a crossover prompt showing both parent genomes."""
        lines = [
            f"Parent A: f(Ri_g) = {parent_a.genome}",
        ]
        if parent_a.fitness is not None:
            lines.append(f"  Fitness A: {parent_a.fitness.primary:.4f}")
        lines.append(f"Parent B: f(Ri_g) = {parent_b.genome}")
        if parent_b.fitness is not None:
            lines.append(f"  Fitness B: {parent_b.fitness.primary:.4f}")
        lines.append(
            "\nCombine the best features of both parents into a new f(Ri_g) expression. "
            "Output only the mathematical expression, no explanation."
        )
        return "\n".join(lines)

    def extract_genome(self, raw_text: str) -> str | None:
        """Extract a valid closure expression from LLM output."""
        for line in raw_text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip common prefixes
            for prefix in ("f(Ri_g) = ", "f(Ri_g)=", "f(Ri_g) ="):
                if line.startswith(prefix):
                    line = line[len(prefix) :]
                    break
            # Strip backticks
            line = line.strip("`").strip()
            ir = parse_closure_expr(line)
            if ir is not None and ir.free_symbols_ok():
                return line
        return None

    # -- Behavior descriptors ------------------------------------------------

    def behavior_descriptor(self, ir: Any, diagnostics: Any) -> tuple[Any, ...]:
        """Compute (complexity_bin, functional_form) for MAP-Elites."""
        assert isinstance(ir, ClosureExpr)
        c = ir.complexity()
        if c <= 5:
            complexity_bin = "simple"
        elif c <= 15:
            complexity_bin = "medium"
        else:
            complexity_bin = "complex"

        form = _classify_form(ir)
        return (complexity_bin, form)

    def behavior_space(self) -> BehaviorSpaceConfig:
        """Return complexity x form behavior space."""
        return BehaviorSpaceConfig(
            dimensions=(
                BehaviorDimension(
                    name="complexity",
                    bins=["simple", "medium", "complex"],
                ),
                BehaviorDimension(
                    name="form",
                    bins=["linear", "exponential", "rational", "power", "composite"],
                ),
            )
        )

    # -- Metadata ------------------------------------------------------------

    def recommended_selection(self) -> str:
        """Recommend lexicase selection for multi-case CFD evaluation."""
        return "lexicase"

    def version(self) -> str:
        """Version string for cache keying."""
        return "cfd_v1"

    def eval_config_hash(self) -> str:
        """Hash of evaluation-relevant config for cache keying."""
        parts = [
            str(self.config.n_cycles),
            str(self.config.grid_N),
            str(self.config.grid_H),
            str(self.config.grid_gamma),
            str(self.config.Sc_t),
        ]
        for case in self.config.benchmark_cases:
            parts.extend([case.name, str(case.Re), str(case.S), str(case.Lambda)])
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def format_reflection_prompt(
        self, population: list[Individual], memory: Any, generation: int
    ) -> str:
        """Format a reflection prompt with top-5 closures."""
        sorted_pop = sorted(
            [ind for ind in population if ind.fitness is not None],
            key=lambda i: i.fitness.primary if i.fitness else 0.0,
            reverse=True,
        )
        top = sorted_pop[:5]
        lines = [f"Generation {generation} — top 5 damping functions:"]
        for i, ind in enumerate(top, 1):
            fit = ind.fitness.primary if ind.fitness else 0.0
            lines.append(f"  {i}. f(Ri_g) = {ind.genome}  (fitness={fit:.4f})")
        lines.append(
            "\nAnalyze patterns in the best closures. "
            "What functional forms work? What Ri_c values? "
            "Suggest 2-3 new expressions to try."
        )
        return "\n".join(lines)

    def default_operator_weights(self) -> dict[str, float]:
        """Default operator weights for CFD mutation ensemble."""
        return {
            "llm_mutate": 0.3,
            "llm_crossover": 0.2,
            "cheap_perturb": 0.2,
            "cheap_simplify": 0.15,
            "cheap_combine": 0.15,
        }

    def format_proof(self, genome: str) -> str:
        """Format a genome as a damping function definition."""
        return f"f(Ri_g) = {genome}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_form(ir: ClosureExpr) -> str:
    """Classify the functional form of a closure expression."""
    s = str(ir.expr)
    has_exp = "exp" in s
    has_pow = "**" in s
    has_div = "/" in s or "Pow" in str(type(ir.expr))

    # Check for Ri_g in a denominator (rational form)
    from sympy import Pow

    is_rational = False
    for sub in ir.expr.atoms(Pow):
        if Ri_g in sub.free_symbols and sub.args[1].is_negative:
            is_rational = True
            break

    if has_exp and (is_rational or has_div):
        return "composite"
    if has_exp:
        return "exponential"
    if is_rational:
        return "rational"
    if has_pow:
        return "power"
    return "linear"
