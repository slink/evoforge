# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.

"""Tests for CFDBackend — turbulence closure evolutionary backend."""

from __future__ import annotations

import pytest

from evoforge.backends.cfd.backend import (
    CaseResult,
    CFDBackend,
    CFDDiagnostics,
    _classify_form,
)
from evoforge.backends.cfd.ir import ClosureExpr, parse_closure_expr
from evoforge.core.config import CFDBackendConfig, CFDBenchmarkCase
from evoforge.core.ir import BehaviorSpaceConfig
from evoforge.core.types import Fitness, Individual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs: object) -> CFDBackendConfig:
    """Build a fast test config (small grid, few cycles)."""
    defaults: dict[str, object] = {
        "grid_N": 32,
        "n_cycles": 2,
        "max_complexity": 30,
        "benchmark_cases": [
            CFDBenchmarkCase(name="test_Re394", Re=394.0, reference_fw=0.226),
        ],
    }
    defaults.update(kwargs)
    return CFDBackendConfig(**defaults)  # type: ignore[arg-type]


def _make_individual(genome: str, fitness: Fitness | None = None) -> Individual:
    """Build a minimal Individual for prompt tests."""
    ir = parse_closure_expr(genome)
    assert ir is not None
    return Individual(
        genome=genome,
        ir=ir,
        ir_hash=ir.structural_hash(),
        generation=0,
        fitness=fitness,
    )


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestCaseResult:
    def test_fields(self) -> None:
        cr = CaseResult(
            name="test",
            predicted_fw=0.22,
            reference_fw=0.226,
            relative_error=0.027,
            converged=True,
        )
        assert cr.name == "test"
        assert cr.converged is True


class TestCFDDiagnostics:
    def test_summary(self) -> None:
        diag = CFDDiagnostics(mean_error=0.05, complexity=7)
        s = diag.summary()
        assert "0.05" in s
        assert "7" in s

    def test_credit_summary_delegates(self) -> None:
        diag = CFDDiagnostics()
        assert isinstance(diag.credit_summary([], 100), str)


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------


class TestParse:
    def test_valid_expression(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        ir = backend.parse("1 - Ri_g/0.25")
        assert ir is not None
        assert isinstance(ir, ClosureExpr)

    def test_invalid_symbol(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        ir = backend.parse("1 - x/0.25")
        assert ir is None

    def test_empty_string(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        assert backend.parse("") is None

    def test_bad_syntax(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        assert backend.parse("1 + + +") is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateStructure:
    def test_valid(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        ir = parse_closure_expr("1 - Ri_g/0.25")
        assert ir is not None
        errors = backend.validate_structure(ir)
        assert errors == []

    def test_bad_symbol(self) -> None:
        # Manually create an IR with bad symbols
        import sympy

        x = sympy.Symbol("x")
        ir = ClosureExpr(1 - x)
        backend = CFDBackend(CFDBackendConfig())
        errors = backend.validate_structure(ir)
        assert any("symbols" in e.lower() for e in errors)

    def test_too_complex(self) -> None:
        backend = CFDBackend(CFDBackendConfig(max_complexity=3))
        ir = parse_closure_expr("1 - Ri_g/0.25 + Ri_g**2")
        assert ir is not None
        errors = backend.validate_structure(ir)
        assert any("complexity" in e.lower() for e in errors)

    def test_not_closure_expr(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        errors = backend.validate_structure("not an IR")
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


class TestSeedPopulation:
    def test_returns_requested_count(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        seeds = backend.seed_population(5)
        assert len(seeds) == 5

    def test_all_parseable(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        for g in backend.seed_population(10):
            ir = parse_closure_expr(g)
            assert ir is not None, f"Failed to parse seed: {g}"

    def test_config_seeds_first(self) -> None:
        cfg = CFDBackendConfig(seeds=["42 - Ri_g"])
        backend = CFDBackend(cfg)
        seeds = backend.seed_population(5)
        assert seeds[0] == "42 - Ri_g"

    def test_padding_when_n_exceeds_bank(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        seeds = backend.seed_population(50)
        assert len(seeds) == 50
        # All should be unique
        assert len(set(seeds)) == len(seeds)


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_system_prompt(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        prompt = backend.system_prompt()
        assert "Ri_g" in prompt
        assert "f(0)" in prompt

    def test_mutation_prompt(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        ind = _make_individual(
            "1 - Ri_g/0.25",
            Fitness(
                primary=0.8,
                auxiliary={"mean_error": 0.1, "complexity": 5.0},
                constraints={"physics_ok": True},
                feasible=True,
            ),
        )
        prompt = backend.format_mutation_prompt(ind, None)
        assert "1 - Ri_g/0.25" in prompt
        assert "0.8" in prompt

    def test_crossover_prompt(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        a = _make_individual("1 - Ri_g/0.25")
        b = _make_individual("exp(-4*Ri_g)")
        prompt = backend.format_crossover_prompt(a, b, None)
        assert "Parent A" in prompt
        assert "Parent B" in prompt

    def test_reflection_prompt(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        pop = [
            _make_individual(
                f"1 - Ri_g/{0.2 + i * 0.01}",
                Fitness(
                    primary=0.5 + i * 0.1,
                    auxiliary={},
                    constraints={},
                    feasible=True,
                ),
            )
            for i in range(7)
        ]
        prompt = backend.format_reflection_prompt(pop, None, 5)
        assert "Generation 5" in prompt
        assert "top 5" in prompt


# ---------------------------------------------------------------------------
# Extract genome
# ---------------------------------------------------------------------------


class TestExtractGenome:
    def test_plain_expression(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        assert backend.extract_genome("1 - Ri_g/0.25") == "1 - Ri_g/0.25"

    def test_with_prefix(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        result = backend.extract_genome("f(Ri_g) = exp(-4*Ri_g)")
        assert result is not None
        # Verify it parses
        assert parse_closure_expr(result) is not None

    def test_with_backticks(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        result = backend.extract_genome("`1/(1 + 4*Ri_g)`")
        assert result is not None

    def test_multiline_picks_first_valid(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        text = "Here is my suggestion:\n# A comment\nexp(-Ri_g)\nmore text"
        result = backend.extract_genome(text)
        assert result == "exp(-Ri_g)"

    def test_no_valid_expression(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        assert backend.extract_genome("no valid expression here!!!") is None


# ---------------------------------------------------------------------------
# Behavior descriptors
# ---------------------------------------------------------------------------


class TestBehaviorDescriptor:
    def test_simple_linear(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        ir = parse_closure_expr("1 - Ri_g/0.25")
        assert ir is not None
        desc = backend.behavior_descriptor(ir, None)
        assert desc[0] in ("simple", "medium", "complex")
        assert desc[1] in ("linear", "exponential", "rational", "power", "composite")

    def test_exponential_form(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        ir = parse_closure_expr("exp(-4*Ri_g)")
        assert ir is not None
        desc = backend.behavior_descriptor(ir, None)
        assert desc[1] == "exponential"

    def test_behavior_space_config(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        space = backend.behavior_space()
        assert isinstance(space, BehaviorSpaceConfig)
        assert len(space.dimensions) == 2


# ---------------------------------------------------------------------------
# Metadata methods
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_recommended_selection(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        assert backend.recommended_selection() == "lexicase"

    def test_version(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        assert backend.version() == "cfd_v1"

    def test_eval_config_hash_deterministic(self) -> None:
        cfg = _make_config()
        b1 = CFDBackend(cfg)
        b2 = CFDBackend(cfg)
        assert b1.eval_config_hash() == b2.eval_config_hash()

    def test_eval_config_hash_changes_with_config(self) -> None:
        b1 = CFDBackend(CFDBackendConfig(n_cycles=5))
        b2 = CFDBackend(CFDBackendConfig(n_cycles=10))
        assert b1.eval_config_hash() != b2.eval_config_hash()

    def test_default_operator_weights(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        weights = backend.default_operator_weights()
        assert isinstance(weights, dict)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_format_proof(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        assert backend.format_proof("1 - Ri_g/0.25") == "f(Ri_g) = 1 - Ri_g/0.25"

    def test_mutation_operators_returns_operators(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        ops = backend.mutation_operators()
        assert len(ops) == 3
        assert all(hasattr(op, "name") for op in ops)

    def test_assign_credit_returns_list(self) -> None:
        backend = CFDBackend(CFDBackendConfig())
        ir = parse_closure_expr("1 - Ri_g/0.25")
        fit = Fitness(primary=0.5, auxiliary={"raw_accuracy": 0.5}, constraints={}, feasible=True)
        result = backend.assign_credit(ir, fit, None, None)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Classify form helper
# ---------------------------------------------------------------------------


class TestClassifyForm:
    def test_linear(self) -> None:
        ir = parse_closure_expr("1 - 4*Ri_g")
        assert ir is not None
        assert _classify_form(ir) == "linear"

    def test_exponential(self) -> None:
        ir = parse_closure_expr("exp(-4*Ri_g)")
        assert ir is not None
        assert _classify_form(ir) == "exponential"

    def test_rational(self) -> None:
        ir = parse_closure_expr("1/(1 + 4*Ri_g)")
        assert ir is not None
        assert _classify_form(ir) == "rational"


# ---------------------------------------------------------------------------
# Evaluate — needs solver, so use small grid + few cycles
# ---------------------------------------------------------------------------


class TestEvaluate:
    @pytest.mark.timeout(120)
    async def test_evaluate_linear_damping(self) -> None:
        """Evaluate a simple linear damping function against one benchmark case."""
        cfg = _make_config()
        backend = CFDBackend(cfg)
        ir = parse_closure_expr("1 - Ri_g/0.25")
        assert ir is not None

        fitness, diag, trace = await backend.evaluate(ir)
        assert isinstance(fitness, Fitness)
        assert isinstance(diag, CFDDiagnostics)
        assert fitness.primary > 0.0
        assert fitness.feasible is True
        assert len(diag.case_results) == 1

    @pytest.mark.timeout(120)
    async def test_evaluate_exponential_damping(self) -> None:
        """Evaluate an exponential damping function."""
        cfg = _make_config()
        backend = CFDBackend(cfg)
        ir = parse_closure_expr("exp(-4*Ri_g)")
        assert ir is not None

        fitness, diag, trace = await backend.evaluate(ir)
        assert isinstance(fitness, Fitness)
        assert fitness.primary > 0.0

    @pytest.mark.timeout(120)
    async def test_evaluate_no_benchmark_cases(self) -> None:
        """With no benchmark cases, evaluate returns physics-only fitness."""
        cfg = CFDBackendConfig(grid_N=32, n_cycles=2)
        backend = CFDBackend(cfg)
        ir = parse_closure_expr("1 - Ri_g/0.25")
        assert ir is not None

        fitness, diag, trace = await backend.evaluate(ir)
        assert fitness.primary == pytest.approx(0.5)

    @pytest.mark.timeout(120)
    async def test_evaluate_bad_physics(self) -> None:
        """f(0) != 1 should get a penalty."""
        cfg = _make_config()
        backend = CFDBackend(cfg)
        # f(0) = 0, not 1
        ir = parse_closure_expr("Ri_g")
        assert ir is not None

        fitness, diag, trace = await backend.evaluate(ir)
        assert isinstance(diag, CFDDiagnostics)
        assert not diag.physics_ok
        # Should still get some fitness, but penalized
        assert fitness.primary >= 0.0

    async def test_evaluate_constant_zero(self) -> None:
        """Constant zero expression: f(0)=0, physics violation."""
        cfg = CFDBackendConfig(grid_N=32, n_cycles=2)
        backend = CFDBackend(cfg)
        ir = parse_closure_expr("0")
        assert ir is not None

        fitness, diag, trace = await backend.evaluate(ir)
        assert not diag.physics_ok

    @pytest.mark.timeout(120)
    async def test_evaluate_stepwise_delegates(self) -> None:
        """evaluate_stepwise should produce the same result as evaluate."""
        cfg = _make_config()
        backend = CFDBackend(cfg)
        ir = parse_closure_expr("1 - Ri_g/0.25")
        assert ir is not None

        f1, _, _ = await backend.evaluate(ir)
        f2, _, _ = await backend.evaluate_stepwise(ir)
        assert f1.primary == pytest.approx(f2.primary)
