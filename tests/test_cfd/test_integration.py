# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.

"""Integration smoke tests for the CFD backend pipeline."""

from __future__ import annotations

import pytest

from evoforge.backends.cfd.backend import CFDBackend, CFDDiagnostics
from evoforge.backends.cfd.ir import ClosureExpr, parse_closure_expr
from evoforge.backends.cfd.operators import ConstantPerturb, SubtreeMutate, TermAddRemove
from evoforge.core.config import CFDBackendConfig, CFDBenchmarkCase
from evoforge.core.types import Fitness, Individual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fast_config() -> CFDBackendConfig:
    """Minimal config for fast integration tests."""
    return CFDBackendConfig(
        n_cycles=2,
        grid_N=32,
        benchmark_cases=[
            CFDBenchmarkCase(name="test_Re394", Re=394.0, reference_fw=0.226),
        ],
    )


def _make_individual(genome: str, fitness: Fitness | None = None) -> Individual:
    """Build an Individual with parsed IR."""
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
# Identity pipeline: parse -> canonicalize -> hash
# ---------------------------------------------------------------------------


class TestIdentityPipeline:
    def test_parse_canonicalize_hash(self) -> None:
        """Parse a genome string, canonicalize, and hash it."""
        genome = "1 - Ri_g/0.25"
        ir = parse_closure_expr(genome)
        assert ir is not None
        assert isinstance(ir, ClosureExpr)

        canon = ir.canonicalize()
        assert isinstance(canon, ClosureExpr)

        h = ir.structural_hash()
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_canonicalize_is_idempotent(self) -> None:
        """Canonicalizing twice gives the same result."""
        ir = parse_closure_expr("exp(-4*Ri_g) + 1 - 1")
        assert ir is not None
        c1 = ir.canonicalize()
        c2 = c1.canonicalize()
        assert c1.structural_hash() == c2.structural_hash()


# ---------------------------------------------------------------------------
# Dedup: equivalent expressions get the same hash
# ---------------------------------------------------------------------------


class TestDedup:
    def test_equivalent_expressions_same_hash(self) -> None:
        """Algebraically equivalent expressions should hash identically."""
        a = parse_closure_expr("1 - Ri_g/0.25")
        b = parse_closure_expr("1 - 4.0*Ri_g")
        assert a is not None and b is not None
        assert a.structural_hash() == b.structural_hash()

    def test_different_expressions_different_hash(self) -> None:
        """Genuinely different expressions should get different hashes."""
        a = parse_closure_expr("1 - Ri_g/0.25")
        b = parse_closure_expr("exp(-4*Ri_g)")
        assert a is not None and b is not None
        assert a.structural_hash() != b.structural_hash()

    def test_commutative_equivalence(self) -> None:
        """a + b and b + a should hash identically."""
        a = parse_closure_expr("Ri_g + 1")
        b = parse_closure_expr("1 + Ri_g")
        assert a is not None and b is not None
        assert a.structural_hash() == b.structural_hash()


# ---------------------------------------------------------------------------
# End-to-end: seed -> parse -> evaluate -> get fitness
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.timeout(120)
    async def test_seed_parse_evaluate(self) -> None:
        """Full pipeline: seed population, parse each, evaluate one."""
        cfg = _fast_config()
        backend = CFDBackend(cfg)

        seeds = backend.seed_population(5)
        assert len(seeds) == 5

        # All seeds should parse
        for genome in seeds:
            ir = backend.parse(genome)
            assert ir is not None, f"Failed to parse seed: {genome}"

        # Evaluate the first seed
        ir = backend.parse(seeds[0])
        assert ir is not None
        fitness, diag, trace = await backend.evaluate(ir)

        assert isinstance(fitness, Fitness)
        assert fitness.primary > 0.0
        assert isinstance(diag, CFDDiagnostics)
        assert len(diag.case_results) == 1

    @pytest.mark.timeout(120)
    async def test_multiple_seeds_evaluate(self) -> None:
        """Evaluate multiple seed types and verify all get finite fitness."""
        cfg = _fast_config()
        backend = CFDBackend(cfg)

        test_genomes = [
            "1 - Ri_g/0.25",
            "exp(-4*Ri_g)",
            "1/(1 + 4*Ri_g)",
        ]

        for genome in test_genomes:
            ir = backend.parse(genome)
            assert ir is not None
            fitness, _diag, _trace = await backend.evaluate(ir)
            assert fitness.primary > 0.0, f"Zero fitness for {genome}"


# ---------------------------------------------------------------------------
# Mutation operators produce valid parseable offspring
# ---------------------------------------------------------------------------


class TestMutationOperatorsIntegration:
    def test_constant_perturb_produces_valid(self) -> None:
        """ConstantPerturb produces parseable offspring."""
        op = ConstantPerturb()
        parent = _make_individual("1 - Ri_g/0.25")

        # Run multiple times — operator is stochastic, some may return None
        valid_count = 0
        for _ in range(20):
            result = op.apply(parent, None)
            if result is not None:
                ir = parse_closure_expr(result)
                assert ir is not None, f"Unparseable offspring: {result}"
                assert ir.free_symbols_ok(), f"Bad symbols in: {result}"
                valid_count += 1

        assert valid_count > 0, "ConstantPerturb never produced a valid offspring"

    def test_subtree_mutate_produces_valid(self) -> None:
        """SubtreeMutate produces parseable offspring."""
        op = SubtreeMutate()
        parent = _make_individual("1 - Ri_g/0.25 + Ri_g**2")

        valid_count = 0
        for _ in range(20):
            result = op.apply(parent, None)
            if result is not None:
                ir = parse_closure_expr(result)
                assert ir is not None, f"Unparseable offspring: {result}"
                assert ir.free_symbols_ok(), f"Bad symbols in: {result}"
                valid_count += 1

        assert valid_count > 0, "SubtreeMutate never produced a valid offspring"

    def test_term_add_remove_produces_valid(self) -> None:
        """TermAddRemove produces parseable offspring."""
        op = TermAddRemove()
        parent = _make_individual("1 - Ri_g/0.25")

        valid_count = 0
        for _ in range(20):
            result = op.apply(parent, None)
            if result is not None:
                ir = parse_closure_expr(result)
                assert ir is not None, f"Unparseable offspring: {result}"
                assert ir.free_symbols_ok(), f"Bad symbols in: {result}"
                valid_count += 1

        assert valid_count > 0, "TermAddRemove never produced a valid offspring"

    def test_all_backend_operators_have_apply(self) -> None:
        """All operators returned by backend.mutation_operators() have apply()."""
        backend = CFDBackend(CFDBackendConfig())
        ops = backend.mutation_operators()
        assert len(ops) == 3
        for op in ops:
            assert hasattr(op, "apply")
            assert callable(op.apply)
            assert hasattr(op, "name")
