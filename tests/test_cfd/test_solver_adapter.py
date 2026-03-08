# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.

"""Tests for the CFD solver adapter."""

from __future__ import annotations

import numpy as np
import pytest

from evoforge.backends.cfd.solver_adapter import (
    JENSEN_CASES,
    compute_nu_t_custom,
    run_case_evolved,
)

# ---------------------------------------------------------------------------
# Test compute_nu_t_custom
# ---------------------------------------------------------------------------


class TestComputeNuTCustom:
    """Test the pure-NumPy custom nu_t computation."""

    def test_linear_damping_matches_builtin(self) -> None:
        """Custom with a linear lambda should match the built-in linear damping."""
        from fluidflow.grid import StretchedGrid
        from fluidflow.models.closures import compute_nu_t

        grid = StretchedGrid(N=32, H=5.0, gamma=2.0)
        rng = np.random.default_rng(42)
        u = rng.standard_normal(grid.N)
        u[0] = 0.0  # no-slip at bed
        C = np.abs(rng.standard_normal(grid.N))
        C[0] = 1.0

        Ri_c = 0.25
        g_prime = 0.5
        Sc_t = 1.0
        epsilon = 1e-10

        # Built-in linear damping
        nu_t_ref, D_t_ref = compute_nu_t(
            u,
            C,
            grid,
            g_prime=g_prime,
            Sc_t=Sc_t,
            Ri_c=Ri_c,
            epsilon=epsilon,
            damping="linear",
        )

        # Custom with equivalent linear lambda (Numba kernel clips to [0, 1])
        def linear_damping(Ri_g: np.ndarray) -> np.ndarray:
            return np.clip(1.0 - Ri_g / Ri_c, 0.0, 1.0)

        nu_t_custom, D_t_custom = compute_nu_t_custom(
            u,
            C,
            grid,
            g_prime=g_prime,
            Sc_t=Sc_t,
            Ri_c=Ri_c,
            epsilon=epsilon,
            damping_fn=linear_damping,
        )

        np.testing.assert_allclose(nu_t_custom, nu_t_ref, rtol=1e-10)
        np.testing.assert_allclose(D_t_custom, D_t_ref, rtol=1e-10)

    def test_zero_damping_kills_viscosity(self) -> None:
        """A damping function that returns zero should produce zero nu_t."""
        from fluidflow.grid import StretchedGrid

        grid = StretchedGrid(N=32, H=5.0, gamma=2.0)
        u = np.exp(-grid.z) * np.sin(grid.z)
        u[0] = 0.0
        C = np.zeros(grid.N)

        def zero_damping(Ri_g: np.ndarray) -> np.ndarray:
            return np.zeros_like(Ri_g)

        nu_t, D_t = compute_nu_t_custom(
            u,
            C,
            grid,
            g_prime=0.0,
            Sc_t=1.0,
            Ri_c=0.25,
            epsilon=1e-10,
            damping_fn=zero_damping,
        )

        np.testing.assert_array_equal(nu_t, 0.0)
        np.testing.assert_array_equal(D_t, 0.0)

    def test_output_shapes(self) -> None:
        """Output arrays should match grid size."""
        from fluidflow.grid import StretchedGrid

        grid = StretchedGrid(N=32, H=5.0, gamma=2.0)
        u = np.zeros(grid.N)
        C = np.zeros(grid.N)

        def identity_damping(Ri_g: np.ndarray) -> np.ndarray:
            return np.ones_like(Ri_g)

        nu_t, D_t = compute_nu_t_custom(
            u,
            C,
            grid,
            g_prime=0.0,
            Sc_t=1.0,
            Ri_c=0.25,
            epsilon=1e-10,
            damping_fn=identity_damping,
        )

        assert nu_t.shape == (grid.N,)
        assert D_t.shape == (grid.N,)

    def test_nu_t_non_negative(self) -> None:
        """nu_t should never be negative (damping clipped to >= 0)."""
        from fluidflow.grid import StretchedGrid

        grid = StretchedGrid(N=32, H=5.0, gamma=2.0)
        rng = np.random.default_rng(99)
        u = rng.standard_normal(grid.N)
        u[0] = 0.0
        C = rng.standard_normal(grid.N)
        C[0] = 1.0

        # Damping that could go negative
        def wild_damping(Ri_g: np.ndarray) -> np.ndarray:
            return 1.0 - 10.0 * Ri_g

        nu_t, _ = compute_nu_t_custom(
            u,
            C,
            grid,
            g_prime=1.0,
            Sc_t=1.0,
            Ri_c=0.25,
            epsilon=1e-10,
            damping_fn=wild_damping,
        )

        assert np.all(nu_t >= 0.0)


# ---------------------------------------------------------------------------
# Test run_case_evolved
# ---------------------------------------------------------------------------


class TestRunCaseEvolved:
    """Test the monkey-patching solver runner."""

    @pytest.mark.timeout(60)
    def test_clear_fluid_runs(self) -> None:
        """Clear-fluid case (S=0, Lambda=0) runs without error."""
        params = {
            "Re": 394.0,
            "S": 0.0,
            "Lambda": 0.0,
            "N": 32,
            "H": 5.0,
            "gamma": 2.0,
            "n_cycles": 2,
        }

        def linear_damping(Ri_g: np.ndarray) -> np.ndarray:
            return np.clip(1.0 - Ri_g / 0.25, 0.0, None)

        result = run_case_evolved(params, linear_damping)

        assert "converged" in result
        assert "params" in result
        assert result["params"]["Re"] == 394.0

    @pytest.mark.timeout(60)
    def test_restores_original_after_run(self) -> None:
        """Module-level compute_nu_t should be restored after run."""
        import fluidflow.models.oscillatory_bl as obl_mod
        import fluidflow.sweep as sweep_mod

        original_obl = obl_mod.compute_nu_t
        original_sweep = sweep_mod.compute_nu_t

        params = {
            "Re": 394.0,
            "S": 0.0,
            "Lambda": 0.0,
            "N": 32,
            "H": 5.0,
            "gamma": 2.0,
            "n_cycles": 2,
        }

        def identity_damping(Ri_g: np.ndarray) -> np.ndarray:
            return np.ones_like(Ri_g)

        run_case_evolved(params, identity_damping)

        # Verify originals were restored
        assert obl_mod.compute_nu_t is original_obl
        assert sweep_mod.compute_nu_t is original_sweep

    @pytest.mark.timeout(60)
    def test_restores_on_error(self) -> None:
        """Module-level compute_nu_t should be restored even if run errors."""
        import fluidflow.models.oscillatory_bl as obl_mod
        import fluidflow.sweep as sweep_mod

        original_obl = obl_mod.compute_nu_t
        original_sweep = sweep_mod.compute_nu_t

        # Invalid params to trigger error
        params = {
            "Re": -1.0,
            "S": 0.0,
            "Lambda": 0.0,
            "N": 32,
            "H": 5.0,
            "gamma": 2.0,
            "n_cycles": 2,
        }

        def identity_damping(Ri_g: np.ndarray) -> np.ndarray:
            return np.ones_like(Ri_g)

        with pytest.raises(ValueError):
            run_case_evolved(params, identity_damping)

        assert obl_mod.compute_nu_t is original_obl
        assert sweep_mod.compute_nu_t is original_sweep


# ---------------------------------------------------------------------------
# Test JENSEN_CASES
# ---------------------------------------------------------------------------


class TestJensenCases:
    """Test the benchmark case definitions."""

    def test_has_at_least_two_cases(self) -> None:
        assert len(JENSEN_CASES) >= 2

    def test_required_fields(self) -> None:
        required = {"name", "Re", "S", "Lambda", "reference_fw", "N", "H", "gamma", "n_cycles"}
        for case in JENSEN_CASES:
            missing = required - set(case.keys())
            assert not missing, f"Case {case.get('name', '?')} missing: {missing}"

    def test_re394_case(self) -> None:
        re394 = [c for c in JENSEN_CASES if "394" in c["name"]]
        assert len(re394) == 1
        assert re394[0]["Re"] == 394.0
        assert re394[0]["reference_fw"] == pytest.approx(0.226, rel=0.01)

    def test_re803_case(self) -> None:
        re803 = [c for c in JENSEN_CASES if "803" in c["name"]]
        assert len(re803) == 1
        assert re803[0]["Re"] == 803.0
        assert re803[0]["reference_fw"] == pytest.approx(0.114, rel=0.01)
