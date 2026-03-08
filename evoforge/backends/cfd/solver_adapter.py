# Copyright (c) 2026 evocode contributors. MIT License. See LICENSE.

"""Solver adapter: runs fluidflow RANS solver with evolved damping functions.

This module provides a pure-NumPy ``compute_nu_t_custom`` that accepts an
arbitrary callable ``damping_fn(Ri_g_array) -> array``, and a
``run_case_evolved`` wrapper that monkey-patches the closure into the
fluidflow modules for a single solver run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from fluidflow.grid import StretchedGrid
from fluidflow.models.closures import KAPPA
from fluidflow.solvers.operators import ddz

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

DampingFn = Callable[[npt.NDArray[np.floating[Any]]], npt.NDArray[np.floating[Any]]]


# ---------------------------------------------------------------------------
# Pure-NumPy turbulent viscosity with arbitrary damping
# ---------------------------------------------------------------------------


def compute_nu_t_custom(
    u: npt.NDArray[np.floating[Any]],
    C: npt.NDArray[np.floating[Any]],
    grid: StretchedGrid,
    g_prime: float,
    Sc_t: float = 1.0,
    Ri_c: float = 0.25,
    epsilon: float = 1e-10,
    damping_fn: DampingFn | None = None,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Compute turbulent viscosity with an arbitrary damping function.

    This is a pure-NumPy (no Numba) reimplementation of
    ``fluidflow.models.closures.compute_nu_t`` that accepts a callable
    ``damping_fn(Ri_g_array) -> array`` instead of a string selector.

    Parameters
    ----------
    u : ndarray, shape (N,)
        Velocity profile.
    C : ndarray, shape (N,)
        Concentration profile.
    grid : StretchedGrid
        Grid object with ``.z`` and ``.N``.
    g_prime : float
        Reduced gravity.
    Sc_t : float
        Turbulent Schmidt number.
    Ri_c : float
        Critical Richardson number (unused by custom fn, kept for API compat).
    epsilon : float
        Regularisation to avoid division by zero.
    damping_fn : callable
        ``f(Ri_g_array) -> damping_array``.  If *None*, uses linear damping.

    Returns
    -------
    nu_t : ndarray, shape (N,)
    D_t : ndarray, shape (N,)
    """
    if damping_fn is None:

        def damping_fn(ri: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
            return np.maximum(1.0 - ri / Ri_c, 0.0)

    # Derivatives on the stretched grid
    dudz = ddz(u, grid)
    dCdz = ddz(C, grid)

    # Base mixing-length viscosity: nu_t0 = kappa^2 * z^2 * |du/dz|
    nu_t0 = KAPPA**2 * grid.z**2 * np.abs(dudz)

    # Gradient Richardson number with safe divide
    dudz_sq = dudz**2
    Ri_g = np.where(
        dudz_sq > epsilon,
        -g_prime * dCdz / dudz_sq,
        0.0,
    )

    # Apply damping and clip to non-negative
    f = np.maximum(damping_fn(Ri_g), 0.0)

    nu_t = nu_t0 * f
    D_t = nu_t / Sc_t

    return nu_t, D_t


# ---------------------------------------------------------------------------
# Monkey-patching runner
# ---------------------------------------------------------------------------


def run_case_evolved(
    params: dict[str, Any],
    damping_fn: DampingFn,
) -> dict[str, Any]:
    """Run ``fluidflow.sweep.single_run`` with an evolved damping function.

    Temporarily replaces the module-level ``compute_nu_t`` in both
    ``fluidflow.models.oscillatory_bl`` and ``fluidflow.sweep`` with a
    wrapper that delegates to :func:`compute_nu_t_custom`, then restores the
    originals (even on error).

    Parameters
    ----------
    params : dict
        Case parameters (Re, S, Lambda, N, H, gamma, n_cycles, ...).
    damping_fn : callable
        Evolved damping function ``f(Ri_g_array) -> array``.

    Returns
    -------
    dict
        Result from ``single_run``.
    """
    import fluidflow.models.oscillatory_bl as obl_mod
    import fluidflow.sweep as sweep_mod

    original_obl = obl_mod.compute_nu_t
    original_sweep = sweep_mod.compute_nu_t

    def _patched_compute_nu_t(
        u: npt.NDArray[np.floating[Any]],
        C: npt.NDArray[np.floating[Any]],
        grid: StretchedGrid,
        g_prime: float = 0.0,
        Sc_t: float = 1.0,
        Ri_c: float = 0.25,
        epsilon: float = 1e-10,
        damping: str = "linear",
    ) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Drop-in replacement that uses the evolved damping_fn."""
        return compute_nu_t_custom(
            u,
            C,
            grid,
            g_prime=g_prime,
            Sc_t=Sc_t,
            Ri_c=Ri_c,
            epsilon=epsilon,
            damping_fn=damping_fn,
        )

    try:
        obl_mod.compute_nu_t = _patched_compute_nu_t
        sweep_mod.compute_nu_t = _patched_compute_nu_t
        result: dict[str, Any] = sweep_mod.single_run(params)
    finally:
        obl_mod.compute_nu_t = original_obl
        sweep_mod.compute_nu_t = original_sweep

    return result


# ---------------------------------------------------------------------------
# Benchmark cases — Jensen et al. (1989)
# ---------------------------------------------------------------------------

JENSEN_CASES: list[dict[str, Any]] = [
    {
        "name": "jensen_Re394",
        "Re": 394.0,
        "S": 0.0,
        "Lambda": 0.0,
        "reference_fw": 0.226,
        "N": 64,
        "H": 5.0,
        "gamma": 2.0,
        "n_cycles": 10,
    },
    {
        "name": "jensen_Re803",
        "Re": 803.0,
        "S": 0.0,
        "Lambda": 0.0,
        "reference_fw": 0.114,
        "N": 64,
        "H": 5.0,
        "gamma": 2.0,
        "n_cycles": 10,
    },
]
