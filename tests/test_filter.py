"""Tests for src/filter.py — filter variance ODE solver."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.filter import solve_variance_ode, get_ode_coefficients
from src.params import BASELINE_PARAMS


def test_variance_ode_nonnegative():
    """v_t must remain non-negative for all t."""
    t, v = solve_variance_ode(BASELINE_PARAMS, T=1.0, n_steps=252, v0=0.0)
    assert np.all(v >= 0), "Filter variance must be non-negative"


def test_variance_ode_initial_condition():
    """v_0 should match the specified initial condition."""
    v0 = 0.01
    t, v = solve_variance_ode(BASELINE_PARAMS, T=1.0, n_steps=252, v0=v0)
    assert abs(v[0] - v0) < 1e-8, f"Initial condition mismatch: v[0]={v[0]} vs v0={v0}"


def test_variance_ode_zero_initial():
    """With v0=0 (dogmatic prior), v_t >= 0 and increases initially."""
    t, v = solve_variance_ode(BASELINE_PARAMS, T=1.0, n_steps=1000, v0=0.0)
    assert v[0] == 0.0
    # sigma_R^2 > 0 implies initial growth
    assert v[1] > 0, "v_t should grow from 0 since sigma_R^2 > 0"


def test_variance_ode_time_grid():
    """Time grid should span [0, T] with correct length."""
    T = 1.0
    n = 252
    t, v = solve_variance_ode(BASELINE_PARAMS, T=T, n_steps=n)
    assert abs(t[0] - 0.0) < 1e-12
    assert abs(t[-1] - T) < 1e-12
    assert len(t) == n + 1
    assert len(v) == n + 1


def test_ode_coefficients_baseline_a2_constant():
    """Under rho_rS=0 baseline, a2 = sigma_R * rho_Rr (constant, not v-dependent)."""
    t, v = solve_variance_ode(BASELINE_PARAMS, T=1.0, n_steps=252)
    a1, a2 = get_ode_coefficients(v, BASELINE_PARAMS)
    expected_a2 = BASELINE_PARAMS['sigma_R'] * BASELINE_PARAMS['rho_Rr']
    assert np.allclose(a2, expected_a2, atol=1e-10), (
        f"a2 not constant under rho_rS=0: max deviation={np.max(np.abs(a2 - expected_a2))}"
    )


def test_variance_ode_single_step():
    """Edge case: single time step should still return two values."""
    t, v = solve_variance_ode(BASELINE_PARAMS, T=1.0, n_steps=1, v0=0.0)
    assert len(t) == 2
    assert len(v) == 2
    assert v[0] == 0.0
    assert v[1] >= 0.0


def test_variance_ode_grid_consistency():
    """Coarse and fine grids should give similar terminal v_T values."""
    _, v_coarse = solve_variance_ode(BASELINE_PARAMS, T=1.0, n_steps=100)
    _, v_fine = solve_variance_ode(BASELINE_PARAMS, T=1.0, n_steps=1000)
    assert abs(v_coarse[-1] - v_fine[-1]) < 1e-4, (
        f"Grid inconsistency: v_T coarse={v_coarse[-1]:.6f}, fine={v_fine[-1]:.6f}"
    )


def test_ode_coefficients_a1_linear_in_v():
    """a1 = sigma_R*rho_RS + beta*v should scale linearly with v."""
    t, v = solve_variance_ode(BASELINE_PARAMS, T=1.0, n_steps=252)
    a1, a2 = get_ode_coefficients(v, BASELINE_PARAMS)
    sigma_R = BASELINE_PARAMS['sigma_R']
    rho_RS = BASELINE_PARAMS['rho_RS']
    beta = BASELINE_PARAMS['beta']
    expected_a1 = sigma_R * rho_RS + beta * v
    assert np.allclose(a1, expected_a1, atol=1e-12)


def test_variance_ode_positive_v0():
    """With positive v0, v_t should remain non-negative."""
    t, v = solve_variance_ode(BASELINE_PARAMS, T=1.0, n_steps=252, v0=0.05)
    assert np.all(v >= 0)
    assert abs(v[0] - 0.05) < 1e-8
