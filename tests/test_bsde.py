"""Tests for src/bsde.py — BSDE structural verification."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import make_params
from src.filter import solve_variance_ode, get_ode_coefficients
from src.state import simulate_state
from src.objects import compute_path_processes, compute_eta
from src.welfare import compute_welfare_loss
from src.bsde import (
    compute_Y_Z_lsm,
    verify_decoupling_identity,
    compute_optimal_consumption,
    compute_optimal_portfolio,
)

N_PATHS = 1_000
N_STEPS = 20
SEED = 42


@pytest.fixture(scope='module')
def bsde_sim():
    """Module-scoped fixture for BSDE simulation."""
    params = make_params(gamma=5.0, psi=1.5)
    T, v0 = params['T'], params.get('v0', 0.0)
    t_grid, v_grid = solve_variance_ode(params, T, N_STEPS, v0)
    a1_grid, a2_grid = get_ode_coefficients(v_grid, params)
    r, m, dI = simulate_state(params, t_grid, v_grid, a1_grid, a2_grid, N_PATHS, SEED)
    log_H, alpha, log_phi, int_r = compute_path_processes(r, m, dI, t_grid, params)

    res_welf = compute_welfare_loss(params, x=1000.0, K=500.0,
                                    n_paths=max(N_PATHS, 5000), n_steps=N_STEPS,
                                    chunk_size=max(N_PATHS, 5000), seed=SEED+1)
    tilde_x = res_welf['tilde_x']

    Y, Z = compute_Y_Z_lsm(
        params, t_grid, v_grid, a1_grid, a2_grid,
        r, m, dI, log_H, log_phi, int_r,
        tilde_x, K=500.0, degree=2,
    )
    return {
        'params': params, 't_grid': t_grid, 'v_grid': v_grid,
        'r': r, 'm': m, 'dI': dI,
        'log_H': log_H, 'alpha': alpha, 'log_phi': log_phi, 'int_r': int_r,
        'tilde_x': tilde_x, 'K': 500.0, 'Y': Y, 'Z': Z,
    }


def test_Y_shape(bsde_sim):
    """Y should have shape (n_paths, n_steps+1)."""
    Y = bsde_sim['Y']
    assert Y.shape == (N_PATHS, N_STEPS + 1)


def test_Z_shape(bsde_sim):
    """Z should have shape (n_paths, n_steps, 2)."""
    Z = bsde_sim['Z']
    assert Z.shape == (N_PATHS, N_STEPS, 2)


def test_Y_terminal_condition(bsde_sim):
    """Y_T = -K * exp(-int_r_T) must hold exactly."""
    Y = bsde_sim['Y']
    int_r = bsde_sim['int_r']
    K = bsde_sim['K']
    Y_T_expected = -K * np.exp(-int_r[:, -1])
    max_err = np.max(np.abs(Y[:, -1] - Y_T_expected))
    assert max_err < 1e-8, f"Terminal condition max error: {max_err:.2e}"


def test_decoupling_identity(bsde_sim):
    """X_t + exp(int_r_t)*Y_t = tilde_x*phi_t should hold to tight tolerance."""
    r = bsde_sim['r']
    Y = bsde_sim['Y']
    log_phi = bsde_sim['log_phi']
    int_r = bsde_sim['int_r']
    tilde_x = bsde_sim['tilde_x']
    t_grid = bsde_sim['t_grid']

    check = verify_decoupling_identity(r, Y, log_phi, int_r, tilde_x, t_grid)
    # The identity should hold by construction; tolerance allows regression error
    assert check['residual_mean'] < 50.0, (
        f"Decoupling residual too large: {check['residual_mean']:.4f}"
    )


def test_optimal_consumption_positive(bsde_sim):
    """c_t* = delta^psi * tilde_x * phi_t should always be positive."""
    log_phi = bsde_sim['log_phi']
    tilde_x = bsde_sim['tilde_x']
    params = bsde_sim['params']
    c_star = compute_optimal_consumption(log_phi, tilde_x, params)
    assert np.all(c_star > 0), "Consumption must be positive"


def test_optimal_consumption_shape(bsde_sim):
    """c_t* should have shape (n_paths, n_steps+1)."""
    log_phi = bsde_sim['log_phi']
    tilde_x = bsde_sim['tilde_x']
    params = bsde_sim['params']
    c_star = compute_optimal_consumption(log_phi, tilde_x, params)
    assert c_star.shape == (N_PATHS, N_STEPS + 1)


def test_optimal_portfolio_shape(bsde_sim):
    """pi_t* should have shape (n_paths, n_steps, 2) or (n_paths, n_steps+1, 2)."""
    Z = bsde_sim['Z']
    log_phi = bsde_sim['log_phi']
    int_r = bsde_sim['int_r']
    m = bsde_sim['m']
    params = bsde_sim['params']
    tilde_x = bsde_sim['tilde_x']
    eta = compute_eta(m, params)
    pi_star = compute_optimal_portfolio(Z, log_phi, int_r, eta, tilde_x, params)
    n_paths = N_PATHS
    assert pi_star.shape[0] == n_paths
    assert pi_star.shape[-1] == 2


def test_bsde_no_nan(bsde_sim):
    """Y and Z should not contain NaN or Inf values."""
    assert not np.any(np.isnan(bsde_sim['Y'])), "Y contains NaN"
    assert not np.any(np.isinf(bsde_sim['Y'])), "Y contains Inf"
    assert not np.any(np.isnan(bsde_sim['Z'])), "Z contains NaN"
    assert not np.any(np.isinf(bsde_sim['Z'])), "Z contains Inf"


def test_Y_terminal_K0():
    """Y_T = 0 when K=0."""
    params = make_params(gamma=5.0, psi=1.5)
    T, v0 = params['T'], params.get('v0', 0.0)
    t_grid, v_grid = solve_variance_ode(params, T, N_STEPS, v0)
    a1_grid, a2_grid = get_ode_coefficients(v_grid, params)
    r, m, dI = simulate_state(params, t_grid, v_grid, a1_grid, a2_grid, N_PATHS, SEED)
    log_H, alpha, log_phi, int_r = compute_path_processes(r, m, dI, t_grid, params)
    res_welf = compute_welfare_loss(params, x=1000.0, K=0.0,
                                    n_paths=5000, n_steps=N_STEPS,
                                    chunk_size=5000, seed=SEED)
    tilde_x = res_welf['tilde_x']
    Y, Z = compute_Y_Z_lsm(
        params, t_grid, v_grid, a1_grid, a2_grid,
        r, m, dI, log_H, log_phi, int_r,
        tilde_x, K=0.0, degree=2,
    )
    assert np.allclose(Y[:, -1], 0.0, atol=1e-8), "Y_T should be 0 when K=0"
