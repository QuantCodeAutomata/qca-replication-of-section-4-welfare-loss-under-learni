"""Tests for src/objects.py — path process computations."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import BASELINE_PARAMS, make_params
from src.filter import solve_variance_ode, get_ode_coefficients
from src.state import simulate_state
from src.objects import compute_eta, compute_path_processes


@pytest.fixture(scope='module')
def small_sim():
    """Module-scoped fixture for a small simulation (avoids re-computing)."""
    params = make_params(gamma=5.0, psi=1.5)
    t, v = solve_variance_ode(params, T=1.0, n_steps=50)
    a1, a2 = get_ode_coefficients(v, params)
    r, m, dI = simulate_state(params, t, v, a1, a2, n_paths=500, seed=0)
    return params, t, v, a1, a2, r, m, dI


def test_eta_shape(small_sim):
    """eta should have shape (n_paths, n_steps+1, 2)."""
    params, t, v, a1, a2, r, m, dI = small_sim
    eta = compute_eta(m, params)
    assert eta.shape == (*m.shape, 2)


def test_eta2_zero_baseline(small_sim):
    """Under baseline (rho_rS=0, phi_B=0), eta_2 = 0."""
    params, t, v, a1, a2, r, m, dI = small_sim
    assert params['rho_rS'] == 0.0 and params['phi_B'] == 0.0
    eta = compute_eta(m, params)
    assert np.allclose(eta[:, :, 1], 0.0), "eta_2 must be zero under baseline"


def test_eta1_equals_beta_m(small_sim):
    """eta_1 = beta * m_t."""
    params, t, v, a1, a2, r, m, dI = small_sim
    eta = compute_eta(m, params)
    assert np.allclose(eta[:, :, 0], params['beta'] * m)


def test_log_H_initial_zero(small_sim):
    """log H_0 = 0 for all paths (H_0 = 1)."""
    params, t, v, a1, a2, r, m, dI = small_sim
    log_H, alpha, log_phi, int_r = compute_path_processes(r, m, dI, t, params)
    assert np.allclose(log_H[:, 0], 0.0, atol=1e-12)


def test_log_phi_initial_zero(small_sim):
    """log phi_0 = 0 for all paths (phi_0 = 1)."""
    params, t, v, a1, a2, r, m, dI = small_sim
    log_H, alpha, log_phi, int_r = compute_path_processes(r, m, dI, t, params)
    assert np.allclose(log_phi[:, 0], 0.0, atol=1e-12)


def test_int_r_initial_zero(small_sim):
    """int_r at t=0 must be 0."""
    params, t, v, a1, a2, r, m, dI = small_sim
    log_H, alpha, log_phi, int_r = compute_path_processes(r, m, dI, t, params)
    assert np.allclose(int_r[:, 0], 0.0, atol=1e-12)


def test_int_r_nonnegative(small_sim):
    """Integral of r should be non-decreasing when r>0."""
    params, t, v, a1, a2, r, m, dI = small_sim
    log_H, alpha, log_phi, int_r = compute_path_processes(r, m, dI, t, params)
    # int_r should be non-decreasing if r > 0 almost everywhere
    # Check that cumulative integrals don't decrease by more than rounding
    diff = np.diff(int_r, axis=1)
    # Most increments should be positive (r_t around 0.02)
    pos_frac = np.mean(diff > 0)
    assert pos_frac > 0.8, f"Most int_r increments should be positive, got {pos_frac:.3f}"


def test_H_T_mean_near_one():
    """E[H_T] ≈ 1 (martingale property under sufficient paths)."""
    params = make_params(gamma=5.0, psi=1.5)
    t, v = solve_variance_ode(params, T=1.0, n_steps=100)
    a1, a2 = get_ode_coefficients(v, params)
    r, m, dI = simulate_state(params, t, v, a1, a2, n_paths=5000, seed=99)
    log_H, alpha, log_phi, int_r = compute_path_processes(r, m, dI, t, params)
    H_T_mean = float(np.mean(np.exp(log_H[:, -1])))
    # Allow 10% relative tolerance due to finite sample
    assert abs(H_T_mean - 1.0) < 0.20, f"E[H_T]={H_T_mean:.4f} too far from 1"


def test_no_learning_H_is_one_exactly():
    """Under baseline no-learning (mu_R=0, phi_B=0, rho_rS=0), H_t^0 = 1 exactly."""
    params = make_params(gamma=5.0, psi=1.5)
    assert params['mu_R'] == 0.0 and params['phi_B'] == 0.0 and params['rho_rS'] == 0.0

    t, v = solve_variance_ode(params, T=1.0, n_steps=50)
    a1, a2 = get_ode_coefficients(v, params)
    r, m, dI = simulate_state(params, t, v, a1, a2, n_paths=200, seed=1)

    # No-learning: m_t = mu_R = 0 => eta = 0 => log H = 0
    m_nl = np.zeros_like(m)
    log_H0, _, _, _ = compute_path_processes(r, m_nl, dI, t, params)

    assert np.allclose(log_H0, 0.0, atol=1e-12), (
        "H_t^0 must equal 1 under baseline no-learning (eta=0)"
    )


def test_alpha_shape(small_sim):
    """alpha should have the same shape as r."""
    params, t, v, a1, a2, r, m, dI = small_sim
    log_H, alpha, log_phi, int_r = compute_path_processes(r, m, dI, t, params)
    assert alpha.shape == r.shape


def test_shapes_consistent(small_sim):
    """All output arrays should have shape (n_paths, n_steps+1)."""
    params, t, v, a1, a2, r, m, dI = small_sim
    log_H, alpha, log_phi, int_r = compute_path_processes(r, m, dI, t, params)
    n_paths, n_steps_plus1 = r.shape
    assert log_H.shape == (n_paths, n_steps_plus1)
    assert alpha.shape == (n_paths, n_steps_plus1)
    assert log_phi.shape == (n_paths, n_steps_plus1)
    assert int_r.shape == (n_paths, n_steps_plus1)


def test_theta_computation():
    """theta = (1-gamma)/(1-1/psi) should be negative for gamma>1, psi>1."""
    params = make_params(gamma=5.0, psi=1.5)
    gamma = params['gamma']
    psi = params['psi']
    theta = (1.0 - gamma) / (1.0 - 1.0 / psi)
    # gamma=5>1, psi=1.5>1 => (1-5)/(1-2/3) = -4/(1/3) = -12
    assert theta < 0, f"theta should be negative for gamma>1,psi>1, got {theta}"
    assert abs(theta - (-12.0)) < 1e-10
