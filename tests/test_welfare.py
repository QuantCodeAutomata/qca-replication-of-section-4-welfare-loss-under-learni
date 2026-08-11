"""Tests for src/welfare.py — welfare loss computation."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import make_params, make_crra_params
from src.welfare import compute_welfare_loss

# Small paths for fast tests
N_PATHS_FAST = 10_000
N_STEPS_FAST = 50
CHUNK_FAST = 10_000
SEED = 42


@pytest.fixture(scope='module')
def welf_K500():
    """Module-scoped: welfare result for K=500."""
    params = make_params(gamma=5.0, psi=1.5)
    return compute_welfare_loss(params, x=1000.0, K=500.0,
                                n_paths=N_PATHS_FAST, n_steps=N_STEPS_FAST,
                                chunk_size=CHUNK_FAST, seed=SEED)


@pytest.fixture(scope='module')
def welf_K0():
    """Module-scoped: welfare result for K=0."""
    params = make_params(gamma=5.0, psi=1.5)
    return compute_welfare_loss(params, x=1000.0, K=0.0,
                                n_paths=N_PATHS_FAST, n_steps=N_STEPS_FAST,
                                chunk_size=CHUNK_FAST, seed=SEED)


def test_welfare_loss_positive(welf_K500):
    """Welfare loss L must be positive (learning has value)."""
    assert welf_K500['L'] > 0, f"L={welf_K500['L']} should be positive"


def test_welfare_loss_less_than_one(welf_K500):
    """Welfare loss L must be < 1 (can't lose more than 100% of wealth)."""
    assert welf_K500['L'] < 1.0, f"L={welf_K500['L']} should be < 1"


def test_A_admissibility(welf_K500):
    """Admissibility requires 1 - A > 0 (A < 1).

    Note: A can be negative because alpha_t contains a negative bracket term
    (delta^psi/(psi-1) + r - delta*theta/(1-gamma)) that dominates at baseline
    parameters. Only the condition 1 - A > 0 is required for the value function
    to be well-defined.
    """
    A = welf_K500['A']
    assert A < 1.0, f"A={A} must satisfy A < 1 for 1-A > 0 (admissibility)"
    assert 1.0 - A > 0.0, f"1-A={1.0 - A} must be positive"


def test_B_nonneg(welf_K500):
    """B = E[H_T * K * exp(-int_r)] should be non-negative for K>=0."""
    assert welf_K500['B'] >= 0, f"B={welf_K500['B']} should be >= 0"


def test_B_zero_when_K0(welf_K0):
    """B = 0 when K=0."""
    assert abs(welf_K0['B']) < 1e-10, f"B should be 0 for K=0, got {welf_K0['B']}"


def test_vhat_formula_K500(welf_K500):
    """Vhat_0 = (1/(1-gamma)) * tilde_x^(1-gamma)."""
    gamma = 5.0
    tilde_x = welf_K500['tilde_x']
    Vhat_expected = (1.0 / (1.0 - gamma)) * tilde_x ** (1.0 - gamma)
    err = abs(welf_K500['Vhat_0'] - Vhat_expected)
    assert err < 1e-8, f"Vhat_0 formula error: {err:.2e}"


def test_vhat_formula_K0(welf_K0):
    """Vhat_0 = (1/(1-gamma)) * tilde_x^(1-gamma) also when K=0."""
    gamma = 5.0
    tilde_x = welf_K0['tilde_x']
    Vhat_expected = (1.0 / (1.0 - gamma)) * tilde_x ** (1.0 - gamma)
    err = abs(welf_K0['Vhat_0'] - Vhat_expected)
    assert err < 1e-8, f"Vhat_0 formula error (K=0): {err:.2e}"


def test_H0_mean_one(welf_K500):
    """Under baseline no-learning, E[H_T^0] must be 1 (within MC error)."""
    # H^0 = 1 identically, so mean must be exactly 1
    assert abs(welf_K500['H0_T_mean'] - 1.0) < 1e-10, (
        f"H0_T_mean={welf_K500['H0_T_mean']} should be exactly 1"
    )


def test_welfare_loss_positive_K0(welf_K0):
    """Welfare loss should still be positive even with K=0."""
    assert welf_K0['L'] > 0


def test_welfare_loss_K0_independent_of_x():
    """With K=0, welfare loss L should be (approximately) independent of x."""
    params = make_params(gamma=5.0, psi=1.5)
    L_vals = []
    for x in [500.0, 1000.0, 2000.0]:
        res = compute_welfare_loss(params, x=x, K=0.0,
                                   n_paths=N_PATHS_FAST, n_steps=N_STEPS_FAST,
                                   chunk_size=CHUNK_FAST, seed=SEED)
        L_vals.append(res['L'])

    std_L = np.std(L_vals)
    # With K=0, L should be essentially constant; allow MC noise
    assert std_L < 0.01, (
        f"K=0: L should be independent of x, but std={std_L:.4f} across "
        f"x=[500, 1000, 2000]"
    )


def test_welfare_loss_decreases_with_gamma():
    """Higher RRA gamma => lower welfare loss."""
    params_low_g = make_params(gamma=2.0, psi=1.5)
    params_high_g = make_params(gamma=10.0, psi=1.5)

    r_low = compute_welfare_loss(params_low_g, x=1000.0, K=500.0,
                                  n_paths=N_PATHS_FAST, n_steps=N_STEPS_FAST,
                                  chunk_size=CHUNK_FAST, seed=SEED)
    r_high = compute_welfare_loss(params_high_g, x=1000.0, K=500.0,
                                   n_paths=N_PATHS_FAST, n_steps=N_STEPS_FAST,
                                   chunk_size=CHUNK_FAST, seed=SEED)

    assert r_low['L'] > r_high['L'], (
        f"L should decrease with gamma: L(gamma=2)={r_low['L']:.6f}, "
        f"L(gamma=10)={r_high['L']:.6f}"
    )


def test_welfare_loss_increases_with_psi():
    """Higher EIS psi => higher welfare loss."""
    params_low_p = make_params(gamma=5.0, psi=1.1)
    params_high_p = make_params(gamma=5.0, psi=2.0)

    r_low = compute_welfare_loss(params_low_p, x=1000.0, K=500.0,
                                  n_paths=N_PATHS_FAST, n_steps=N_STEPS_FAST,
                                  chunk_size=CHUNK_FAST, seed=SEED)
    r_high = compute_welfare_loss(params_high_p, x=1000.0, K=500.0,
                                   n_paths=N_PATHS_FAST, n_steps=N_STEPS_FAST,
                                   chunk_size=CHUNK_FAST, seed=SEED)

    assert r_low['L'] < r_high['L'], (
        f"L should increase with psi: L(psi=1.1)={r_low['L']:.6f}, "
        f"L(psi=2.0)={r_high['L']:.6f}"
    )


def test_crra_smaller_welfare_loss():
    """Welfare loss under CRRA (gamma*psi=1) should be smaller than EZ."""
    gamma = 5.0
    K = 500.0
    x = 1000.0

    params_ez = make_params(gamma=gamma, psi=1.5)
    params_crra = make_crra_params(gamma=gamma)

    res_ez = compute_welfare_loss(params_ez, x=x, K=K,
                                  n_paths=N_PATHS_FAST, n_steps=N_STEPS_FAST,
                                  chunk_size=CHUNK_FAST, seed=SEED)
    res_crra = compute_welfare_loss(params_crra, x=x, K=K,
                                    n_paths=N_PATHS_FAST, n_steps=N_STEPS_FAST,
                                    chunk_size=CHUNK_FAST, seed=SEED)

    assert res_ez['L'] >= res_crra['L'], (
        f"EZ welfare loss {res_ez['L']:.6f} should be >= CRRA {res_crra['L']:.6f}"
    )


def test_admissibility_x_gt_B(welf_K500):
    """tilde_x > 0 implies x > B (admissibility)."""
    assert welf_K500['tilde_x'] > 0


def test_mc_standard_errors_reported(welf_K500):
    """Monte Carlo standard errors should be present in the result dict."""
    assert 'mc_std_A' in welf_K500
    assert 'mc_std_B' in welf_K500
    assert 'mc_std_A0' in welf_K500
    assert 'mc_std_B0' in welf_K500
    assert welf_K500['mc_std_A'] >= 0
    assert welf_K500['mc_std_B'] >= 0


def test_welfare_loss_increases_with_x_K500():
    """With K=500, welfare loss L should increase with initial wealth x."""
    params = make_params(gamma=5.0, psi=1.5)
    x_vals = [700.0, 1000.0, 1500.0, 2000.0]
    L_vals = []
    for x in x_vals:
        res = compute_welfare_loss(params, x=x, K=500.0,
                                   n_paths=N_PATHS_FAST, n_steps=N_STEPS_FAST,
                                   chunk_size=CHUNK_FAST, seed=SEED)
        L_vals.append(res['L'])

    # L should be monotonically increasing in x when K=500
    for i in range(len(L_vals) - 1):
        assert L_vals[i] < L_vals[i+1], (
            f"L should increase with x (K=500): L({x_vals[i]})={L_vals[i]:.6f} "
            f">= L({x_vals[i+1]})={L_vals[i+1]:.6f}"
        )
