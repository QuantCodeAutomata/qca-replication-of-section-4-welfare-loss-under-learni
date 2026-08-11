"""BSDE solver for Experiment 2 - structural verification.

Implements backward SDE Y_t via least-squares Monte Carlo (LSM) and
verifies the decoupling identity:
    X_t + exp(int_0^t r_s ds) * Y_t = tilde_x * phi_t

Terminal condition:
    Y_T = -K * exp(-int_0^T r_s ds)

Expectation representation:
    H_t * Y_t = E[-H_T*K*exp(-int_r_T)
                  + tilde_x*int_t^T H_s*alpha_s*phi_s ds | F_t]

The LSM method regresses H_t*Y_t on polynomial basis functions of (r_t, m_t, t).

Custom - Context7 found no library equivalent (paper Section 3 FBSDE decoupling)
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from typing import Dict, Any, Tuple

from .objects import compute_eta


def _build_basis(
    r_t: np.ndarray,
    m_t: np.ndarray,
    t: float,
    degree: int = 2,
) -> np.ndarray:
    """Build polynomial basis matrix for LSM regression at a single time step.

    Columns: [1, r, m, t, r^2, m^2, t^2, rm, rt, mt, ...]

    Args:
        r_t: short rate at time t, shape (n_paths,)
        m_t: filtered mean at time t, shape (n_paths,)
        t: current time (scalar)
        degree: maximum polynomial degree

    Returns:
        design matrix (n_paths, n_features)
    """
    t_arr = np.full_like(r_t, t)
    X_raw = np.column_stack([r_t, m_t, t_arr])
    pf = PolynomialFeatures(degree=degree, include_bias=True)
    return pf.fit_transform(X_raw)


def compute_Y_Z_lsm(
    params: Dict[str, Any],
    t_grid: np.ndarray,
    v_grid: np.ndarray,
    a1_grid: np.ndarray,
    a2_grid: np.ndarray,
    r: np.ndarray,
    m: np.ndarray,
    dI: np.ndarray,
    log_H: np.ndarray,
    log_phi: np.ndarray,
    int_r: np.ndarray,
    tilde_x: float,
    K: float,
    degree: int = 2,
    ridge_alpha: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate BSDE solution Y_t and control Z_t via least-squares Monte Carlo.

    Algorithm (backward recursion):
      1. Set Y_T = -K * exp(-int_r_T)  exactly.
      2. Define V_t = H_t * Y_t.
      3. At each t_i (going backward):
           target_{i} = V_{i+1} + tilde_x * (integral of H*alpha*phi from t_i to t_{i+1})
           Regress target on basis(r_{t_i}, m_{t_i}, t_i) -> V_i estimate
           Y_i = V_i / H_i
      4. Z_i estimated from regression of Y_{i+1}*dI_i/dt on same basis.

    Args:
        params: model parameter dictionary
        t_grid: time grid (n_steps+1,)
        v_grid: filter variance (n_steps+1,)
        a1_grid: a_1 grid (n_steps+1,)
        a2_grid: a_2 grid (n_steps+1,)
        r: short rate paths (n_paths, n_steps+1)
        m: filtered mean paths (n_paths, n_steps+1)
        dI: innovation increments (n_paths, n_steps, 2)
        log_H: log stochastic exponential (n_paths, n_steps+1)
        log_phi: log phi process (n_paths, n_steps+1)
        int_r: cumulative r-integral (n_paths, n_steps+1)
        tilde_x: adjusted initial wealth (scalar)
        K: terminal liability (scalar)
        degree: polynomial degree for regression basis
        ridge_alpha: Ridge regression regularisation parameter

    Returns:
        Y: BSDE solution (n_paths, n_steps+1)
        Z: control process (n_paths, n_steps, 2)
    """
    n_paths, n_steps_plus1 = r.shape
    n_steps = n_steps_plus1 - 1
    dt = float(t_grid[1] - t_grid[0])

    gamma = params['gamma']
    psi = params['psi']
    delta = params['delta']
    theta = (1.0 - gamma) / (1.0 - 1.0 / psi)

    H = np.exp(log_H)
    phi = np.exp(log_phi)

    # Recompute alpha (needed for integrand)
    eta = compute_eta(m, params)
    eta_norm2 = eta[:, :, 0] ** 2 + eta[:, :, 1] ** 2
    bracket = (
        delta ** psi / (psi - 1.0)
        + r
        + (1.0 / (2.0 * gamma)) * eta_norm2
        - delta * theta / (1.0 - gamma)
    )
    alpha = np.exp(-int_r) * bracket

    # Running integrand H_s * alpha_s * phi_s
    integrand = H * alpha * phi  # (n_paths, n_steps+1)

    # Terminal condition (exact)
    Y_T = -K * np.exp(-int_r[:, -1])  # (n_paths,)

    Y = np.zeros((n_paths, n_steps_plus1))
    Y[:, -1] = Y_T

    Z = np.zeros((n_paths, n_steps, 2))

    # V_t = H_t * Y_t
    V_next = H[:, -1] * Y_T  # V at T

    # Cumulative future integral (backward accumulation from T)
    future_int = np.zeros(n_paths)

    for i in range(n_steps - 1, -1, -1):
        # Trapezoidal contribution from [t_i, t_{i+1}]
        contrib = tilde_x * 0.5 * (integrand[:, i] + integrand[:, i + 1]) * dt
        future_int = future_int + contrib

        # Regression target at t_i: E[V_T + tilde_x * int_{t_i}^T integrand | F_{t_i}]
        target = V_next + future_int

        # Build basis at t_i
        basis = _build_basis(r[:, i], m[:, i], t_grid[i], degree)

        # Ridge regression for numerical stability
        reg = Ridge(alpha=ridge_alpha, fit_intercept=False)
        reg.fit(basis, target)
        V_i = reg.predict(basis)

        # Y_i = V_i / H_i
        H_i = H[:, i]
        Y_i = np.where(H_i > 1e-300, V_i / H_i, 0.0)
        Y[:, i] = Y_i

        # Z_i: regression of Y_{i+1} * dI_i / dt (martingale representation)
        Y_ip1 = Y[:, i + 1]
        for k in range(2):
            cov_target = Y_ip1 * dI[:, i, k] / dt
            reg_Z = Ridge(alpha=ridge_alpha, fit_intercept=False)
            reg_Z.fit(basis, cov_target)
            Z[:, i, k] = reg_Z.predict(basis)

        V_next = V_i

    return Y, Z


def verify_decoupling_identity(
    r: np.ndarray,
    Y: np.ndarray,
    log_phi: np.ndarray,
    int_r: np.ndarray,
    tilde_x: float,
    t_grid: np.ndarray,
) -> Dict[str, Any]:
    """Verify the decoupling identity X_t + exp(int_r_t)*Y_t = tilde_x*phi_t.

    X_t is recovered as X_t = tilde_x*phi_t - exp(int_r_t)*Y_t.
    The identity then holds by construction, so we verify:
      - Residual of (X_t + exp(int_r_t)*Y_t - tilde_x*phi_t) is zero to machine precision
      - Implied wealth X_t >= 0 (admissibility check)

    Args:
        r: short rate (n_paths, n_steps+1)
        Y: BSDE solution (n_paths, n_steps+1)
        log_phi: log phi (n_paths, n_steps+1)
        int_r: cumulative r-integral (n_paths, n_steps+1)
        tilde_x: adjusted initial wealth
        t_grid: time grid (n_steps+1,)

    Returns:
        dict with:
            X: implied wealth process (n_paths, n_steps+1)
            tilde_phi: tilde_x * phi (n_paths, n_steps+1)
            residual_mean: mean absolute decoupling residual
            residual_max: max absolute decoupling residual
            X_negative_frac: fraction of (path, time) pairs with X < 0
            X_mean: mean of X
    """
    phi = np.exp(log_phi)
    tilde_phi = tilde_x * phi
    exp_int_r = np.exp(int_r)

    # Implied wealth
    X = tilde_phi - exp_int_r * Y

    # Decoupling residual (algebraic identity)
    lhs = X + exp_int_r * Y
    residual = lhs - tilde_phi  # should be zero by construction

    return {
        'X': X,
        'tilde_phi': tilde_phi,
        'residual_mean': float(np.mean(np.abs(residual))),
        'residual_max': float(np.max(np.abs(residual))),
        'X_negative_frac': float(np.mean(X < 0)),
        'X_mean': float(np.mean(X)),
        'X_terminal_mean': float(np.mean(X[:, -1])),
    }


def compute_optimal_consumption(
    log_phi: np.ndarray,
    tilde_x: float,
    params: Dict[str, Any],
) -> np.ndarray:
    """Compute the explicit optimal consumption rule c_t* = delta^psi * tilde_x * phi_t.

    Args:
        log_phi: log phi process (n_paths, n_steps+1)
        tilde_x: adjusted initial wealth
        params: model parameter dictionary

    Returns:
        c_star: optimal consumption (n_paths, n_steps+1)
    """
    delta = params['delta']
    psi = params['psi']
    phi = np.exp(log_phi)
    c_star = delta ** psi * tilde_x * phi
    return c_star


def compute_optimal_portfolio(
    Z: np.ndarray,
    log_phi: np.ndarray,
    int_r: np.ndarray,
    eta: np.ndarray,
    tilde_x: float,
    params: Dict[str, Any],
) -> np.ndarray:
    """Compute the optimal aggregate portfolio exposure in the innovation basis.

    pi_t* = -exp(int_r_t) * Z_t + (1/gamma) * tilde_x * phi_t * eta_t

    Args:
        Z: control process (n_paths, n_steps, 2)
        log_phi: log phi (n_paths, n_steps+1)
        int_r: cumulative r-integral (n_paths, n_steps+1)
        eta: filtered MPR (n_paths, n_steps+1, 2)
        tilde_x: adjusted initial wealth
        params: model parameter dictionary

    Returns:
        pi_star: optimal portfolio (n_paths, n_steps, 2) in innovation basis
    """
    gamma = params['gamma']
    phi = np.exp(log_phi[:, :-1])  # (n_paths, n_steps) left-point
    exp_int_r = np.exp(int_r[:, :-1])  # (n_paths, n_steps)

    # Liability hedging term: -exp(int_r) * Z_t
    liability_hedge = -exp_int_r[:, :, np.newaxis] * Z  # (n_paths, n_steps, 2)

    # Myopic + intertemporal hedging: (1/gamma) * tilde_x * phi_t * eta_t
    myopic = (1.0 / gamma) * tilde_x * phi[:, :, np.newaxis] * eta[:, :-1, :]

    pi_star = liability_hedge + myopic
    return pi_star
