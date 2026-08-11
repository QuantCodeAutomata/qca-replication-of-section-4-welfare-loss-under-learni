"""Path process objects for the Epstein-Zin ALM model.

Implements the custom processes:
  - eta_t:   filtered market price of risk vector
  - log H_t: log of the stochastic exponential
  - alpha_t: discounted bracket process
  - log phi_t: log of the phi process
  - int_r:   cumulative integral of the short rate

All processes match the explicit formulas from the paper (Section 3 Theorem).

Custom - Context7 found no library equivalent (paper Section 3 Theorem formulas)
"""
import numpy as np
from typing import Dict, Any, Tuple


def compute_eta(
    m: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """Compute the filtered market price of risk vector eta_t.

    eta_t = (beta*m_t,  (1-rho_rS^2)^{-1/2}*(-beta*rho_rS*m_t + phi_B))'

    Under baseline (rho_rS=0, phi_B=0):
        eta_t = (beta*m_t, 0)'

    Args:
        m: filtered mean paths, shape (..., n_steps+1) or (n_paths, n_steps+1)
        params: model parameter dictionary

    Returns:
        eta: shape (*m.shape, 2)
    """
    beta = params['beta']
    rho_rS = params['rho_rS']
    phi_B = params['phi_B']

    eta1 = beta * m
    if rho_rS == 0.0:
        # factor = 1; eta2 = phi_B
        eta2 = np.full_like(m, phi_B)
    else:
        factor = (1.0 - rho_rS ** 2) ** (-0.5)
        eta2 = factor * (-beta * rho_rS * m + phi_B)

    return np.stack([eta1, eta2], axis=-1)


def compute_path_processes(
    r: np.ndarray,
    m: np.ndarray,
    dI: np.ndarray,
    t_grid: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute log H_t, alpha_t, log phi_t, and cumulative integral of r.

    Processes (paper Section 3):
      H_t = E(-int_0^t eta' dI):
        log H_{t+dt} - log H_t = -eta_t' dI_t - 0.5*||eta_t||^2 dt

      alpha_t = exp(-int_0^t r_s ds) * [delta^psi/(psi-1) + r_t
                + (1/(2*gamma))||eta_t||^2 - delta*theta/(1-gamma)]

      phi_t = exp(int_0^t [-delta^psi*psi/(psi-1)
                + ((gamma-1)/(2*gamma^2))||eta_s||^2
                + delta*theta/(1-gamma)] ds
                + (1/gamma)*int_0^t eta_s' dI_s)

      theta = (1-gamma) / (1 - 1/psi)

    Discretization: left-point Euler for stochastic integrals;
    trapezoidal rule for the short-rate integral.

    Args:
        r: short rate paths (n_paths, n_steps+1)
        m: filtered mean paths (n_paths, n_steps+1)
        dI: innovation increments (n_paths, n_steps, 2)
        t_grid: time grid (n_steps+1,)
        params: model parameter dictionary

    Returns:
        log_H:   log H process (n_paths, n_steps+1)
        alpha:   alpha process (n_paths, n_steps+1)
        log_phi: log phi process (n_paths, n_steps+1)
        int_r:   cumulative int_0^t r_s ds (n_paths, n_steps+1)
    """
    n_paths, n_steps_plus1 = r.shape
    dt = float(t_grid[1] - t_grid[0])

    gamma = params['gamma']
    psi = params['psi']
    delta = params['delta']
    # Preference aggregator: theta = (1-gamma)/(1 - 1/psi)
    theta = (1.0 - gamma) / (1.0 - 1.0 / psi)

    # eta: (n_paths, n_steps+1, 2)
    eta = compute_eta(m, params)
    # ||eta||^2: (n_paths, n_steps+1)
    eta_norm2 = eta[:, :, 0] ** 2 + eta[:, :, 1] ** 2

    # --- Cumulative integral of r (trapezoidal rule) ---
    # delta_r_i = 0.5*(r_i + r_{i+1})*dt  for i=0,...,n_steps-1
    r_trap = 0.5 * (r[:, :-1] + r[:, 1:]) * dt  # (n_paths, n_steps)
    int_r = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(r_trap, axis=1)],
        axis=1,
    )  # (n_paths, n_steps+1)

    # --- log H: left-point Euler ---
    # eta_dot_dI using left-point values eta[:, i, :] and dI[:, i, :]
    eta_dot_dI = (
        eta[:, :-1, 0] * dI[:, :, 0]
        + eta[:, :-1, 1] * dI[:, :, 1]
    )  # (n_paths, n_steps)

    dlog_H = -eta_dot_dI - 0.5 * eta_norm2[:, :-1] * dt  # (n_paths, n_steps)
    log_H = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(dlog_H, axis=1)],
        axis=1,
    )

    # --- log phi: left-point Euler ---
    # Deterministic drift per step
    drift_phi = (
        -delta ** psi * psi / (psi - 1.0)
        + (gamma - 1.0) / (2.0 * gamma ** 2) * eta_norm2[:, :-1]
        + delta * theta / (1.0 - gamma)
    )  # (n_paths, n_steps)

    # Stochastic term (same eta_dot_dI scaled by 1/gamma)
    stoch_phi = (1.0 / gamma) * eta_dot_dI  # (n_paths, n_steps)

    dlog_phi = drift_phi * dt + stoch_phi
    log_phi = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(dlog_phi, axis=1)],
        axis=1,
    )

    # --- alpha_t ---
    bracket = (
        delta ** psi / (psi - 1.0)
        + r
        + (1.0 / (2.0 * gamma)) * eta_norm2
        - delta * theta / (1.0 - gamma)
    )
    alpha = np.exp(-int_r) * bracket

    return log_H, alpha, log_phi, int_r
