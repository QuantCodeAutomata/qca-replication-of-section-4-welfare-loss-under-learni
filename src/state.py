"""State variable simulation for the Epstein-Zin ALM model.

Simulates the short rate r_t and filtered market price of risk m_t
under the observable innovation filtration using Euler-Maruyama.

SDEs:
    dr_t = kappa_r(mu_r - r_t)dt + sigma_r[rho_rS dI^1 + sqrt(1-rho_rS^2) dI^2]
    dm_t = kappa_R(mu_R - m_t)dt + a_1(t) dI^1 + a_2(t) dI^2

Under baseline (rho_rS=0):
    dr_t = kappa_r(mu_r - r_t)dt + sigma_r dI^2

Custom - Context7 found no library equivalent (paper Section 2-3 SDE system)
"""
import numpy as np
from typing import Dict, Any, Tuple


def simulate_state(
    params: Dict[str, Any],
    t_grid: np.ndarray,
    v_grid: np.ndarray,
    a1_grid: np.ndarray,
    a2_grid: np.ndarray,
    n_paths: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate short rate r_t and filtered mean m_t via Euler-Maruyama.

    Args:
        params: model parameter dictionary
        t_grid: time grid of shape (n_steps+1,)
        v_grid: filter variance on the time grid (n_steps+1,)  [unused, kept for API consistency]
        a1_grid: a_1 coefficients (n_steps+1,)
        a2_grid: a_2 coefficients (n_steps+1,)
        n_paths: number of Monte Carlo paths
        seed: random seed for reproducibility

    Returns:
        r: short rate paths (n_paths, n_steps+1)
        m: filtered mean paths (n_paths, n_steps+1)
        dI: innovation Brownian increments (n_paths, n_steps, 2)
    """
    rng = np.random.default_rng(seed)
    n_steps = len(t_grid) - 1
    dt = float(t_grid[1] - t_grid[0])
    sqrt_dt = np.sqrt(dt)

    kappa_r = params['kappa_r']
    mu_r = params['mu_r']
    sigma_r = params['sigma_r']
    rho_rS = params['rho_rS']
    kappa_R = params['kappa_R']
    mu_R = params['mu_R']
    r0 = params.get('r0', mu_r)
    m0 = params.get('m0', mu_R)

    sqrt_1_rho2 = np.sqrt(max(1.0 - rho_rS ** 2, 0.0))

    # Innovation increments: each column is dI^1 or dI^2, distributed N(0, dt)
    dI = rng.standard_normal((n_paths, n_steps, 2)) * sqrt_dt

    r = np.empty((n_paths, n_steps + 1))
    m = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0
    m[:, 0] = m0

    for i in range(n_steps):
        dI1 = dI[:, i, 0]
        dI2 = dI[:, i, 1]

        # Short rate (Vasicek-type mean-reverting)
        r[:, i + 1] = (
            r[:, i]
            + kappa_r * (mu_r - r[:, i]) * dt
            + sigma_r * (rho_rS * dI1 + sqrt_1_rho2 * dI2)
        )

        # Filtered market price of risk (Kalman-Bucy dynamics)
        m[:, i + 1] = (
            m[:, i]
            + kappa_R * (mu_R - m[:, i]) * dt
            + a1_grid[i] * dI1
            + a2_grid[i] * dI2
        )

    return r, m, dI
