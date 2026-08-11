"""Filter variance ODE solver for the Epstein-Zin ALM model.

Solves the deterministic Riccati-type ODE for the filter variance v_t:
    dv_t = [sigma_R^2 - 2*kappa_R*v_t - a_1(t)^2 - a_2(t)^2] dt
where:
    a_1(t) = sigma_R * rho_RS + beta * v_t
    a_2(t) = sigma_R * rho_Rr - rho_rS * beta * (1 - rho_rS^2)^(-1/2) * v_t

Under baseline (rho_rS=0):  a_2 = sigma_R * rho_Rr  (constant)

Custom - Context7 found no library equivalent (paper Appendix C ODE)
"""
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Tuple


def _ode_rhs(t: float, v: list, params: Dict[str, Any]) -> list:
    """Right-hand side of the filter variance ODE.

    Args:
        t: current time (ODE is autonomous so t is unused)
        v: current state [v_t]
        params: model parameter dictionary

    Returns:
        List [dv/dt]
    """
    kappa_R = params['kappa_R']
    sigma_R = params['sigma_R']
    rho_RS = params['rho_RS']
    rho_Rr = params['rho_Rr']
    rho_rS = params['rho_rS']
    beta = params['beta']

    v_val = max(v[0], 0.0)  # enforce non-negativity

    # Innovation gain coefficients
    a1 = sigma_R * rho_RS + beta * v_val
    if rho_rS == 0.0:
        a2 = sigma_R * rho_Rr
    else:
        a2 = sigma_R * rho_Rr - rho_rS * beta * (1.0 - rho_rS ** 2) ** (-0.5) * v_val

    dvdt = sigma_R ** 2 - 2.0 * kappa_R * v_val - a1 ** 2 - a2 ** 2
    return [dvdt]


def get_ode_coefficients(
    v: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a_1(t) and a_2(t) from the filter variance array.

    Args:
        v: filter variance array, shape (n_steps+1,)
        params: model parameter dictionary

    Returns:
        Tuple (a1, a2) each of shape (n_steps+1,)
    """
    sigma_R = params['sigma_R']
    rho_RS = params['rho_RS']
    rho_Rr = params['rho_Rr']
    rho_rS = params['rho_rS']
    beta = params['beta']

    a1 = sigma_R * rho_RS + beta * v
    if rho_rS == 0.0:
        a2 = np.full_like(v, sigma_R * rho_Rr)
    else:
        a2 = sigma_R * rho_Rr - rho_rS * beta * (1.0 - rho_rS ** 2) ** (-0.5) * v
    return a1, a2


def solve_variance_ode(
    params: Dict[str, Any],
    T: float,
    n_steps: int,
    v0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the deterministic filter variance ODE on [0, T].

    Uses scipy solve_ivp with RK45. Enforces v_t >= 0 pointwise.

    Args:
        params: model parameter dictionary
        T: terminal time
        n_steps: number of discretization steps
        v0: initial filter variance (default 0, dogmatic prior)

    Returns:
        Tuple (t_grid, v_grid) both of shape (n_steps+1,)
    """
    t_grid = np.linspace(0.0, T, n_steps + 1)
    sol = solve_ivp(
        _ode_rhs,
        [0.0, T],
        [v0],
        args=(params,),
        t_eval=t_grid,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
    )
    v_grid = np.maximum(sol.y[0], 0.0)
    return t_grid, v_grid
