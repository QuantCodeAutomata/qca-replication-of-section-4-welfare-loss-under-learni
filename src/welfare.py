"""Welfare loss computation for the Epstein-Zin ALM model.

Implements the Section 4 welfare loss metric:
    L = 1 - (1/x)*[((1-A)/(1-A^0))*(x - B^0) + B]

where:
    A  = E[int_0^T H_s * alpha_s * phi_s ds]       (learning case)
    B  = E[H_T * K * exp(-int_0^T r_s ds)]           (learning case)
    A^0, B^0 = no-learning analogues (m_t = mu_R constant)

The no-learning case uses the same dI paths (common random numbers) for
variance reduction.

Custom - Context7 found no library equivalent (paper Section 4 formula)
"""
import numpy as np
from typing import Dict, Any, List

from .filter import solve_variance_ode, get_ode_coefficients
from .state import simulate_state
from .objects import compute_path_processes


def _run_chunk(
    params: Dict[str, Any],
    t_grid: np.ndarray,
    v_grid: np.ndarray,
    a1_grid: np.ndarray,
    a2_grid: np.ndarray,
    K: float,
    n_paths: int,
    seed: int,
) -> Dict[str, float]:
    """Run one Monte Carlo chunk, returning contributions to A, B, A^0, B^0.

    Uses common random numbers across learning and no-learning cases.

    Args:
        params: model parameter dictionary
        t_grid: time grid (n_steps+1,)
        v_grid: filter variance (n_steps+1,)
        a1_grid: a_1 coefficients (n_steps+1,)
        a2_grid: a_2 coefficients (n_steps+1,)
        K: terminal liability
        n_paths: paths in this chunk
        seed: random seed

    Returns:
        dict with keys: A, B, A0, B0, H_T_mean, H0_T_mean
    """
    # Simulate state (learning case)
    r, m, dI = simulate_state(params, t_grid, v_grid, a1_grid, a2_grid, n_paths, seed)

    # --- Learning case ---
    log_H, alpha, log_phi, int_r = compute_path_processes(r, m, dI, t_grid, params)
    H = np.exp(log_H)
    phi = np.exp(log_phi)

    # A = E[int_0^T H_s * alpha_s * phi_s ds]  (trapezoidal rule)
    integrand = H * alpha * phi  # (n_paths, n_steps+1)
    A = float(np.mean(np.trapz(integrand, t_grid, axis=1)))

    # B = E[H_T * K * exp(-int_r_T)]
    B = float(np.mean(H[:, -1] * K * np.exp(-int_r[:, -1])))
    H_T_mean = float(np.mean(H[:, -1]))

    # --- No-learning benchmark: m_t = mu_R (constant) ---
    mu_R = params['mu_R']
    m_nl = np.full_like(m, mu_R)
    log_H0, alpha0, log_phi0, int_r0 = compute_path_processes(r, m_nl, dI, t_grid, params)
    H0 = np.exp(log_H0)
    phi0 = np.exp(log_phi0)

    integrand0 = H0 * alpha0 * phi0
    A0 = float(np.mean(np.trapz(integrand0, t_grid, axis=1)))
    B0 = float(np.mean(H0[:, -1] * K * np.exp(-int_r0[:, -1])))
    H0_T_mean = float(np.mean(H0[:, -1]))

    return {
        'A': A, 'B': B, 'A0': A0, 'B0': B0,
        'H_T_mean': H_T_mean, 'H0_T_mean': H0_T_mean,
    }


def compute_welfare_loss(
    params: Dict[str, Any],
    x: float,
    K: float,
    n_paths: int = 200_000,
    n_steps: int = 252,
    chunk_size: int = 50_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute the welfare loss from ignoring learning about the latent MPR.

    Computes:
        tilde_x = (x - B) / (1 - A)
        L = 1 - (1/x)*[((1-A)/(1-A^0))*(x - B^0) + B]
        Vhat_0 = (1/(1-gamma)) * tilde_x^(1-gamma)

    Args:
        params: model parameter dictionary (see src/params.py)
        x: initial wealth
        K: terminal liability (use K=0 for liability-free case)
        n_paths: total Monte Carlo paths (paper uses 1_000_000)
        n_steps: time discretization steps (default 252 = daily over T=1)
        chunk_size: paths per chunk for memory management
        seed: base random seed

    Returns:
        dict containing: A, B, A0, B0, tilde_x, L, Vhat_0,
                         H_T_mean, H0_T_mean, mc_std_A, mc_std_B,
                         mc_std_A0, mc_std_B0, n_paths_actual,
                         t_grid, v_grid
    """
    T = params.get('T', 1.0)
    v0 = params.get('v0', 0.0)

    # Solve deterministic filter variance ODE
    t_grid, v_grid = solve_variance_ode(params, T, n_steps, v0)
    a1_grid, a2_grid = get_ode_coefficients(v_grid, params)

    n_chunks = max(1, n_paths // chunk_size)

    A_chunks: List[float] = []
    B_chunks: List[float] = []
    A0_chunks: List[float] = []
    B0_chunks: List[float] = []
    H_T_means: List[float] = []
    H0_T_means: List[float] = []

    for chunk in range(n_chunks):
        chunk_seed = seed * 10_000 + chunk
        res = _run_chunk(
            params, t_grid, v_grid, a1_grid, a2_grid,
            K, chunk_size, chunk_seed,
        )
        A_chunks.append(res['A'])
        B_chunks.append(res['B'])
        A0_chunks.append(res['A0'])
        B0_chunks.append(res['B0'])
        H_T_means.append(res['H_T_mean'])
        H0_T_means.append(res['H0_T_mean'])

    A = float(np.mean(A_chunks))
    B = float(np.mean(B_chunks))
    A0 = float(np.mean(A0_chunks))
    B0 = float(np.mean(B0_chunks))

    mc_std_A = float(np.std(A_chunks) / np.sqrt(n_chunks)) if n_chunks > 1 else 0.0
    mc_std_B = float(np.std(B_chunks) / np.sqrt(n_chunks)) if n_chunks > 1 else 0.0
    mc_std_A0 = float(np.std(A0_chunks) / np.sqrt(n_chunks)) if n_chunks > 1 else 0.0
    mc_std_B0 = float(np.std(B0_chunks) / np.sqrt(n_chunks)) if n_chunks > 1 else 0.0

    # Admissibility checks
    assert x > B, f"Admissibility: x={x} must exceed B={B:.6f}"
    assert 1.0 - A > 0.0, f"Admissibility: 1-A={1.0 - A:.6f} must be positive"
    assert 1.0 - A0 > 0.0, f"Admissibility: 1-A0={1.0 - A0:.6f} must be positive"

    gamma = params['gamma']
    tilde_x = (x - B) / (1.0 - A)
    L = 1.0 - (1.0 / x) * (((1.0 - A) / (1.0 - A0)) * (x - B0) + B)
    Vhat_0 = (1.0 / (1.0 - gamma)) * tilde_x ** (1.0 - gamma)

    return {
        'A': A, 'B': B, 'A0': A0, 'B0': B0,
        'tilde_x': tilde_x,
        'L': L,
        'Vhat_0': Vhat_0,
        'H_T_mean': float(np.mean(H_T_means)),
        'H0_T_mean': float(np.mean(H0_T_means)),
        'mc_std_A': mc_std_A,
        'mc_std_B': mc_std_B,
        'mc_std_A0': mc_std_A0,
        'mc_std_B0': mc_std_B0,
        'n_paths_actual': n_chunks * chunk_size,
        't_grid': t_grid,
        'v_grid': v_grid,
    }


def compute_mc_expectations(
    params: Dict[str, Any],
    K: float,
    n_paths: int = 200_000,
    n_steps: int = 252,
    chunk_size: int = 50_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute Monte Carlo expectations A, B, A⁰, B⁰ independent of initial wealth x.

    Since L = 1 - (1/x)*[((1-A)/(1-A^0))*(x-B^0)+B] depends on x only through
    this closed-form expression, calling this function once per (params, K)
    combination allows computing L for any x without re-running Monte Carlo.

    Args:
        params: model parameter dictionary
        K: terminal liability
        n_paths: total Monte Carlo paths
        n_steps: time discretization steps
        chunk_size: paths per chunk
        seed: base random seed

    Returns:
        dict with keys: A, B, A0, B0, H_T_mean, H0_T_mean,
                        mc_std_A, mc_std_B, mc_std_A0, mc_std_B0,
                        n_paths_actual, t_grid, v_grid
    """
    T = params.get('T', 1.0)
    v0 = params.get('v0', 0.0)

    t_grid, v_grid = solve_variance_ode(params, T, n_steps, v0)
    a1_grid, a2_grid = get_ode_coefficients(v_grid, params)

    n_chunks = max(1, n_paths // chunk_size)
    A_chunks: List[float] = []
    B_chunks: List[float] = []
    A0_chunks: List[float] = []
    B0_chunks: List[float] = []
    H_T_means: List[float] = []
    H0_T_means: List[float] = []

    for chunk in range(n_chunks):
        chunk_seed = seed * 10_000 + chunk
        res = _run_chunk(
            params, t_grid, v_grid, a1_grid, a2_grid,
            K, chunk_size, chunk_seed,
        )
        A_chunks.append(res['A'])
        B_chunks.append(res['B'])
        A0_chunks.append(res['A0'])
        B0_chunks.append(res['B0'])
        H_T_means.append(res['H_T_mean'])
        H0_T_means.append(res['H0_T_mean'])

    n_chunks_actual = n_chunks
    A = float(np.mean(A_chunks))
    B = float(np.mean(B_chunks))
    A0 = float(np.mean(A0_chunks))
    B0 = float(np.mean(B0_chunks))

    return {
        'A': A, 'B': B, 'A0': A0, 'B0': B0,
        'H_T_mean': float(np.mean(H_T_means)),
        'H0_T_mean': float(np.mean(H0_T_means)),
        'mc_std_A': float(np.std(A_chunks) / np.sqrt(n_chunks_actual)) if n_chunks_actual > 1 else 0.0,
        'mc_std_B': float(np.std(B_chunks) / np.sqrt(n_chunks_actual)) if n_chunks_actual > 1 else 0.0,
        'mc_std_A0': float(np.std(A0_chunks) / np.sqrt(n_chunks_actual)) if n_chunks_actual > 1 else 0.0,
        'mc_std_B0': float(np.std(B0_chunks) / np.sqrt(n_chunks_actual)) if n_chunks_actual > 1 else 0.0,
        'n_paths_actual': n_chunks_actual * chunk_size,
        't_grid': t_grid,
        'v_grid': v_grid,
    }


def welfare_loss_from_expectations(
    x: float,
    A: float,
    B: float,
    A0: float,
    B0: float,
    gamma: float,
) -> Dict[str, float]:
    """Evaluate welfare loss L and value Vhat_0 from pre-computed expectations.

    This is a pure O(1) function; use after compute_mc_expectations to sweep
    over a grid of x values without re-running Monte Carlo.

    Args:
        x: initial wealth
        A, B: learning-case expectations
        A0, B0: no-learning expectations
        gamma: relative risk aversion

    Returns:
        dict with tilde_x, L, Vhat_0
    """
    assert x > B, f"Admissibility: x={x:.4f} must exceed B={B:.6f}"
    assert 1.0 - A > 0.0, f"Admissibility: 1-A={1.0-A:.6f} must be positive"
    assert 1.0 - A0 > 0.0, f"Admissibility: 1-A0={1.0-A0:.6f} must be positive"

    tilde_x = (x - B) / (1.0 - A)
    L = 1.0 - (1.0 / x) * (((1.0 - A) / (1.0 - A0)) * (x - B0) + B)
    Vhat_0 = (1.0 / (1.0 - gamma)) * tilde_x ** (1.0 - gamma)
    return {'tilde_x': tilde_x, 'L': L, 'Vhat_0': Vhat_0}


def compute_welfare_loss_x_grid(
    params: Dict[str, Any],
    x_grid: np.ndarray,
    K: float,
    n_paths: int = 200_000,
    n_steps: int = 252,
    chunk_size: int = 50_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute welfare loss over a grid of initial wealth values.

    Runs Monte Carlo **once** per (params, K) then evaluates L analytically
    for each x in x_grid, which is 8x faster than calling compute_welfare_loss
    once per x value.

    Args:
        params: model parameter dictionary
        x_grid: array of initial wealth values
        K: terminal liability
        n_paths: total Monte Carlo paths
        n_steps: time discretization steps
        chunk_size: paths per chunk
        seed: base random seed

    Returns:
        dict with keys: A, B, A0, B0, x_grid, L_grid (array), and MC metadata
    """
    exp = compute_mc_expectations(params, K, n_paths, n_steps, chunk_size, seed)
    A, B, A0, B0 = exp['A'], exp['B'], exp['A0'], exp['B0']
    gamma = params['gamma']

    L_grid = np.full(len(x_grid), np.nan)
    for i, x in enumerate(x_grid):
        try:
            res = welfare_loss_from_expectations(x, A, B, A0, B0, gamma)
            L_grid[i] = res['L']
        except AssertionError:
            L_grid[i] = np.nan

    return {**exp, 'x_grid': np.asarray(x_grid), 'L_grid': L_grid}
