"""Default parameter sets for the Epstein-Zin ALM model.

All values are the Section 4 baseline from the paper unless noted.
Reproducibility notes are documented here for transparency.

Reproducibility file (as required by implementation notes):
  - sigma_r = -0.03 as reported (negative); robustness with |sigma_r| run in exp_1
  - r0 = mu_r = 0.02  (Appendix C recommendation)
  - m0 = mu_R = 0.0   (dogmatic prior)
  - v0 = 0.0          (zero initial filter variance = dogmatic prior at mu_R)
  - Time grid: 252 steps over T=1 (daily discretization)
  - Base random seed: 42
"""
from typing import Dict, Any


BASELINE_PARAMS: Dict[str, Any] = {
    # --- Market dynamics parameters (Section 4 baseline) ---
    'kappa_r': 0.5,      # mean-reversion speed of short rate r_t
    'kappa_R': 1.5,      # mean-reversion speed of latent MPR R_t
    'mu_r': 0.02,        # long-run mean of short rate
    'mu_R': 0.0,         # long-run mean of latent MPR
    'phi_B': 0.0,        # bond market price of risk
    'rho_rS': 0.0,       # correlation of short rate with stock innovations
    'sigma_r': -0.03,    # short rate volatility (as reported in paper)
    'sigma_R': 0.2,      # MPR volatility
    'beta': 4.0,         # Kalman-Bucy innovation gain
    'rho_RS': -0.95,     # correlation of MPR with stock return
    'rho_Rr': 0.1,       # correlation of MPR with short rate
    'T': 1.0,            # investment horizon (years)
    # --- Preference parameters ---
    'delta': 0.08,       # subjective discount rate
    'gamma': 5.0,        # relative risk aversion (RRA)
    'psi': 1.5,          # elasticity of intertemporal substitution (EIS)
    # --- Initial conditions (not specified by paper; documented defaults) ---
    'r0': 0.02,          # r0 = mu_r  (Appendix C)
    'm0': 0.0,           # m0 = mu_R  (Appendix C)
    'v0': 0.0,           # initial filter variance (dogmatic prior)
}


def make_params(
    gamma: float = 5.0,
    psi: float = 1.5,
    delta: float = 0.08,
    sigma_r: float = -0.03,
) -> Dict[str, Any]:
    """Create a parameter dict from the baseline with custom preference values.

    Args:
        gamma: relative risk aversion (must be > 1 for the theorem conditions)
        psi: elasticity of intertemporal substitution (must be > 1)
        delta: subjective discount rate
        sigma_r: short rate volatility (default as reported in paper)

    Returns:
        Parameter dictionary based on BASELINE_PARAMS
    """
    p = dict(BASELINE_PARAMS)
    p['gamma'] = gamma
    p['psi'] = psi
    p['delta'] = delta
    p['sigma_r'] = sigma_r
    # Keep initial conditions consistent with mu_r, mu_R
    p['r0'] = p['mu_r']
    p['m0'] = p['mu_R']
    return p


# CRRA special case: gamma * psi = 1  =>  psi = 1/gamma
# For gamma=5: psi = 0.2  (EIS < 1, unlike the Epstein-Zin general case)
def make_crra_params(gamma: float = 5.0, delta: float = 0.08) -> Dict[str, Any]:
    """Create CRRA parameters satisfying gamma * psi = 1.

    Under CRRA, theta = (1-gamma)/(1-1/psi) = (1-gamma)/(1-gamma) = 1.

    Args:
        gamma: relative risk aversion
        delta: subjective discount rate

    Returns:
        CRRA parameter dictionary
    """
    psi_crra = 1.0 / gamma
    return make_params(gamma=gamma, psi=psi_crra, delta=delta)
