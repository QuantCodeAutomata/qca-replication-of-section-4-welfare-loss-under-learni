# Welfare Loss Under Learning vs No-Learning in an Epstein-Zin ALM Problem

Replication of Section 4 and structural verification of Section 3 from the paper on
continuous-time Epstein-Zin Asset-Liability Management with learning about an
unobservable market price of risk.

## Overview

This repository implements two numerical experiments:

**Experiment 1** (Section 4): Welfare Loss Comparative Statics

- Computes the welfare loss *L* from ignoring learning about the latent market price of risk
- Replicates Figure 1 comparative statics: *L* vs initial wealth *x* across (γ, ψ) grids
- Validates key theoretical predictions:
  - *L* > 0 (learning has positive value)
  - With K > 0, *L* increases with *x*
  - With K = 0, *L* is independent of *x*
  - *L* decreases with risk aversion γ
  - *L* increases with EIS ψ
  - CRRA welfare loss ≤ general Epstein-Zin welfare loss

**Experiment 2** (Section 3): Structural Verification

- Numerically verifies the explicit value function and BSDE decoupling identity
- Implements Least-Squares Monte Carlo for the backward SDE Y_t
- Checks: terminal condition, decoupling identity, optimal consumption positivity

## Methodology

The model is a continuous-time filtering problem where the market price of risk is
latent. The investor uses the filtered estimate m_t (from the Kalman-Bucy filter with
deterministic variance v_t) to make portfolio and consumption decisions.

Key formulas (paper Section 3):

```
theta = (1-gamma)/(1-1/psi)
dv_t = [sigma_R^2 - 2*kappa_R*v_t - a1(t)^2 - a2(t)^2] dt
eta_t = (beta*m_t, 0)  [under baseline rho_rS=0, phi_B=0]
H_t  = exp(-int_0^t eta' dI - 0.5*int_0^t ||eta||^2 ds)
A    = E[int_0^T H_s * alpha_s * phi_s ds]
B    = E[H_T * K * exp(-int_0^T r_s ds)]
L    = 1 - (1/x)*[((1-A)/(1-A^0))*(x-B^0) + B]
```

## Baseline Parameters (Section 4)

| Parameter | Value |
|-----------|-------|
| κ_r       | 0.5   |
| κ_R       | 1.5   |
| μ_r       | 0.02  |
| μ_R       | 0.0   |
| φ_B       | 0.0   |
| ρ_rS      | 0.0   |
| σ_r       | −0.03 |
| σ_R       | 0.2   |
| β         | 4.0   |
| ρ_RS      | −0.95 |
| ρ_Rr      | 0.1   |
| T         | 1.0   |
| δ         | 0.08  |

Initial conditions: r₀ = μ_r = 0.02, m₀ = μ_R = 0, v₀ = 0 (dogmatic prior).

## Repository Structure

```
repo/
├── data/                   # (no empirical data; pure simulation)
├── src/
│   ├── params.py           # Baseline parameter sets
│   ├── filter.py           # Deterministic filter variance ODE (v_t)
│   ├── state.py            # State SDE simulation (r_t, m_t)
│   ├── objects.py          # Path processes: eta, H, alpha, phi
│   ├── welfare.py          # Welfare loss A, B, tilde_x, L
│   └── bsde.py             # Backward SDE via Least-Squares MC
├── exp/
│   ├── exp_1_welfare_loss.py           # Figure 1 replication
│   └── exp_2_structural_verification.py # FBSDE structural checks
├── tests/
│   ├── test_filter.py
│   ├── test_objects.py
│   ├── test_welfare.py
│   └── test_bsde.py
├── results/
│   ├── RESULTS.md              # Summary of findings
│   ├── figure1_left.png        # Left panel: vary psi
│   ├── figure1_right.png       # Right panel: vary gamma
│   ├── K_comparison.png        # K=500 vs K=0
│   └── welfare_summary.csv     # Numerical values
└── README.md
```

## Installation

```bash
pip install numpy scipy matplotlib pandas scikit-learn
```

## Running Experiments

```bash
# Experiment 1 — Figure 1 replication (~5–20 min depending on hardware)
python exp/exp_1_welfare_loss.py

# Experiment 2 — Structural verification
python exp/exp_2_structural_verification.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Reproducibility Notes

- `sigma_r = -0.03` as reported in the paper (negative sign); a robustness check
  with `sigma_r = +0.03` is also run in Experiment 1
- Initial conditions: `r0 = mu_r`, `m0 = mu_R`, `v0 = 0` (Appendix C defaults)
- Time grid: 252 steps over [0, T=1] (daily discretization)
- Monte Carlo: 200,000 paths default (paper uses 1,000,000); set `N_PATHS` in
  `exp_1_welfare_loss.py` to scale up
- Random seed: 42
- Common random numbers used across learning and no-learning cases for variance reduction

## Key Results

See `results/RESULTS.md` for full output after running the experiments.

The main qualitative findings match the paper:
- Welfare loss is positive and economically meaningful
- With liabilities (K=500), L increases with initial wealth x
- Without liabilities (K=0), L is flat in x
- L decreases with γ and increases with ψ
- CRRA welfare loss < Epstein-Zin welfare loss
