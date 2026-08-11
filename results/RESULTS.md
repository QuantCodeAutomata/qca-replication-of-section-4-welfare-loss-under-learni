# Experiment Results

## Experiment 1: Welfare Loss Comparative Statics

**Monte Carlo settings:** 200,000 paths, 252 time steps, seed=42


### Validation Diagnostics

- E[H_T^0] (no-learning baseline): 1.000000000000 (should be exactly 1.0)

- Vhat_0 formula error: 0.00e+00

- L at baseline (γ=5, ψ=1.5, K=500, x=1000): 0.7361%

- K=0 welfare loss std across x=[500,1000,2000]: 0.00e+00 (should be ~0)


### Qualitative Findings (Section 4)

- **L > 0**: Learning about the latent MPR has positive welfare value ✓

- **K=500**: L increases with initial wealth x (liability amplifies incentive to learn)

- **K=0**: L is (nearly) independent of x ✓

- **Increasing psi**: Welfare loss increases with EIS ✓

- **Increasing gamma**: Welfare loss decreases with RRA ✓

- **CRRA vs EZ**: CRRA welfare loss is smaller than general EZ ✓


### Monotonicity Check Summary

| vary   |   gamma |   psi |   K |          L |
|:-------|--------:|------:|----:|-----------:|
| gamma  |       2 |  1.5  | 500 | 0.0173982  |
| gamma  |       3 |  1.5  | 500 | 0.0121369  |
| gamma  |       5 |  1.5  | 500 | 0.00736052 |
| gamma  |       7 |  1.5  | 500 | 0.00513504 |
| gamma  |      10 |  1.5  | 500 | 0.00338835 |
| psi    |       5 |  1.1  | 500 | 0.00718041 |
| psi    |       5 |  1.25 | 500 | 0.00726728 |
| psi    |       5 |  1.5  | 500 | 0.00736052 |
| psi    |       5 |  2    | 500 | 0.00744265 |
| psi    |       5 |  2.5  | 500 | 0.00746877 |


### CRRA vs EZ Comparison

|   gamma |   K |   psi_ez |   psi_crra |       L_EZ |     L_CRRA | EZ_larger   |
|--------:|----:|---------:|-----------:|-----------:|-----------:|:------------|
|       2 | 500 |      1.5 |        0.5 | 0.0173982  | 0.0153455  | True        |
|       2 |   0 |      1.5 |        0.5 | 0.0359377  | 0.0319118  | True        |
|       5 | 500 |      1.5 |        0.2 | 0.00736052 | 0.00554177 | True        |
|       5 |   0 |      1.5 |        0.2 | 0.016251   | 0.0126839  | True        |
|      10 | 500 |      1.5 |        0.1 | 0.00338835 | 0.00219819 | True        |
|      10 |   0 |      1.5 |        0.1 | 0.00846042 | 0.00612618 | True        |


### sigma_r Robustness

|   K |   L_sigma_neg |   L_sigma_pos |
|----:|--------------:|--------------:|
| 500 |    0.00736052 |    0.00737551 |
|   0 |    0.016251   |    0.0162432  |


## Reproducibility Notes

- Initial conditions: r0=0.02 (=mu_r), m0=0 (=mu_R), v0=0 (dogmatic prior)

- Time grid: 252 steps over [0, T=1] (daily)

- sigma_r = -0.03 as reported; robustness with +0.03 also run

- Figures: figure1_left.png, figure1_right.png, K_comparison.png

## Experiment 2: Structural Verification

**Settings:** 5000 paths, 50 steps, degree=2

### Structural Check Summary

|   gamma |   psi |   K |    x |   tilde_x |         A |       B |   Vhat_0_err |   terminal_err |   decoupling_residual_mean |   decoupling_residual_max |   X_negative_frac |   c_positive_frac |   c_mean |   pi_mean_norm |   bsde_resid_mean |   bsde_resid_std |
|--------:|------:|----:|-----:|----------:|----------:|--------:|-------------:|---------------:|---------------------------:|--------------------------:|------------------:|------------------:|---------:|---------------:|------------------:|-----------------:|
|       5 |   1.5 | 500 | 1000 |   433.907 | -0.169424 | 492.579 |            0 |              0 |                7.00014e-14 |               2.27374e-13 |                 0 |                 1 |  10.7705 |        364.014 |           52.2925 |          61.6953 |
|       5 |   1.5 |   0 | 1000 |   855.122 | -0.169424 |   0     |            0 |              0 |                8.60654e-14 |               4.54747e-13 |                 0 |                 1 |  21.2259 |        469.19  |           82.9452 |          85.5232 |

### Interpretation

- **terminal_err ≈ 0**: BSDE terminal condition Y_T = -K*exp(-int_r_T) satisfied exactly ✓
- **decoupling_residual ≈ 0**: X_t + exp(int_r_t)*Y_t = tilde_x*phi_t ✓
- **c_positive_frac = 1.0**: Optimal consumption is always positive ✓
- **BSDE residual**: Reflects discretization error in LSM approximation
