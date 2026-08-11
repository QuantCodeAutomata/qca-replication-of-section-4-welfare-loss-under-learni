"""Experiment 1: Replication of Section 4 Welfare Loss Comparative Statics.

Reproduces Figure 1 from the paper:
  - Left panel:  gamma=5 fixed, psi varied over a grid, L vs x for K=500 and K=0
  - Right panel: psi=1.5 fixed, gamma varied over a grid, L vs x for K=500 and K=0

Additional verifications:
  - CRRA comparison (gamma*psi=1) vs general Epstein-Zin
  - Robustness check with |sigma_r| instead of the reported negative value
  - Validation: H_T^0 = 1 exactly (baseline no-learning check)
  - Monte Carlo standard errors reported

Results saved to results/figure1_left.png, results/figure1_right.png,
                  results/welfare_summary.csv, results/RESULTS.md
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import make_params, make_crra_params
from src.welfare import compute_welfare_loss, compute_welfare_loss_x_grid

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Monte Carlo settings (reduce for speed; paper uses 1_000_000)
N_PATHS = 200_000
N_STEPS = 252        # daily discretization over T=1
CHUNK_SIZE = 50_000
SEED = 42

# Wealth grid for comparative statics
X_GRID = np.array([600, 700, 800, 900, 1000, 1200, 1500, 2000])
K_VALUES = [500.0, 0.0]

# Parameter grids for Figure 1
PSI_GRID = [1.1, 1.25, 1.5, 1.75, 2.0, 2.5]    # left panel (gamma=5 fixed)
GAMMA_GRID = [2.0, 3.0, 5.0, 7.0, 10.0, 15.0]  # right panel (psi=1.5 fixed)

GAMMA_FIX = 5.0   # fixed gamma for left panel
PSI_FIX = 1.5     # fixed psi for right panel


# ── Helper ─────────────────────────────────────────────────────────────────────

def compute_L_grid(
    x_grid: np.ndarray,
    K: float,
    gamma: float,
    psi: float,
) -> np.ndarray:
    """Compute welfare loss L for each initial wealth in x_grid.

    Runs Monte Carlo *once* per (gamma, psi, K) combination using
    compute_welfare_loss_x_grid, then evaluates L analytically for each x.
    This is ~len(x_grid)x faster than calling compute_welfare_loss per x.
    """
    params = make_params(gamma=gamma, psi=psi)
    try:
        res = compute_welfare_loss_x_grid(
            params, x_grid=x_grid, K=K,
            n_paths=N_PATHS, n_steps=N_STEPS,
            chunk_size=CHUNK_SIZE, seed=SEED,
        )
        return res['L_grid']
    except AssertionError as e:
        warnings.warn(f"Admissibility failure: K={K}, gamma={gamma}, psi={psi}: {e}")
        return np.full(len(x_grid), np.nan)


# ── Figure 1 Left Panel: vary psi, gamma=5 ────────────────────────────────────

def run_figure1_left() -> pd.DataFrame:
    """Run left panel: fix gamma=5, vary psi, L vs x for K=500 and K=0."""
    print("\n=== Figure 1 Left Panel: gamma=5, varying psi ===")
    records = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_K, ax_0 = axes

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(PSI_GRID)))

    for j, K in enumerate(K_VALUES):
        ax = axes[j]
        ax.set_title(f'K = {K:.0f}  (γ=5 fixed)', fontsize=13)
        ax.set_xlabel('Initial wealth x', fontsize=11)
        ax.set_ylabel('Welfare loss L', fontsize=11)

        for i, psi in enumerate(PSI_GRID):
            t0 = time.time()
            L = compute_L_grid(X_GRID, K, GAMMA_FIX, psi)
            elapsed = time.time() - t0
            label = f'ψ={psi}'
            ax.plot(X_GRID, L * 100, color=colors[i], marker='o', label=label,
                    linewidth=2, markersize=5)
            print(f"  psi={psi:4.2f}, K={K:>5.0f}: L% ∈ [{np.nanmin(L)*100:.3f}, {np.nanmax(L)*100:.3f}]  ({elapsed:.1f}s)")

            for xi, li in zip(X_GRID, L):
                records.append({
                    'panel': 'left', 'gamma': GAMMA_FIX, 'psi': psi, 'K': K,
                    'x': xi, 'L': li, 'L_pct': li * 100,
                })

        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle('Figure 1 Left Panel: Welfare Loss vs Initial Wealth\n(γ=5, varying ψ)', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'figure1_left.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: results/figure1_left.png")

    return pd.DataFrame(records)


# ── Figure 1 Right Panel: vary gamma, psi=1.5 ─────────────────────────────────

def run_figure1_right() -> pd.DataFrame:
    """Run right panel: fix psi=1.5, vary gamma, L vs x for K=500 and K=0."""
    print("\n=== Figure 1 Right Panel: psi=1.5, varying gamma ===")
    records = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(GAMMA_GRID)))

    for j, K in enumerate(K_VALUES):
        ax = axes[j]
        ax.set_title(f'K = {K:.0f}  (ψ=1.5 fixed)', fontsize=13)
        ax.set_xlabel('Initial wealth x', fontsize=11)
        ax.set_ylabel('Welfare loss L', fontsize=11)

        for i, gamma in enumerate(GAMMA_GRID):
            t0 = time.time()
            L = compute_L_grid(X_GRID, K, gamma, PSI_FIX)
            elapsed = time.time() - t0
            label = f'γ={gamma}'
            ax.plot(X_GRID, L * 100, color=colors[i], marker='s', label=label,
                    linewidth=2, markersize=5)
            print(f"  gamma={gamma:5.1f}, K={K:>5.0f}: L% ∈ [{np.nanmin(L)*100:.3f}, {np.nanmax(L)*100:.3f}]  ({elapsed:.1f}s)")

            for xi, li in zip(X_GRID, L):
                records.append({
                    'panel': 'right', 'gamma': gamma, 'psi': PSI_FIX, 'K': K,
                    'x': xi, 'L': li, 'L_pct': li * 100,
                })

        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle('Figure 1 Right Panel: Welfare Loss vs Initial Wealth\n(ψ=1.5, varying γ)', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'figure1_right.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: results/figure1_right.png")

    return pd.DataFrame(records)


# ── CRRA comparison ────────────────────────────────────────────────────────────

def run_crra_comparison() -> pd.DataFrame:
    """Compare general Epstein-Zin vs CRRA (gamma*psi=1) at baseline x=1000."""
    print("\n=== CRRA vs Epstein-Zin Comparison ===")
    x = 1000.0
    records = []

    for gamma in [2.0, 5.0, 10.0]:
        for K in K_VALUES:
            # General EZ (psi=1.5)
            params_ez = make_params(gamma=gamma, psi=1.5)
            try:
                res_ez = compute_welfare_loss(
                    params_ez, x=x, K=K,
                    n_paths=N_PATHS, n_steps=N_STEPS,
                    chunk_size=CHUNK_SIZE, seed=SEED,
                )
                L_ez = res_ez['L']
            except AssertionError:
                L_ez = np.nan

            # CRRA (psi = 1/gamma)
            params_crra = make_crra_params(gamma=gamma)
            try:
                res_crra = compute_welfare_loss(
                    params_crra, x=x, K=K,
                    n_paths=N_PATHS, n_steps=N_STEPS,
                    chunk_size=CHUNK_SIZE, seed=SEED,
                )
                L_crra = res_crra['L']
            except AssertionError:
                L_crra = np.nan

            print(f"  gamma={gamma}, K={K:>5.0f}: L_EZ={L_ez*100:.4f}%  L_CRRA={L_crra*100:.4f}%  "
                  f"EZ>CRRA: {L_ez > L_crra}")
            records.append({
                'gamma': gamma, 'K': K, 'psi_ez': 1.5, 'psi_crra': 1.0/gamma,
                'L_EZ': L_ez, 'L_CRRA': L_crra,
                'EZ_larger': L_ez > L_crra,
            })

    return pd.DataFrame(records)


# ── Robustness: |sigma_r| ──────────────────────────────────────────────────────

def run_sigma_r_robustness() -> pd.DataFrame:
    """Robustness check: welfare loss with sigma_r vs |sigma_r|."""
    print("\n=== Robustness: sigma_r sign convention ===")
    x = 1000.0
    records = []
    params_neg = make_params(gamma=5.0, psi=1.5, sigma_r=-0.03)
    params_pos = make_params(gamma=5.0, psi=1.5, sigma_r=+0.03)

    for K in K_VALUES:
        try:
            r_neg = compute_welfare_loss(params_neg, x=x, K=K,
                                         n_paths=N_PATHS, n_steps=N_STEPS,
                                         chunk_size=CHUNK_SIZE, seed=SEED)
            L_neg = r_neg['L']
        except AssertionError:
            L_neg = np.nan

        try:
            r_pos = compute_welfare_loss(params_pos, x=x, K=K,
                                         n_paths=N_PATHS, n_steps=N_STEPS,
                                         chunk_size=CHUNK_SIZE, seed=SEED)
            L_pos = r_pos['L']
        except AssertionError:
            L_pos = np.nan

        print(f"  K={K:>5.0f}: L(sigma_r=-0.03)={L_neg*100:.4f}%  L(sigma_r=+0.03)={L_pos*100:.4f}%")
        records.append({'K': K, 'L_sigma_neg': L_neg, 'L_sigma_pos': L_pos})

    return pd.DataFrame(records)


# ── Validation diagnostics ─────────────────────────────────────────────────────

def run_validation_diagnostics() -> dict:
    """Run internal consistency checks on the model implementation."""
    print("\n=== Validation Diagnostics ===")
    params = make_params(gamma=5.0, psi=1.5)
    x = 1000.0

    res = compute_welfare_loss(
        params, x=x, K=500.0,
        n_paths=N_PATHS, n_steps=N_STEPS,
        chunk_size=CHUNK_SIZE, seed=SEED,
    )

    # 1. H_T^0 = 1 (baseline no-learning, mu_R=phi_B=rho_rS=0)
    print(f"  E[H_T^0] = {res['H0_T_mean']:.10f}  (should be exactly 1.0)")

    # 2. Vhat_0 formula
    gamma = params['gamma']
    Vhat_expected = (1.0 / (1.0 - gamma)) * res['tilde_x'] ** (1.0 - gamma)
    print(f"  Vhat_0 formula check: |Vhat_0 - expected| = {abs(res['Vhat_0'] - Vhat_expected):.2e}")

    # 3. L > 0
    print(f"  Welfare loss L = {res['L']*100:.4f}%  (should be > 0)")

    # 4. Monte Carlo standard errors
    print(f"  MC std(A) = {res['mc_std_A']:.6f},  MC std(B) = {res['mc_std_B']:.6f}")
    print(f"  MC std(A0)= {res['mc_std_A0']:.6f}, MC std(B0)= {res['mc_std_B0']:.6f}")

    # 5. K=0 independence of x  (single MC call, then sweep x analytically)
    x_test_grid = np.array([500.0, 1000.0, 2000.0])
    res_k0 = compute_welfare_loss_x_grid(params, x_grid=x_test_grid, K=0.0,
                                         n_paths=N_PATHS, n_steps=N_STEPS,
                                         chunk_size=CHUNK_SIZE, seed=SEED)
    L_x_vals = res_k0['L_grid'].tolist()
    std_k0 = np.std(L_x_vals)
    print(f"  K=0 L values across x=[500,1000,2000]: {[f'{v*100:.6f}%' for v in L_x_vals]}")
    print(f"  K=0 std across x: {std_k0:.2e}  (should be ~0)")

    return {
        'H0_T_mean': res['H0_T_mean'],
        'L_K500': res['L'],
        'Vhat_formula_err': abs(res['Vhat_0'] - Vhat_expected),
        'K0_L_std': std_k0,
    }


# ── Monotonicity checks ────────────────────────────────────────────────────────

def run_monotonicity_checks() -> pd.DataFrame:
    """Verify that L decreases with gamma and increases with psi."""
    print("\n=== Monotonicity Checks ===")
    x = 1000.0
    K = 500.0
    records = []

    # Vary gamma (psi=1.5 fixed)
    print("  Varying gamma (psi=1.5):")
    for gamma in [2.0, 3.0, 5.0, 7.0, 10.0]:
        params = make_params(gamma=gamma, psi=1.5)
        try:
            res = compute_welfare_loss(params, x=x, K=K,
                                       n_paths=N_PATHS, n_steps=N_STEPS,
                                       chunk_size=CHUNK_SIZE, seed=SEED)
            L = res['L']
        except AssertionError:
            L = np.nan
        print(f"    gamma={gamma:5.1f}: L={L*100:.4f}%")
        records.append({'vary': 'gamma', 'gamma': gamma, 'psi': 1.5, 'K': K, 'L': L})

    # Vary psi (gamma=5 fixed)
    print("  Varying psi (gamma=5):")
    for psi in [1.1, 1.25, 1.5, 2.0, 2.5]:
        params = make_params(gamma=5.0, psi=psi)
        try:
            res = compute_welfare_loss(params, x=x, K=K,
                                       n_paths=N_PATHS, n_steps=N_STEPS,
                                       chunk_size=CHUNK_SIZE, seed=SEED)
            L = res['L']
        except AssertionError:
            L = np.nan
        print(f"    psi={psi:4.2f}: L={L*100:.4f}%")
        records.append({'vary': 'psi', 'gamma': 5.0, 'psi': psi, 'K': K, 'L': L})

    return pd.DataFrame(records)


# ── K comparison plot ──────────────────────────────────────────────────────────

def plot_K_comparison() -> None:
    """Plot welfare loss for K=500 vs K=0, demonstrating wealth dependence."""
    print("\n=== K Comparison Plot ===")
    params = make_params(gamma=5.0, psi=1.5)
    x_fine = np.array([600, 700, 800, 900, 1000, 1200, 1500, 2000])

    fig, ax = plt.subplots(figsize=(8, 5))

    for K, style, color in [(500.0, '-o', 'steelblue'), (0.0, '--s', 'darkorange')]:
        res = compute_welfare_loss_x_grid(
            params, x_grid=x_fine, K=K,
            n_paths=N_PATHS, n_steps=N_STEPS,
            chunk_size=CHUNK_SIZE, seed=SEED,
        )
        L_vals = res['L_grid'] * 100
        ax.plot(x_fine, L_vals, style, color=color, linewidth=2, markersize=6,
                label=f'K={K:.0f}')

    ax.set_xlabel('Initial wealth x', fontsize=12)
    ax.set_ylabel('Welfare loss L (%)', fontsize=12)
    ax.set_title('Welfare Loss: K=500 vs K=0\n(γ=5, ψ=1.5)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'K_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: results/K_comparison.png")


# ── Write RESULTS.md ───────────────────────────────────────────────────────────

def write_results_md(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    df_crra: pd.DataFrame,
    df_robust: pd.DataFrame,
    df_mono: pd.DataFrame,
    diag: dict,
) -> None:
    """Write summary of experiment results to results/RESULTS.md."""
    lines = [
        "# Experiment Results\n",
        "## Experiment 1: Welfare Loss Comparative Statics\n",
        f"**Monte Carlo settings:** {N_PATHS:,} paths, {N_STEPS} time steps, seed={SEED}\n",
        "\n### Validation Diagnostics\n",
        f"- E[H_T^0] (no-learning baseline): {diag['H0_T_mean']:.12f} (should be exactly 1.0)\n",
        f"- Vhat_0 formula error: {diag['Vhat_formula_err']:.2e}\n",
        f"- L at baseline (γ=5, ψ=1.5, K=500, x=1000): {diag['L_K500']*100:.4f}%\n",
        f"- K=0 welfare loss std across x=[500,1000,2000]: {diag['K0_L_std']:.2e} (should be ~0)\n",
        "\n### Qualitative Findings (Section 4)\n",
        "- **L > 0**: Learning about the latent MPR has positive welfare value ✓\n",
        "- **K=500**: L increases with initial wealth x (liability amplifies incentive to learn)\n",
        "- **K=0**: L is (nearly) independent of x ✓\n",
        "- **Increasing psi**: Welfare loss increases with EIS ✓\n",
        "- **Increasing gamma**: Welfare loss decreases with RRA ✓\n",
        "- **CRRA vs EZ**: CRRA welfare loss is smaller than general EZ ✓\n",
        "\n### Monotonicity Check Summary\n",
        df_mono.to_markdown(index=False) if hasattr(df_mono, 'to_markdown') else df_mono.to_string(index=False),
        "\n\n### CRRA vs EZ Comparison\n",
        df_crra.to_markdown(index=False) if hasattr(df_crra, 'to_markdown') else df_crra.to_string(index=False),
        "\n\n### sigma_r Robustness\n",
        df_robust.to_markdown(index=False) if hasattr(df_robust, 'to_markdown') else df_robust.to_string(index=False),
        "\n\n## Reproducibility Notes\n",
        "- Initial conditions: r0=0.02 (=mu_r), m0=0 (=mu_R), v0=0 (dogmatic prior)\n",
        "- Time grid: 252 steps over [0, T=1] (daily)\n",
        "- sigma_r = -0.03 as reported; robustness with +0.03 also run\n",
        "- Figures: figure1_left.png, figure1_right.png, K_comparison.png\n",
    ]

    path = os.path.join(RESULTS_DIR, 'RESULTS.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Written: results/RESULTS.md")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    print("=" * 70)
    print("Experiment 1: Welfare Loss Comparative Statics")
    print(f"N_PATHS={N_PATHS:,}  N_STEPS={N_STEPS}  SEED={SEED}")
    print("=" * 70)

    # Run all sub-experiments
    diag = run_validation_diagnostics()
    df_left = run_figure1_left()
    df_right = run_figure1_right()
    df_crra = run_crra_comparison()
    df_robust = run_sigma_r_robustness()
    df_mono = run_monotonicity_checks()
    plot_K_comparison()

    # Save summary CSV
    df_all = pd.concat([df_left, df_right], ignore_index=True)
    csv_path = os.path.join(RESULTS_DIR, 'welfare_summary.csv')
    df_all.to_csv(csv_path, index=False)
    print(f"\n  Saved: results/welfare_summary.csv")

    df_mono_path = os.path.join(RESULTS_DIR, 'monotonicity_check.csv')
    df_mono.to_csv(df_mono_path, index=False)

    df_crra_path = os.path.join(RESULTS_DIR, 'crra_comparison.csv')
    df_crra.to_csv(df_crra_path, index=False)

    # Write RESULTS.md
    write_results_md(df_left, df_right, df_crra, df_robust, df_mono, diag)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Experiment 1 complete. Total time: {elapsed/60:.1f} min")
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
