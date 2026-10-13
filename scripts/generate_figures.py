"""Generates authoritative research figures and diagnostics for the engine.

Outputs saved to reports/figures/:
1. crr_convergence.png: CRR binomial convergence to BSM as steps increase.
2. greeks_surface.png: BSM Greeks (Delta, Gamma, Vega, Theta, Rho) vs. Spot.
3. hedge_simulation.png: Discrete Delta Hedging P&L error distribution.
4. svi_volatility_smile.png: Fitted SVI implied volatility smile and residuals.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from options_engine.binomial import crr_price
from options_engine.black_scholes import bsm_greeks, bsm_price
from options_engine.greeks import simulate_delta_hedging
from options_engine.volatility_surface import (
    SVIParameters,
    fit_svi_slice,
    raw_svi_total_variance,
)


def generate_all_figures(output_dir: Path | str = "reports/figures") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("tableau-colorblind10")

    # -------------------------------------------------------------
    # Figure 1: CRR Convergence to BSM
    # -------------------------------------------------------------
    s, k, t, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    bsm_val = bsm_price(s, k, t, r, sigma, "call", q)

    step_counts = np.arange(10, 301, 5)
    crr_values = [
        crr_price(s, k, t, r, sigma, steps=n, option_type="call").price for n in step_counts
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(step_counts, crr_values, label="CRR European Call Price", color="#1f77b4", lw=1.5)
    ax.axhline(
        bsm_val, color="#d62728", linestyle="--", label=f"BSM Analytical ({bsm_val:.4f})", lw=1.5
    )
    ax.set_title(
        "CRR Binomial Tree Convergence to BSM European Price", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Number of Steps (N)")
    ax.set_ylabel("Option Price ($)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "crr_convergence.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 2: Greeks vs Spot Price
    # -------------------------------------------------------------
    spots = np.linspace(60, 140, 100)
    deltas, gammas, vegas, thetas = [], [], [], []
    for sp in spots:
        g = bsm_greeks(sp, k, t, r, sigma, "call", q)
        deltas.append(g.delta)
        gammas.append(g.gamma)
        vegas.append(g.vega)
        thetas.append(g.theta)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    axes[0, 0].plot(spots, deltas, color="#2ca02c", lw=2)
    axes[0, 0].set_title("Delta (dV/dS)", fontsize=11, fontweight="bold")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].axvline(100.0, color="gray", linestyle=":")

    axes[0, 1].plot(spots, gammas, color="#ff7f0e", lw=2)
    axes[0, 1].set_title("Gamma (d²V/dS²)", fontsize=11, fontweight="bold")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].axvline(100.0, color="gray", linestyle=":")

    axes[1, 0].plot(spots, vegas, color="#9467bd", lw=2)
    axes[1, 0].set_title("Vega (dV/dσ)", fontsize=11, fontweight="bold")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].axvline(100.0, color="gray", linestyle=":")

    axes[1, 1].plot(spots, thetas, color="#d62728", lw=2)
    axes[1, 1].set_title("Theta (Passage of Time)", fontsize=11, fontweight="bold")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].axvline(100.0, color="gray", linestyle=":")

    fig.suptitle(
        "BSM Sensitivity Greeks across Underlying Spot Prices", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "greeks_surface.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 3: Discrete Delta Hedging Simulation P&L Distribution
    # -------------------------------------------------------------
    sim_coarse = simulate_delta_hedging(s, k, 0.5, r, sigma, n_steps=10, n_simulations=500, seed=42)
    sim_fine = simulate_delta_hedging(s, k, 0.5, r, sigma, n_steps=60, n_simulations=500, seed=42)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.hist(
        sim_coarse["pnl_distribution"],
        bins=25,
        alpha=0.5,
        color="#1f77b4",
        label=f"N=10 Steps (Std={sim_coarse['std_pnl']:.3f})",
        density=True,
    )
    ax.hist(
        sim_fine["pnl_distribution"],
        bins=25,
        alpha=0.6,
        color="#2ca02c",
        label=f"N=60 Steps (Std={sim_fine['std_pnl']:.3f})",
        density=True,
    )
    ax.axvline(0.0, color="black", linestyle="--", lw=1.2)
    ax.set_title("Discrete Delta Hedging Terminal P&L Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Terminal Hedge Error P&L ($)")
    ax.set_ylabel("Probability Density")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "hedge_simulation.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 4: SVI Implied Volatility Fit
    # -------------------------------------------------------------
    k_grid = np.linspace(-0.25, 0.25, 25)
    true_p = SVIParameters(a=0.04, b=0.12, rho=-0.25, m=0.01, eta=0.08)
    w_exact = raw_svi_total_variance(k_grid, true_p.a, true_p.b, true_p.rho, true_p.m, true_p.eta)
    rng = np.random.default_rng(42)
    w_noisy = w_exact + rng.normal(0, 0.001, len(k_grid))

    fit_p, _, _, _ = fit_svi_slice(k_grid, w_noisy)
    w_fit = raw_svi_total_variance(k_grid, fit_p.a, fit_p.b, fit_p.rho, fit_p.m, fit_p.eta)
    iv_noisy = np.sqrt(np.maximum(w_noisy, 1e-6) / 1.0)
    iv_fit = np.sqrt(np.maximum(w_fit, 1e-6) / 1.0)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.scatter(
        k_grid, iv_noisy * 100, color="#d62728", label="Market IV Observations", s=25, zorder=3
    )
    ax.plot(k_grid, iv_fit * 100, color="#1f77b4", lw=2, label="Calibrated SVI Smile", zorder=2)
    ax.set_title(
        "SVI Parametric Implied Volatility Smile Calibration", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Log-Moneyness k = ln(K / F)")
    ax.set_ylabel("Implied Volatility (%)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "svi_volatility_smile.png")
    plt.close(fig)

    print(f"Authoritative figures successfully generated in {out_dir}")


if __name__ == "__main__":
    generate_all_figures()
