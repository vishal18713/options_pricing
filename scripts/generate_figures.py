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

    # -------------------------------------------------------------
    # Figure 5: Interactive Binomial Tree (CRR with Early Exercise)
    # -------------------------------------------------------------
    from options_engine.binomial import build_crr_tree_visualization

    tree_data = build_crr_tree_visualization(
        spot=100.0,
        strike=105.0,
        time_to_expiry=0.5,
        rate=0.05,
        volatility=0.25,
        steps=5,
        option_type="put",
        exercise_style="american",
    )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    # Draw branches
    for edge in tree_data["edges"]:
        ax.plot(
            [edge["from_step"], edge["to_step"]],
            [
                edge["from_idx"] - edge["from_step"] / 2.0,
                edge["to_idx"] - edge["to_step"] / 2.0,
            ],
            color="#8c96a6",
            lw=1.2,
            zorder=1,
        )

    # Draw nodes
    cont_x, cont_y, cont_txt = [], [], []
    ee_x, ee_y, ee_txt = [], [], []

    for node in tree_data["nodes"]:
        x = node["step"]
        y = node["node_idx"] - node["step"] / 2.0
        label = f"S:{node['stock_price']:.1f}\nV:{node['option_value']:.1f}"
        if node["early_exercised"]:
            ee_x.append(x)
            ee_y.append(y)
            ee_txt.append(label)
        else:
            cont_x.append(x)
            cont_y.append(y)
            cont_txt.append(label)

    if cont_x:
        ax.scatter(
            cont_x,
            cont_y,
            s=550,
            color="#1f77b4",
            edgecolors="#0d47a1",
            lw=1.5,
            zorder=3,
            label="Continuation / Expiry Node",
        )
    if ee_x:
        ax.scatter(
            ee_x,
            ee_y,
            s=650,
            color="#e53935",
            marker="D",
            edgecolors="#b71c1c",
            lw=1.8,
            zorder=4,
            label="American Early-Exercise Decision",
        )

    for node in tree_data["nodes"]:
        x = node["step"]
        y = node["node_idx"] - node["step"] / 2.0
        txt = f"${node['stock_price']:.1f}\n${node['option_value']:.1f}"
        ax.annotate(
            txt,
            (x, y),
            fontsize=7,
            ha="center",
            va="center",
            color="white",
            weight="bold",
            zorder=5,
        )

    ax.set_title(
        "CRR Binomial Lattice (American Put, K=105, T=0.5y, N=5 steps)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Time Step (i)")
    ax.set_ylabel("Normalized Tree Height (j - i/2)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "binomial_tree.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 6: Volatility Surface (3D Surface Map)
    # -------------------------------------------------------------
    t_vals = np.linspace(0.1, 2.0, 30)
    k_vals = np.linspace(70.0, 130.0, 35)
    t_mesh, k_mesh = np.meshgrid(t_vals, k_vals)
    moneyness = np.log(k_mesh / 100.0)

    # Parametric synthetic surface (skew and term structure)
    iv_mesh = 0.20 - 0.08 * moneyness + 0.05 * (moneyness**2) + 0.03 * np.sqrt(t_mesh)

    fig = plt.figure(figsize=(10, 6.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        k_mesh,
        t_mesh,
        iv_mesh * 100.0,
        cmap="viridis",
        edgecolor="none",
        alpha=0.92,
        antialiased=True,
    )
    ax.set_title("Implied Volatility Surface σ_IV(K, T)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Strike Price ($)", labelpad=8)
    ax.set_ylabel("Maturity T (Years)", labelpad=8)
    ax.set_zlabel("Implied Volatility (%)", labelpad=8)
    ax.view_init(elev=28, azim=-125)
    fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, label="Implied Volatility (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "volatility_surface.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 7: Monte Carlo Simulation Explorer
    # -------------------------------------------------------------
    from options_engine.monte_carlo import simulate_monte_carlo_paths

    mc_data = simulate_monte_carlo_paths(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        rate=0.05,
        volatility=0.20,
        n_display_paths=40,
        n_total_paths=15_000,
        n_time_steps=40,
        option_type="call",
        seed=42,
    )

    fig, (ax_path, ax_dist, ax_conv) = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

    # 1. Trajectories
    time_grid = mc_data["time_grid"]
    sample_trajectories = mc_data["sample_paths"][:40]
    for path in sample_trajectories:
        ax_path.plot(time_grid, path, lw=0.9, alpha=0.55)
    ax_path.axhline(100.0, color="black", linestyle="--", lw=1.2, label="Strike K=100")
    ax_path.set_title("Simulated GBM Price Paths", fontsize=11, fontweight="bold")
    ax_path.set_xlabel("Time t (Years)")
    ax_path.set_ylabel("Underlying Price ($)")
    ax_path.grid(True, linestyle=":", alpha=0.5)
    ax_path.legend(loc="upper left")

    # 2. Terminal Price Distribution
    term_prices = mc_data["terminal_prices"]
    ax_dist.hist(
        term_prices,
        bins=35,
        density=True,
        color="#3b82f6",
        alpha=0.65,
        edgecolor="#1d4ed8",
        label="S_T Density",
    )
    ax_dist.axvline(100.0, color="#dc2626", linestyle="--", lw=1.5, label="Strike K=100 (ITM/OTM)")
    ax_dist.set_title("Terminal Price S_T Distribution", fontsize=11, fontweight="bold")
    ax_dist.set_xlabel("Terminal Price S_T ($)")
    ax_dist.set_ylabel("Probability Density")
    ax_dist.grid(True, linestyle=":", alpha=0.5)
    ax_dist.legend(loc="upper right")

    # 3. Monte Carlo Running Convergence
    conv_counts = mc_data["sample_subsets"]
    running_est = mc_data["running_means"]
    bsm_bench = bsm_price(100.0, 100.0, 1.0, 0.05, 0.20, "call")
    ax_conv.plot(
        conv_counts,
        running_est,
        color="#059669",
        lw=1.8,
        label=f"MC Estimate (${mc_data['final_price']:.3f})",
    )
    ax_conv.axhline(
        bsm_bench,
        color="#dc2626",
        linestyle="--",
        lw=1.5,
        label=f"BSM Analytical (${bsm_bench:.3f})",
    )
    ax_conv.set_title("Monte Carlo Price Convergence", fontsize=11, fontweight="bold")
    ax_conv.set_xlabel("Number of Simulation Paths")
    ax_conv.set_ylabel("Option Price ($)")
    ax_conv.grid(True, linestyle=":", alpha=0.5)
    ax_conv.legend(loc="lower right")

    fig.suptitle(
        "Monte Carlo Risk-Neutral Simulation Explorer & Pricing Convergence",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "monte_carlo_simulation.png")
    plt.close(fig)

    print(f"Authoritative figures successfully generated in {out_dir}")


if __name__ == "__main__":
    generate_all_figures()
