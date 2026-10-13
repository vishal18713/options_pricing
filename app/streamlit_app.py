"""Interactive Streamlit Demo UI for Options Pricing & Greeks Engine.

Includes:
1. Interactive Binomial Tree (up/down branches, stock prices, option values, American early-exercise highlights).
2. Interactive 3D Volatility Surface (rotatable Plotly 3D surface across moneyness and maturities).
3. Monte Carlo Simulation Explorer (path trajectories, terminal distribution, payoffs, and running price convergence).
4. Multi-engine pricer and analytical Greeks comparisons.
5. Bracketed Brent implied volatility solver.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from options_engine.binomial import build_crr_tree_visualization, crr_price
from options_engine.black_scholes import bsm_price
from options_engine.greeks import compare_greeks
from options_engine.implied_volatility import implied_volatility
from options_engine.monte_carlo import monte_carlo_price, simulate_monte_carlo_paths
from options_engine.volatility_surface import (
    SVIParameters,
    check_butterfly_arbitrage_slice,
    check_calendar_arbitrage,
    raw_svi_total_variance,
)

st.set_page_config(
    page_title="Options Pricing & Greeks Engine",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Quantitative Options Pricing & Greeks Engine")
st.markdown(
    "A research-grade quantitative options laboratory featuring **interactive binomial trees**, "
    "**rotatable 3D volatility surfaces**, and a **Monte Carlo path explorer**."
)

# -------------------------------------------------------------
# Sidebar Contract Inputs
# -------------------------------------------------------------
st.sidebar.header("Option Contract Parameters")
spot = st.sidebar.number_input("Spot Price ($)", min_value=1.0, value=100.0, step=1.0)
strike = st.sidebar.number_input("Strike Price ($)", min_value=1.0, value=100.0, step=1.0)
time_to_expiry = st.sidebar.slider(
    "Time to Expiry (Years)", min_value=0.05, max_value=5.0, value=1.0, step=0.05
)
rate_pct = st.sidebar.slider(
    "Risk-Free Rate (%)", min_value=-2.0, max_value=20.0, value=5.0, step=0.25
)
dividend_pct = st.sidebar.slider(
    "Dividend Yield (%)", min_value=0.0, max_value=15.0, value=1.0, step=0.25
)
vol_pct = st.sidebar.slider("Volatility (%)", min_value=1.0, max_value=150.0, value=20.0, step=1.0)
opt_type = st.sidebar.selectbox("Option Type", ["call", "put"])

rate = rate_pct / 100.0
dividend_yield = dividend_pct / 100.0
volatility = vol_pct / 100.0

# -------------------------------------------------------------
# Top Metrics Bar
# -------------------------------------------------------------
bsm_val = bsm_price(spot, strike, time_to_expiry, rate, volatility, opt_type, dividend_yield)
crr_val = crr_price(
    spot,
    strike,
    time_to_expiry,
    rate,
    volatility,
    steps=300,
    option_type=opt_type,
    exercise_style="american",
    dividend_yield=dividend_yield,
).price
mc_val = monte_carlo_price(
    spot,
    strike,
    time_to_expiry,
    rate,
    volatility,
    n_paths=50_000,
    seed=42,
    option_type=opt_type,
    dividend_yield=dividend_yield,
    antithetic=True,
)

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("BSM European Price", f"${bsm_val:.4f}")
col_m2.metric(
    "CRR American Price (300 steps)",
    f"${crr_val:.4f}",
    delta=f"${crr_val - bsm_val:+.4f} early ex.",
)
col_m3.metric(
    "Monte Carlo (50k paths)", f"${mc_val.price:.4f}", delta=f"±${mc_val.standard_error:.4f} SE"
)

st.divider()

# -------------------------------------------------------------
# Main Tabs
# -------------------------------------------------------------
tab_tree, tab_surface, tab_mc, tab_greeks, tab_iv = st.tabs(
    [
        "🌲 1. Interactive Binomial Tree",
        "🌐 2. 3D Volatility Surface",
        "🎲 3. Monte Carlo Explorer",
        "📐 4. Greeks Comparison",
        "🎯 5. Implied Volatility Solver",
    ]
)

# =============================================================
# TAB 1: Interactive Binomial Tree
# =============================================================
with tab_tree:
    st.subheader("Interactive CRR Binomial Tree")
    st.markdown(
        "Explore up/down stock price branches, option continuation values, and **highlighted American early-exercise decisions** at each node."
    )

    col_t1, col_t2 = st.columns([1, 4])
    with col_t1:
        tree_steps = st.slider("Tree Depth (Steps)", min_value=2, max_value=10, value=5, step=1)
        tree_ex_style = st.selectbox("Exercise Style", ["american", "european"], index=0)

    tree_data = build_crr_tree_visualization(
        spot,
        strike,
        time_to_expiry,
        rate,
        volatility,
        steps=tree_steps,
        option_type=opt_type,
        exercise_style=tree_ex_style,
        dividend_yield=dividend_yield,
    )

    # Plotly interactive tree graph
    fig_tree = go.Figure()

    # Draw branch edges
    for edge in tree_data["edges"]:
        # Find coordinates
        from_step, from_idx = edge["from_step"], edge["from_idx"]
        to_step, to_idx = edge["to_step"], edge["to_idx"]

        # Y position centered: j - step/2
        y_from = from_idx - from_step / 2.0
        y_to = to_idx - to_step / 2.0

        fig_tree.add_trace(
            go.Scatter(
                x=[from_step, to_step],
                y=[y_from, y_to],
                mode="lines",
                line=dict(color="#64748b", width=1.5),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Draw nodes
    x_nodes, y_nodes, hover_texts, node_colors, node_symbols, node_sizes = [], [], [], [], [], []

    for node in tree_data["nodes"]:
        step = node["step"]
        idx = node["node_idx"]
        s_node = node["stock_price"]
        v_node = node["option_value"]
        intr_node = node["intrinsic_value"]
        exercised = node["early_exercised"]

        x_nodes.append(step)
        y_nodes.append(idx - step / 2.0)

        status_txt = "⚡ EARLY EXERCISED" if exercised else "Continuation"
        hover_txt = (
            f"<b>Step {step} (t={node['time']:.2f}y)</b><br>"
            f"Stock Price: ${s_node:.2f}<br>"
            f"Option Value: ${v_node:.2f}<br>"
            f"Intrinsic Value: ${intr_node:.2f}<br>"
            f"Decision: <b>{status_txt}</b>"
        )
        hover_texts.append(hover_txt)

        if exercised:
            node_colors.append("#ef4444")  # Red for early exercise
            node_symbols.append("diamond")
            node_sizes.append(18)
        else:
            node_colors.append("#3b82f6")  # Blue for continuation
            node_symbols.append("circle")
            node_sizes.append(14)

    fig_tree.add_trace(
        go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode="markers+text",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                symbol=node_symbols,
                line=dict(color="#ffffff", width=1.5),
            ),
            text=[
                f"${node['stock_price']:.1f}<br><b>${node['option_value']:.2f}</b>"
                for node in tree_data["nodes"]
            ],
            textposition="top center",
            hovertext=hover_texts,
            hoverinfo="text",
            name="Nodes",
            showlegend=False,
        )
    )

    fig_tree.update_layout(
        title=f"CRR Tree (N={tree_steps} Steps) | Up Factor u={tree_data['u']:.3f}, Down d={tree_data['d']:.3f}, Prob p={tree_data['p']:.3f}",
        xaxis=dict(title="Time Step (i)", tickmode="linear", tick0=0, dtick=1, showgrid=True),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=520,
        margin=dict(l=20, r=20, t=50, b=30),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#f8fafc"),
    )

    st.plotly_chart(fig_tree, width="stretch")
    st.info(
        "🔵 **Blue circles** represent regular continuation nodes. "
        "🔴 **Red diamonds** highlight nodes where American early exercise is optimal ($V_{i, j} = \\text{Intrinsic} > \\text{Continuation}$)."
    )

# =============================================================
# TAB 2: Interactive 3D Volatility Surface
# =============================================================
with tab_surface:
    st.subheader("Interactive 3D Implied Volatility Surface")
    st.markdown(
        "Plot log-moneyness $k = \\ln(K/F)$ or strike price against time-to-maturity $T$ with annualized implied volatility $\\sigma_{\\text{IV}}$ on the vertical axis. "
        "**Click and drag to rotate the 3D surface in 360 degrees.**"
    )

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        x_axis_mode = st.radio(
            "Moneyness Axis", ["Log-Moneyness ln(K/F)", "Strike Price ($)"], horizontal=True
        )
    with col_s2:
        surface_preset = st.selectbox(
            "Surface Regime",
            [
                "Equity Downside Skew (Typical SPX/Equities)",
                "FX Smile (Symmetric Kurtosis)",
                "Flat Black-Scholes Baseline",
            ],
        )

    # Generate multi-slice SVI parameter surface across maturities
    maturities = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    k_grid = np.linspace(-0.35, 0.35, 40)

    slices_dict = {}
    for mat in maturities:
        if surface_preset == "Equity Downside Skew (Typical SPX/Equities)":
            # Realistic term structure: ATM vol grows, skew flattens with sqrt(T)
            a_t = 0.04 * mat
            b_t = 0.12 * np.sqrt(mat)
            rho_t = -0.35
            m_t = 0.02 * np.sqrt(mat)
            eta_t = 0.08 * np.sqrt(mat)
        elif surface_preset == "FX Smile (Symmetric Kurtosis)":
            a_t = 0.035 * mat
            b_t = 0.15 * np.sqrt(mat)
            rho_t = 0.0
            m_t = 0.0
            eta_t = 0.07 * np.sqrt(mat)
        else:
            # Flat BSM surface
            a_t = (volatility**2) * mat
            b_t = 0.001
            rho_t = 0.0
            m_t = 0.0
            eta_t = 0.01

        slices_dict[mat] = SVIParameters(a=a_t, b=b_t, rho=rho_t, m=m_t, eta=eta_t)

    # Evaluate grid
    K_mesh, T_mesh = np.meshgrid(k_grid, maturities)
    IV_mesh = np.zeros_like(K_mesh)

    for r_idx, mat in enumerate(maturities):
        p_slice = slices_dict[mat]
        w_slice = raw_svi_total_variance(
            k_grid, p_slice.a, p_slice.b, p_slice.rho, p_slice.m, p_slice.eta
        )
        IV_mesh[r_idx, :] = np.sqrt(np.maximum(w_slice, 1e-6) / mat) * 100.0

    # Optional x-axis strike conversion: K = F * exp(k)
    fwd_1y = spot * np.exp((rate - dividend_yield) * 1.0)
    X_plot = (fwd_1y * np.exp(K_mesh)) if "Strike" in x_axis_mode else K_mesh
    x_title = "Strike Price ($)" if "Strike" in x_axis_mode else "Log-Moneyness k = ln(K / F)"

    fig_surf = go.Figure(
        data=[
            go.Surface(
                x=X_plot,
                y=T_mesh,
                z=IV_mesh,
                colorscale="Viridis",
                colorbar=dict(title="IV (%)"),
                contours_z=dict(
                    show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
                ),
            )
        ]
    )

    fig_surf.update_layout(
        title=f"3D Volatility Surface ({surface_preset})",
        scene=dict(
            xaxis=dict(title=x_title, backgroundcolor="#0f172a"),
            yaxis=dict(title="Maturity T (Years)", backgroundcolor="#0f172a"),
            zaxis=dict(title="Implied Volatility (%)", backgroundcolor="#0f172a"),
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2)),
        ),
        height=620,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0f172a",
        font=dict(color="#f8fafc"),
    )

    st.plotly_chart(fig_surf, width="stretch")

    # Static Arbitrage check on the surface
    is_cal_free, cal_diag = check_calendar_arbitrage(slices_dict, k_grid)
    is_bf_free, bf_diag = check_butterfly_arbitrage_slice(slices_dict[1.0], k_grid)

    col_ar1, col_ar2 = st.columns(2)
    with col_ar1:
        if is_cal_free:
            st.success(
                "✅ **Calendar Arbitrage Free**: Total variance $\\partial w / \\partial T \\ge 0$ across all maturity pairs."
            )
        else:
            st.error(
                f"❌ Calendar Arbitrage Detected: {len(cal_diag['violations'])} violations found."
            )
    with col_ar2:
        if is_bf_free:
            st.success(
                "✅ **Butterfly Arbitrage Free**: Implied probability density $g(k) \\ge 0$ everywhere."
            )
        else:
            st.error("❌ Butterfly Arbitrage Detected: Negative risk-neutral density region found.")

# =============================================================
# TAB 3: Monte Carlo Simulation Explorer
# =============================================================
with tab_mc:
    st.subheader("Monte Carlo Simulation Explorer")
    st.markdown(
        "Visualize multi-step GBM underlying price trajectories, examine terminal price and option payoff distributions, "
        "and inspect running sample mean convergence toward the theoretical BSM value."
    )

    col_mc1, col_mc2, col_mc3 = st.columns(3)
    with col_mc1:
        mc_n_paths = st.slider(
            "Total Simulation Paths", min_value=1_000, max_value=50_000, value=15_000, step=1_000
        )
    with col_mc2:
        mc_disp_paths = st.slider(
            "Display Trajectories", min_value=10, max_value=150, value=40, step=10
        )
    with col_mc3:
        mc_steps = st.slider(
            "Time Discretization Steps", min_value=10, max_value=100, value=40, step=10
        )

    mc_data = simulate_monte_carlo_paths(
        spot,
        strike,
        time_to_expiry,
        rate,
        volatility,
        n_display_paths=mc_disp_paths,
        n_total_paths=mc_n_paths,
        n_time_steps=mc_steps,
        seed=42,
        option_type=opt_type,
        dividend_yield=dividend_yield,
    )

    # Row 1: Stock Price Paths
    fig_paths = go.Figure()
    time_grid = mc_data["time_grid"]
    for p_idx in range(mc_disp_paths):
        fig_paths.add_trace(
            go.Scatter(
                x=time_grid,
                y=mc_data["sample_paths"][p_idx],
                mode="lines",
                line=dict(width=1.0),
                opacity=0.6,
                showlegend=False,
                hoverinfo="none",
            )
        )

    # Strike line
    fig_paths.add_hline(
        y=strike,
        line=dict(color="#f59e0b", width=2, dash="dash"),
        annotation_text=f"Strike ${strike:.1f}",
        annotation_position="bottom right",
    )

    fig_paths.update_layout(
        title=f"Sample Asset Price Paths S_t ({mc_disp_paths} trajectories)",
        xaxis=dict(title="Time (Years)", showgrid=True),
        yaxis=dict(title="Stock Price ($)", showgrid=True),
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#f8fafc"),
    )
    st.plotly_chart(fig_paths, width="stretch")

    # Row 2: Two columns: Terminal Distribution + Payoff, and Convergence
    col_dist, col_conv = st.columns(2)

    with col_dist:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=mc_data["terminal_prices"],
                nbinsx=40,
                marker=dict(color="#3b82f6"),
                name="Terminal Stock S_T",
                opacity=0.75,
            )
        )
        fig_hist.add_vline(
            x=strike, line=dict(color="#f59e0b", width=2, dash="dash"), annotation_text="Strike"
        )
        fig_hist.update_layout(
            title=f"Terminal Stock Price Distribution (N={mc_n_paths:,})",
            xaxis=dict(title="Terminal Price S_T ($)"),
            yaxis=dict(title="Frequency"),
            height=340,
            margin=dict(l=10, r=10, t=40, b=20),
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font=dict(color="#f8fafc"),
        )
        st.plotly_chart(fig_hist, width="stretch")

    with col_conv:
        fig_conv = go.Figure()
        fig_conv.add_trace(
            go.Scatter(
                x=mc_data["sample_subsets"],
                y=mc_data["running_means"],
                mode="lines",
                name="MC Running Estimate",
                line=dict(color="#10b981", width=2),
            )
        )
        fig_conv.add_hline(
            y=bsm_val,
            line=dict(color="#ef4444", width=2, dash="dash"),
            annotation_text=f"BSM Analytical (${bsm_val:.4f})",
        )
        fig_conv.update_layout(
            title="Monte Carlo Price Convergence vs Path Count",
            xaxis=dict(title="Number of Simulated Paths"),
            yaxis=dict(title="Estimated Option Price ($)"),
            height=340,
            margin=dict(l=10, r=10, t=40, b=20),
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font=dict(color="#f8fafc"),
        )
        st.plotly_chart(fig_conv, width="stretch")

    st.caption(
        f"Final Monte Carlo Price: **${mc_data['final_price']:.4f}** (SE: ±${mc_data['final_se']:.4f}) vs Analytical BSM: **${bsm_val:.4f}**."
    )

# =============================================================
# TAB 4: Greeks Comparison
# =============================================================
with tab_greeks:
    st.subheader("Cross-Method Greeks Comparison")
    st.markdown(
        "Sensitivities evaluated across Analytical, Central Finite Difference, and Pathwise MC estimators:"
    )
    df_greeks = compare_greeks(
        spot, strike, time_to_expiry, rate, volatility, opt_type, dividend_yield
    )
    st.dataframe(df_greeks.style.format(precision=4), width="stretch")

# =============================================================
# TAB 5: Implied Volatility Solver
# =============================================================
with tab_iv:
    st.subheader("Bracketed Brent Implied Volatility Solver")
    mkt_price = st.number_input(
        "Observed Option Price ($)", min_value=0.01, value=float(round(bsm_val, 2)), step=0.1
    )
    if st.button("Solve Implied Volatility"):
        iv_res = implied_volatility(
            mkt_price, spot, strike, time_to_expiry, rate, opt_type, dividend_yield
        )
        if iv_res.status == "CONVERGED":
            st.success(
                f"Converged Implied Volatility: **{iv_res.implied_volatility * 100:.2f}%** "
                f"(Iterations: {iv_res.iterations}, Residual: {iv_res.residual:.2e})"
            )
        else:
            st.error(f"Solver Failed: **{iv_res.reason_code}**")
