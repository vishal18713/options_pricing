"""Interactive Streamlit Demo UI for Options Pricing & Greeks Engine.

Thin presentation layer delegating all calculations to the core engine.
No duplicate formulas in UI code.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from options_engine.binomial import crr_price
from options_engine.black_scholes import bsm_price
from options_engine.greeks import compare_greeks
from options_engine.implied_volatility import implied_volatility
from options_engine.monte_carlo import monte_carlo_price
from options_engine.volatility_surface import (
    check_butterfly_arbitrage_slice,
    fit_svi_slice,
    raw_svi_total_variance,
)

st.set_page_config(
    page_title="Options Pricing & Greeks Engine",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Quantitative Options Pricing & Greeks Engine")
st.markdown(
    "A modular quantitative options engine implementing **BSM**, **CRR Binomial Trees**, "
    "**Monte Carlo with Variance Reduction**, **Analytical & Numerical Greeks**, "
    "and **SVI Volatility Surface Calibration**."
)

st.sidebar.header("Option Contract Parameters")
spot = st.sidebar.number_input("Spot Price ($)", min_value=1.0, value=100.0, step=1.0)
strike = st.sidebar.number_input("Strike Price ($)", min_value=1.0, value=100.0, step=1.0)
time_to_expiry = st.sidebar.slider(
    "Time to Expiry (Years)", min_value=0.01, max_value=5.0, value=1.0, step=0.05
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

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Pricer Comparison",
        "Greeks Analysis",
        "Implied Volatility Solver",
        "SVI Volatility Smile",
    ]
)

with tab1:
    st.subheader("Multi-Engine Pricing Comparison")
    col1, col2, col3 = st.columns(3)

    # BSM
    bsm_val = bsm_price(spot, strike, time_to_expiry, rate, volatility, opt_type, dividend_yield)
    with col1:
        st.metric(label="Black-Scholes-Merton (European)", value=f"${bsm_val:.4f}")

    # CRR American
    crr_res = crr_price(
        spot,
        strike,
        time_to_expiry,
        rate,
        volatility,
        steps=400,
        option_type=opt_type,
        exercise_style="american",
        dividend_yield=dividend_yield,
    )
    with col2:
        st.metric(
            label="CRR Binomial (American, 400 steps)",
            value=f"${crr_res.price:.4f}",
            delta=f"${crr_res.price - bsm_val:+.4f} early ex.",
        )

    # Monte Carlo
    mc_res = monte_carlo_price(
        spot,
        strike,
        time_to_expiry,
        rate,
        volatility,
        n_paths=100_000,
        seed=42,
        option_type=opt_type,
        dividend_yield=dividend_yield,
        antithetic=True,
    )
    with col3:
        st.metric(
            label="Monte Carlo (100k paths, Antithetic)",
            value=f"${mc_res.price:.4f}",
            delta=f"±${mc_res.standard_error:.4f} SE",
        )

    st.caption(
        f"Monte Carlo 95% CI: [{mc_res.confidence_interval[0]:.4f}, {mc_res.confidence_interval[1]:.4f}]"
    )

with tab2:
    st.subheader("Cross-Method Greeks Comparison")
    df_greeks = compare_greeks(
        spot, strike, time_to_expiry, rate, volatility, opt_type, dividend_yield
    )
    st.dataframe(df_greeks.style.format(precision=4), width="stretch")

with tab3:
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
                f"Converged Implied Volatility: **{iv_res.implied_volatility * 100:.2f}%** (Iterations: {iv_res.iterations}, Residual: {iv_res.residual:.2e})"
            )
        else:
            st.error(f"Solver Failed: **{iv_res.reason_code}**")

with tab4:
    st.subheader("SVI Parametric Smile Fitting & Arbitrage Check")
    k_eval = np.linspace(-0.25, 0.25, 25)
    w_demo = 0.04 + 0.12 * (-0.25 * (k_eval - 0.01) + np.sqrt((k_eval - 0.01) ** 2 + 0.08**2))
    fit_p, obj, success, diag = fit_svi_slice(k_eval, w_demo)
    is_free, arb_diag = check_butterfly_arbitrage_slice(fit_p, k_eval)

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Calibrated Parameters:**", fit_p.to_dict())
    with col_b:
        if is_free:
            st.success("✅ Arbitrage Diagnostic: Slice is Butterfly-Arbitrage Free!")
        else:
            st.warning("⚠️ Warning: Butterfly arbitrage detected on slice.")

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
    w_fit = raw_svi_total_variance(k_eval, fit_p.a, fit_p.b, fit_p.rho, fit_p.m, fit_p.eta)
    ax.plot(
        k_eval,
        np.sqrt(w_fit / time_to_expiry) * 100,
        label="Calibrated SVI Smile",
        color="#1f77b4",
        lw=2,
    )
    ax.set_xlabel("Log-Moneyness k = ln(K / F)")
    ax.set_ylabel("Implied Volatility (%)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    st.pyplot(fig)
