"""Greeks calculation by independent numerical methods and discrete delta-hedging simulation.

Features:
- Configurable central finite-difference estimators for Spot, Volatility, Rate, and Time.
- Support for Finite Differences using BSM, CRR, or Monte Carlo (with Common Random Numbers).
- Pathwise derivative estimators for continuous payoffs (European vanilla Delta and Vega).
- Cross-method comparison harness (analytical vs. finite differences vs. pathwise).
- Discrete delta-hedging simulation under geometric Brownian motion.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from options_engine.black_scholes import bsm_greeks, bsm_price
from options_engine.types import GreeksResult, OptionType
from options_engine.validation import validate_market_inputs


def finite_difference_greeks(
    pricer_func: Callable[..., float],
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
    ds_rel: float = 1e-3,
    dvol_abs: float = 1e-4,
    dr_abs: float = 1e-4,
    dt_abs: float = 1e-4,
    extra_pricer_kwargs: dict[str, Any] | None = None,
) -> GreeksResult:
    """Compute central finite-difference Greeks for any pricing engine.

    Args:
        pricer_func: Pricing callable accepting (spot, strike, time_to_expiry, rate, volatility, option_type, dividend_yield, **kwargs)
        spot: Spot price S
        strike: Strike price K
        time_to_expiry: Time T in years
        rate: Risk-free rate r
        volatility: Volatility sigma
        option_type: "call" or "put"
        dividend_yield: Dividend yield q
        ds_rel: Relative spot bump fraction (h_s = spot * ds_rel)
        dvol_abs: Absolute volatility bump
        dr_abs: Absolute rate bump
        dt_abs: Absolute time bump
        extra_pricer_kwargs: Extra keyword args passed to pricer_func (e.g. steps or seed)

    Returns:
        GreeksResult containing numerical sensitivities.
    """
    opt_type = OptionType(option_type)
    kwargs = extra_pricer_kwargs or {}

    validate_market_inputs(spot, strike, time_to_expiry, volatility, rate, dividend_yield)

    hs = max(spot * ds_rel, 1e-4)
    hvol = max(dvol_abs, 1e-6)
    hr = max(dr_abs, 1e-6)
    ht = min(max(dt_abs, 1e-6), max(time_to_expiry / 2.0, 1e-6)) if time_to_expiry > 0 else 1e-6

    p_base = pricer_func(
        spot, strike, time_to_expiry, rate, volatility, opt_type, dividend_yield, **kwargs
    )

    # Spot bumps (Delta & Gamma)
    p_s_up = pricer_func(
        spot + hs, strike, time_to_expiry, rate, volatility, opt_type, dividend_yield, **kwargs
    )
    p_s_dn = pricer_func(
        spot - hs, strike, time_to_expiry, rate, volatility, opt_type, dividend_yield, **kwargs
    )
    delta = (p_s_up - p_s_dn) / (2.0 * hs)
    gamma = (p_s_up - 2.0 * p_base + p_s_dn) / (hs**2)

    # Volatility bump (Vega)
    p_v_up = pricer_func(
        spot,
        strike,
        time_to_expiry,
        rate,
        max(volatility + hvol, 1e-6),
        opt_type,
        dividend_yield,
        **kwargs,
    )
    p_v_dn = pricer_func(
        spot,
        strike,
        time_to_expiry,
        rate,
        max(volatility - hvol, 1e-6),
        opt_type,
        dividend_yield,
        **kwargs,
    )
    vega = (p_v_up - p_v_dn) / (2.0 * hvol)

    # Rate bump (Rho)
    p_r_up = pricer_func(
        spot, strike, time_to_expiry, rate + hr, volatility, opt_type, dividend_yield, **kwargs
    )
    p_r_dn = pricer_func(
        spot, strike, time_to_expiry, rate - hr, volatility, opt_type, dividend_yield, **kwargs
    )
    rho = (p_r_up - p_r_dn) / (2.0 * hr)

    # Time passage bump (Theta)
    # Theta convention: passage of calendar time -> -(p(T+ht) - p(T-ht))/(2*ht)
    if time_to_expiry > ht:
        p_t_up = pricer_func(
            spot, strike, time_to_expiry + ht, rate, volatility, opt_type, dividend_yield, **kwargs
        )
        p_t_dn = pricer_func(
            spot, strike, time_to_expiry - ht, rate, volatility, opt_type, dividend_yield, **kwargs
        )
        theta = -(p_t_up - p_t_dn) / (2.0 * ht)
    else:
        # One-sided forward difference near expiry
        p_t_up = pricer_func(
            spot, strike, time_to_expiry + ht, rate, volatility, opt_type, dividend_yield, **kwargs
        )
        theta = -(p_t_up - p_base) / ht

    return GreeksResult(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
        method="finite_difference",
        diagnostics={"hs": hs, "hvol": hvol, "hr": hr, "ht": ht},
    )


def pathwise_mc_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    n_paths: int = 150_000,
    seed: int | None = 42,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
) -> GreeksResult:
    """Compute Greeks using Pathwise Sensitivity (Infinitesimal Perturbation Analysis).

    Pathwise derivatives are mathematically valid for continuous European payoffs:
    - Delta = e^(-rT) * (S_T / S) * 1_{payoff > 0}
    - Vega  = e^(-rT) * S_T * [ -sigma*T + sqrt(T)*Z ] * 1_{payoff > 0}

    Note: Gamma involves discontinuous Dirac delta payoff derivative, so pathwise estimators
    require Likelihood Ratio Method (LRM) or Malliavin calculus. Here Gamma, Theta, and Rho
    are annotated in diagnostics or fallback.
    """
    s = float(spot)
    k = float(strike)
    t = float(time_to_expiry)
    r = float(rate)
    v = float(volatility)
    q = float(dividend_yield)
    opt_type = OptionType(option_type)

    validate_market_inputs(s, k, t, v, r, q)

    if t <= 0 or v <= 0:
        return bsm_greeks(s, k, t, r, v, opt_type, q)

    rng = np.random.default_rng(seed)
    # Antithetic pairs for variance reduction
    n_pairs = n_paths // 2
    z = rng.standard_normal(n_pairs)
    z_all = np.concatenate([z, -z])

    drift = (r - q - 0.5 * v**2) * t
    diff = v * np.sqrt(t)
    st = s * np.exp(drift + diff * z_all)
    discount = np.exp(-r * t)

    # Indicator of terminal ITM condition
    if opt_type == OptionType.CALL:
        in_the_money = st > k
        # d(S_T)/d(S) = S_T / S
        # For call: d(payoff)/d(S_T) = 1
        delta_paths = discount * (st / s) * in_the_money
        # d(S_T)/d(sigma) = S_T * (-sigma*T + sqrt(T)*Z)
        vega_paths = discount * st * (-v * t + np.sqrt(t) * z_all) * in_the_money
    else:
        in_the_money = st < k
        # For put: d(payoff)/d(S_T) = -1
        delta_paths = -discount * (st / s) * in_the_money
        vega_paths = -discount * st * (-v * t + np.sqrt(t) * z_all) * in_the_money

    delta = float(np.mean(delta_paths))
    vega = float(np.mean(vega_paths))

    # Analytical values for sensitivities not suitable for simple pathwise continuous derivative
    ref_greeks = bsm_greeks(s, k, t, r, v, opt_type, q)

    return GreeksResult(
        delta=delta,
        gamma=ref_greeks.gamma,
        vega=vega,
        theta=ref_greeks.theta,
        rho=ref_greeks.rho,
        method="pathwise_mc",
        diagnostics={
            "delta_standard_error": float(np.std(delta_paths, ddof=1) / np.sqrt(len(z_all))),
            "vega_standard_error": float(np.std(vega_paths, ddof=1) / np.sqrt(len(z_all))),
            "discontinuity_note": "Gamma not pathwise continuous; referenced to analytical.",
        },
    )


def compare_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Compare Greeks computed by Analytical, Central Finite Difference, and Pathwise MC."""
    g_ana = bsm_greeks(spot, strike, time_to_expiry, rate, volatility, option_type, dividend_yield)

    def bsm_pricer_wrapper(s, k, t, r, v, ot, dy):
        return bsm_price(s, k, t, r, v, ot, dy)

    g_fd = finite_difference_greeks(
        bsm_pricer_wrapper,
        spot,
        strike,
        time_to_expiry,
        rate,
        volatility,
        option_type,
        dividend_yield,
    )
    g_pw = pathwise_mc_greeks(
        spot,
        strike,
        time_to_expiry,
        rate,
        volatility,
        n_paths=100_000,
        seed=42,
        option_type=option_type,
        dividend_yield=dividend_yield,
    )

    data = [
        {
            "Greek": "Delta",
            "Analytical": g_ana.delta,
            "Finite_Diff": g_fd.delta,
            "Pathwise_MC": g_pw.delta,
            "FD_Diff": abs(g_fd.delta - g_ana.delta),
            "PW_Diff": abs(g_pw.delta - g_ana.delta),
        },
        {
            "Greek": "Gamma",
            "Analytical": g_ana.gamma,
            "Finite_Diff": g_fd.gamma,
            "Pathwise_MC": g_pw.gamma,
            "FD_Diff": abs(g_fd.gamma - g_ana.gamma),
            "PW_Diff": abs(g_pw.gamma - g_ana.gamma),
        },
        {
            "Greek": "Vega",
            "Analytical": g_ana.vega,
            "Finite_Diff": g_fd.vega,
            "Pathwise_MC": g_pw.vega,
            "FD_Diff": abs(g_fd.vega - g_ana.vega),
            "PW_Diff": abs(g_pw.vega - g_ana.vega),
        },
        {
            "Greek": "Theta",
            "Analytical": g_ana.theta,
            "Finite_Diff": g_fd.theta,
            "Pathwise_MC": g_pw.theta,
            "FD_Diff": abs(g_fd.theta - g_ana.theta),
            "PW_Diff": abs(g_pw.theta - g_ana.theta),
        },
        {
            "Greek": "Rho",
            "Analytical": g_ana.rho,
            "Finite_Diff": g_fd.rho,
            "Pathwise_MC": g_pw.rho,
            "FD_Diff": abs(g_fd.rho - g_ana.rho),
            "PW_Diff": abs(g_pw.rho - g_ana.rho),
        },
    ]
    return pd.DataFrame(data)


def simulate_delta_hedging(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    n_steps: int = 50,
    n_simulations: int = 500,
    seed: int = 42,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
) -> dict[str, Any]:
    """Simulate discrete delta hedging of a short option position.

    Under continuous rebalancing in Black-Scholes world, hedge tracking error is zero.
    Under discrete rebalancing, the terminal hedge P&L variance scales as O(1/N_steps).
    """
    rng = np.random.default_rng(seed)
    opt_type = OptionType(option_type)
    dt = time_to_expiry / n_steps

    init_opt_price = bsm_price(
        spot, strike, time_to_expiry, rate, volatility, opt_type, dividend_yield
    )

    terminal_pnls = []

    for _ in range(n_simulations):
        s_t = spot
        t_rem = time_to_expiry

        # Initial delta and cash position:
        # We short 1 option, receive init_opt_price.
        # Long delta shares of underlying stock: cost = delta * S.
        # Cash account = init_opt_price - delta * S.
        delta_0 = bsm_greeks(s_t, strike, t_rem, rate, volatility, opt_type, dividend_yield).delta
        shares = delta_0
        cash = init_opt_price - shares * s_t

        for _step in range(n_steps):
            t_rem -= dt
            # Step stock price forward under geometric brownian motion
            z = rng.standard_normal()
            drift = (rate - dividend_yield - 0.5 * volatility**2) * dt
            diff = volatility * np.sqrt(dt) * z
            s_t = s_t * np.exp(drift + diff)

            # Cash account grows at risk-free rate, plus continuous dividend on held shares
            cash = cash * np.exp(rate * dt) + shares * s_t * (np.exp(dividend_yield * dt) - 1.0)

            if t_rem > 1e-6:
                # Rebalance delta
                new_delta = bsm_greeks(
                    s_t, strike, t_rem, rate, volatility, opt_type, dividend_yield
                ).delta
                d_shares = new_delta - shares
                cash -= d_shares * s_t
                shares = new_delta

        # Terminal payoff to fulfill short option
        if opt_type == OptionType.CALL:
            payoff = max(s_t - strike, 0.0)
        else:
            payoff = max(strike - s_t, 0.0)

        # Liquidate portfolio: shares * S + cash - payoff
        pnl = shares * s_t + cash - payoff
        terminal_pnls.append(pnl)

    pnl_arr = np.array(terminal_pnls)
    return {
        "n_steps": n_steps,
        "n_simulations": n_simulations,
        "mean_pnl": float(np.mean(pnl_arr)),
        "std_pnl": float(np.std(pnl_arr, ddof=1)),
        "median_pnl": float(np.median(pnl_arr)),
        "pnl_distribution": pnl_arr,
    }
