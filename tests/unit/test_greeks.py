"""Unit tests for Phase 4: Greeks by independent methods and hedging experiment."""

import numpy as np

from options_engine.binomial import crr_price
from options_engine.black_scholes import bsm_greeks, bsm_price
from options_engine.greeks import (
    compare_greeks,
    finite_difference_greeks,
    pathwise_mc_greeks,
    simulate_delta_hedging,
)
from options_engine.types import OptionType


class TestGreeksAndHedging:
    def test_finite_difference_greeks_on_bsm(self):
        s = 100.0
        k = 100.0
        t = 1.0
        r = 0.05
        sigma = 0.20
        q = 0.01

        g_ana = bsm_greeks(s, k, t, r, sigma, OptionType.CALL, q)

        def pricer(s_, k_, t_, r_, v_, ot_, dy_):
            return bsm_price(s_, k_, t_, r_, v_, ot_, dy_)

        g_fd = finite_difference_greeks(pricer, s, k, t, r, sigma, OptionType.CALL, q)

        assert np.isclose(g_fd.delta, g_ana.delta, rtol=1e-3)
        assert np.isclose(g_fd.gamma, g_ana.gamma, rtol=1e-3)
        assert np.isclose(g_fd.vega, g_ana.vega, rtol=1e-3)
        assert np.isclose(g_fd.theta, g_ana.theta, rtol=1e-3)
        assert np.isclose(g_fd.rho, g_ana.rho, rtol=1e-3)

    def test_finite_difference_greeks_on_crr(self):
        s = 100.0
        k = 100.0
        t = 1.0
        r = 0.05
        sigma = 0.20
        q = 0.0

        def crr_pricer(s_, k_, t_, r_, v_, ot_, dy_):
            return crr_price(
                s_, k_, t_, r_, v_, steps=400, option_type=ot_, dividend_yield=dy_
            ).price

        g_crr_fd = finite_difference_greeks(
            crr_pricer, s, k, t, r, sigma, OptionType.CALL, q, ds_rel=0.01
        )
        g_ana = bsm_greeks(s, k, t, r, sigma, OptionType.CALL, q)

        # CRR finite difference delta should be close to analytical delta
        assert np.isclose(g_crr_fd.delta, g_ana.delta, atol=0.01)
        assert np.isclose(g_crr_fd.vega, g_ana.vega, atol=0.5)

    def test_pathwise_mc_greeks_agreement(self):
        s = 100.0
        k = 105.0
        t = 0.5
        r = 0.04
        sigma = 0.25
        q = 0.0

        g_ana_call = bsm_greeks(s, k, t, r, sigma, OptionType.CALL, q)
        g_pw_call = pathwise_mc_greeks(
            s, k, t, r, sigma, n_paths=150_000, seed=42, option_type="call", dividend_yield=q
        )

        # Pathwise Delta and Vega agreement within Monte Carlo tolerance
        assert np.isclose(g_pw_call.delta, g_ana_call.delta, atol=0.01)
        assert np.isclose(g_pw_call.vega, g_ana_call.vega, atol=0.5)

        g_ana_put = bsm_greeks(s, k, t, r, sigma, OptionType.PUT, q)
        g_pw_put = pathwise_mc_greeks(
            s, k, t, r, sigma, n_paths=150_000, seed=42, option_type="put", dividend_yield=q
        )

        assert np.isclose(g_pw_put.delta, g_ana_put.delta, atol=0.01)
        assert np.isclose(g_pw_put.vega, g_ana_put.vega, atol=0.5)

    def test_compare_greeks_harness(self):
        df = compare_greeks(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
        assert len(df) == 5
        assert set(df["Greek"]) == {"Delta", "Gamma", "Vega", "Theta", "Rho"}
        assert (df["FD_Diff"] < 0.05).all()

    def test_delta_hedging_simulation(self):
        # In discrete delta hedging, higher frequency rebalancing (more steps) reduces P&L standard deviation
        s = 100.0
        k = 100.0
        t = 0.25
        r = 0.05
        sigma = 0.20

        sim_coarse = simulate_delta_hedging(
            s, k, t, r, sigma, n_steps=10, n_simulations=300, seed=42
        )
        sim_fine = simulate_delta_hedging(s, k, t, r, sigma, n_steps=60, n_simulations=300, seed=42)

        # Mean P&L is centered around 0 (fair hedge)
        assert abs(sim_coarse["mean_pnl"]) < 0.5
        assert abs(sim_fine["mean_pnl"]) < 0.5

        # P&L dispersion is significantly smaller with finer rebalancing
        assert sim_fine["std_pnl"] < sim_coarse["std_pnl"]
