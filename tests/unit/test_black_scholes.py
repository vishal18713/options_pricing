"""Comprehensive unit tests for BSM analytical pricing and Greeks."""

import numpy as np
import pytest

from options_engine.black_scholes import bsm_greeks, bsm_price
from options_engine.types import OptionType
from options_engine.validation import InputValidationError


class TestBSMAnalyticalCore:
    def test_textbook_call_put_values(self):
        # Classic Hull textbook example: S=49, K=50, r=0.05, sigma=0.20, T=0.3846 (~20 weeks), q=0
        s = 49.0
        k = 50.0
        r = 0.05
        sigma = 0.20
        t = 0.3846
        q = 0.0

        call = bsm_price(s, k, t, r, sigma, option_type="call", dividend_yield=q)
        put = bsm_price(s, k, t, r, sigma, option_type="put", dividend_yield=q)

        # Hull textbook reference: Call ~ 2.4005, Put ~ 2.4481 (by put-call parity)
        assert np.isclose(call, 2.4005, atol=1e-3)
        assert np.isclose(put, 2.4481, atol=1e-3)

    def test_put_call_parity_exactness(self):
        # Parity: Call - Put = S * exp(-q*T) - K * exp(-r*T)
        test_cases = [
            (100.0, 100.0, 1.0, 0.05, 0.20, 0.0),
            (120.0, 100.0, 0.5, 0.03, 0.35, 0.02),
            (80.0, 100.0, 2.0, 0.01, 0.15, 0.04),
            (50.0, 150.0, 0.25, 0.06, 0.50, 0.03),
        ]

        for s, k, t, r, sigma, q in test_cases:
            call = bsm_price(s, k, t, r, sigma, option_type=OptionType.CALL, dividend_yield=q)
            put = bsm_price(s, k, t, r, sigma, option_type=OptionType.PUT, dividend_yield=q)

            forward_diff = call - put
            theoretical_diff = s * np.exp(-q * t) - k * np.exp(-r * t)

            assert np.isclose(forward_diff, theoretical_diff, atol=1e-12)

    def test_vectorization_and_array_broadcasting(self):
        spots = np.array([90.0, 100.0, 110.0])
        strikes = np.array([100.0, 100.0, 100.0])
        calls = bsm_price(spots, strikes, 1.0, 0.05, 0.20, option_type="call")

        assert isinstance(calls, np.ndarray)
        assert calls.shape == (3,)
        assert calls[0] < calls[1] < calls[2]  # Call price monotonic in spot

    def test_monotonicity_properties(self):
        s_base = 100.0
        k_base = 100.0
        t_base = 1.0
        r_base = 0.05
        sigma_base = 0.20

        # Call increases with spot; Put decreases with spot
        c_low_s = bsm_price(s_base - 5, k_base, t_base, r_base, sigma_base, "call")
        c_mid_s = bsm_price(s_base, k_base, t_base, r_base, sigma_base, "call")
        c_high_s = bsm_price(s_base + 5, k_base, t_base, r_base, sigma_base, "call")
        assert c_low_s < c_mid_s < c_high_s

        p_low_s = bsm_price(s_base - 5, k_base, t_base, r_base, sigma_base, "put")
        p_mid_s = bsm_price(s_base, k_base, t_base, r_base, sigma_base, "put")
        p_high_s = bsm_price(s_base + 5, k_base, t_base, r_base, sigma_base, "put")
        assert p_low_s > p_mid_s > p_high_s

        # Call decreases with strike; Put increases with strike
        c_low_k = bsm_price(s_base, k_base - 5, t_base, r_base, sigma_base, "call")
        c_high_k = bsm_price(s_base, k_base + 5, t_base, r_base, sigma_base, "call")
        assert c_low_k > c_high_k

        p_low_k = bsm_price(s_base, k_base - 5, t_base, r_base, sigma_base, "put")
        p_high_k = bsm_price(s_base, k_base + 5, t_base, r_base, sigma_base, "put")
        assert p_low_k < p_high_k

        # Both increase with volatility
        c_low_vol = bsm_price(s_base, k_base, t_base, r_base, sigma_base - 0.05, "call")
        c_high_vol = bsm_price(s_base, k_base, t_base, r_base, sigma_base + 0.05, "call")
        assert c_low_vol < c_high_vol

    def test_boundary_zero_expiry_and_zero_volatility(self):
        # Expiry T=0
        assert bsm_price(110.0, 100.0, 0.0, 0.05, 0.20, "call") == 10.0
        assert bsm_price(90.0, 100.0, 0.0, 0.05, 0.20, "call") == 0.0
        assert bsm_price(90.0, 100.0, 0.0, 0.05, 0.20, "put") == 10.0
        assert bsm_price(110.0, 100.0, 0.0, 0.05, 0.20, "put") == 0.0

        # Zero volatility T > 0 (discounted intrinsic based on forward)
        call_0vol = bsm_price(100.0, 100.0, 1.0, 0.05, 0.0, "call", dividend_yield=0.0)
        # Forward = 100 * exp(0.05) > 100, Call = 100 - 100*exp(-0.05)
        expected_call = 100.0 - 100.0 * np.exp(-0.05)
        assert np.isclose(call_0vol, expected_call, atol=1e-10)

    def test_analytical_greeks_against_central_finite_differences(self):
        s = 100.0
        k = 105.0
        t = 0.75
        r = 0.04
        sigma = 0.25
        q = 0.02

        for opt_type in [OptionType.CALL, OptionType.PUT]:
            g = bsm_greeks(s, k, t, r, sigma, option_type=opt_type, dividend_yield=q)

            # Delta bump
            ds = 1e-4
            p_up = bsm_price(s + ds, k, t, r, sigma, opt_type, q)
            p_dn = bsm_price(s - ds, k, t, r, sigma, opt_type, q)
            num_delta = (p_up - p_dn) / (2.0 * ds)
            assert np.isclose(g.delta, num_delta, rtol=1e-4)

            # Gamma bump (second derivative has machine epsilon amplification with tiny steps)
            p_mid = bsm_price(s, k, t, r, sigma, opt_type, q)
            ds_gamma = 1e-3
            p_g_up = bsm_price(s + ds_gamma, k, t, r, sigma, opt_type, q)
            p_g_dn = bsm_price(s - ds_gamma, k, t, r, sigma, opt_type, q)
            num_gamma = (p_g_up - 2.0 * p_mid + p_g_dn) / (ds_gamma**2)
            assert np.isclose(g.gamma, num_gamma, rtol=1e-3)
            assert g.gamma > 0.0

            # Vega bump
            dvol = 1e-4
            p_v_up = bsm_price(s, k, t, r, sigma + dvol, opt_type, q)
            p_v_dn = bsm_price(s, k, t, r, sigma - dvol, opt_type, q)
            num_vega = (p_v_up - p_v_dn) / (2.0 * dvol)
            assert np.isclose(g.vega, num_vega, rtol=1e-4)
            assert g.vega > 0.0

            # Theta passage of time bump: dPrice/dt = -(p(t+dt) - p(t-dt))/(2*dt)
            dt = 1e-5
            p_t_up = bsm_price(s, k, t + dt, r, sigma, opt_type, q)
            p_t_dn = bsm_price(s, k, t - dt, r, sigma, opt_type, q)
            num_theta = -(p_t_up - p_t_dn) / (2.0 * dt)
            assert np.isclose(g.theta, num_theta, rtol=1e-4)

            # Rho bump
            dr = 1e-4
            p_r_up = bsm_price(s, k, t, r + dr, sigma, opt_type, q)
            p_r_dn = bsm_price(s, k, t, r - dr, sigma, opt_type, q)
            num_rho = (p_r_up - p_r_dn) / (2.0 * dr)
            assert np.isclose(g.rho, num_rho, rtol=1e-4)

    def test_invalid_input_domain_raises(self):
        with pytest.raises(InputValidationError):
            bsm_price(-10.0, 100.0, 1.0, 0.05, 0.20)

        with pytest.raises(InputValidationError):
            bsm_price(100.0, -100.0, 1.0, 0.05, 0.20)

        with pytest.raises(InputValidationError):
            bsm_price(100.0, 100.0, -0.5, 0.05, 0.20)

        with pytest.raises(InputValidationError):
            bsm_price(100.0, 100.0, 1.0, 0.05, -0.20)

        with pytest.raises(InputValidationError):
            bsm_price(np.nan, 100.0, 1.0, 0.05, 0.20)
