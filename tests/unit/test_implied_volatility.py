"""Unit tests for Phase 6: Forward estimation and Implied Volatility solver."""

import numpy as np
import pandas as pd

from options_engine.black_scholes import bsm_price
from options_engine.forwards import estimate_forward_from_parity, forward_price_theoretical
from options_engine.implied_volatility import check_price_bounds, implied_volatility
from options_engine.types import OptionType


class TestForwardAndImpliedVolatility:
    def test_round_trip_volatility_inversion(self):
        # Generate exact BSM price with known sigma -> invert -> recover sigma within tight tolerance
        test_cases = [
            (100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL, 0.0),
            (100.0, 110.0, 0.5, 0.03, 0.35, OptionType.CALL, 0.02),
            (100.0, 90.0, 0.5, 0.03, 0.15, OptionType.PUT, 0.01),
            (250.0, 200.0, 2.0, 0.04, 0.45, OptionType.PUT, 0.03),
            (50.0, 70.0, 0.25, 0.02, 0.60, OptionType.CALL, 0.0),
        ]

        for s, k, t, r, true_sigma, opt_type, q in test_cases:
            price = bsm_price(s, k, t, r, true_sigma, opt_type, q)
            res = implied_volatility(price, s, k, t, r, opt_type, q)

            assert res.status == "CONVERGED"
            assert res.implied_volatility is not None
            assert np.isclose(res.implied_volatility, true_sigma, atol=1e-6)
            assert abs(res.residual) < 1e-7

    def test_price_bounds_checking(self):
        s = 100.0
        k = 100.0
        t = 1.0
        r = 0.05
        q = 0.0

        # Theoretical bounds for call: max(0, 100 - 100*e^-0.05) ~ 4.877 <= C <= 100
        # Price below lower bound (e.g. 2.0)
        valid, lb, ub, reason = check_price_bounds(2.0, s, k, t, r, OptionType.CALL, q)
        assert not valid
        assert reason == "IV_PRICE_BELOW_LOWER_BOUND"

        # Price above upper bound (e.g. 105.0)
        valid, lb, ub, reason = check_price_bounds(105.0, s, k, t, r, OptionType.CALL, q)
        assert not valid
        assert reason == "IV_PRICE_ABOVE_UPPER_BOUND"

        # Legitimate price
        valid, lb, ub, reason = check_price_bounds(10.0, s, k, t, r, OptionType.CALL, q)
        assert valid
        assert reason == "VALID"

    def test_out_of_bounds_solver_failure_returns_reason_code(self):
        # Solve for price below lower bound -> FAILED with explicit reason_code (never 0.0)
        res = implied_volatility(1.0, 100.0, 100.0, 1.0, 0.05, OptionType.CALL)
        assert res.status == "FAILED"
        assert res.implied_volatility is None
        assert res.reason_code == "IV_PRICE_BELOW_LOWER_BOUND"

        # Negative price -> FAILED
        res_neg = implied_volatility(-5.0, 100.0, 100.0, 1.0, 0.05, OptionType.CALL)
        assert res_neg.status == "FAILED"
        assert res_neg.implied_volatility is None
        assert res_neg.reason_code == "IV_NON_POSITIVE_PRICE"

    def test_forward_price_theoretical_and_parity_estimation(self):
        s = 100.0
        r = 0.05
        q = 0.02
        t = 1.0
        sigma = 0.20

        fwd_true = forward_price_theoretical(s, t, r, q)
        # S * exp((r - q) * T) = 100 * exp(0.03) = 103.045453
        assert np.isclose(fwd_true, 100.0 * np.exp(0.03), atol=1e-8)

        # Construct quotes adhering to put-call parity
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        calls_data = []
        puts_data = []
        for k in strikes:
            c = bsm_price(s, k, t, r, sigma, OptionType.CALL, q)
            p = bsm_price(s, k, t, r, sigma, OptionType.PUT, q)
            calls_data.append({"strike": k, "mid": c})
            puts_data.append({"strike": k, "mid": p})

        fwd_est = estimate_forward_from_parity(
            pd.DataFrame(calls_data),
            pd.DataFrame(puts_data),
            time_to_expiry=t,
            rate=r,
            spot_estimate=s,
        )

        assert fwd_est is not None
        assert np.isclose(fwd_est, fwd_true, atol=1e-6)
