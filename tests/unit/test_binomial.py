import numpy as np
import pytest

from options_engine.binomial import crr_price
from options_engine.black_scholes import bsm_price
from options_engine.types import ExerciseStyle, OptionType
from options_engine.validation import ModelDiscretizationError


class TestCRRBinomialCore:
    def test_european_crr_convergence_to_bsm(self):
        # As N increases, European CRR price converges to BSM price
        s = 100.0
        k = 100.0
        t = 1.0
        r = 0.05
        sigma = 0.20
        q = 0.02

        bsm_call = bsm_price(s, k, t, r, sigma, OptionType.CALL, q)
        bsm_put = bsm_price(s, k, t, r, sigma, OptionType.PUT, q)

        # Evaluate at step counts
        crr_100_call = crr_price(s, k, t, r, sigma, steps=100, option_type="call", dividend_yield=q)
        crr_500_call = crr_price(s, k, t, r, sigma, steps=500, option_type="call", dividend_yield=q)

        crr_100_put = crr_price(s, k, t, r, sigma, steps=100, option_type="put", dividend_yield=q)
        crr_500_put = crr_price(s, k, t, r, sigma, steps=500, option_type="put", dividend_yield=q)

        # Difference should decrease / be within tight documented tolerance
        assert abs(crr_500_call.price - bsm_call) < 0.02
        assert abs(crr_500_put.price - bsm_put) < 0.02
        assert abs(crr_500_call.price - bsm_call) < abs(crr_100_call.price - bsm_call)
        assert abs(crr_500_put.price - bsm_put) < abs(crr_100_put.price - bsm_put)

    def test_american_greater_than_or_equal_to_european(self):
        # American option value must never be strictly less than European value
        test_grid = [
            (100.0, 100.0, 1.0, 0.05, 0.20, 0.0),
            (100.0, 120.0, 0.5, 0.08, 0.30, 0.0),
            (
                120.0,
                100.0,
                0.5,
                0.02,
                0.25,
                0.08,
            ),  # High dividend yield creates early call exercise
            (80.0, 100.0, 1.0, 0.08, 0.20, 0.0),  # Deep ITM put creates early put exercise
        ]

        for s, k, t, r, sigma, q in test_grid:
            for opt_type in ["call", "put"]:
                euro = crr_price(
                    s,
                    k,
                    t,
                    r,
                    sigma,
                    steps=200,
                    option_type=opt_type,
                    exercise_style=ExerciseStyle.EUROPEAN,
                    dividend_yield=q,
                )
                amer = crr_price(
                    s,
                    k,
                    t,
                    r,
                    sigma,
                    steps=200,
                    option_type=opt_type,
                    exercise_style=ExerciseStyle.AMERICAN,
                    dividend_yield=q,
                )
                assert amer.price >= euro.price - 1e-12

    def test_american_early_exercise_premium(self):
        # Case 1: Deep ITM American Put with positive risk-free rate has strictly positive early-exercise premium
        s = 70.0
        k = 100.0
        t = 1.0
        r = 0.10
        sigma = 0.20
        q = 0.0

        euro_put = crr_price(
            s,
            k,
            t,
            r,
            sigma,
            steps=300,
            option_type="put",
            exercise_style="european",
            dividend_yield=q,
        )
        amer_put = crr_price(
            s,
            k,
            t,
            r,
            sigma,
            steps=300,
            option_type="put",
            exercise_style="american",
            dividend_yield=q,
        )

        assert amer_put.price > euro_put.price
        # American put must be at least intrinsic value (30.0)
        assert amer_put.price >= (k - s)

        # Case 2: ITM American Call with substantial dividend yield has early-exercise premium
        s_div = 120.0
        k_div = 100.0
        t_div = 1.0
        r_div = 0.02
        sigma_div = 0.20
        q_div = 0.10  # high dividend

        euro_call = crr_price(
            s_div,
            k_div,
            t_div,
            r_div,
            sigma_div,
            steps=300,
            option_type="call",
            exercise_style="european",
            dividend_yield=q_div,
        )
        amer_call = crr_price(
            s_div,
            k_div,
            t_div,
            r_div,
            sigma_div,
            steps=300,
            option_type="call",
            exercise_style="american",
            dividend_yield=q_div,
        )

        assert amer_call.price > euro_call.price
        assert amer_call.price >= (s_div - k_div)

    def test_non_dividend_american_call_equals_european_call(self):
        # Under standard theory, with q=0 and r >= 0, it is never optimal to early-exercise an American Call.
        # Hence American Call price == European Call price.
        s = 100.0
        k = 100.0
        t = 1.0
        r = 0.05
        sigma = 0.25
        q = 0.0

        euro_call = crr_price(
            s,
            k,
            t,
            r,
            sigma,
            steps=250,
            option_type="call",
            exercise_style="european",
            dividend_yield=q,
        )
        amer_call = crr_price(
            s,
            k,
            t,
            r,
            sigma,
            steps=250,
            option_type="call",
            exercise_style="american",
            dividend_yield=q,
        )

        assert np.isclose(amer_call.price, euro_call.price, atol=1e-6)

    def test_invalid_probability_domain_raises(self):
        # If dt is large, volatility is tiny, and drift is high: p = (exp((r-q)dt) - d)/(u-d) can exceed [0, 1]
        with pytest.raises(ModelDiscretizationError):
            crr_price(100.0, 100.0, 5.0, 0.50, 0.01, steps=1)

    def test_crr_boundary_cases(self):
        # Expiry T=0
        res = crr_price(110.0, 100.0, 0.0, 0.05, 0.20, steps=10, option_type="call")
        assert res.price == 10.0
        assert res.diagnostics.get("at_expiry") is True

        # Volatility = 0
        res_vol0 = crr_price(100.0, 100.0, 1.0, 0.05, 0.0, steps=10, option_type="call")
        assert res_vol0.price > 0.0
        assert res_vol0.diagnostics.get("zero_volatility") is True
