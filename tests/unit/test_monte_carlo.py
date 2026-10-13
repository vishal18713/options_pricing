"""Unit tests for Monte Carlo option pricing engine and variance reduction."""

import numpy as np

from options_engine.black_scholes import bsm_price
from options_engine.monte_carlo import monte_carlo_price
from options_engine.types import OptionType


class TestMonteCarloCore:
    def test_deterministic_reproducibility_by_seed(self):
        # Multiple runs with the same seed must produce identical price and standard error
        res1 = monte_carlo_price(
            100.0, 100.0, 1.0, 0.05, 0.20, n_paths=50_000, seed=12345, antithetic=True
        )
        res2 = monte_carlo_price(
            100.0, 100.0, 1.0, 0.05, 0.20, n_paths=50_000, seed=12345, antithetic=True
        )

        assert res1.price == res2.price
        assert res1.standard_error == res2.standard_error
        assert res1.confidence_interval == res2.confidence_interval

    def test_mc_agrees_with_bsm_within_standard_error(self):
        # A statistically valid test: |MC - BSM| should typically be within ~3 standard errors
        s = 100.0
        k = 105.0
        t = 0.5
        r = 0.03
        sigma = 0.22
        q = 0.01

        bsm_call = bsm_price(s, k, t, r, sigma, OptionType.CALL, q)
        bsm_put = bsm_price(s, k, t, r, sigma, OptionType.PUT, q)

        mc_call = monte_carlo_price(
            s,
            k,
            t,
            r,
            sigma,
            n_paths=150_000,
            seed=42,
            option_type="call",
            dividend_yield=q,
            antithetic=True,
        )
        mc_put = monte_carlo_price(
            s,
            k,
            t,
            r,
            sigma,
            n_paths=150_000,
            seed=42,
            option_type="put",
            dividend_yield=q,
            antithetic=True,
        )

        # Statistical agreement in units of standard error
        call_error_in_se = abs(mc_call.price - bsm_call) / mc_call.standard_error
        put_error_in_se = abs(mc_put.price - bsm_put) / mc_put.standard_error

        assert call_error_in_se < 3.0
        assert put_error_in_se < 3.0

        # Confidence intervals cover BSM
        assert mc_call.confidence_interval[0] <= bsm_call <= mc_call.confidence_interval[1]
        assert mc_put.confidence_interval[0] <= bsm_put <= mc_put.confidence_interval[1]

    def test_antithetic_variance_reduction(self):
        # Antithetic variate should result in lower or equal standard error compared to naive MC
        s = 100.0
        k = 100.0
        t = 1.0
        r = 0.05
        sigma = 0.20

        res_naive = monte_carlo_price(
            s, k, t, r, sigma, n_paths=100_000, seed=999, antithetic=False
        )
        res_anti = monte_carlo_price(s, k, t, r, sigma, n_paths=100_000, seed=999, antithetic=True)

        assert res_anti.standard_error < res_naive.standard_error
        assert "antithetic" in res_anti.variance_reduction

    def test_control_variate_variance_reduction(self):
        # Control variate further reduces standard error
        s = 100.0
        k = 100.0
        t = 1.0
        r = 0.05
        sigma = 0.20

        res_anti = monte_carlo_price(
            s, k, t, r, sigma, n_paths=80_000, seed=777, antithetic=True, control_variate=False
        )
        res_cv = monte_carlo_price(
            s, k, t, r, sigma, n_paths=80_000, seed=777, antithetic=True, control_variate=True
        )

        assert res_cv.standard_error < res_anti.standard_error
        assert "control_variate" in res_cv.variance_reduction

    def test_standard_error_scaling_with_sample_count(self):
        # Standard error scales approximately as O(1/sqrt(N))
        # N2 = 4 * N1 -> SE(N2) ~ 0.5 * SE(N1)
        s = 100.0
        k = 100.0
        t = 1.0
        r = 0.05
        sigma = 0.20

        res1 = monte_carlo_price(s, k, t, r, sigma, n_paths=25_000, seed=10, antithetic=False)
        res2 = monte_carlo_price(s, k, t, r, sigma, n_paths=100_000, seed=10, antithetic=False)

        se_ratio = res2.standard_error / res1.standard_error
        # Theoretical ratio is 1/sqrt(4) = 0.5
        assert np.isclose(se_ratio, 0.5, atol=0.08)

    def test_boundary_conditions(self):
        # T=0
        res_exp = monte_carlo_price(110.0, 100.0, 0.0, 0.05, 0.20, n_paths=1000)
        assert res_exp.price == 10.0
        assert res_exp.standard_error == 0.0

        # vol=0
        res_vol0 = monte_carlo_price(100.0, 100.0, 1.0, 0.05, 0.0, n_paths=1000)
        assert res_vol0.price > 0.0
        assert res_vol0.standard_error == 0.0
