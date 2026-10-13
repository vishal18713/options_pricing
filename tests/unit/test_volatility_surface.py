"""Unit tests for Phase 7: SVI surface calibration and static arbitrage diagnostics."""

import numpy as np
import pytest

from options_engine.volatility_surface import (
    SVIParameters,
    check_butterfly_arbitrage_slice,
    check_calendar_arbitrage,
    fit_svi_slice,
    raw_svi_total_variance,
)


class TestSVISurfaceAndArbitrage:
    def test_svi_parameter_evaluation_and_fit(self):
        # Known ground truth SVI parameters
        true_params = SVIParameters(a=0.04, b=0.15, rho=-0.3, m=0.02, eta=0.1)

        k_grid = np.linspace(-0.3, 0.3, 21)
        w_clean = raw_svi_total_variance(
            k_grid,
            true_params.a,
            true_params.b,
            true_params.rho,
            true_params.m,
            true_params.eta,
        )

        # Fit SVI to exact generated points
        fitted_params, obj, success, diag = fit_svi_slice(k_grid, w_clean)

        assert success
        assert diag["rmse"] < 1e-3

        # Predicted variance matches ground truth
        w_fitted = raw_svi_total_variance(
            k_grid,
            fitted_params.a,
            fitted_params.b,
            fitted_params.rho,
            fitted_params.m,
            fitted_params.eta,
        )
        assert np.allclose(w_clean, w_fitted, atol=1e-3)

    def test_butterfly_arbitrage_detection_on_good_and_bad_slices(self):
        k_grid = np.linspace(-0.5, 0.5, 51)

        # 1. Healthy arbitrage-free slice
        good_params = SVIParameters(a=0.04, b=0.1, rho=-0.2, m=0.0, eta=0.1)
        is_free, diag_good = check_butterfly_arbitrage_slice(good_params, k_grid)
        assert is_free
        assert diag_good["arbitrage_free"] is True
        assert diag_good["min_density"] >= -1e-6

        # 2. Deliberately constructed butterfly arbitrage violation
        # (Extreme b or negative a creating negative curvature or density)
        bad_params = SVIParameters(a=-0.02, b=1.5, rho=0.8, m=0.1, eta=0.01)
        is_bad_free, diag_bad = check_butterfly_arbitrage_slice(bad_params, k_grid)
        assert not is_bad_free
        assert diag_bad["arbitrage_free"] is False

    def test_calendar_arbitrage_detection(self):
        k_grid = np.linspace(-0.3, 0.3, 31)

        # Correctly ordered maturities with non-decreasing total variance
        slice_t1 = SVIParameters(a=0.02, b=0.1, rho=-0.2, m=0.0, eta=0.1)  # T=0.25
        slice_t2 = SVIParameters(a=0.05, b=0.1, rho=-0.2, m=0.0, eta=0.1)  # T=0.50
        slice_t3 = SVIParameters(a=0.09, b=0.1, rho=-0.2, m=0.0, eta=0.1)  # T=1.00

        slices_good = {0.25: slice_t1, 0.50: slice_t2, 1.00: slice_t3}
        is_cal_free, diag_cal = check_calendar_arbitrage(slices_good, k_grid)
        assert is_cal_free
        assert len(diag_cal["violations"]) == 0

        # Inverted maturities violating calendar arbitrage: T=1.00 total variance is lower than T=0.50
        slice_t_inverted = SVIParameters(a=0.03, b=0.1, rho=-0.2, m=0.0, eta=0.1)
        slices_bad = {0.25: slice_t1, 0.50: slice_t2, 1.00: slice_t_inverted}
        is_bad_cal, diag_bad_cal = check_calendar_arbitrage(slices_bad, k_grid)
        assert not is_bad_cal
        assert len(diag_bad_cal["violations"]) > 0
        assert diag_bad_cal["violations"][0]["t1"] == 0.50
        assert diag_bad_cal["violations"][0]["t2"] == 1.00

    def test_sparse_slice_error(self):
        k_small = np.array([-0.1, 0.0, 0.1])
        w_small = np.array([0.04, 0.03, 0.04])
        with pytest.raises(ValueError, match="At least 5 quotes required"):
            fit_svi_slice(k_small, w_small)
