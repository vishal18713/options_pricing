"""Unit and regression tests for Phase 8: QuantLib validation and performance benchmark."""

from options_engine.validation_suite import price_with_quantlib, run_quantlib_cross_validation_grid
from scripts.benchmark import benchmark_engine_runtimes


class TestQuantLibCrossValidation:
    def test_quantlib_european_bsm_exact_alignment(self):
        # Aligned European Call
        ql_price, _ = price_with_quantlib(
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            rate=0.05,
            volatility=0.20,
            dividend_yield=0.01,
        )
        from options_engine.black_scholes import bsm_price

        engine_price = bsm_price(100.0, 100.0, 1.0, 0.05, 0.20, dividend_yield=0.01)

        # Tolerances matched to within QuantLib standard numerical rounding
        assert abs(engine_price - ql_price) < 1e-4

    def test_quantlib_american_crr_alignment(self):
        # Aligned American Put with early exercise
        ql_price, _ = price_with_quantlib(
            spot=70.0,
            strike=100.0,
            time_to_expiry=1.0,
            rate=0.08,
            volatility=0.20,
            exercise_style="american",
            option_type="put",
            binomial_steps=500,
        )
        from options_engine.binomial import crr_price

        engine_res = crr_price(
            70.0,
            100.0,
            1.0,
            0.08,
            0.20,
            steps=500,
            exercise_style="american",
            option_type="put",
        )

        assert abs(engine_res.price - ql_price) < 0.05

    def test_quantlib_grid_cross_validation(self):
        df = run_quantlib_cross_validation_grid()
        assert len(df) == 7
        # European absolute difference should be < 1e-3
        euro_rows = df[df["Style"] == "european"]
        assert (euro_rows["Abs_Diff"] < 1e-3).all()
        # American absolute difference with CRR should be < 0.08
        amer_rows = df[df["Style"] == "american"]
        assert (amer_rows["Abs_Diff"] < 0.08).all()

    def test_benchmark_runs_cleanly(self):
        bench_df = benchmark_engine_runtimes(repetitions=2)
        assert len(bench_df) == 6
        assert (bench_df["Median_Time_Sec"] > 0).all()
