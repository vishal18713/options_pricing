"""Independent cross-validation suite against QuantLib and benchmark comparisons.

Aligns QuantLib instrument, exercise style, day count (ACT/365), risk-free rate,
continuous dividend yield, and expiry conventions with our engine models.
"""

from typing import Any

import pandas as pd

from options_engine.binomial import crr_price
from options_engine.black_scholes import bsm_price
from options_engine.types import ExerciseStyle, OptionType


def price_with_quantlib(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType = OptionType.CALL,
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
    dividend_yield: float = 0.0,
    binomial_steps: int = 500,
) -> float:
    """Price option using QuantLib with aligned market conventions."""
    import QuantLib as ql  # noqa: N813

    # Evaluation date
    today = ql.Date(1, ql.January, 2026)
    ql.Settings.instance().evaluationDate = today

    # Expiry calculation under ACT/365
    day_count = ql.Actual365Fixed()
    days = int(round(time_to_expiry * 365.0))
    maturity_date = today + days
    actual_t = day_count.yearFraction(today, maturity_date)

    # Option type
    ql_option_type = ql.Option.Call if option_type == OptionType.CALL else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(ql_option_type, strike)

    # Quotes and curves
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    rate_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(today, rate, day_count, ql.Continuous, ql.Annual)
    )
    div_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(today, dividend_yield, day_count, ql.Continuous, ql.Annual)
    )
    vol_curve = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), volatility, day_count)
    )

    bsm_process = ql.BlackScholesMertonProcess(spot_handle, div_curve, rate_curve, vol_curve)

    if exercise_style == ExerciseStyle.EUROPEAN:
        exercise = ql.EuropeanExercise(maturity_date)
        option = ql.VanillaOption(payoff, exercise)
        engine = ql.AnalyticEuropeanEngine(bsm_process)
        option.setPricingEngine(engine)
        return float(option.NPV()), actual_t
    else:
        exercise = ql.AmericanExercise(today, maturity_date)
        option = ql.VanillaOption(payoff, exercise)
        engine = ql.BinomialVanillaEngine(bsm_process, "crr", binomial_steps)
        option.setPricingEngine(engine)
        return float(option.NPV()), actual_t


def run_quantlib_cross_validation_grid() -> pd.DataFrame:
    """Run an automated grid cross-validation comparing project models against QuantLib."""
    test_grid = [
        # (S, K, T, r, sigma, q, opt_type, ex_style)
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, OptionType.CALL, ExerciseStyle.EUROPEAN),
        (100.0, 100.0, 1.0, 0.05, 0.20, 0.0, OptionType.PUT, ExerciseStyle.EUROPEAN),
        (110.0, 100.0, 0.5, 0.03, 0.25, 0.02, OptionType.CALL, ExerciseStyle.EUROPEAN),
        (90.0, 100.0, 0.5, 0.03, 0.25, 0.02, OptionType.PUT, ExerciseStyle.EUROPEAN),
        (100.0, 95.0, 1.5, 0.04, 0.30, 0.01, OptionType.CALL, ExerciseStyle.AMERICAN),
        (100.0, 105.0, 1.5, 0.04, 0.30, 0.01, OptionType.PUT, ExerciseStyle.AMERICAN),
        (70.0, 100.0, 1.0, 0.08, 0.20, 0.0, OptionType.PUT, ExerciseStyle.AMERICAN),
    ]

    records: list[dict[str, Any]] = []

    for s, k, t, r, v, q, opt_type, ex_style in test_grid:
        ql_price, actual_t = price_with_quantlib(
            spot=s,
            strike=k,
            time_to_expiry=t,
            rate=r,
            volatility=v,
            option_type=opt_type,
            exercise_style=ex_style,
            dividend_yield=q,
            binomial_steps=500,
        )

        if ex_style == ExerciseStyle.EUROPEAN:
            engine_price = bsm_price(s, k, actual_t, r, v, opt_type, q)
            model_name = "BSM"
        else:
            engine_price = crr_price(
                s,
                k,
                actual_t,
                r,
                v,
                steps=500,
                option_type=opt_type,
                exercise_style=ex_style,
                dividend_yield=q,
            ).price
            model_name = "CRR"

        abs_diff = abs(engine_price - ql_price)
        rel_diff = abs_diff / ql_price if ql_price > 0 else abs_diff

        records.append(
            {
                "Spot": s,
                "Strike": k,
                "T": t,
                "Rate": r,
                "Vol": v,
                "Div": q,
                "Type": opt_type.value,
                "Style": ex_style.value,
                "Model": model_name,
                "Engine_Price": engine_price,
                "QuantLib_Price": ql_price,
                "Abs_Diff": abs_diff,
                "Rel_Diff": rel_diff,
            }
        )

    return pd.DataFrame(records)
