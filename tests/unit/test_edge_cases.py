"""Unit tests for edge cases and Greek zero-vol/expiry branches."""

from options_engine.black_scholes import bsm_greeks
from options_engine.conventions import (
    scale_rho_per_basis_point,
    scale_theta_per_day,
    scale_vega_per_percentage_point,
    year_fraction_act365,
)
from options_engine.types import OptionType


def test_greeks_at_expiry():
    # Call ITM at expiry: delta = exp(-q*0) = 1.0
    g_call_itm = bsm_greeks(110.0, 100.0, 0.0, 0.05, 0.20, OptionType.CALL)
    assert g_call_itm.delta == 1.0
    assert g_call_itm.gamma == 0.0
    assert g_call_itm.diagnostics.get("at_expiry") is True

    # Call OTM at expiry: delta = 0.0
    g_call_otm = bsm_greeks(90.0, 100.0, 0.0, 0.05, 0.20, OptionType.CALL)
    assert g_call_otm.delta == 0.0

    # Put ITM at expiry: delta = -1.0
    g_put_itm = bsm_greeks(90.0, 100.0, 0.0, 0.05, 0.20, OptionType.PUT)
    assert g_put_itm.delta == -1.0

    # Put OTM at expiry: delta = 0.0
    g_put_otm = bsm_greeks(110.0, 100.0, 0.0, 0.05, 0.20, OptionType.PUT)
    assert g_put_otm.delta == 0.0


def test_greeks_zero_volatility():
    # Volatility == 0, forward > strike
    g_call = bsm_greeks(120.0, 100.0, 1.0, 0.05, 0.0, OptionType.CALL)
    assert g_call.delta == 1.0
    assert g_call.vega == 0.0
    assert g_call.diagnostics.get("zero_volatility") is True

    # Volatility == 0, forward < strike
    g_put = bsm_greeks(80.0, 100.0, 1.0, 0.05, 0.0, OptionType.PUT)
    assert g_put.delta == -1.0
    assert g_put.vega == 0.0


def test_conventions_scaling():
    assert year_fraction_act365(182.5) == 0.5
    assert scale_theta_per_day(36.5) == 0.1
    assert scale_vega_per_percentage_point(100.0) == 1.0
    assert scale_rho_per_basis_point(10000.0) == 1.0
