"""Smoke and contract tests for Phase 0."""

import pytest

from options_engine import __version__
from options_engine.conventions import scale_theta_per_day, year_fraction_act365
from options_engine.types import (
    ExerciseStyle,
    MarketInputs,
    OptionContract,
    OptionType,
)
from options_engine.validation import InputValidationError, validate_market_inputs


def test_package_version():
    assert __version__ == "0.1.0"


def test_option_contract_creation():
    contract = OptionContract(
        underlying="AAPL",
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
        strike=100.0,
        expiry=1.0,
    )
    assert contract.strike == 100.0
    assert contract.option_type == OptionType.CALL
    assert contract.exercise_style == ExerciseStyle.EUROPEAN

    with pytest.raises(ValueError):
        OptionContract("AAPL", OptionType.CALL, ExerciseStyle.EUROPEAN, strike=-10.0, expiry=1.0)


def test_market_inputs_creation():
    mkt = MarketInputs(spot=100.0, rate=0.05, dividend_yield=0.02)
    assert mkt.spot == 100.0
    assert mkt.rate == 0.05
    assert mkt.dividend_yield == 0.02

    with pytest.raises(ValueError):
        MarketInputs(spot=-50.0, rate=0.05)


def test_conventions():
    assert year_fraction_act365(365) == 1.0
    assert scale_theta_per_day(365.0) == 1.0


def test_validation_domain():
    # Valid
    validate_market_inputs(100.0, 100.0, 1.0, 0.20, 0.05, 0.0)

    # Invalid spot
    with pytest.raises(InputValidationError):
        validate_market_inputs(-100.0, 100.0, 1.0, 0.20, 0.05)

    # Invalid negative volatility
    with pytest.raises(InputValidationError):
        validate_market_inputs(100.0, 100.0, 1.0, -0.20, 0.05)
