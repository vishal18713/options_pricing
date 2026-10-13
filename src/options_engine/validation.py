"""Input domain validation and error taxonomy."""

import numpy as np


class InputValidationError(ValueError):
    """Raised when an input violates domain constraints (e.g. negative spot/strike/time/vol)."""

    pass


class ModelDiscretizationError(ValueError):
    """Raised when numerical discretization parameters lead to invalid model states (e.g. CRR p not in [0, 1])."""

    pass


def validate_market_inputs(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
    volatility: float | np.ndarray,
    rate: float | np.ndarray,
    dividend_yield: float | np.ndarray = 0.0,
) -> None:
    """Validate numerical domains for option inputs."""
    s = np.asarray(spot)
    k = np.asarray(strike)
    t = np.asarray(time_to_expiry)
    v = np.asarray(volatility)
    r = np.asarray(rate)
    q = np.asarray(dividend_yield)

    if (
        np.any(np.isnan(s))
        or np.any(np.isnan(k))
        or np.any(np.isnan(t))
        or np.any(np.isnan(v))
        or np.any(np.isnan(r))
        or np.any(np.isnan(q))
    ):
        raise InputValidationError("INPUT_NAN: NaN encountered in input arrays.")

    if (
        np.any(np.isinf(s))
        or np.any(np.isinf(k))
        or np.any(np.isinf(t))
        or np.any(np.isinf(v))
        or np.any(np.isinf(r))
        or np.any(np.isinf(q))
    ):
        raise InputValidationError("INPUT_INF: Infinite values encountered in input arrays.")

    if np.any(s <= 0):
        raise InputValidationError(
            f"INPUT_SPOT_NON_POSITIVE: Spot must be strictly positive. Got {spot}"
        )

    if np.any(k <= 0):
        raise InputValidationError(
            f"INPUT_STRIKE_NON_POSITIVE: Strike must be strictly positive. Got {strike}"
        )

    if np.any(t < 0):
        raise InputValidationError(
            f"INPUT_TIME_NEGATIVE: Time to expiry must be non-negative. Got {time_to_expiry}"
        )

    if np.any(v < 0):
        raise InputValidationError(
            f"INPUT_VOL_NEGATIVE: Volatility must be non-negative. Got {volatility}"
        )
