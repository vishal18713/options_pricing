"""Black-Scholes-Merton (BSM) analytical pricing and closed-form Greeks.

Supports continuous dividend yield, scalar and NumPy-array inputs,
and robust boundary/limiting behaviors for zero volatility, expiration, and deep ITM/OTM regimes.
"""

import numpy as np
from scipy.stats import norm

from options_engine.types import GreeksResult, OptionType
from options_engine.validation import validate_market_inputs


def bsm_d1_d2(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
    dividend_yield: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate d1 and d2 with broadcasting and domain validation.

    Returns:
        tuple[np.ndarray, np.ndarray]: (d1, d2)
    """
    validate_market_inputs(spot, strike, time_to_expiry, volatility, rate, dividend_yield)

    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    t = np.asarray(time_to_expiry, dtype=np.float64)
    r = np.asarray(rate, dtype=np.float64)
    v = np.asarray(volatility, dtype=np.float64)
    q = np.asarray(dividend_yield, dtype=np.float64)

    # Avoid division by zero where t <= 0 or v <= 0
    vol_sqrt_t = v * np.sqrt(t)

    # Where vol_sqrt_t is tiny or zero, d1 and d2 tend to +/- infinity depending on forward moneyness
    # Forward = s * exp((r - q) * t)
    with np.errstate(divide="ignore", invalid="ignore"):
        forward = s * np.exp((r - q) * t)
        log_fk = np.log(forward / k)
        d1 = np.where(
            vol_sqrt_t > 0,
            (np.log(s / k) + (r - q + 0.5 * v**2) * t) / vol_sqrt_t,
            np.where(log_fk > 0, np.inf, np.where(log_fk < 0, -np.inf, 0.0)),
        )
        d2 = np.where(vol_sqrt_t > 0, d1 - vol_sqrt_t, d1)

    return d1, d2


def bsm_price(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    time_to_expiry: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Calculate analytical Black-Scholes-Merton option price.

    Args:
        spot: Spot price S > 0
        strike: Strike price K > 0
        time_to_expiry: Time to expiration T >= 0 in years
        rate: Continuously compounded risk-free rate r
        volatility: Annualized volatility sigma >= 0
        option_type: "call" or "put" (OptionType or string)
        dividend_yield: Continuously compounded dividend yield q >= 0

    Returns:
        Option price as a float (if all inputs are scalar) or ndarray.
    """
    is_scalar = (
        np.isscalar(spot)
        and np.isscalar(strike)
        and np.isscalar(time_to_expiry)
        and np.isscalar(rate)
        and np.isscalar(volatility)
        and np.isscalar(dividend_yield)
    )

    validate_market_inputs(spot, strike, time_to_expiry, volatility, rate, dividend_yield)

    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    t = np.asarray(time_to_expiry, dtype=np.float64)
    r = np.asarray(rate, dtype=np.float64)
    v = np.asarray(volatility, dtype=np.float64)
    q = np.asarray(dividend_yield, dtype=np.float64)

    opt_type = OptionType(option_type)

    # Expiry T == 0 or zero volatility limiting cases
    # Intrinsic value under forward discounting:
    # At T=0: Call = max(S - K, 0), Put = max(K - S, 0)
    # When vol=0 and T > 0: discounted intrinsic based on forward:
    # Call = max(S*e^(-qT) - K*e^(-rT), 0)
    # Put = max(K*e^(-rT) - S*e^(-qT), 0)
    disc_s = s * np.exp(-q * t)
    disc_k = k * np.exp(-r * t)

    vol_sqrt_t = v * np.sqrt(t)
    zero_vol_or_expiry = (vol_sqrt_t <= 0) | (t <= 0)

    d1, d2 = bsm_d1_d2(s, k, t, r, v, q)

    if opt_type == OptionType.CALL:
        standard_price = disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
        zero_vol_price = np.maximum(disc_s - disc_k, 0.0)
    else:
        standard_price = disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)
        zero_vol_price = np.maximum(disc_k - disc_s, 0.0)

    price = np.where(zero_vol_or_expiry, zero_vol_price, standard_price)

    # Ensure precision issues never produce negative option values
    price = np.maximum(price, 0.0)

    if is_scalar:
        return float(price.item())
    return price


def bsm_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
) -> GreeksResult:
    """Compute analytical Black-Scholes-Merton Greeks (Delta, Gamma, Vega, Theta, Rho).

    Conventions:
    - Theta: passage of calendar time convention (dPrice/dt = -dPrice/dT). ATM typically negative.
    - Vega: dPrice/dVol for 1.00 change in volatility (100% vol).
    - Rho: dPrice/dRate for 1.00 change in interest rate (100% rate).
    """
    s = float(spot)
    k = float(strike)
    t = float(time_to_expiry)
    r = float(rate)
    v = float(volatility)
    q = float(dividend_yield)
    opt_type = OptionType(option_type)

    validate_market_inputs(s, k, t, v, r, q)

    # Boundary: T == 0
    if t <= 0:
        if opt_type == OptionType.CALL:
            delta = np.exp(-q * t) if s > k else (0.5 * np.exp(-q * t) if s == k else 0.0)
        else:
            delta = -np.exp(-q * t) if s < k else (-0.5 * np.exp(-q * t) if s == k else 0.0)
        return GreeksResult(
            delta=float(delta),
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            rho=0.0,
            method="analytical",
            diagnostics={"at_expiry": True},
        )

    # Boundary: Volatility == 0
    if v <= 0:
        forward = s * np.exp((r - q) * t)
        if opt_type == OptionType.CALL:
            delta = np.exp(-q * t) if forward > k else 0.0
            rho = t * k * np.exp(-r * t) if forward > k else 0.0
            theta = -(q * s * np.exp(-q * t) - r * k * np.exp(-r * t)) if forward > k else 0.0
        else:
            delta = -np.exp(-q * t) if forward < k else 0.0
            rho = -t * k * np.exp(-r * t) if forward < k else 0.0
            theta = -(-q * s * np.exp(-q * t) + r * k * np.exp(-r * t)) if forward < k else 0.0
        return GreeksResult(
            delta=float(delta),
            gamma=0.0,
            vega=0.0,
            theta=float(theta),
            rho=float(rho),
            method="analytical",
            diagnostics={"zero_volatility": True},
        )

    d1, d2 = bsm_d1_d2(s, k, t, r, v, q)
    d1 = float(d1)
    d2 = float(d2)

    sqrt_t = np.sqrt(t)
    pdf_d1 = norm.pdf(d1)
    disc_s = s * np.exp(-q * t)
    disc_k = k * np.exp(-r * t)

    gamma = float(np.exp(-q * t) * pdf_d1 / (s * v * sqrt_t))
    vega = float(disc_s * pdf_d1 * sqrt_t)

    if opt_type == OptionType.CALL:
        delta = float(np.exp(-q * t) * norm.cdf(d1))
        theta = float(
            -disc_s * pdf_d1 * v / (2.0 * sqrt_t)
            - r * disc_k * norm.cdf(d2)
            + q * disc_s * norm.cdf(d1)
        )
        rho = float(k * t * np.exp(-r * t) * norm.cdf(d2))
    else:
        delta = float(np.exp(-q * t) * (norm.cdf(d1) - 1.0))
        theta = float(
            -disc_s * pdf_d1 * v / (2.0 * sqrt_t)
            + r * disc_k * norm.cdf(-d2)
            - q * disc_s * norm.cdf(-d1)
        )
        rho = float(-k * t * np.exp(-r * t) * norm.cdf(-d2))

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        method="analytical",
        diagnostics={"d1": d1, "d2": d2},
    )
