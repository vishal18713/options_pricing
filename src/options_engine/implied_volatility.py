"""Implied volatility calculation using Brent's root-finding method.

Includes:
- Theoretical no-arbitrage price bounds checking
- Structured IVResult reporting with explicit diagnostic failure reason codes
- Never returning numerical zero on failed solves or out-of-bounds quotes
- Round-trip accuracy guarantees
"""

import numpy as np
from scipy.optimize import brentq

from options_engine.black_scholes import bsm_price
from options_engine.types import IVResult, OptionType
from options_engine.validation import validate_market_inputs


def check_price_bounds(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
) -> tuple[bool, float, float, str]:
    """Check whether market price satisfies European no-arbitrage lower and upper bounds.

    Bounds:
    - Call: max(0, S*e^(-qT) - K*e^(-rT)) <= Call <= S*e^(-qT)
    - Put:  max(0, K*e^(-rT) - S*e^(-qT)) <= Put <= K*e^(-rT)

    Returns:
        tuple[bool, float, float, str]: (is_valid, lower_bound, upper_bound, reason)
    """
    opt_type = OptionType(option_type)
    disc_s = spot * np.exp(-dividend_yield * time_to_expiry)
    disc_k = strike * np.exp(-rate * time_to_expiry)

    if opt_type == OptionType.CALL:
        lower_bound = max(0.0, disc_s - disc_k)
        upper_bound = disc_s
    else:
        lower_bound = max(0.0, disc_k - disc_s)
        upper_bound = disc_k

    # Floating point tolerance
    eps = 1e-7
    if price < (lower_bound - eps):
        return False, lower_bound, upper_bound, "IV_PRICE_BELOW_LOWER_BOUND"
    if price > (upper_bound + eps):
        return False, lower_bound, upper_bound, "IV_PRICE_ABOVE_UPPER_BOUND"

    return True, lower_bound, upper_bound, "VALID"


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
    sigma_min: float = 1e-5,
    sigma_max: float = 5.0,
    max_iterations: int = 200,
    tol: float = 1e-8,
) -> IVResult:
    """Solve for implied volatility using bracketed Brent root finder.

    Args:
        price: Observed option market price (e.g. quote midpoint)
        spot: Underlying spot price S > 0
        strike: Strike price K > 0
        time_to_expiry: Time T in years
        rate: Continuously compounded risk-free rate r
        option_type: "call" or "put"
        dividend_yield: Continuous dividend yield q
        sigma_min: Lower bound of search bracket (default 1e-5)
        sigma_max: Upper bound of search bracket (default 5.0)
        max_iterations: Maximum Brent iterations
        tol: Convergence tolerance

    Returns:
        IVResult containing implied_volatility (or None if failed), status, residual, and diagnostics.
    """
    opt_type = OptionType(option_type)

    try:
        validate_market_inputs(spot, strike, time_to_expiry, 0.1, rate, dividend_yield)
    except Exception as e:
        return IVResult(
            implied_volatility=None,
            status="FAILED",
            reason_code=f"INPUT_ERROR: {e}",
        )

    if price <= 0:
        return IVResult(
            implied_volatility=None,
            status="FAILED",
            reason_code="IV_NON_POSITIVE_PRICE",
        )

    if time_to_expiry <= 0:
        return IVResult(
            implied_volatility=None,
            status="FAILED",
            reason_code="IV_AT_EXPIRY",
        )

    # Verify no-arbitrage theoretical bounds
    is_valid, lower_b, upper_b, reason = check_price_bounds(
        price, spot, strike, time_to_expiry, rate, opt_type, dividend_yield
    )
    if not is_valid:
        return IVResult(
            implied_volatility=None,
            status="FAILED",
            bracket=(lower_b, upper_b),
            reason_code=reason,
        )

    # Objective function: bsm_price(sigma) - target_price
    def objective(vol: float) -> float:
        return bsm_price(spot, strike, time_to_expiry, rate, vol, opt_type, dividend_yield) - price

    f_min = objective(sigma_min)
    f_max = objective(sigma_max)

    if f_min * f_max > 0:
        # Not bracketed within [sigma_min, sigma_max]
        if abs(f_min) < tol:
            return IVResult(
                implied_volatility=sigma_min,
                status="CONVERGED",
                iterations=0,
                residual=float(f_min),
                bracket=(sigma_min, sigma_max),
            )
        if abs(f_max) < tol:
            return IVResult(
                implied_volatility=sigma_max,
                status="CONVERGED",
                iterations=0,
                residual=float(f_max),
                bracket=(sigma_min, sigma_max),
            )
        return IVResult(
            implied_volatility=None,
            status="FAILED",
            bracket=(sigma_min, sigma_max),
            residual=float(min(abs(f_min), abs(f_max))),
            reason_code="IV_NOT_BRACKETED",
        )

    try:
        sol, r_res = brentq(
            objective,
            sigma_min,
            sigma_max,
            maxiter=max_iterations,
            xtol=tol,
            full_output=True,
            disp=False,
        )
        if r_res.converged:
            res_val = float(objective(sol))
            return IVResult(
                implied_volatility=float(sol),
                status="CONVERGED",
                iterations=r_res.iterations,
                residual=res_val,
                bracket=(sigma_min, sigma_max),
            )
        else:
            return IVResult(
                implied_volatility=None,
                status="FAILED",
                iterations=r_res.iterations,
                bracket=(sigma_min, sigma_max),
                reason_code="IV_MAX_ITERATIONS_EXCEEDED",
            )
    except Exception as e:
        return IVResult(
            implied_volatility=None,
            status="FAILED",
            bracket=(sigma_min, sigma_max),
            reason_code=f"IV_SOLVER_EXCEPTION: {e}",
        )
