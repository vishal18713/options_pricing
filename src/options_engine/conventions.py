"""Conventions, unit conversions, and scaling logic."""

import numpy as np


def year_fraction_act365(days: int | float | np.ndarray) -> float | np.ndarray:
    """Calculate ACT/365 year fraction."""
    return days / 365.0


def scale_theta_per_day(theta_annual: float | np.ndarray) -> float | np.ndarray:
    """Scale passage-of-time Theta from 1 year to 1 calendar day (ACT/365)."""
    return theta_annual / 365.0


def scale_vega_per_percentage_point(vega_1_0: float | np.ndarray) -> float | np.ndarray:
    """Scale Vega from 1.00 vol unit to 1 percentage point (0.01 vol)."""
    return vega_1_0 * 0.01


def scale_rho_per_basis_point(rho_1_0: float | np.ndarray) -> float | np.ndarray:
    """Scale Rho from 1.00 rate unit to 1 basis point (0.0001 rate)."""
    return rho_1_0 * 0.0001
