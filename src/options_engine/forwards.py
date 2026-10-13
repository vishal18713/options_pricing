"""Forward estimation and put-call parity utilities.

Calculates forward price F(T) = S * exp((r - q) * T) or estimates synthetic forward
from matched put-call parity quotes:
    C(K) - P(K) = exp(-r * T) * (F - K)
=>  F = K + exp(r * T) * (C(K) - P(K))
"""

import numpy as np
import pandas as pd


def forward_price_theoretical(
    spot: float,
    time_to_expiry: float,
    rate: float,
    dividend_yield: float = 0.0,
) -> float:
    """Theoretical forward price under cost of carry."""
    return float(spot * np.exp((rate - dividend_yield) * time_to_expiry))


def estimate_forward_from_parity(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    time_to_expiry: float,
    rate: float,
    max_strike_distance_pct: float = 0.15,
    spot_estimate: float | None = None,
) -> float | None:
    """Estimate forward price from matched call and put mid-quotes using put-call parity.

    Matches identical strikes within a moneyness window and calculates median forward estimate.
    """
    if len(calls_df) == 0 or len(puts_df) == 0 or time_to_expiry <= 0:
        return None

    # Merge calls and puts on strike
    merged = pd.merge(
        calls_df[["strike", "mid"]].rename(columns={"mid": "call_mid"}),
        puts_df[["strike", "mid"]].rename(columns={"mid": "put_mid"}),
        on="strike",
    )

    if len(merged) == 0:
        return None

    # If spot estimate provided, focus on near-the-money strikes where liquidity and parity hold best
    if spot_estimate is not None and spot_estimate > 0:
        atm_mask = (merged["strike"] >= spot_estimate * (1.0 - max_strike_distance_pct)) & (
            merged["strike"] <= spot_estimate * (1.0 + max_strike_distance_pct)
        )
        subset = merged[atm_mask]
        if len(subset) > 0:
            merged = subset

    # F = K + exp(r * T) * (C - P)
    discount_inv = np.exp(rate * time_to_expiry)
    estimated_forwards = merged["strike"] + discount_inv * (merged["call_mid"] - merged["put_mid"])

    # Robust median to protect against quote noise or bid-ask bounce
    median_fwd = float(np.median(estimated_forwards))
    return median_fwd
