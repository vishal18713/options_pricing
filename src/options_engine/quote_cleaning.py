"""Quote cleaning rules, validation filters, and reason code auditing.

Filters:
- DATA_ZERO_OR_NEGATIVE_BID: bid <= 0
- DATA_CROSSED_MARKET: bid > ask
- DATA_EXCESSIVE_SPREAD: (ask - bid) / mid > max_relative_spread
- DATA_MISSING_VALUE: NaN/None in strike, bid, or ask
- DATA_LOW_LIQUIDITY: volume or open interest below threshold (optional heuristic)
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CleaningFilterConfig:
    """Configuration options for market quote cleaning."""

    require_positive_bid: bool = True
    reject_crossed_market: bool = True
    max_relative_spread: float = 0.50  # 50% relative spread limit
    min_volume: int | None = None
    min_open_interest: int | None = None
    min_strike: float | None = 0.01


@dataclass
class CleaningAuditSummary:
    """Summary of quote cleaning rejection statistics."""

    total_input_quotes: int
    accepted_quotes: int
    rejected_quotes: int
    rejections_by_reason: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_quotes": self.total_input_quotes,
            "accepted_quotes": self.accepted_quotes,
            "rejected_quotes": self.rejected_quotes,
            "rejections_by_reason": self.rejections_by_reason,
            "acceptance_rate_pct": (
                (self.accepted_quotes / self.total_input_quotes * 100.0)
                if self.total_input_quotes > 0
                else 0.0
            ),
        }


def clean_option_quotes(
    quotes_df: pd.DataFrame,
    config: CleaningFilterConfig | None = None,
) -> tuple[pd.DataFrame, CleaningAuditSummary]:
    """Filter raw quotes and record itemized rejection reasons.

    Args:
        quotes_df: DataFrame containing at least ['strike', 'bid', 'ask', 'expiry', 'call_put']
        config: CleaningFilterConfig parameters

    Returns:
        tuple[pd.DataFrame, CleaningAuditSummary]:
            - Cleaned DataFrame containing accepted records with calculated 'mid'
            - Audit summary object recording rejection reasons and counts
    """
    cfg = config or CleaningFilterConfig()

    df = quotes_df.copy()
    total_quotes = len(df)
    rejection_reasons: dict[str, int] = {}

    if total_quotes == 0:
        summary = CleaningAuditSummary(0, 0, 0, {})
        return pd.DataFrame(), summary

    # Ensure required columns exist
    required_cols = ["strike", "bid", "ask", "expiry", "call_put"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from quotes DataFrame.")

    # Initialize rejection reason tracking
    df["rejection_reason"] = None
    df["is_accepted"] = True

    # Filter 1: Missing values
    missing_mask = df["strike"].isna() | df["bid"].isna() | df["ask"].isna()
    if missing_mask.any():
        df.loc[missing_mask & df["is_accepted"], "rejection_reason"] = "DATA_MISSING_VALUE"
        df.loc[missing_mask, "is_accepted"] = False

    # Filter 2: Non-positive strike
    if cfg.min_strike is not None:
        invalid_strike = df["strike"] < cfg.min_strike
        if invalid_strike.any():
            df.loc[invalid_strike & df["is_accepted"], "rejection_reason"] = "DATA_INVALID_STRIKE"
            df.loc[invalid_strike, "is_accepted"] = False

    # Filter 3: Zero or negative bid
    if cfg.require_positive_bid:
        zero_bid = df["bid"] <= 0.0
        if zero_bid.any():
            df.loc[zero_bid & df["is_accepted"], "rejection_reason"] = "DATA_ZERO_OR_NEGATIVE_BID"
            df.loc[zero_bid, "is_accepted"] = False

    # Filter 4: Crossed market (bid > ask)
    if cfg.reject_crossed_market:
        crossed = df["bid"] > df["ask"]
        if crossed.any():
            df.loc[crossed & df["is_accepted"], "rejection_reason"] = "DATA_CROSSED_MARKET"
            df.loc[crossed, "is_accepted"] = False

    # Calculate mid price and relative spread
    mid = 0.5 * (df["bid"] + df["ask"])
    df["mid"] = mid

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_spread = np.where(mid > 0, (df["ask"] - df["bid"]) / mid, np.inf)
    df["relative_spread"] = rel_spread

    # Filter 5: Excessive relative spread
    if cfg.max_relative_spread is not None:
        wide_spread = df["relative_spread"] > cfg.max_relative_spread
        if wide_spread.any():
            df.loc[wide_spread & df["is_accepted"], "rejection_reason"] = "DATA_EXCESSIVE_SPREAD"
            df.loc[wide_spread, "is_accepted"] = False

    # Filter 6: Low liquidity (volume / open interest)
    if cfg.min_volume is not None and "volume" in df.columns:
        low_vol = df["volume"].fillna(0) < cfg.min_volume
        if low_vol.any():
            df.loc[low_vol & df["is_accepted"], "rejection_reason"] = "DATA_LOW_VOLUME"
            df.loc[low_vol, "is_accepted"] = False

    if cfg.min_open_interest is not None and "open_interest" in df.columns:
        low_oi = df["open_interest"].fillna(0) < cfg.min_open_interest
        if low_oi.any():
            df.loc[low_oi & df["is_accepted"], "rejection_reason"] = "DATA_LOW_OPEN_INTEREST"
            df.loc[low_oi, "is_accepted"] = False

    # Aggregate rejection reasons
    rejected_rows = df[~df["is_accepted"]]
    for reason, count in rejected_rows["rejection_reason"].value_counts().items():
        rejection_reasons[str(reason)] = int(count)

    accepted_df = (
        df[df["is_accepted"]]
        .drop(columns=["rejection_reason", "is_accepted"])
        .reset_index(drop=True)
    )
    accepted_count = len(accepted_df)
    rejected_count = total_quotes - accepted_count

    summary = CleaningAuditSummary(
        total_input_quotes=total_quotes,
        accepted_quotes=accepted_count,
        rejected_quotes=rejected_count,
        rejections_by_reason=rejection_reasons,
    )

    return accepted_df, summary
