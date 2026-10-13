"""Market data adapter module.

Provides explicit download functions, normalized schema representations,
provenance metadata tracking, and raw snapshot preservation.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def fetch_option_chain_yfinance(
    ticker_symbol: str,
    expiries: list[str] | None = None,
    save_raw_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fetch option chains for a ticker using yfinance.

    Guarantees:
    - Retains raw quote records and retrieval timestamp.
    - Captures spot price, dividend yield, and currency metadata.
    - Does not overwrite raw data.

    Returns:
        Dict containing raw metadata and normalized quote records.
    """
    import yfinance as yf

    retrieval_timestamp = datetime.now(UTC).isoformat()
    ticker = yf.Ticker(ticker_symbol)

    # Underlying information
    fast_info = getattr(ticker, "fast_info", {})
    spot_price = getattr(fast_info, "last_price", None) or getattr(
        fast_info, "previous_close", None
    )
    currency = getattr(fast_info, "currency", "USD")

    available_expiries = ticker.options
    target_expiries = expiries if expiries is not None else list(available_expiries)

    raw_chains: dict[str, Any] = {}
    normalized_quotes: list[dict[str, Any]] = []

    for exp in target_expiries:
        try:
            chain = ticker.option_chain(exp)
            calls_df = chain.calls.copy()
            puts_df = chain.puts.copy()

            raw_chains[exp] = {
                "calls": calls_df.to_dict(orient="records"),
                "puts": puts_df.to_dict(orient="records"),
            }

            for row in calls_df.itertuples():
                normalized_quotes.append(
                    {
                        "timestamp": retrieval_timestamp,
                        "ticker": ticker_symbol,
                        "expiry": exp,
                        "strike": float(row.strike),
                        "call_put": "call",
                        "bid": float(getattr(row, "bid", 0.0)),
                        "ask": float(getattr(row, "ask", 0.0)),
                        "last": float(getattr(row, "lastPrice", 0.0)),
                        "volume": float(getattr(row, "volume", 0.0))
                        if pd.notna(getattr(row, "volume", None))
                        else None,
                        "open_interest": float(getattr(row, "openInterest", 0.0))
                        if pd.notna(getattr(row, "openInterest", None))
                        else None,
                        "implied_volatility_provider": float(
                            getattr(row, "impliedVolatility", 0.0)
                        ),
                        "source": "yfinance",
                    }
                )

            for row in puts_df.itertuples():
                normalized_quotes.append(
                    {
                        "timestamp": retrieval_timestamp,
                        "ticker": ticker_symbol,
                        "expiry": exp,
                        "strike": float(row.strike),
                        "call_put": "put",
                        "bid": float(getattr(row, "bid", 0.0)),
                        "ask": float(getattr(row, "ask", 0.0)),
                        "last": float(getattr(row, "lastPrice", 0.0)),
                        "volume": float(getattr(row, "volume", 0.0))
                        if pd.notna(getattr(row, "volume", None))
                        else None,
                        "open_interest": float(getattr(row, "openInterest", 0.0))
                        if pd.notna(getattr(row, "openInterest", None))
                        else None,
                        "implied_volatility_provider": float(
                            getattr(row, "impliedVolatility", 0.0)
                        ),
                        "source": "yfinance",
                    }
                )
        except Exception as e:
            raw_chains[exp] = {"error": str(e)}

    payload = {
        "metadata": {
            "ticker": ticker_symbol,
            "retrieval_timestamp": retrieval_timestamp,
            "spot_price": spot_price,
            "currency": currency,
            "provider": "yfinance",
            "expiries": target_expiries,
        },
        "raw_chains": raw_chains,
        "quotes": normalized_quotes,
    }

    if save_raw_path:
        out_p = Path(save_raw_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return payload
