"""Unit tests for Phase 5: Market data ingestion and quote quality cleaning."""

import json
from pathlib import Path

import pandas as pd
import pytest

from options_engine.quote_cleaning import CleaningFilterConfig, clean_option_quotes


@pytest.fixture
def sample_fixture_data():
    fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_chain.json"
    with open(fixture_path, encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data["quotes"])


class TestMarketDataAndQuoteQuality:
    def test_cleaning_with_reason_codes_audit(self, sample_fixture_data):
        cfg = CleaningFilterConfig(
            require_positive_bid=True,
            reject_crossed_market=True,
            max_relative_spread=0.50,
        )

        clean_df, audit = clean_option_quotes(sample_fixture_data, cfg)

        # Total quotes in fixture = 10
        assert audit.total_input_quotes == 10

        # We know:
        # Strike 520 has bid=0 -> DATA_ZERO_OR_NEGATIVE_BID
        # Strike 530 has bid=2.50 > ask=2.00 -> DATA_CROSSED_MARKET
        # Strike 540 has bid=0.05, ask=0.80 -> rel_spread = (0.75)/0.425 = 1.76 > 0.50 -> DATA_EXCESSIVE_SPREAD
        # Strike 550 has bid=None -> DATA_MISSING_VALUE
        assert audit.rejected_quotes == 4
        assert audit.accepted_quotes == 6
        assert len(clean_df) == 6

        reasons = audit.rejections_by_reason
        assert reasons.get("DATA_ZERO_OR_NEGATIVE_BID") == 1
        assert reasons.get("DATA_CROSSED_MARKET") == 1
        assert reasons.get("DATA_EXCESSIVE_SPREAD") == 1
        assert reasons.get("DATA_MISSING_VALUE") == 1

        # Check calculated columns
        assert "mid" in clean_df.columns
        assert "relative_spread" in clean_df.columns
        assert (clean_df["bid"] > 0).all()
        assert (clean_df["bid"] <= clean_df["ask"]).all()
        assert (clean_df["relative_spread"] <= 0.50).all()

    def test_audit_summary_dict(self, sample_fixture_data):
        clean_df, audit = clean_option_quotes(sample_fixture_data)
        d = audit.to_dict()
        assert "total_input_quotes" in d
        assert "accepted_quotes" in d
        assert "acceptance_rate_pct" in d
        assert d["acceptance_rate_pct"] == 60.0

    def test_empty_dataframe_handling(self):
        empty_df = pd.DataFrame()
        clean_df, audit = clean_option_quotes(empty_df)
        assert len(clean_df) == 0
        assert audit.total_input_quotes == 0
        assert audit.accepted_quotes == 0

    def test_missing_required_columns_raises(self):
        bad_df = pd.DataFrame({"some_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="Required column"):
            clean_option_quotes(bad_df)
