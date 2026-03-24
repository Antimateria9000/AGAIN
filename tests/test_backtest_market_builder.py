from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from app.backtest_market_builder import MarketFrameBuilder
from scripts.data_fetcher import FreshDataRequiredError
from scripts.utils.universe_integrity import UniverseIntegrityReport, UniverseTickerIntegrity


class FakeFetcher:
    def __init__(self, frame: pd.DataFrame, report: UniverseIntegrityReport):
        self.frame = frame
        self.report = report

    def fetch_many_stocks_with_report(self, tickers, start_date, end_date):
        del tickers, start_date, end_date
        return self.frame.copy(), self.report


def _build_report(*, source: str, fallback_tickers: list[str] | None = None) -> UniverseIntegrityReport:
    ticker_integrity = {
        "AAA": UniverseTickerIntegrity(
            ticker="AAA",
            final_status="ok" if source == "fresh_network" else "fallback_cache",
            source=source,
            rows_obtained=2,
            trainable=True,
            used_fallback=source == "local_cache",
            freshness="cached" if source == "local_cache" else "fresh",
        )
    }
    return UniverseIntegrityReport(
        requested_tickers=["AAA"],
        successful_tickers=["AAA"],
        discarded_tickers=[],
        discarded_details={},
        ticker_integrity=ticker_integrity,
        fresh_tickers=[] if source == "local_cache" else ["AAA"],
        fallback_tickers=fallback_tickers or ([] if source == "fresh_network" else ["AAA"]),
        decision="DEGRADED_ALLOWED" if source == "local_cache" else "CONTINUE_CLEAN",
        training_allowed=True,
        degraded=source == "local_cache",
    )


def test_market_frame_builder_builds_market_frame_and_detects_critical_gap():
    frame = pd.DataFrame(
        [
            {"Date": "2024-01-01", "Open": 10, "High": 11, "Low": 9, "Close": 10, "Volume": 1000, "Ticker": "AAA", "Sector": "Unknown"},
            {"Date": "2024-01-20", "Open": 12, "High": 13, "Low": 11, "Close": 12, "Volume": 1000, "Ticker": "AAA", "Sector": "Unknown"},
        ]
    )
    builder = MarketFrameBuilder(FakeFetcher(frame, _build_report(source="fresh_network")), max_gap_days=10)

    result = builder.build(
        ["AAA"],
        datetime(2024, 1, 1),
        datetime(2024, 1, 31),
        allow_local_fallback=True,
    )

    assert result.market_frame.instruments() == ("AAA",)
    assert result.source_summary == {"fresh_network": 1}
    assert result.provenance_by_ticker["AAA"]["source"] == "fresh_network"
    assert any("AAA presenta un hueco temporal" in warning for warning in result.warnings)


def test_market_frame_builder_rejects_local_fallback_when_mode_forbids_it():
    frame = pd.DataFrame(
        [
            {"Date": "2024-01-01", "Open": 10, "High": 11, "Low": 9, "Close": 10, "Volume": 1000, "Ticker": "AAA", "Sector": "Unknown"},
            {"Date": "2024-01-02", "Open": 11, "High": 12, "Low": 10, "Close": 11, "Volume": 1000, "Ticker": "AAA", "Sector": "Unknown"},
        ]
    )
    builder = MarketFrameBuilder(FakeFetcher(frame, _build_report(source="local_cache", fallback_tickers=["AAA"])))

    with pytest.raises(FreshDataRequiredError):
        builder.build(
            ["AAA"],
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
            allow_local_fallback=False,
        )
