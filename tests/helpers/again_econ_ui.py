from __future__ import annotations

from datetime import datetime

import pandas as pd

from again_econ.config import BacktestConfig, ExecutionConfig, WalkforwardConfig
from again_econ.contracts import PositionTarget, SignalRecord
from again_econ.runner import run_backtest
from tests.helpers.again_econ import build_single_symbol_market


def market_to_frame(market) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": bar.timestamp,
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Volume": bar.volume,
                "Ticker": bar.instrument_id,
                "Sector": "Unknown",
            }
            for bar in market.bars
        ]
    )


def build_result(*, label: str, exit_index: int):
    market = build_single_symbol_market([10, 10, 10, 11, 12], [10, 10, 10, 11, 12], start=datetime(2024, 1, 1))
    timestamps = market.timestamps()
    config = BacktestConfig(
        walkforward=WalkforwardConfig(train_size=2, test_size=3, step_size=3),
        execution=ExecutionConfig(initial_cash=100.0, allocation_fraction=1.0, allow_fractional_shares=False),
        label=label,
    )
    signals = (
        SignalRecord(
            instrument_id="AAA",
            decision_timestamp=timestamps[2],
            available_at=timestamps[2],
            target_state=PositionTarget.LONG,
        ),
        SignalRecord(
            instrument_id="AAA",
            decision_timestamp=timestamps[exit_index],
            available_at=timestamps[exit_index],
            target_state=PositionTarget.FLAT,
        ),
    )
    result = run_backtest(market, config, signals=signals, adapter_name="test_ui")
    return result, market_to_frame(market)
