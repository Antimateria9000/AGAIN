from __future__ import annotations

import json

import pytest

from again_econ.config import BacktestConfig, ExecutionConfig, SignalConfig, WalkforwardConfig
from again_econ.runner import run_backtest_from_bundle

from tests.again_econ_test_utils import build_single_symbol_market


def test_backtest_integration_runs_end_to_end_from_again_bundle(tmp_path):
    market = build_single_symbol_market([10, 10, 11, 12, 12], [10, 10, 12, 12, 12])
    timestamps = market.timestamps()
    bundle_path = tmp_path / "again_forecasts.json"
    bundle_path.write_text(
        json.dumps(
            {
                "bundle_version": 1,
                "payload_type": "forecast_records",
                "records": [
                    {
                        "instrument_id": "AAA",
                        "decision_timestamp": timestamps[1].isoformat(),
                        "available_at": timestamps[1].isoformat(),
                        "target_kind": "return",
                        "value": 0.05,
                    },
                    {
                        "instrument_id": "AAA",
                        "decision_timestamp": timestamps[2].isoformat(),
                        "available_at": timestamps[2].isoformat(),
                        "target_kind": "return",
                        "value": -0.01,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = BacktestConfig(
        walkforward=WalkforwardConfig(train_size=2, test_size=3, step_size=3),
        signal=SignalConfig(long_threshold=0.0),
        execution=ExecutionConfig(initial_cash=100.0, allocation_fraction=1.0, allow_fractional_shares=False),
        label="integration_test",
    )

    result = run_backtest_from_bundle(market, config, bundle_path)

    assert result.manifest.adapter_name == "again_json_bundle"
    assert len(result.windows) == 1
    assert result.summary_metrics.trade_count == 1
    assert result.windows[0].trades[0].net_pnl == 9.0
    assert result.summary_metrics.total_return == pytest.approx(0.09)
