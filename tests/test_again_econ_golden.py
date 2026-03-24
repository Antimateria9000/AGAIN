from __future__ import annotations

import json

import pytest

from again_econ.config import BacktestConfig, ExecutionConfig, WalkforwardConfig
from again_econ.runner import run_backtest_from_bundle

from tests.again_econ_test_utils import build_single_symbol_market


def test_again_econ_golden_run_stabilizes_oos_manifest_and_discards(tmp_path):
    market = build_single_symbol_market([10, 10, 10, 11, 12, 13], [10, 10, 10, 11, 12, 13])
    timestamps = market.timestamps()
    bundle_path = tmp_path / "golden_signals.json"
    bundle_path.write_text(
        json.dumps(
            {
                "bundle_version": 1,
                "payload_type": "signal_records",
                "records": [
                    {
                        "instrument_id": "AAA",
                        "decision_timestamp": timestamps[2].isoformat(),
                        "available_at": timestamps[2].isoformat(),
                        "target_state": "long",
                        "score": 0.9,
                    },
                    {
                        "instrument_id": "AAA",
                        "decision_timestamp": timestamps[3].isoformat(),
                        "available_at": timestamps[3].isoformat(),
                        "target_state": "flat",
                        "score": -0.3,
                    },
                    {
                        "instrument_id": "AAA",
                        "decision_timestamp": timestamps[5].isoformat(),
                        "available_at": timestamps[5].isoformat(),
                        "target_state": "long",
                        "score": 0.2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = BacktestConfig(
        walkforward=WalkforwardConfig(train_size=2, test_size=4, step_size=4),
        execution=ExecutionConfig(initial_cash=100.0, allocation_fraction=1.0, allow_fractional_shares=False),
        label="golden_again_econ",
    )

    result = run_backtest_from_bundle(market, config, bundle_path)

    assert len(result.windows) == 1
    assert result.summary_metrics.trade_count == 1
    assert result.summary_metrics.total_return == pytest.approx(0.09)
    assert [point.equity for point in result.oos_curve] == pytest.approx([100.0, 100.0, 109.0, 109.0])
    assert result.windows[0].manifest is not None
    assert result.windows[0].manifest.input_record_count == 3
    assert result.windows[0].manifest.scheduled_signal_count == 2
    assert result.windows[0].manifest.discarded_signal_count == 1
    assert result.windows[0].manifest.discarded_reason_counts == {"no_next_open_available": 1}
