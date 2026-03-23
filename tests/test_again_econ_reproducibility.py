from __future__ import annotations

import json

from again_econ.config import BacktestConfig, ExecutionConfig, WalkforwardConfig
from again_econ.runner import run_backtest_from_bundle

from tests.again_econ_test_utils import build_single_symbol_market


def test_backtest_manifest_and_outputs_are_reproducible(tmp_path):
    market = build_single_symbol_market([10, 10, 10, 11, 12], [10, 10, 10, 11, 12])
    timestamps = market.timestamps()
    bundle_path = tmp_path / "repro_forecasts.json"
    payload = {
        "bundle_version": 1,
        "payload_type": "forecast_records",
        "records": [
            {
                "instrument_id": "AAA",
                "decision_timestamp": timestamps[2].isoformat(),
                "available_at": timestamps[2].isoformat(),
                "target_kind": "return",
                "value": 0.05,
            },
            {
                "instrument_id": "AAA",
                "decision_timestamp": timestamps[3].isoformat(),
                "available_at": timestamps[3].isoformat(),
                "target_kind": "return",
                "value": -0.01,
            },
        ],
    }
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    config = BacktestConfig(
        walkforward=WalkforwardConfig(train_size=2, test_size=3, step_size=3),
        execution=ExecutionConfig(initial_cash=100.0, allocation_fraction=1.0, allow_fractional_shares=False),
        label="repro_test",
    )

    first = run_backtest_from_bundle(market, config, bundle_path)
    second = run_backtest_from_bundle(market, config, bundle_path)

    assert first.manifest.run_id == second.manifest.run_id
    assert first.manifest.input_fingerprint == second.manifest.input_fingerprint
    assert first.summary_metrics == second.summary_metrics
    assert first.oos_curve == second.oos_curve
    assert first.windows[0].trades == second.windows[0].trades
