from __future__ import annotations

from datetime import datetime

import pandas as pd

from again_econ.config import BacktestConfig, ExecutionConfig, WalkforwardConfig
from again_econ.contracts import PositionTarget, SignalRecord
from again_econ.runner import run_backtest
from again_econ.storage import BacktestStorage
from again_econ.ui_adapter import BacktestUIAdapter
from tests.again_econ_test_utils import build_single_symbol_market


def _market_to_frame(market) -> pd.DataFrame:
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


def _build_result(*, label: str, exit_index: int):
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
    return result, _market_to_frame(market)


def test_backtest_ui_adapter_persists_lists_and_loads_runs(tmp_path):
    storage = BacktestStorage(tmp_path / "econ_ui")
    ui = BacktestUIAdapter(storage)
    result, market_frame = _build_result(label="econ_ui_left", exit_index=3)

    run_view = ui.persist_result(
        result,
        mode="exploratory_live",
        preset_name="exploratory",
        methodology_label="exploratory_global_model_replay",
        model_name="Gen6_1",
        config_reference="config/config.yaml",
        requested_universe=("AAA",),
        effective_universe=("AAA",),
        market_data=market_frame,
        market_context={"source_summary": {"fresh_network": 1}, "methodology_note": "nota"},
    )

    assert (storage.run_dir(result.manifest.run_id) / "run_manifest.json").exists()
    assert run_view["summary"]["run_id"] == result.manifest.run_id
    assert not run_view["oos_curve"].empty
    rows = ui.list_runs()
    assert len(rows) == 1
    assert rows[0]["mode"] == "exploratory_live"
    assert rows[0]["effective_universe_size"] == 1
    loaded = ui.load_run_view(result.manifest.run_id)
    assert loaded["manifest"]["run_id"] == result.manifest.run_id
    assert loaded["artifact_audit"]["market_sha256"]


def test_backtest_ui_adapter_compares_runs(tmp_path):
    storage = BacktestStorage(tmp_path / "econ_ui_compare")
    ui = BacktestUIAdapter(storage)
    left_result, left_market = _build_result(label="econ_ui_cmp_left", exit_index=3)
    right_result, right_market = _build_result(label="econ_ui_cmp_right", exit_index=4)

    left_view = ui.persist_result(
        left_result,
        mode="exploratory_live",
        preset_name="exploratory",
        methodology_label="exploratory_global_model_replay",
        model_name="Gen6_1",
        config_reference="config/config.yaml",
        requested_universe=("AAA",),
        effective_universe=("AAA",),
        market_data=left_market,
        market_context={"source_summary": {"fresh_network": 1}},
    )
    right_view = ui.persist_result(
        right_result,
        mode="official_frozen",
        preset_name="strict_frozen",
        methodology_label="frozen_global_model_replay",
        model_name="Gen6_1",
        config_reference="config/config.yaml",
        requested_universe=("AAA",),
        effective_universe=("AAA",),
        market_data=right_market,
        market_context={"source_summary": {"fresh_network": 1}},
    )

    comparison = ui.compare_runs(left_view["manifest"]["run_id"], right_view["manifest"]["run_id"])

    assert comparison["left_run_id"] == left_view["manifest"]["run_id"]
    assert comparison["right_run_id"] == right_view["manifest"]["run_id"]
    assert "total_return" in comparison["summary_delta"]
