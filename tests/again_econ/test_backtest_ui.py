from __future__ import annotations

from again_econ.storage import BacktestStorage
from again_econ.ui_adapter import BacktestUIAdapter
from tests.helpers.again_econ_ui import build_result


def test_backtest_ui_adapter_persists_lists_and_loads_runs(tmp_path):
    storage = BacktestStorage(tmp_path / "econ_ui")
    ui = BacktestUIAdapter(storage)
    result, market_frame = build_result(label="econ_ui_left", exit_index=3)

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
    left_result, left_market = build_result(label="econ_ui_cmp_left", exit_index=3)
    right_result, right_market = build_result(label="econ_ui_cmp_right", exit_index=4)

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
