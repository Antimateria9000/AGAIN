import pandas as pd

from again_benchmark.storage import BenchmarkStorage
from again_benchmark.ui_adapter import BenchmarkUIAdapter
from again_benchmark.runner import BenchmarkRunner
from tests.again_benchmark_test_utils import FakeBenchmarkAdapter, build_definition, build_market_data, build_storage_root


def test_ui_adapter_prepares_runs_and_plot_payloads(tmp_path):
    storage = BenchmarkStorage(build_storage_root(tmp_path))
    runner = BenchmarkRunner(storage, FakeBenchmarkAdapter(build_market_data(), bias=1.0))
    ui = BenchmarkUIAdapter(storage, runner)
    definition = build_definition()
    storage.write_definition(definition)

    run_bundle = ui.run_live(definition.definition_id, as_of_timestamp=pd.Timestamp("2024-01-31").to_pydatetime())
    run_view = ui.load_run_view(run_bundle.manifest.run_id)

    assert run_view["manifest"].run_id == run_bundle.manifest.run_id
    assert "AAA" in run_view["plot_payload"]["tickers"]
    rows = ui.list_runs()
    assert len(rows) == 1
    assert rows[0]["validation_state"] == "live_exploratory"
    assert rows[0]["effective_universe_size"] == 2


def test_run_comparison_produces_summary_and_per_ticker_deltas(tmp_path):
    storage = BenchmarkStorage(build_storage_root(tmp_path))
    definition = build_definition()
    storage.write_definition(definition)
    left_runner = BenchmarkRunner(storage, FakeBenchmarkAdapter(build_market_data(), bias=0.0))
    left_ui = BenchmarkUIAdapter(storage, left_runner)
    right_runner = BenchmarkRunner(storage, FakeBenchmarkAdapter(build_market_data(), bias=2.0))
    right_ui = BenchmarkUIAdapter(storage, right_runner)

    left = left_ui.run_live(definition.definition_id, as_of_timestamp=pd.Timestamp("2024-01-31").to_pydatetime())
    right = right_ui.run_live(definition.definition_id, as_of_timestamp=pd.Timestamp("2024-02-01").to_pydatetime())
    comparison = right_ui.compare_runs(left.manifest.run_id, right.manifest.run_id)

    assert comparison.benchmark_id == definition.benchmark_id
    assert "RMSE" in comparison.summary_delta
    assert any(item["ticker"] == "AAA" for item in comparison.ticker_deltas)
