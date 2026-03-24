import sys

import pandas as pd
import pytest

from again_benchmark.contracts import BenchmarkMode
from again_benchmark.runner import BenchmarkRunner
from again_benchmark.storage import BenchmarkStorage
from tests.again_benchmark_test_utils import FakeBenchmarkAdapter, build_definition, build_market_data, build_storage_root


def test_frozen_benchmark_runs_from_materialized_snapshot_and_is_rerunnable(tmp_path):
    storage = BenchmarkStorage(build_storage_root(tmp_path))
    definition = build_definition()
    runner = BenchmarkRunner(storage, FakeBenchmarkAdapter(build_market_data(), bias=0.0))

    snapshot_manifest = runner.create_frozen_snapshot(definition, as_of_timestamp=pd.Timestamp("2024-01-31").to_pydatetime())
    first = runner.run_frozen_from_snapshot(definition, snapshot_manifest.snapshot_id)
    second = runner.rerun_from_run_id(first.manifest.run_id)

    assert first.manifest.mode == BenchmarkMode.FROZEN
    assert first.manifest.snapshot_id == snapshot_manifest.snapshot_id
    assert first.manifest.run_id == second.manifest.run_id
    assert first.summary.metrics == second.summary.metrics


def test_live_and_frozen_modes_are_explicitly_distinguished(tmp_path):
    storage = BenchmarkStorage(build_storage_root(tmp_path))
    definition = build_definition()
    runner = BenchmarkRunner(storage, FakeBenchmarkAdapter(build_market_data(), bias=1.0))

    live = runner.run_live(definition, as_of_timestamp=pd.Timestamp("2024-01-31").to_pydatetime())
    snapshot_manifest = runner.create_frozen_snapshot(definition, as_of_timestamp=pd.Timestamp("2024-01-31").to_pydatetime())
    frozen = runner.run_frozen_from_snapshot(definition, snapshot_manifest.snapshot_id)

    assert live.manifest.mode == BenchmarkMode.LIVE
    assert live.manifest.snapshot_id is None
    assert frozen.manifest.mode == BenchmarkMode.FROZEN
    assert frozen.manifest.snapshot_id == snapshot_manifest.snapshot_id


def test_runner_is_decoupled_from_app_runtime_with_fake_adapter(tmp_path):
    storage = BenchmarkStorage(build_storage_root(tmp_path))
    definition = build_definition()
    runner = BenchmarkRunner(storage, FakeBenchmarkAdapter(build_market_data(), bias=0.0))

    bundle = runner.run_live(definition, as_of_timestamp=pd.Timestamp("2024-01-31").to_pydatetime())

    assert bundle.summary.metrics["RMSE"] == pytest.approx(0.0)
    assert "streamlit" not in sys.modules


def test_golden_metrics_fixture_is_stable_for_zero_bias_predictions(tmp_path):
    storage = BenchmarkStorage(build_storage_root(tmp_path))
    definition = build_definition()
    runner = BenchmarkRunner(storage, FakeBenchmarkAdapter(build_market_data(), bias=0.0))

    bundle = runner.run_live(definition, as_of_timestamp=pd.Timestamp("2024-01-31").to_pydatetime())

    assert bundle.summary.metrics == {"MAPE": 0.0, "MAE": 0.0, "RMSE": 0.0, "DirAcc": 100.0}
