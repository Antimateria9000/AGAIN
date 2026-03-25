import json

import pandas as pd
import pytest

from again_benchmark.contracts import BenchmarkDefinition
from again_benchmark.runner import BenchmarkRunner
from again_benchmark.storage import BenchmarkStorage
from again_benchmark.errors import BenchmarkStorageError, BenchmarkValidationError
from tests.helpers.again_benchmark import FakeBenchmarkAdapter, build_definition, build_market_data, build_storage_root


def test_snapshot_creation_persists_parquet_manifest_and_checksum(tmp_path):
    storage = BenchmarkStorage(build_storage_root(tmp_path))
    definition = build_definition()
    runner = BenchmarkRunner(storage, FakeBenchmarkAdapter(build_market_data()))

    snapshot_manifest = runner.create_frozen_snapshot(definition, as_of_timestamp=pd.Timestamp("2024-01-31").to_pydatetime())

    snapshot_dir = storage.snapshot_dir(snapshot_manifest.snapshot_id)
    assert (snapshot_dir / "market_data.parquet").exists()
    assert (snapshot_dir / "market_data.parquet.sha256").exists()
    assert (snapshot_dir / "snapshot_manifest.json").exists()


def test_corrupted_snapshot_checksum_is_rejected(tmp_path):
    storage = BenchmarkStorage(build_storage_root(tmp_path))
    definition = build_definition()
    runner = BenchmarkRunner(storage, FakeBenchmarkAdapter(build_market_data()))

    snapshot_manifest = runner.create_frozen_snapshot(definition, as_of_timestamp=pd.Timestamp("2024-01-31").to_pydatetime())
    snapshot_file = storage.snapshot_dir(snapshot_manifest.snapshot_id) / "market_data.parquet"
    snapshot_file.write_bytes(b"corrupto")

    with pytest.raises(BenchmarkStorageError):
        storage.load_snapshot(snapshot_manifest.snapshot_id)


def test_manifest_serialization_and_deserialization_roundtrip(tmp_path):
    storage = BenchmarkStorage(build_storage_root(tmp_path))
    definition = build_definition()
    storage.write_definition(definition)

    loaded = storage.load_definition(definition.definition_id)
    catalog = storage.list_definitions()

    assert loaded == definition
    assert storage.definition_path(definition.definition_id).suffix == ".yaml"
    assert len(catalog) == 1
    assert catalog[0].updated_at.tzinfo is None


def test_manifest_corrupt_run_json_fails_to_load(tmp_path):
    storage = BenchmarkStorage(build_storage_root(tmp_path))
    definition = build_definition()
    runner = BenchmarkRunner(storage, FakeBenchmarkAdapter(build_market_data()))
    bundle = runner.run_live(definition, as_of_timestamp=pd.Timestamp("2024-01-31").to_pydatetime())

    run_manifest_path = storage.run_dir(bundle.manifest.run_id) / "run_manifest.json"
    run_manifest_path.write_text(json.dumps({"run_id": bundle.manifest.run_id}), encoding="utf-8")

    with pytest.raises(Exception):
        storage.load_run_bundle(bundle.manifest.run_id)


def test_frozen_snapshot_rejects_incompatible_definition(tmp_path):
    storage = BenchmarkStorage(build_storage_root(tmp_path))
    definition = build_definition()
    runner = BenchmarkRunner(storage, FakeBenchmarkAdapter(build_market_data()))
    snapshot_manifest = runner.create_frozen_snapshot(definition, as_of_timestamp=pd.Timestamp("2024-01-31").to_pydatetime())

    incompatible = BenchmarkDefinition(
        benchmark_id="other_benchmark",
        benchmark_version=1,
        definition_id="other_definition",
        label="Otra definicion",
        tickers=("AAA",),
        horizon=2,
    )

    with pytest.raises(BenchmarkValidationError):
        runner.run_frozen_from_snapshot(incompatible, snapshot_manifest.snapshot_id)


def test_run_bundle_persists_artifact_checksums_in_manifest(tmp_path):
    storage = BenchmarkStorage(build_storage_root(tmp_path))
    definition = build_definition()
    runner = BenchmarkRunner(storage, FakeBenchmarkAdapter(build_market_data()))

    bundle = runner.run_live(definition, as_of_timestamp=pd.Timestamp("2024-01-31").to_pydatetime())
    loaded = storage.load_run_bundle(bundle.manifest.run_id)

    assert loaded.manifest.summary_sha256
    assert loaded.manifest.metrics_sha256
    assert loaded.manifest.ticker_results_sha256
    assert loaded.manifest.plot_payload_sha256


def test_snapshot_is_immutable_when_rewritten_with_same_id_and_different_manifest(tmp_path):
    storage = BenchmarkStorage(build_storage_root(tmp_path))
    definition = build_definition()
    runner = BenchmarkRunner(storage, FakeBenchmarkAdapter(build_market_data()))

    snapshot_manifest = runner.create_frozen_snapshot(definition, as_of_timestamp=pd.Timestamp("2024-01-31").to_pydatetime())
    manifest_path = storage.snapshot_dir(snapshot_manifest.snapshot_id) / "snapshot_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["source_adapter"] = "other_adapter"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkStorageError):
        storage.write_snapshot(build_market_data(), snapshot_manifest)
