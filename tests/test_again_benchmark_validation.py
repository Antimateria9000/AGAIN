from datetime import datetime

import pytest

from again_benchmark.contracts import BenchmarkDefinition, BenchmarkMode, BenchmarkRunManifest
from again_benchmark.errors import BenchmarkValidationError
from again_benchmark.validation import validate_definition, validate_run_manifest


def test_validate_definition_rejects_invalid_benchmark_id():
    definition = BenchmarkDefinition(
        benchmark_id="benchmark invalido",
        benchmark_version=1,
        definition_id="def-1",
        label="Invalido",
        tickers=("AAA",),
        horizon=2,
    )

    with pytest.raises(BenchmarkValidationError):
        validate_definition(definition)


def test_validate_definition_rejects_empty_universe():
    with pytest.raises(BenchmarkValidationError):
        BenchmarkDefinition(
            benchmark_id="valid_id",
            benchmark_version=1,
            definition_id="def-1",
            label="Sin universo",
            tickers=(),
            horizon=2,
        )


def test_validate_run_manifest_rejects_incoherent_split_date():
    manifest = BenchmarkRunManifest(
        run_id="run-1",
        benchmark_id="bench",
        benchmark_version=1,
        definition_id="def-1",
        mode=BenchmarkMode.LIVE,
        created_at=datetime(2024, 1, 5),
        as_of_timestamp=datetime(2024, 1, 5),
        split_date=datetime(2024, 1, 6),
        tickers=("AAA",),
        effective_universe=("AAA",),
        horizon=2,
        metrics=("MAPE",),
        model_name="model",
        profile_path=None,
        config_fingerprint=None,
        model_sha256=None,
        normalizers_sha256=None,
        dataset_sha256=None,
        snapshot_id=None,
        snapshot_sha256=None,
        code_commit_sha=None,
        python_version=None,
        torch_version=None,
        backend=None,
        device=None,
    )

    with pytest.raises(BenchmarkValidationError):
        validate_run_manifest(manifest)
