from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any

from again_benchmark.contracts import (
    BenchmarkComparisonResult,
    BenchmarkDefinition,
    BenchmarkMode,
    BenchmarkRunManifest,
    BenchmarkSnapshotManifest,
    BenchmarkSummary,
    BenchmarkTickerResult,
    DefinitionCatalogEntry,
    RunCatalogEntry,
    SplitPolicy,
)


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(val) for key, val in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_serialize(payload), ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def definition_from_dict(payload: dict[str, Any]) -> BenchmarkDefinition:
    return BenchmarkDefinition(
        benchmark_id=str(payload["benchmark_id"]),
        benchmark_version=int(payload["benchmark_version"]),
        definition_id=str(payload["definition_id"]),
        label=str(payload["label"]),
        tickers=tuple(str(value) for value in payload["tickers"]),
        horizon=int(payload["horizon"]),
        metrics=tuple(str(value) for value in payload["metrics"]),
        split_policy=SplitPolicy(str(payload["split_policy"])),
        lookback_years=int(payload["lookback_years"]),
        historical_display_days=int(payload["historical_display_days"]),
        notes=str(payload["notes"]) if payload.get("notes") is not None else None,
    )


def snapshot_manifest_from_dict(payload: dict[str, Any]) -> BenchmarkSnapshotManifest:
    return BenchmarkSnapshotManifest(
        snapshot_id=str(payload["snapshot_id"]),
        benchmark_id=str(payload["benchmark_id"]),
        benchmark_version=int(payload["benchmark_version"]),
        definition_id=str(payload["definition_id"]),
        created_at=datetime.fromisoformat(payload["created_at"]),
        as_of_timestamp=datetime.fromisoformat(payload["as_of_timestamp"]),
        tickers=tuple(str(value) for value in payload["tickers"]),
        effective_universe=tuple(str(value) for value in payload["effective_universe"]),
        horizon=int(payload["horizon"]),
        metrics=tuple(str(value) for value in payload["metrics"]),
        split_policy=SplitPolicy(str(payload["split_policy"])),
        row_count=int(payload["row_count"]),
        data_min_timestamp=datetime.fromisoformat(payload["data_min_timestamp"]),
        data_max_timestamp=datetime.fromisoformat(payload["data_max_timestamp"]),
        data_path=str(payload["data_path"]),
        data_sha256=str(payload["data_sha256"]),
    )


def run_manifest_from_dict(payload: dict[str, Any]) -> BenchmarkRunManifest:
    return BenchmarkRunManifest(
        run_id=str(payload["run_id"]),
        benchmark_id=str(payload["benchmark_id"]),
        benchmark_version=int(payload["benchmark_version"]),
        definition_id=str(payload["definition_id"]),
        mode=BenchmarkMode(str(payload["mode"])),
        created_at=datetime.fromisoformat(payload["created_at"]),
        as_of_timestamp=datetime.fromisoformat(payload["as_of_timestamp"]),
        split_date=datetime.fromisoformat(payload["split_date"]),
        tickers=tuple(str(value) for value in payload["tickers"]),
        effective_universe=tuple(str(value) for value in payload["effective_universe"]),
        horizon=int(payload["horizon"]),
        metrics=tuple(str(value) for value in payload["metrics"]),
        model_name=str(payload["model_name"]),
        profile_path=str(payload["profile_path"]) if payload.get("profile_path") is not None else None,
        config_fingerprint=str(payload["config_fingerprint"]) if payload.get("config_fingerprint") is not None else None,
        model_sha256=str(payload["model_sha256"]) if payload.get("model_sha256") is not None else None,
        normalizers_sha256=str(payload["normalizers_sha256"]) if payload.get("normalizers_sha256") is not None else None,
        dataset_sha256=str(payload["dataset_sha256"]) if payload.get("dataset_sha256") is not None else None,
        snapshot_id=str(payload["snapshot_id"]) if payload.get("snapshot_id") is not None else None,
        snapshot_sha256=str(payload["snapshot_sha256"]) if payload.get("snapshot_sha256") is not None else None,
        code_commit_sha=str(payload["code_commit_sha"]) if payload.get("code_commit_sha") is not None else None,
        python_version=str(payload["python_version"]) if payload.get("python_version") is not None else None,
        torch_version=str(payload["torch_version"]) if payload.get("torch_version") is not None else None,
        backend=str(payload["backend"]) if payload.get("backend") is not None else None,
        device=str(payload["device"]) if payload.get("device") is not None else None,
    )


def ticker_result_from_dict(payload: dict[str, Any]) -> BenchmarkTickerResult:
    return BenchmarkTickerResult(
        ticker=str(payload["ticker"]),
        split_date=datetime.fromisoformat(payload["split_date"]),
        forecast_dates=tuple(datetime.fromisoformat(value) for value in payload["forecast_dates"]),
        historical_dates=tuple(datetime.fromisoformat(value) for value in payload["historical_dates"]),
        historical_close=tuple(float(value) for value in payload["historical_close"]),
        actual_close=tuple(float(value) for value in payload["actual_close"]),
        predicted_close=tuple(float(value) for value in payload["predicted_close"]),
        metrics={str(key): float(value) for key, value in payload["metrics"].items()},
        last_observed_close=float(payload["last_observed_close"]),
    )


def summary_from_dict(payload: dict[str, Any]) -> BenchmarkSummary:
    return BenchmarkSummary(
        run_id=str(payload["run_id"]),
        benchmark_id=str(payload["benchmark_id"]),
        benchmark_version=int(payload["benchmark_version"]),
        mode=BenchmarkMode(str(payload["mode"])),
        requested_tickers=tuple(str(value) for value in payload["requested_tickers"]),
        effective_universe=tuple(str(value) for value in payload["effective_universe"]),
        completed_tickers=tuple(str(value) for value in payload["completed_tickers"]),
        failed_tickers=tuple(str(value) for value in payload["failed_tickers"]),
        metrics={str(key): float(value) for key, value in payload["metrics"].items()},
    )


def comparison_from_dict(payload: dict[str, Any]) -> BenchmarkComparisonResult:
    return BenchmarkComparisonResult(
        benchmark_id=str(payload["benchmark_id"]),
        left_run_id=str(payload["left_run_id"]),
        right_run_id=str(payload["right_run_id"]),
        summary_delta={str(key): float(value) for key, value in payload["summary_delta"].items()},
        ticker_deltas=tuple(dict(item) for item in payload["ticker_deltas"]),
    )


def run_catalog_entry_from_dict(payload: dict[str, Any]) -> RunCatalogEntry:
    return RunCatalogEntry(
        run_id=str(payload["run_id"]),
        benchmark_id=str(payload["benchmark_id"]),
        benchmark_version=int(payload["benchmark_version"]),
        definition_id=str(payload["definition_id"]),
        mode=BenchmarkMode(str(payload["mode"])),
        created_at=datetime.fromisoformat(payload["created_at"]),
        snapshot_id=str(payload["snapshot_id"]) if payload.get("snapshot_id") is not None else None,
        model_name=str(payload["model_name"]),
        summary_metrics={str(key): float(value) for key, value in (payload.get("summary_metrics") or {}).items()},
    )


def definition_catalog_entry_from_dict(payload: dict[str, Any]) -> DefinitionCatalogEntry:
    return DefinitionCatalogEntry(
        definition_id=str(payload["definition_id"]),
        benchmark_id=str(payload["benchmark_id"]),
        benchmark_version=int(payload["benchmark_version"]),
        label=str(payload["label"]),
        updated_at=datetime.fromisoformat(payload["updated_at"]),
    )
