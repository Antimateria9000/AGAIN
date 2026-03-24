from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from pathlib import Path
from typing import Any

from again_benchmark.errors import BenchmarkValidationError


def normalize_naive_utc_timestamp(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise BenchmarkValidationError(f"{field_name} debe ser datetime")
    if value.tzinfo is not None and value.utcoffset() is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _ensure_finite_mapping(metrics: dict[str, float], *, field_name: str) -> None:
    if not isinstance(metrics, dict):
        raise BenchmarkValidationError(f"{field_name} debe ser un dict")
    for key, value in metrics.items():
        if not key:
            raise BenchmarkValidationError(f"{field_name} contiene una metrica sin nombre")
        if not math.isfinite(float(value)):
            raise BenchmarkValidationError(f"{field_name}.{key} debe ser finita")


class BenchmarkMode(str, Enum):
    LIVE = "live"
    FROZEN = "frozen"


class SplitPolicy(str, Enum):
    COMMON_HISTORY_CUTOFF = "common_history_cutoff"


DEFAULT_METRICS = ("MAPE", "MAE", "RMSE", "DirAcc")


@dataclass(frozen=True)
class BenchmarkDefinition:
    benchmark_id: str
    benchmark_version: int
    definition_id: str
    label: str
    tickers: tuple[str, ...]
    horizon: int
    metrics: tuple[str, ...] = DEFAULT_METRICS
    split_policy: SplitPolicy = SplitPolicy.COMMON_HISTORY_CUTOFF
    lookback_years: int = 3
    historical_display_days: int = 365
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise BenchmarkValidationError("benchmark_id es obligatorio")
        if self.benchmark_version <= 0:
            raise BenchmarkValidationError("benchmark_version debe ser > 0")
        if not self.definition_id:
            raise BenchmarkValidationError("definition_id es obligatorio")
        if not self.label:
            raise BenchmarkValidationError("label es obligatorio")
        if not self.tickers:
            raise BenchmarkValidationError("La definicion del benchmark requiere al menos un ticker")
        if len(set(self.tickers)) != len(self.tickers):
            raise BenchmarkValidationError("La definicion del benchmark no puede repetir tickers")
        if self.horizon <= 0:
            raise BenchmarkValidationError("horizon debe ser > 0")
        if self.lookback_years <= 0:
            raise BenchmarkValidationError("lookback_years debe ser > 0")
        if self.historical_display_days <= 0:
            raise BenchmarkValidationError("historical_display_days debe ser > 0")
        if not self.metrics:
            raise BenchmarkValidationError("metrics no puede estar vacio")


@dataclass(frozen=True)
class BenchmarkSnapshotManifest:
    snapshot_id: str
    benchmark_id: str
    benchmark_version: int
    definition_id: str
    created_at: datetime
    as_of_timestamp: datetime
    tickers: tuple[str, ...]
    effective_universe: tuple[str, ...]
    horizon: int
    metrics: tuple[str, ...]
    split_policy: SplitPolicy
    row_count: int
    data_min_timestamp: datetime
    data_max_timestamp: datetime
    data_path: str
    data_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", normalize_naive_utc_timestamp(self.created_at, field_name="created_at"))
        object.__setattr__(self, "as_of_timestamp", normalize_naive_utc_timestamp(self.as_of_timestamp, field_name="as_of_timestamp"))
        object.__setattr__(
            self,
            "data_min_timestamp",
            normalize_naive_utc_timestamp(self.data_min_timestamp, field_name="data_min_timestamp"),
        )
        object.__setattr__(
            self,
            "data_max_timestamp",
            normalize_naive_utc_timestamp(self.data_max_timestamp, field_name="data_max_timestamp"),
        )
        if not self.snapshot_id:
            raise BenchmarkValidationError("snapshot_id es obligatorio")
        if self.row_count <= 0:
            raise BenchmarkValidationError("row_count debe ser > 0")
        if not self.data_sha256:
            raise BenchmarkValidationError("data_sha256 es obligatorio")


@dataclass(frozen=True)
class BenchmarkRunManifest:
    run_id: str
    benchmark_id: str
    benchmark_version: int
    definition_id: str
    mode: BenchmarkMode
    created_at: datetime
    as_of_timestamp: datetime
    split_date: datetime
    tickers: tuple[str, ...]
    effective_universe: tuple[str, ...]
    horizon: int
    metrics: tuple[str, ...]
    model_name: str
    profile_path: str | None
    config_fingerprint: str | None
    model_sha256: str | None
    normalizers_sha256: str | None
    dataset_sha256: str | None
    snapshot_id: str | None
    snapshot_sha256: str | None
    code_commit_sha: str | None
    python_version: str | None
    torch_version: str | None
    backend: str | None
    device: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", normalize_naive_utc_timestamp(self.created_at, field_name="created_at"))
        object.__setattr__(self, "as_of_timestamp", normalize_naive_utc_timestamp(self.as_of_timestamp, field_name="as_of_timestamp"))
        object.__setattr__(self, "split_date", normalize_naive_utc_timestamp(self.split_date, field_name="split_date"))
        if not self.run_id:
            raise BenchmarkValidationError("run_id es obligatorio")
        if self.horizon <= 0:
            raise BenchmarkValidationError("horizon debe ser > 0")
        if not self.tickers:
            raise BenchmarkValidationError("tickers no puede estar vacio")
        if self.mode == BenchmarkMode.FROZEN and (not self.snapshot_id or not self.snapshot_sha256):
            raise BenchmarkValidationError("Los frozen benchmarks requieren snapshot_id y snapshot_sha256")


@dataclass(frozen=True)
class BenchmarkTickerResult:
    ticker: str
    split_date: datetime
    forecast_dates: tuple[datetime, ...]
    historical_dates: tuple[datetime, ...]
    historical_close: tuple[float, ...]
    actual_close: tuple[float, ...]
    predicted_close: tuple[float, ...]
    metrics: dict[str, float]
    last_observed_close: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_date", normalize_naive_utc_timestamp(self.split_date, field_name="split_date"))
        object.__setattr__(
            self,
            "forecast_dates",
            tuple(normalize_naive_utc_timestamp(value, field_name="forecast_date") for value in self.forecast_dates),
        )
        object.__setattr__(
            self,
            "historical_dates",
            tuple(normalize_naive_utc_timestamp(value, field_name="historical_date") for value in self.historical_dates),
        )
        if not self.ticker:
            raise BenchmarkValidationError("ticker es obligatorio")
        if len(self.forecast_dates) != len(self.actual_close) or len(self.forecast_dates) != len(self.predicted_close):
            raise BenchmarkValidationError("Las series forecast_dates, actual_close y predicted_close deben estar alineadas")
        if not self.historical_dates or not self.historical_close:
            raise BenchmarkValidationError("BenchmarkTickerResult requiere historial observado")
        _ensure_finite_mapping(self.metrics, field_name="metrics")


@dataclass(frozen=True)
class BenchmarkSummary:
    run_id: str
    benchmark_id: str
    benchmark_version: int
    mode: BenchmarkMode
    requested_tickers: tuple[str, ...]
    effective_universe: tuple[str, ...]
    completed_tickers: tuple[str, ...]
    failed_tickers: tuple[str, ...]
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        if not self.run_id:
            raise BenchmarkValidationError("BenchmarkSummary.run_id es obligatorio")
        _ensure_finite_mapping(self.metrics, field_name="metrics")


@dataclass(frozen=True)
class BenchmarkComparisonResult:
    benchmark_id: str
    left_run_id: str
    right_run_id: str
    summary_delta: dict[str, float]
    ticker_deltas: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        _ensure_finite_mapping(self.summary_delta, field_name="summary_delta")


@dataclass(frozen=True)
class BenchmarkRunBundle:
    manifest: BenchmarkRunManifest
    summary: BenchmarkSummary
    ticker_results: tuple[BenchmarkTickerResult, ...]
    plot_payload: dict[str, Any]


@dataclass(frozen=True)
class RunCatalogEntry:
    run_id: str
    benchmark_id: str
    benchmark_version: int
    definition_id: str
    mode: BenchmarkMode
    created_at: datetime
    snapshot_id: str | None
    model_name: str
    summary_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", normalize_naive_utc_timestamp(self.created_at, field_name="created_at"))
        _ensure_finite_mapping(self.summary_metrics, field_name="summary_metrics")


@dataclass(frozen=True)
class DefinitionCatalogEntry:
    definition_id: str
    benchmark_id: str
    benchmark_version: int
    label: str
    updated_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "updated_at", normalize_naive_utc_timestamp(self.updated_at, field_name="updated_at"))


@dataclass(frozen=True)
class BenchmarkStorageLayout:
    root: Path
    definitions_dir: Path
    snapshots_dir: Path
    runs_dir: Path
    catalog_path: Path
