from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from again_benchmark.contracts import (
    BenchmarkDefinition,
    BenchmarkMode,
    BenchmarkRunManifest,
    BenchmarkSnapshotManifest,
    DEFAULT_METRICS,
    TimeNormalizationPolicy,
    ValidationState,
)
from again_benchmark.errors import BenchmarkStorageError, BenchmarkValidationError

ALLOWED_METRICS = set(DEFAULT_METRICS)
BENCHMARK_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_definition(definition: BenchmarkDefinition) -> None:
    if not BENCHMARK_ID_RE.match(definition.benchmark_id):
        raise BenchmarkValidationError("benchmark_id contiene caracteres no permitidos")
    unsupported = set(definition.metrics) - ALLOWED_METRICS
    if unsupported:
        raise BenchmarkValidationError(f"Metricas no soportadas: {sorted(unsupported)}")
    if len(set(definition.tickers)) != len(definition.tickers):
        raise BenchmarkValidationError("La definicion no puede repetir tickers")


def validate_snapshot_manifest(manifest: BenchmarkSnapshotManifest, definition: BenchmarkDefinition | None = None) -> None:
    if manifest.data_min_timestamp > manifest.data_max_timestamp:
        raise BenchmarkValidationError("El snapshot declara data_min_timestamp > data_max_timestamp")
    if manifest.time_normalization_policy != TimeNormalizationPolicy.NAIVE_UTC:
        raise BenchmarkValidationError("again_benchmark solo soporta timestamps naive UTC-normalized")
    if definition is not None:
        if manifest.definition_id != definition.definition_id:
            raise BenchmarkValidationError("El snapshot no pertenece a la definicion solicitada")
        if manifest.benchmark_id != definition.benchmark_id:
            raise BenchmarkValidationError("El snapshot no coincide con benchmark_id")
        if manifest.horizon != definition.horizon:
            raise BenchmarkValidationError("El snapshot no coincide con el horizon de la definicion")
        if tuple(manifest.tickers) != tuple(definition.tickers):
            raise BenchmarkValidationError("El snapshot no coincide con el universo solicitado")
        unsupported = set(manifest.effective_universe) - set(definition.tickers)
        if unsupported:
            raise BenchmarkValidationError(f"El snapshot contiene tickers fuera de la definicion: {sorted(unsupported)}")


def validate_snapshot_frame(frame: pd.DataFrame, manifest: BenchmarkSnapshotManifest) -> None:
    observed_columns = tuple(str(column) for column in frame.columns)
    if observed_columns != tuple(manifest.expected_columns):
        raise BenchmarkValidationError("El snapshot materializado no coincide con las columnas esperadas del manifiesto")
    if len(frame) != manifest.row_count:
        raise BenchmarkValidationError("El snapshot materializado no coincide con row_count del manifiesto")


def validate_run_manifest(manifest: BenchmarkRunManifest, definition: BenchmarkDefinition | None = None) -> None:
    if manifest.split_date > manifest.as_of_timestamp:
        raise BenchmarkValidationError("split_date no puede ser posterior a as_of_timestamp")
    if manifest.mode == BenchmarkMode.FROZEN and manifest.validation_state != ValidationState.FROZEN_VALIDATED:
        raise BenchmarkValidationError("Los frozen benchmarks deben marcarse como frozen_validated")
    if manifest.mode == BenchmarkMode.LIVE and manifest.validation_state != ValidationState.LIVE_EXPLORATORY:
        raise BenchmarkValidationError("Los live benchmarks deben marcarse como live_exploratory")
    if definition is not None and manifest.definition_id != definition.definition_id:
        raise BenchmarkValidationError("La corrida no corresponde a la definicion indicada")
    if definition is not None and tuple(manifest.tickers) != tuple(definition.tickers):
        raise BenchmarkValidationError("La corrida no coincide con el universo de la definicion")
    unsupported = set(manifest.effective_universe) - set(manifest.tickers)
    if unsupported:
        raise BenchmarkValidationError(f"effective_universe contiene tickers fuera de tickers: {sorted(unsupported)}")
    discarded = {item.ticker for item in manifest.discarded_tickers}
    completed = set(manifest.effective_universe)
    if discarded & completed:
        raise BenchmarkValidationError("Un ticker no puede estar a la vez completado y descartado")


def validate_checksum_file(path: Path, expected_sha256: str) -> None:
    checksum_path = Path(f"{path}.sha256")
    if not checksum_path.exists():
        raise BenchmarkStorageError(f"Falta el checksum {checksum_path}")
    stored = checksum_path.read_text(encoding="utf-8").strip()
    if stored != expected_sha256:
        raise BenchmarkStorageError(f"Checksum invalido para {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    current = digest.hexdigest()
    if current != expected_sha256:
        raise BenchmarkStorageError(f"El contenido actual de {path} no coincide con el checksum esperado")


def validate_sidecar_checksum(path: Path) -> None:
    checksum_path = Path(f"{path}.sha256")
    if not checksum_path.exists():
        raise BenchmarkStorageError(f"Falta el checksum {checksum_path}")
    expected = checksum_path.read_text(encoding="utf-8").strip()
    validate_checksum_file(path, expected)
