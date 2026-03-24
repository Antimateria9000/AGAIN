from __future__ import annotations

import hashlib
import re
from pathlib import Path

from again_benchmark.contracts import BenchmarkDefinition, BenchmarkRunManifest, BenchmarkSnapshotManifest, DEFAULT_METRICS
from again_benchmark.errors import BenchmarkStorageError, BenchmarkValidationError

ALLOWED_METRICS = set(DEFAULT_METRICS)
BENCHMARK_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_definition(definition: BenchmarkDefinition) -> None:
    if not BENCHMARK_ID_RE.match(definition.benchmark_id):
        raise BenchmarkValidationError("benchmark_id contiene caracteres no permitidos")
    unsupported = set(definition.metrics) - ALLOWED_METRICS
    if unsupported:
        raise BenchmarkValidationError(f"Metricas no soportadas: {sorted(unsupported)}")


def validate_snapshot_manifest(manifest: BenchmarkSnapshotManifest, definition: BenchmarkDefinition | None = None) -> None:
    if manifest.data_min_timestamp > manifest.data_max_timestamp:
        raise BenchmarkValidationError("El snapshot declara data_min_timestamp > data_max_timestamp")
    if definition is not None:
        if manifest.definition_id != definition.definition_id:
            raise BenchmarkValidationError("El snapshot no pertenece a la definicion solicitada")
        if manifest.benchmark_id != definition.benchmark_id:
            raise BenchmarkValidationError("El snapshot no coincide con benchmark_id")
        if manifest.horizon != definition.horizon:
            raise BenchmarkValidationError("El snapshot no coincide con el horizon de la definicion")


def validate_run_manifest(manifest: BenchmarkRunManifest, definition: BenchmarkDefinition | None = None) -> None:
    if manifest.split_date > manifest.as_of_timestamp:
        raise BenchmarkValidationError("split_date no puede ser posterior a as_of_timestamp")
    if definition is not None and manifest.definition_id != definition.definition_id:
        raise BenchmarkValidationError("La corrida no corresponde a la definicion indicada")


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
