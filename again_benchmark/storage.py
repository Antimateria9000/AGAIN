from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd

from again_benchmark.contracts import (
    BenchmarkDefinition,
    BenchmarkMode,
    BenchmarkRunBundle,
    BenchmarkSnapshotManifest,
    BenchmarkStorageLayout,
    DefinitionCatalogEntry,
    RunCatalogEntry,
    ValidationState,
)
from again_benchmark.errors import BenchmarkStorageError
from again_benchmark.manifests import (
    definition_from_dict,
    read_json,
    read_yaml,
    run_manifest_from_dict,
    snapshot_manifest_from_dict,
    summary_from_dict,
    ticker_result_from_dict,
    write_json,
    write_yaml,
)
from again_benchmark.validation import validate_checksum_file, validate_sidecar_checksum, validate_snapshot_frame


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256(path: str | Path) -> str:
    target = Path(path)
    checksum = compute_sha256(target)
    Path(f"{target}.sha256").write_text(checksum, encoding="utf-8")
    return checksum


class BenchmarkStorage:
    def __init__(self, root: Path):
        self.layout = BenchmarkStorageLayout(
            root=root,
            definitions_dir=root / "definitions",
            snapshots_dir=root / "snapshots",
            runs_dir=root / "runs",
            catalog_path=root / "catalog.sqlite",
        )
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        self.layout.root.mkdir(parents=True, exist_ok=True)
        self.layout.definitions_dir.mkdir(parents=True, exist_ok=True)
        self.layout.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.layout.runs_dir.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.layout.catalog_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS definitions (
                    definition_id TEXT PRIMARY KEY,
                    benchmark_id TEXT NOT NULL,
                    benchmark_version INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    path TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    benchmark_id TEXT NOT NULL,
                    definition_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    as_of_timestamp TEXT NOT NULL,
                    data_sha256 TEXT NOT NULL,
                    path TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    benchmark_id TEXT NOT NULL,
                    benchmark_version INTEGER NOT NULL,
                    definition_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    snapshot_id TEXT,
                    model_name TEXT NOT NULL,
                    effective_universe_json TEXT NOT NULL DEFAULT '[]',
                    validation_state TEXT NOT NULL DEFAULT 'live_exploratory',
                    discarded_count INTEGER NOT NULL DEFAULT 0,
                    path TEXT NOT NULL,
                    summary_metrics_json TEXT NOT NULL
                )
                """
            )
            self._ensure_column(connection, "runs", "effective_universe_json", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column(connection, "runs", "validation_state", "TEXT NOT NULL DEFAULT 'live_exploratory'")
            self._ensure_column(connection, "runs", "discarded_count", "INTEGER NOT NULL DEFAULT 0")
            connection.commit()

    @staticmethod
    def _ensure_column(connection: sqlite3.Connection, table_name: str, column_name: str, column_sql: str) -> None:
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()}
        if column_name not in columns:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")

    def definition_path(self, definition_id: str) -> Path:
        return self.layout.definitions_dir / f"{definition_id}.yaml"

    def legacy_definition_path(self, definition_id: str) -> Path:
        return self.layout.definitions_dir / f"{definition_id}.json"

    def write_definition(self, definition: BenchmarkDefinition) -> Path:
        path = self.definition_path(definition.definition_id)
        if path.exists():
            existing = definition_from_dict(read_yaml(path))
            if existing != definition:
                raise BenchmarkStorageError(f"La definicion {definition.definition_id} ya existe con contenido distinto")
        else:
            write_yaml(path, definition)
        with sqlite3.connect(self.layout.catalog_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO definitions (definition_id, benchmark_id, benchmark_version, label, updated_at, path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    definition.definition_id,
                    definition.benchmark_id,
                    definition.benchmark_version,
                    definition.label,
                    datetime.now(UTC).replace(tzinfo=None).isoformat(),
                    str(path),
                ),
            )
            connection.commit()
        return path

    def load_definition(self, definition_id: str) -> BenchmarkDefinition:
        yaml_path = self.definition_path(definition_id)
        if yaml_path.exists():
            return definition_from_dict(read_yaml(yaml_path))
        return definition_from_dict(read_json(self.legacy_definition_path(definition_id)))

    def list_definitions(self) -> tuple[DefinitionCatalogEntry, ...]:
        with sqlite3.connect(self.layout.catalog_path) as connection:
            rows = connection.execute(
                "SELECT definition_id, benchmark_id, benchmark_version, label, updated_at FROM definitions ORDER BY updated_at DESC"
            ).fetchall()
        return tuple(
            DefinitionCatalogEntry(
                definition_id=row[0],
                benchmark_id=row[1],
                benchmark_version=int(row[2]),
                label=row[3],
                updated_at=datetime.fromisoformat(row[4]),
            )
            for row in rows
        )

    def snapshot_dir(self, snapshot_id: str) -> Path:
        return self.layout.snapshots_dir / snapshot_id

    def write_snapshot(self, frame: pd.DataFrame, manifest: BenchmarkSnapshotManifest) -> Path:
        target_dir = self.snapshot_dir(manifest.snapshot_id)
        data_path = target_dir / "market_data.parquet"
        manifest_path = target_dir / "snapshot_manifest.json"
        if target_dir.exists() and data_path.exists() and manifest_path.exists():
            existing_manifest = snapshot_manifest_from_dict(read_json(manifest_path))
            validate_checksum_file(data_path, manifest.data_sha256)
            if existing_manifest != manifest:
                raise BenchmarkStorageError(f"El snapshot {manifest.snapshot_id} ya existe con un manifiesto distinto")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(data_path, index=False)
            write_sha256(data_path)
            write_json(manifest_path, manifest)
            write_sha256(manifest_path)
        with sqlite3.connect(self.layout.catalog_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO snapshots (snapshot_id, benchmark_id, definition_id, created_at, as_of_timestamp, data_sha256, path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.snapshot_id,
                    manifest.benchmark_id,
                    manifest.definition_id,
                    manifest.created_at.isoformat(),
                    manifest.as_of_timestamp.isoformat(),
                    manifest.data_sha256,
                    str(target_dir),
                ),
            )
            connection.commit()
        return target_dir

    def load_snapshot(self, snapshot_id: str) -> tuple[pd.DataFrame, BenchmarkSnapshotManifest]:
        target_dir = self.snapshot_dir(snapshot_id)
        manifest_path = target_dir / "snapshot_manifest.json"
        if manifest_path.exists() and Path(f"{manifest_path}.sha256").exists():
            validate_sidecar_checksum(manifest_path)
        manifest = snapshot_manifest_from_dict(read_json(manifest_path))
        data_path = target_dir / "market_data.parquet"
        validate_checksum_file(data_path, manifest.data_sha256)
        frame = pd.read_parquet(data_path)
        validate_snapshot_frame(frame, manifest)
        return frame, manifest

    def run_dir(self, run_id: str) -> Path:
        return self.layout.runs_dir / run_id

    def write_run_bundle(self, bundle: BenchmarkRunBundle) -> BenchmarkRunBundle:
        target_dir = self.run_dir(bundle.manifest.run_id)
        if target_dir.exists() and (target_dir / "run_manifest.json").exists():
            return self.load_run_bundle(bundle.manifest.run_id)

        target_dir.mkdir(parents=True, exist_ok=True)
        summary_path = target_dir / "summary.json"
        plot_payload_path = target_dir / "plot_payload.json"
        write_json(summary_path, bundle.summary)
        summary_sha256 = write_sha256(summary_path)
        write_json(plot_payload_path, bundle.plot_payload)
        plot_payload_sha256 = write_sha256(plot_payload_path)

        metrics_df = pd.DataFrame(
            [{"ticker": result.ticker, **result.metrics, "observed_points": len(result.actual_close)} for result in bundle.ticker_results]
        )
        metrics_path = target_dir / "metrics.parquet"
        metrics_df.to_parquet(metrics_path, index=False)
        metrics_sha256 = write_sha256(metrics_path)

        ticker_results_df = pd.DataFrame(
            [
                {
                    "ticker": result.ticker,
                    "split_date": result.split_date.isoformat(),
                    "forecast_dates_json": json.dumps([value.isoformat() for value in result.forecast_dates]),
                    "historical_dates_json": json.dumps([value.isoformat() for value in result.historical_dates]),
                    "historical_close_json": json.dumps(list(result.historical_close)),
                    "actual_close_json": json.dumps(list(result.actual_close)),
                    "predicted_close_json": json.dumps(list(result.predicted_close)),
                    "metrics_json": json.dumps(result.metrics, sort_keys=True),
                    "last_observed_close": result.last_observed_close,
                    "status": result.status,
                }
                for result in bundle.ticker_results
            ]
        )
        ticker_results_path = target_dir / "ticker_results.parquet"
        ticker_results_df.to_parquet(ticker_results_path, index=False)
        ticker_results_sha256 = write_sha256(ticker_results_path)

        persisted_manifest = replace(
            bundle.manifest,
            summary_sha256=summary_sha256,
            metrics_sha256=metrics_sha256,
            ticker_results_sha256=ticker_results_sha256,
            plot_payload_sha256=plot_payload_sha256,
        )
        manifest_path = target_dir / "run_manifest.json"
        write_json(manifest_path, persisted_manifest)
        write_sha256(manifest_path)

        with sqlite3.connect(self.layout.catalog_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs (run_id, benchmark_id, benchmark_version, definition_id, mode, created_at, snapshot_id, model_name, effective_universe_json, validation_state, discarded_count, path, summary_metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persisted_manifest.run_id,
                    persisted_manifest.benchmark_id,
                    persisted_manifest.benchmark_version,
                    persisted_manifest.definition_id,
                    persisted_manifest.mode.value,
                    persisted_manifest.created_at.isoformat(),
                    persisted_manifest.snapshot_id,
                    persisted_manifest.model_name,
                    json.dumps(list(persisted_manifest.effective_universe), sort_keys=True),
                    persisted_manifest.validation_state.value,
                    len(persisted_manifest.discarded_tickers),
                    str(target_dir),
                    json.dumps(bundle.summary.metrics, sort_keys=True),
                ),
            )
            connection.commit()
        return BenchmarkRunBundle(
            manifest=persisted_manifest,
            summary=bundle.summary,
            ticker_results=bundle.ticker_results,
            plot_payload=bundle.plot_payload,
        )

    def list_runs(self) -> tuple[RunCatalogEntry, ...]:
        with sqlite3.connect(self.layout.catalog_path) as connection:
            rows = connection.execute(
                """
                SELECT run_id, benchmark_id, benchmark_version, definition_id, mode, created_at, snapshot_id, model_name, effective_universe_json, validation_state, discarded_count, summary_metrics_json
                FROM runs
                ORDER BY created_at DESC, run_id DESC
                """
            ).fetchall()
        return tuple(
            RunCatalogEntry(
                run_id=row[0],
                benchmark_id=row[1],
                benchmark_version=int(row[2]),
                definition_id=row[3],
                mode=BenchmarkMode(str(row[4])),
                created_at=datetime.fromisoformat(row[5]),
                snapshot_id=row[6],
                model_name=row[7],
                effective_universe=tuple(json.loads(row[8] or "[]")),
                validation_state=ValidationState(str(row[9] or ValidationState.LIVE_EXPLORATORY.value)),
                discarded_count=int(row[10] or 0),
                summary_metrics=json.loads(row[11]),
            )
            for row in rows
        )

    def load_run_bundle(self, run_id: str) -> BenchmarkRunBundle:
        target_dir = self.run_dir(run_id)
        manifest_path = target_dir / "run_manifest.json"
        if manifest_path.exists() and Path(f"{manifest_path}.sha256").exists():
            validate_sidecar_checksum(manifest_path)
        manifest = run_manifest_from_dict(read_json(manifest_path))
        summary_path = target_dir / "summary.json"
        plot_payload_path = target_dir / "plot_payload.json"
        metrics_path = target_dir / "metrics.parquet"
        ticker_results_path = target_dir / "ticker_results.parquet"
        if manifest.summary_sha256:
            validate_checksum_file(summary_path, manifest.summary_sha256)
        else:
            validate_sidecar_checksum(summary_path)
        if manifest.plot_payload_sha256:
            validate_checksum_file(plot_payload_path, manifest.plot_payload_sha256)
        else:
            validate_sidecar_checksum(plot_payload_path)
        if manifest.metrics_sha256:
            validate_checksum_file(metrics_path, manifest.metrics_sha256)
        else:
            validate_sidecar_checksum(metrics_path)
        if manifest.ticker_results_sha256:
            validate_checksum_file(ticker_results_path, manifest.ticker_results_sha256)
        else:
            validate_sidecar_checksum(ticker_results_path)
        summary = summary_from_dict(read_json(summary_path))
        plot_payload = read_json(plot_payload_path)
        ticker_results_df = pd.read_parquet(ticker_results_path)
        ticker_results = []
        for row in ticker_results_df.to_dict(orient="records"):
            ticker_results.append(
                ticker_result_from_dict(
                    {
                        "ticker": row["ticker"],
                        "split_date": row["split_date"],
                        "forecast_dates": json.loads(row["forecast_dates_json"]),
                        "historical_dates": json.loads(row["historical_dates_json"]),
                        "historical_close": json.loads(row["historical_close_json"]),
                        "actual_close": json.loads(row["actual_close_json"]),
                        "predicted_close": json.loads(row["predicted_close_json"]),
                        "metrics": json.loads(row["metrics_json"]),
                        "last_observed_close": row["last_observed_close"],
                        "status": row.get("status", "completed"),
                    }
                )
            )
        return BenchmarkRunBundle(
            manifest=manifest,
            summary=summary,
            ticker_results=tuple(ticker_results),
            plot_payload=plot_payload,
        )
