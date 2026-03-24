from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd


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


def validate_sidecar_checksum(path: str | Path) -> str:
    target = Path(path)
    checksum_path = Path(f"{target}.sha256")
    if not checksum_path.exists():
        raise FileNotFoundError(f"Falta el sidecar de checksum para {target}")
    expected = checksum_path.read_text(encoding="utf-8").strip()
    current = compute_sha256(target)
    if expected != current:
        raise ValueError(f"Checksum invalido para {target}")
    return current


class BacktestStorage:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.catalog_path = self.root / "catalog.sqlite"
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.catalog_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    preset_name TEXT NOT NULL,
                    methodology_label TEXT NOT NULL,
                    label TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    provider_name TEXT NOT NULL,
                    provider_version TEXT NOT NULL,
                    config_reference TEXT,
                    requested_universe_json TEXT NOT NULL,
                    effective_universe_json TEXT NOT NULL,
                    market_source_summary_json TEXT NOT NULL,
                    window_count INTEGER NOT NULL,
                    discarded_signal_count INTEGER NOT NULL,
                    total_return REAL NOT NULL,
                    annualized_return REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    trade_count INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    exposure_ratio REAL NOT NULL,
                    path TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def write_run(
        self,
        *,
        manifest: dict[str, Any],
        summary: dict[str, Any],
        market_data: pd.DataFrame,
        oos_curve: pd.DataFrame,
        windows: pd.DataFrame,
        trades: pd.DataFrame,
        fills: pd.DataFrame,
        discards: pd.DataFrame,
    ) -> dict[str, Any]:
        run_id = str(manifest["run_id"])
        target_dir = self.run_dir(run_id)
        if target_dir.exists() and (target_dir / "run_manifest.json").exists():
            return self.load_run(run_id)

        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = target_dir / "run_manifest.json"
        summary_path = target_dir / "summary.json"
        market_path = target_dir / "market_data.parquet"
        oos_path = target_dir / "oos_curve.parquet"
        windows_path = target_dir / "windows.parquet"
        trades_path = target_dir / "trades.parquet"
        fills_path = target_dir / "fills.parquet"
        discards_path = target_dir / "discards.parquet"

        self._write_json(manifest_path, manifest)
        self._write_json(summary_path, summary)
        self._prepare_dataframe_for_parquet(market_data).to_parquet(market_path, index=False)
        self._prepare_dataframe_for_parquet(oos_curve).to_parquet(oos_path, index=False)
        self._prepare_dataframe_for_parquet(windows).to_parquet(windows_path, index=False)
        self._prepare_dataframe_for_parquet(trades).to_parquet(trades_path, index=False)
        self._prepare_dataframe_for_parquet(fills).to_parquet(fills_path, index=False)
        self._prepare_dataframe_for_parquet(discards).to_parquet(discards_path, index=False)

        write_sha256(manifest_path)
        write_sha256(summary_path)
        write_sha256(market_path)
        write_sha256(oos_path)
        write_sha256(windows_path)
        write_sha256(trades_path)
        write_sha256(fills_path)
        write_sha256(discards_path)

        metrics = dict(summary["summary_metrics"])
        with sqlite3.connect(self.catalog_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, created_at, mode, preset_name, methodology_label, label, model_name,
                    provider_name, provider_version, config_reference, requested_universe_json,
                    effective_universe_json, market_source_summary_json, window_count,
                    discarded_signal_count, total_return, annualized_return, max_drawdown,
                    sharpe_ratio, trade_count, win_rate, exposure_ratio, path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(summary["created_at"]),
                    str(summary["mode"]),
                    str(summary["preset_name"]),
                    str(summary["methodology_label"]),
                    str(manifest["label"]),
                    str(summary["model_name"]),
                    str(manifest["provider"]["name"]),
                    str(manifest["provider"]["version"]),
                    summary.get("config_reference"),
                    json.dumps(list(summary["requested_universe"]), sort_keys=True),
                    json.dumps(list(summary["effective_universe"]), sort_keys=True),
                    json.dumps(dict(summary["market_context"]["source_summary"]), sort_keys=True),
                    int(manifest["window_count"]),
                    int(manifest["discarded_signal_count"]),
                    float(metrics["total_return"]),
                    float(metrics["annualized_return"]),
                    float(metrics["max_drawdown"]),
                    float(metrics["sharpe_ratio"]),
                    int(metrics["trade_count"]),
                    float(metrics["win_rate"]),
                    float(metrics["exposure_ratio"]),
                    str(target_dir),
                ),
            )
            connection.commit()
        return self.load_run(run_id)

    def list_runs(self) -> tuple[dict[str, Any], ...]:
        with sqlite3.connect(self.catalog_path) as connection:
            rows = connection.execute(
                """
                SELECT
                    run_id, created_at, mode, preset_name, methodology_label, label, model_name,
                    provider_name, provider_version, config_reference, requested_universe_json,
                    effective_universe_json, market_source_summary_json, window_count,
                    discarded_signal_count, total_return, annualized_return, max_drawdown,
                    sharpe_ratio, trade_count, win_rate, exposure_ratio
                FROM runs
                ORDER BY created_at DESC, run_id DESC
                """
            ).fetchall()
        entries = []
        for row in rows:
            entries.append(
                {
                    "run_id": row[0],
                    "created_at": row[1],
                    "mode": row[2],
                    "preset_name": row[3],
                    "methodology_label": row[4],
                    "label": row[5],
                    "model_name": row[6],
                    "provider_name": row[7],
                    "provider_version": row[8],
                    "config_reference": row[9],
                    "requested_universe": tuple(json.loads(row[10] or "[]")),
                    "effective_universe": tuple(json.loads(row[11] or "[]")),
                    "market_source_summary": json.loads(row[12] or "{}"),
                    "window_count": int(row[13]),
                    "discarded_signal_count": int(row[14]),
                    "summary_metrics": {
                        "total_return": float(row[15]),
                        "annualized_return": float(row[16]),
                        "max_drawdown": float(row[17]),
                        "sharpe_ratio": float(row[18]),
                        "trade_count": int(row[19]),
                        "win_rate": float(row[20]),
                        "exposure_ratio": float(row[21]),
                    },
                }
            )
        return tuple(entries)

    def load_run(self, run_id: str) -> dict[str, Any]:
        target_dir = self.run_dir(run_id)
        manifest_path = target_dir / "run_manifest.json"
        summary_path = target_dir / "summary.json"
        market_path = target_dir / "market_data.parquet"
        oos_path = target_dir / "oos_curve.parquet"
        windows_path = target_dir / "windows.parquet"
        trades_path = target_dir / "trades.parquet"
        fills_path = target_dir / "fills.parquet"
        discards_path = target_dir / "discards.parquet"

        artifact_audit = {
            "manifest_sha256": validate_sidecar_checksum(manifest_path),
            "summary_sha256": validate_sidecar_checksum(summary_path),
            "market_sha256": validate_sidecar_checksum(market_path),
            "oos_curve_sha256": validate_sidecar_checksum(oos_path),
            "windows_sha256": validate_sidecar_checksum(windows_path),
            "trades_sha256": validate_sidecar_checksum(trades_path),
            "fills_sha256": validate_sidecar_checksum(fills_path),
            "discards_sha256": validate_sidecar_checksum(discards_path),
        }
        return {
            "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
            "market_data": pd.read_parquet(market_path),
            "oos_curve": pd.read_parquet(oos_path),
            "windows": pd.read_parquet(windows_path),
            "trades": pd.read_parquet(trades_path),
            "fills": pd.read_parquet(fills_path),
            "discards": pd.read_parquet(discards_path),
            "artifact_audit": artifact_audit,
        }

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _prepare_dataframe_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        for column in prepared.columns:
            if prepared[column].dtype != "object":
                continue
            if prepared[column].map(lambda value: isinstance(value, (dict, list))).any():
                prepared[column] = prepared[column].map(
                    lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                )
        return prepared
