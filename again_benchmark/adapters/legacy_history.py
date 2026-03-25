from __future__ import annotations

from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd

from app.benchmark_store import delete_benchmark_row as delete_benchmark_row_by_id
from app.benchmark_store import ensure_database, load_benchmark_rows, save_benchmark_payload
from scripts.utils.repo_layout import resolve_repo_path

logger = logging.getLogger(__name__)

METRICS = ["MAPE", "MAE", "RMSE", "DirAcc"]


def _db_path(config: dict) -> Path:
    return resolve_repo_path(config, config["paths"]["benchmark_history_db_path"])


def _serialize_results(all_results: dict) -> str:
    serializable = {}
    for ticker, data in all_results.items():
        serializable[ticker] = {
            "historical_dates": [pd.Timestamp(value).isoformat() for value in data["historical_dates"]],
            "historical_close": list(map(float, data["historical_close"])),
            "pred_dates": [pd.Timestamp(value).isoformat() for value in data["pred_dates"]],
            "predictions": list(map(float, data["predictions"])),
            "historical_pred_close": list(map(float, data["historical_pred_close"])),
            "metrics": {key: float(value) for key, value in data["metrics"].items()},
            "last_date": pd.Timestamp(data["last_date"]).isoformat(),
        }
    return json.dumps(serializable, sort_keys=True)


def _deserialize_results(payload_json: str) -> dict:
    raw = json.loads(payload_json)
    results = {}
    for ticker, data in raw.items():
        results[ticker] = {
            "historical_dates": [pd.Timestamp(value) for value in data["historical_dates"]],
            "historical_close": data["historical_close"],
            "pred_dates": [pd.Timestamp(value) for value in data["pred_dates"]],
            "predictions": data["predictions"],
            "historical_pred_close": data["historical_pred_close"],
            "metrics": data["metrics"],
            "last_date": pd.Timestamp(data["last_date"]),
        }
    return results


def save_benchmark_to_store(config: dict, benchmark_date: str, all_results: dict, model_name: str) -> None:
    db_path = _db_path(config)
    ensure_database(db_path)
    save_benchmark_payload(db_path, benchmark_date, model_name, _serialize_results(all_results))
    logger.info("Benchmark guardado en %s", db_path)


def load_benchmark_history(config: dict, benchmark_tickers: list[str]):
    db_path = _db_path(config)
    rows = load_benchmark_rows(db_path)
    history_rows = []
    entries = []

    for row_id, benchmark_date, model_name, payload_json in rows:
        results = _deserialize_results(payload_json)
        entries.append({"id": row_id, "Date": benchmark_date, "Model_Name": model_name})
        row = {"Date": benchmark_date, "Model_Name": model_name}
        avg_metrics = {metric: [] for metric in METRICS}
        for ticker in benchmark_tickers:
            ticker_metrics = results.get(ticker, {}).get("metrics", {})
            for metric in METRICS:
                value = float(ticker_metrics.get(metric, 0.0))
                row[f"{ticker}_{metric}"] = value
                if value != 0.0:
                    avg_metrics[metric].append(value)
        for metric in METRICS:
            values = avg_metrics[metric]
            row[f"Avg_{metric}"] = float(np.mean(values)) if values else 0.0
        history_rows.append(row)

    columns = [("Basico", "Date"), ("Basico", "Model_Name")]
    for metric in METRICS:
        columns.append(("Media", metric))
    for ticker in benchmark_tickers:
        for metric in METRICS:
            columns.append((ticker, metric))

    if not history_rows:
        return pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns)), entries

    ordered_columns = ["Date", "Model_Name"] + [f"Avg_{metric}" for metric in METRICS] + [
        f"{ticker}_{metric}"
        for ticker in benchmark_tickers
        for metric in METRICS
    ]
    history_df = pd.DataFrame(history_rows)
    for column in ordered_columns:
        if column not in history_df.columns:
            history_df[column] = 0.0 if column not in {"Date", "Model_Name"} else ""
    history_df = history_df[ordered_columns]
    history_df.columns = pd.MultiIndex.from_tuples(columns)
    return history_df, entries


def delete_benchmark_row(config: dict, row_id: int) -> bool:
    db_path = _db_path(config)
    deleted = delete_benchmark_row_by_id(db_path, row_id)
    if deleted:
        logger.info("Se elimino la fila de benchmark con id=%s", row_id)
    return deleted
