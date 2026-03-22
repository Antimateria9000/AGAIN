from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch

from app.benchmark_store import delete_benchmark_row as delete_benchmark_row_by_id
from app.benchmark_store import ensure_database, load_benchmark_rows, save_benchmark_payload
from scripts.data_fetcher import DataFetcher
from scripts.prediction_engine import generate_predictions, load_data_and_model, preprocess_data
from scripts.runtime_config import ConfigManager
from scripts.utils.device_utils import resolve_execution_context
from scripts.utils.prediction_utils import compute_directional_accuracy, price_path_to_step_returns

logger = logging.getLogger(__name__)
METRICS = ["MAPE", "MAE", "RMSE", "DirAcc"]


def _db_path(config: dict) -> Path:
    return Path(config["paths"]["benchmark_history_db_path"])


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


def _process_ticker(ticker: str, full_data: pd.DataFrame, config: dict, trim_date: pd.Timestamp, dataset, normalizers, model):
    full_data = full_data[full_data["Ticker"] == ticker].copy()
    full_data["Date"] = pd.to_datetime(full_data["Date"]).dt.tz_localize(None)
    full_data = full_data.sort_values("Date").set_index("Date")
    historical_close = full_data["Close"]

    new_data = full_data[full_data.index <= trim_date].copy().reset_index()
    if new_data.empty:
        logger.warning("No hay datos previos al recorte para %s", ticker)
        return None

    ticker_data, original_close = preprocess_data(config, new_data, ticker, normalizers, historical_mode=True)
    with torch.no_grad():
        median, _, _, details = generate_predictions(
            config,
            dataset,
            model,
            ticker_data,
            return_details=True,
            raw_ticker_data=new_data,
        )

    future_actual = historical_close[historical_close.index > trim_date].sort_index().iloc[: len(median)]
    if len(future_actual) != len(median):
        logger.warning("Horizonte insuficiente para %s: pred=%s real=%s", ticker, len(median), len(future_actual))
        return None

    pred_dates = [pd.Timestamp(date).to_pydatetime() for date in future_actual.index]
    actual_close = future_actual.to_numpy(dtype=float)
    predictions = np.asarray(median, dtype=float)
    differences = np.abs(predictions - actual_close)
    denominators = np.where(actual_close == 0.0, 1e-12, actual_close)
    relative_diff = (differences / denominators) * 100.0
    last_observed_close = float(original_close.iloc[-1])
    actual_returns = price_path_to_step_returns(last_observed_close, actual_close)
    directional_accuracy = compute_directional_accuracy(details["relative_returns_median"], actual_returns)

    return {
        "historical_dates": ticker_data["Date"].dt.tz_localize(None).tolist(),
        "historical_close": original_close.tolist(),
        "pred_dates": pred_dates,
        "predictions": predictions.tolist(),
        "historical_pred_close": actual_close.tolist(),
        "metrics": {
            "MAPE": float(np.mean(relative_diff)),
            "MAE": float(np.mean(differences)),
            "RMSE": float(np.sqrt(np.mean(np.square(predictions - actual_close)))),
            "DirAcc": float(directional_accuracy),
        },
        "last_date": pd.Timestamp(ticker_data["Date"].iloc[-1]).to_pydatetime(),
    }


def _build_benchmark_figure(all_results: dict, historical_period_days: int, split_date: pd.Timestamp | None):
    fig = go.Figure()
    colors = ["#0B5FFF", "#00A36C", "#E74C3C", "#8E44AD", "#FF8C00", "#00B8D9", "#C2185B", "#B7950B", "#6D4C41", "#546E7A"]

    for idx, (ticker, data) in enumerate(all_results.items()):
        color = colors[idx % len(colors)]
        cutoff_date = pd.Timestamp(data["pred_dates"][0]) - pd.Timedelta(days=historical_period_days)
        historical_series = pd.Series(data["historical_close"], index=pd.to_datetime(data["historical_dates"]))
        filtered_history = historical_series[historical_series.index >= cutoff_date]
        combined_dates = filtered_history.index.tolist() + list(data["pred_dates"])
        combined_close = filtered_history.tolist() + list(data["historical_pred_close"])
        combined_pred_close = [None] * len(filtered_history) + list(data["predictions"])

        fig.add_trace(go.Scatter(x=combined_dates, y=combined_close, mode="lines", name=f"{ticker} (real)", line=dict(color=color), legendgroup=ticker))
        fig.add_trace(go.Scatter(x=combined_dates, y=combined_pred_close, mode="lines", name=f"{ticker} (prediccion)", line=dict(color=color, dash="dash"), legendgroup=ticker))

    if split_date is not None:
        split_str = pd.Timestamp(split_date).isoformat()
        fig.add_shape(type="line", x0=split_str, x1=split_str, y0=0, y1=1, xref="x", yref="paper", line=dict(color="red", width=2, dash="dash"))
        fig.add_annotation(x=split_str, y=1.05, xref="x", yref="paper", text="Inicio del horizonte de validacion", showarrow=False, font=dict(size=12), align="center")

    fig.update_layout(
        title="Benchmark historico de predicciones",
        xaxis_title="Fecha",
        yaxis_title="Precio de cierre",
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
    )
    return fig


def build_metrics_dataframe(all_results: dict) -> pd.DataFrame:
    rows = []
    for ticker, data in all_results.items():
        row = {"Ticker": ticker}
        row.update(data["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def run_benchmark(
    config: dict,
    benchmark_tickers: list[str],
    years: int,
    historical_period_days: int = 365,
    run_timestamp: datetime | None = None,
    config_manager: ConfigManager | None = None,
):
    all_results: dict[str, dict] = {}
    max_prediction_length = config["model"]["max_prediction_length"]
    benchmark_timestamp = run_timestamp or datetime.now().replace(tzinfo=None)
    start_date = pd.Timestamp(benchmark_timestamp).tz_localize(None) - pd.Timedelta(days=years * 365)
    end_date = benchmark_timestamp

    runtime_config_manager = config_manager or ConfigManager()
    fetcher = DataFetcher(runtime_config_manager, config["prediction"]["years"])
    combined_data = fetcher.fetch_many_stocks(benchmark_tickers, start_date, end_date)
    if combined_data.empty:
        raise ValueError("No se ha podido descargar ningun ticker del benchmark")

    ticker_data_dict = {ticker: frame.copy() for ticker, frame in combined_data.groupby("Ticker")}
    trim_candidates = []
    for data in ticker_data_dict.values():
        unique_dates = pd.Series(pd.to_datetime(data["Date"]).dt.tz_localize(None).sort_values().unique())
        if len(unique_dates) > max_prediction_length:
            trim_candidates.append(unique_dates.iloc[-(max_prediction_length + 1)])
    if not trim_candidates:
        raise ValueError("No hay suficiente historial para construir el benchmark")

    trim_date = min(trim_candidates)
    first_ticker = next(iter(ticker_data_dict))
    first_data = ticker_data_dict[first_ticker].copy()
    _, dataset, normalizers, model = load_data_and_model(config, first_ticker, raw_data=first_data, historical_mode=True)

    try:
        for ticker, data in ticker_data_dict.items():
            processed = _process_ticker(ticker, data, config, trim_date, dataset, normalizers, model)
            if processed is not None:
                all_results[ticker] = processed
    finally:
        del model
        del dataset
        del normalizers
        if resolve_execution_context(config, purpose="predict").uses_cuda:
            torch.cuda.empty_cache()

    metrics_df = build_metrics_dataframe(all_results)
    fig = _build_benchmark_figure(all_results, historical_period_days, pd.Timestamp(trim_date))
    return all_results, fig, metrics_df


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

    ordered_columns = ["Date", "Model_Name"] + [f"Avg_{metric}" for metric in METRICS] + [f"{ticker}_{metric}" for ticker in benchmark_tickers for metric in METRICS]
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
