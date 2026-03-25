from __future__ import annotations

from datetime import datetime
import logging

import numpy as np
import pandas as pd

from again_benchmark.adapters.legacy_history import (
    METRICS,
    delete_benchmark_row,
    load_benchmark_history,
    save_benchmark_to_store,
)
from scripts.utils.prediction_utils import compute_directional_accuracy, price_path_to_step_returns

logger = logging.getLogger(__name__)


def _process_ticker(ticker: str, full_data: pd.DataFrame, config: dict, trim_date: pd.Timestamp, dataset, normalizers, model):
    import torch

    from scripts.prediction_engine import generate_predictions, preprocess_data

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
    import plotly.graph_objs as go

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
    import torch

    from scripts.data_fetcher import DataFetcher
    from scripts.prediction_engine import load_data_and_model
    from scripts.runtime_config import ConfigManager
    from scripts.utils.device_utils import resolve_execution_context

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
