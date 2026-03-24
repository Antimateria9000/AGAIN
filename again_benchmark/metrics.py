from __future__ import annotations

import math

import numpy as np

from again_benchmark.contracts import BenchmarkSummary, BenchmarkTickerResult


def compute_metric_values(predicted_close, actual_close, predicted_step_returns, actual_step_returns) -> dict[str, float]:
    predicted = np.asarray(predicted_close, dtype=float).reshape(-1)
    actual = np.asarray(actual_close, dtype=float).reshape(-1)
    predicted_returns = np.asarray(predicted_step_returns, dtype=float).reshape(-1)
    actual_returns = np.asarray(actual_step_returns, dtype=float).reshape(-1)
    if predicted.size == 0 or actual.size == 0:
        return {"MAPE": 0.0, "MAE": 0.0, "RMSE": 0.0, "DirAcc": 0.0}
    horizon = min(predicted.size, actual.size)
    predicted = predicted[:horizon]
    actual = actual[:horizon]
    predicted_returns = predicted_returns[:horizon]
    actual_returns = actual_returns[:horizon]
    diff = predicted - actual
    denom = np.where(actual == 0.0, 1e-12, actual)
    return {
        "MAPE": float(np.mean(np.abs(diff) / np.abs(denom)) * 100.0),
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(math.sqrt(float(np.mean(np.square(diff))))),
        "DirAcc": float(np.mean(np.sign(predicted_returns) == np.sign(actual_returns)) * 100.0),
    }


def summarize_results(
    *,
    run_id: str,
    benchmark_id: str,
    benchmark_version: int,
    mode,
    requested_tickers: tuple[str, ...],
    effective_universe: tuple[str, ...],
    ticker_results: tuple[BenchmarkTickerResult, ...],
    failed_tickers: tuple[str, ...],
) -> BenchmarkSummary:
    if not ticker_results:
        metrics = {"MAPE": 0.0, "MAE": 0.0, "RMSE": 0.0, "DirAcc": 0.0}
    else:
        predicted = np.concatenate([np.asarray(result.predicted_close, dtype=float) for result in ticker_results])
        actual = np.concatenate([np.asarray(result.actual_close, dtype=float) for result in ticker_results])
        actual_returns = []
        predicted_returns = []
        for result in ticker_results:
            actual_prices = np.asarray(result.actual_close, dtype=float)
            predicted_prices = np.asarray(result.predicted_close, dtype=float)
            prior_actual = np.concatenate(([float(result.last_observed_close)], actual_prices[:-1]))
            prior_pred = np.concatenate(([float(result.last_observed_close)], predicted_prices[:-1]))
            actual_returns.append((actual_prices / np.where(prior_actual == 0.0, 1e-12, prior_actual)) - 1.0)
            predicted_returns.append((predicted_prices / np.where(prior_pred == 0.0, 1e-12, prior_pred)) - 1.0)
        metrics = compute_metric_values(predicted, actual, np.concatenate(predicted_returns), np.concatenate(actual_returns))
    return BenchmarkSummary(
        run_id=run_id,
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        mode=mode,
        requested_tickers=requested_tickers,
        effective_universe=effective_universe,
        completed_tickers=tuple(result.ticker for result in ticker_results),
        failed_tickers=failed_tickers,
        metrics=metrics,
    )
