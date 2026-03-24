from __future__ import annotations

from again_benchmark.contracts import BenchmarkComparisonResult, BenchmarkRunBundle
from again_benchmark.errors import BenchmarkValidationError


def compare_run_bundles(left: BenchmarkRunBundle, right: BenchmarkRunBundle) -> BenchmarkComparisonResult:
    if left.manifest.benchmark_id != right.manifest.benchmark_id:
        raise BenchmarkValidationError("Solo se pueden comparar corridas del mismo benchmark_id")
    summary_keys = sorted(set(left.summary.metrics) | set(right.summary.metrics))
    summary_delta = {
        key: float(right.summary.metrics.get(key, 0.0) - left.summary.metrics.get(key, 0.0))
        for key in summary_keys
    }

    left_by_ticker = {result.ticker: result for result in left.ticker_results}
    right_by_ticker = {result.ticker: result for result in right.ticker_results}
    common_tickers = sorted(set(left_by_ticker) & set(right_by_ticker))
    ticker_deltas = []
    for ticker in common_tickers:
        left_metrics = left_by_ticker[ticker].metrics
        right_metrics = right_by_ticker[ticker].metrics
        metric_keys = sorted(set(left_metrics) | set(right_metrics))
        ticker_deltas.append(
            {
                "ticker": ticker,
                "left_metrics": dict(left_metrics),
                "right_metrics": dict(right_metrics),
                "delta_metrics": {key: float(right_metrics.get(key, 0.0) - left_metrics.get(key, 0.0)) for key in metric_keys},
            }
        )

    return BenchmarkComparisonResult(
        benchmark_id=left.manifest.benchmark_id,
        left_run_id=left.manifest.run_id,
        right_run_id=right.manifest.run_id,
        summary_delta=summary_delta,
        ticker_deltas=tuple(ticker_deltas),
    )
