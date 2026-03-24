from __future__ import annotations

from again_benchmark.contracts import BenchmarkRunBundle, BenchmarkTickerResult


def build_plot_payload(ticker_results: tuple[BenchmarkTickerResult, ...]) -> dict:
    return {
        "tickers": {
            result.ticker: {
                "split_date": result.split_date.isoformat(),
                "historical_dates": [value.isoformat() for value in result.historical_dates],
                "historical_close": list(result.historical_close),
                "forecast_dates": [value.isoformat() for value in result.forecast_dates],
                "actual_close": list(result.actual_close),
                "predicted_close": list(result.predicted_close),
                "metrics": dict(result.metrics),
            }
            for result in ticker_results
        }
    }


def build_runs_table_rows(entries) -> list[dict]:
    return [
        {
            "run_id": entry.run_id,
            "benchmark_id": entry.benchmark_id,
            "benchmark_version": entry.benchmark_version,
            "definition_id": entry.definition_id,
            "mode": entry.mode.value,
            "created_at": entry.created_at.isoformat(),
            "snapshot_id": entry.snapshot_id,
            "model_name": entry.model_name,
            "validation_state": entry.validation_state.value,
            "effective_universe_size": len(entry.effective_universe),
            "effective_universe": ", ".join(entry.effective_universe),
            "discarded_count": entry.discarded_count,
            **entry.summary_metrics,
        }
        for entry in entries
    ]


def build_run_view(bundle: BenchmarkRunBundle) -> dict:
    return {
        "manifest": bundle.manifest,
        "summary": bundle.summary,
        "ticker_results": bundle.ticker_results,
        "discarded_tickers": bundle.summary.discarded_tickers,
        "artifact_audit": {
            "snapshot_sha256": bundle.manifest.snapshot_sha256,
            "summary_sha256": bundle.manifest.summary_sha256,
            "metrics_sha256": bundle.manifest.metrics_sha256,
            "ticker_results_sha256": bundle.manifest.ticker_results_sha256,
            "plot_payload_sha256": bundle.manifest.plot_payload_sha256,
        },
        "plot_payload": bundle.plot_payload,
    }
