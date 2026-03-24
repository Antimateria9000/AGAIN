from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from again_benchmark.contracts import BenchmarkDefinition, BenchmarkTickerResult
from again_benchmark.metrics import compute_metric_values


def build_market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=6)
    rows = []
    for ticker, start in (("AAA", 100.0), ("BBB", 50.0)):
        price = start
        for idx, date in enumerate(dates):
            price = price + (2.0 if ticker == "AAA" else 1.0)
            rows.append(
                {
                    "Date": date.to_pydatetime(),
                    "Open": price - 0.5,
                    "High": price + 0.5,
                    "Low": price - 1.0,
                    "Close": price,
                    "Volume": 1_000.0 + idx,
                    "Ticker": ticker,
                    "Sector": "Technology" if ticker == "AAA" else "Industrials",
                }
            )
    return pd.DataFrame(rows)


def build_definition() -> BenchmarkDefinition:
    return BenchmarkDefinition(
        benchmark_id="again_benchmark_test",
        benchmark_version=1,
        definition_id="definition_test_v1",
        label="Benchmark de prueba",
        tickers=("AAA", "BBB"),
        horizon=2,
        lookback_years=1,
        historical_display_days=30,
    )


@dataclass
class FakeBenchmarkAdapter:
    market_data: pd.DataFrame
    bias: float = 0.0

    adapter_name: str = "fake_adapter"

    def fetch_market_data(self, definition: BenchmarkDefinition, as_of_timestamp: datetime) -> pd.DataFrame:
        return self.market_data.copy()

    def evaluate_ticker(
        self,
        definition: BenchmarkDefinition,
        market_data: pd.DataFrame,
        ticker: str,
        split_date: datetime,
    ) -> BenchmarkTickerResult:
        ticker_frame = market_data[market_data["Ticker"] == ticker].copy().sort_values("Date")
        observed = ticker_frame[ticker_frame["Date"] <= split_date].copy()
        future = ticker_frame[ticker_frame["Date"] > split_date].copy().head(definition.horizon)
        actual_close = future["Close"].astype(float).to_numpy()
        predicted_close = actual_close + float(self.bias)
        last_observed_close = float(observed["Close"].iloc[-1])
        actual_returns = (actual_close / pd.Series([last_observed_close, *actual_close[:-1]]).to_numpy()) - 1.0
        predicted_returns = (predicted_close / pd.Series([last_observed_close, *predicted_close[:-1]]).to_numpy()) - 1.0
        metrics = compute_metric_values(predicted_close, actual_close, predicted_returns, actual_returns)
        return BenchmarkTickerResult(
            ticker=ticker,
            split_date=split_date,
            forecast_dates=tuple(pd.Timestamp(value).to_pydatetime() for value in future["Date"].tolist()),
            historical_dates=tuple(pd.Timestamp(value).to_pydatetime() for value in observed["Date"].tolist()),
            historical_close=tuple(float(value) for value in observed["Close"].tolist()),
            actual_close=tuple(float(value) for value in actual_close.tolist()),
            predicted_close=tuple(float(value) for value in predicted_close.tolist()),
            metrics=metrics,
            last_observed_close=last_observed_close,
        )

    def get_model_metadata(self) -> dict[str, str | None]:
        return {
            "model_name": "fake_model",
            "profile_path": "config/config.yaml",
            "model_sha256": "model-hash",
            "normalizers_sha256": "normalizer-hash",
            "dataset_sha256": "dataset-hash",
            "code_commit_sha": "deadbeef",
        }

    def get_runtime_metadata(self) -> dict[str, str | None]:
        return {
            "python_version": "3.11.0",
            "torch_version": "2.7.0",
            "backend": "CPU",
            "device": "CPU",
        }

    def get_config_fingerprint(self) -> str | None:
        return "config-hash"

    def reset(self) -> None:
        return None


def build_storage_root(tmp_path: Path) -> Path:
    root = tmp_path / "benchmarks"
    root.mkdir(parents=True, exist_ok=True)
    return root
