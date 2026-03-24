from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from again_benchmark.contracts import BenchmarkDefinition, BenchmarkTickerResult
from again_benchmark.errors import BenchmarkAdapterError
from again_benchmark.metrics import compute_metric_values
from scripts.data_fetcher import DataFetcher
from scripts.prediction_engine import generate_predictions, load_data_and_model, preprocess_data
from scripts.runtime_config import ConfigManager
from scripts.utils.device_utils import resolve_execution_context
from scripts.utils.prediction_utils import price_path_to_step_returns


def _compute_sha256_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_git_commit_sha(repo_root: Path) -> str | None:
    git_dir = repo_root / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    head_value = head_path.read_text(encoding="utf-8").strip()
    if head_value.startswith("ref: "):
        ref_path = git_dir / head_value[5:]
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip() or None
        return None
    return head_value or None


def _fingerprint_mapping(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class AgainInferenceAdapter:
    adapter_name = "again_inference"

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.config
        self._sessions_by_ticker: dict[str, tuple[object, dict, object]] = {}
        self._runtime = None

    def reset(self) -> None:
        for _, _, model in self._sessions_by_ticker.values():
            del model
        self._sessions_by_ticker = {}
        if self._runtime is None:
            self._runtime = resolve_execution_context(self.config, purpose="predict")
        if self._runtime.uses_cuda:
            torch.cuda.empty_cache()

    def fetch_market_data(self, definition: BenchmarkDefinition, as_of_timestamp: datetime) -> pd.DataFrame:
        fetcher = DataFetcher(self.config_manager, definition.lookback_years)
        start_date = pd.Timestamp(as_of_timestamp).tz_localize(None) - pd.Timedelta(days=definition.lookback_years * 365)
        market_data = fetcher.fetch_many_stocks(list(definition.tickers), start_date.to_pydatetime(), as_of_timestamp)
        if market_data.empty:
            raise BenchmarkAdapterError("No se han podido descargar datos para el benchmark")
        market_data = market_data.copy()
        market_data["Date"] = pd.to_datetime(market_data["Date"]).dt.tz_localize(None)
        return market_data

    def _ensure_session(self, ticker: str, observed_data: pd.DataFrame) -> tuple[object, dict, object]:
        if ticker not in self._sessions_by_ticker:
            _, dataset, normalizers, model = load_data_and_model(
                self.config,
                ticker,
                historical_mode=True,
                years=self.config["prediction"]["years"],
                raw_data=observed_data,
            )
            self._sessions_by_ticker[ticker] = (dataset, normalizers, model)
        return self._sessions_by_ticker[ticker]

    def evaluate_ticker(
        self,
        definition: BenchmarkDefinition,
        market_data: pd.DataFrame,
        ticker: str,
        split_date: datetime,
    ) -> BenchmarkTickerResult:
        full_data = market_data[market_data["Ticker"] == ticker].copy()
        if full_data.empty:
            raise BenchmarkAdapterError(f"El snapshot no contiene datos para {ticker}")
        full_data["Date"] = pd.to_datetime(full_data["Date"]).dt.tz_localize(None)
        full_data = full_data.sort_values("Date")
        observed_data = full_data[full_data["Date"] <= split_date].copy()
        if observed_data.empty:
            raise BenchmarkAdapterError(f"No hay historial observado para {ticker} antes del split")

        dataset, normalizers, model = self._ensure_session(ticker, observed_data)
        processed_df, original_close = preprocess_data(
            self.config,
            observed_data.copy(),
            ticker,
            normalizers,
            historical_mode=True,
        )
        with torch.no_grad():
            median, _, _, details = generate_predictions(
                self.config,
                dataset,
                model,
                processed_df,
                return_details=True,
                raw_ticker_data=observed_data.copy(),
            )

        future_actual = full_data.set_index("Date")["Close"]
        future_actual = future_actual[future_actual.index > pd.Timestamp(split_date)].sort_index().iloc[: definition.horizon]
        if len(future_actual) != len(median):
            raise BenchmarkAdapterError(
                f"El ticker {ticker} no tiene suficientes observaciones reales para el horizonte {definition.horizon}"
            )

        pred_dates = tuple(pd.Timestamp(date).to_pydatetime() for date in future_actual.index)
        actual_close = future_actual.to_numpy(dtype=float)
        predicted_close = np.asarray(median, dtype=float)
        last_observed_close = float(original_close.iloc[-1])
        actual_returns = price_path_to_step_returns(last_observed_close, actual_close)
        predicted_returns = details.get("relative_returns_median")
        if predicted_returns is None:
            predicted_returns = price_path_to_step_returns(last_observed_close, predicted_close)
        metrics = compute_metric_values(predicted_close, actual_close, predicted_returns, actual_returns)
        return BenchmarkTickerResult(
            ticker=ticker,
            split_date=pd.Timestamp(split_date).to_pydatetime(),
            forecast_dates=pred_dates,
            historical_dates=tuple(pd.Timestamp(value).to_pydatetime() for value in observed_data["Date"].tolist()),
            historical_close=tuple(float(value) for value in observed_data["Close"].tolist()),
            actual_close=tuple(float(value) for value in actual_close.tolist()),
            predicted_close=tuple(float(value) for value in predicted_close.tolist()),
            metrics=metrics,
            last_observed_close=last_observed_close,
        )

    def get_model_metadata(self) -> dict[str, str | None]:
        config_path = Path(str(self.config.get("_meta", {}).get("config_path", "config/config.yaml"))).resolve()
        repo_root = config_path.parents[1]
        model_name = str(self.config["model_name"])
        model_path = Path(self.config["paths"]["models_dir"]) / f"{model_name}.pth"
        normalizers_path = Path(self.config["paths"]["normalizers_dir"]) / f"{model_name}_normalizers.pkl"
        dataset_path = Path(self.config["data"]["processed_data_path"])
        return {
            "model_name": model_name,
            "profile_path": str(config_path),
            "model_sha256": _compute_sha256_if_exists(model_path),
            "normalizers_sha256": _compute_sha256_if_exists(normalizers_path),
            "dataset_sha256": _compute_sha256_if_exists(dataset_path),
            "code_commit_sha": _resolve_git_commit_sha(repo_root),
        }

    def get_runtime_metadata(self) -> dict[str, str | None]:
        if self._runtime is None:
            self._runtime = resolve_execution_context(self.config, purpose="predict")
        runtime_display = self._runtime.to_display_dict()
        return {
            "python_version": sys.version.split()[0],
            "torch_version": getattr(torch, "__version__", None),
            "backend": runtime_display.get("backend"),
            "device": runtime_display.get("gpu_name"),
        }

    def get_config_fingerprint(self) -> str | None:
        payload = {
            "model_name": self.config.get("model_name"),
            "config_path": self.config.get("_meta", {}).get("config_path"),
            "model": self.config.get("model"),
            "prediction": self.config.get("prediction"),
            "paths": {
                "models_dir": self.config.get("paths", {}).get("models_dir"),
                "normalizers_dir": self.config.get("paths", {}).get("normalizers_dir"),
            },
        }
        return _fingerprint_mapping(payload)
