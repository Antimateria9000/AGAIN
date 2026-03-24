from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from again_benchmark import BenchmarkModuleConfig, BenchmarkRunner, BenchmarkStorage, BenchmarkUIAdapter
from again_benchmark.adapters import AgainInferenceAdapter, load_legacy_benchmark_rows
from scripts.data_fetcher import DataFetcher
from scripts.prediction_engine import generate_predictions, load_data_and_model, preprocess_data
from scripts.runtime_config import ConfigManager
from scripts.utils.device_utils import resolve_execution_context
from scripts.utils.model_registry import list_model_profiles
from scripts.utils.model_readiness import assess_model_readiness
from scripts.utils.training_universe import resolve_training_universe
from start_training import start_training


class ForecastService:
    def __init__(self, config_manager: ConfigManager, years: int):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.years = years
        self.max_prediction_length = self.config["model"]["max_prediction_length"]
        self.fetcher = DataFetcher(config_manager, years)

    def get_model_readiness(self):
        return assess_model_readiness(self.config)

    def get_runtime_status(self):
        return resolve_execution_context(self.config, purpose="predict").to_display_dict()

    def _ensure_model_ready(self):
        report = self.get_model_readiness()
        if not report.ready:
            details = " | ".join(report.issues)
            raise RuntimeError(f"{report.summary} Detalles: {details}")

    def fetch_stock_data(self, ticker, start_date, end_date):
        return self.fetcher.fetch_stock_data(ticker, start_date, end_date)

    def predict(self, ticker, start_date, end_date):
        self._ensure_model_ready()
        new_data = self.fetch_stock_data(ticker, start_date, end_date)
        if new_data.empty:
            raise ValueError(f"No hay datos para {ticker}")
        new_data["Date"] = pd.to_datetime(new_data["Date"]).dt.tz_localize(None)
        _, dataset, normalizers, model = load_data_and_model(self.config, ticker, raw_data=new_data)
        ticker_data, original_close = preprocess_data(self.config, new_data, ticker, normalizers)
        with torch.no_grad():
            median, lower_bound, upper_bound, details = generate_predictions(
                self.config,
                dataset,
                model,
                ticker_data,
                return_details=True,
                raw_ticker_data=new_data,
            )
        return ticker_data, original_close, median, lower_bound, upper_bound, details

    def predict_historical(self, ticker, start_date, end_date):
        self._ensure_model_ready()
        full_data = self.fetch_stock_data(ticker, start_date, end_date)
        if full_data.empty:
            raise ValueError(f"No hay datos para {ticker}")

        full_data = full_data[full_data["Ticker"] == ticker].copy()
        full_data["Date"] = pd.to_datetime(full_data["Date"]).dt.tz_localize(None)
        full_data = full_data.sort_values("Date")
        unique_dates = full_data["Date"].drop_duplicates().reset_index(drop=True)
        if len(unique_dates) <= self.max_prediction_length:
            raise ValueError("No hay suficiente historial para comparacion historica")

        trim_date = unique_dates.iloc[-(self.max_prediction_length + 1)]
        new_data = full_data[full_data["Date"] <= trim_date].copy()
        _, dataset, normalizers, model = load_data_and_model(self.config, ticker, raw_data=new_data, historical_mode=True)
        ticker_data, original_close = preprocess_data(self.config, new_data, ticker, normalizers, historical_mode=True)
        with torch.no_grad():
            median, lower_bound, upper_bound = generate_predictions(
                self.config,
                dataset,
                model,
                ticker_data,
                raw_ticker_data=new_data,
            )

        historical_close = full_data.set_index("Date")["Close"]
        return ticker_data, original_close, median, lower_bound, upper_bound, historical_close


class BenchmarkService:
    def __init__(self, config_manager: ConfigManager, years: int):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.years = years
        module_config = BenchmarkModuleConfig.from_again_config(self.config)
        storage = BenchmarkStorage(module_config.storage_root)
        adapter = AgainInferenceAdapter(config_manager)
        runner = BenchmarkRunner(storage, adapter)
        self.ui = BenchmarkUIAdapter(storage, runner)

    def get_model_readiness(self):
        return assess_model_readiness(self.config)

    def get_runtime_status(self):
        return resolve_execution_context(self.config, purpose="predict").to_display_dict()

    def ensure_default_definition(self):
        return self.ui.ensure_default_definition(self.config)

    def list_definitions(self):
        return self.ui.list_definitions()

    def run_live(self, definition_id: str, as_of_timestamp: datetime):
        report = self.get_model_readiness()
        if not report.ready:
            details = " | ".join(report.issues)
            raise RuntimeError(f"{report.summary} Detalles: {details}")
        return self.ui.run_live(definition_id, as_of_timestamp=as_of_timestamp)

    def run_frozen(self, definition_id: str, as_of_timestamp: datetime):
        report = self.get_model_readiness()
        if not report.ready:
            details = " | ".join(report.issues)
            raise RuntimeError(f"{report.summary} Detalles: {details}")
        return self.ui.run_frozen(definition_id, as_of_timestamp=as_of_timestamp)

    def rerun_exact(self, run_id: str):
        return self.ui.rerun_from_run_id(run_id)

    def list_runs(
        self,
        *,
        benchmark_id: str | None = None,
        mode: str | None = None,
        model_name: str | None = None,
        run_id: str | None = None,
    ):
        return self.ui.list_runs(benchmark_id=benchmark_id, mode=mode, model_name=model_name, run_id=run_id)

    def load_run_view(self, run_id: str):
        return self.ui.load_run_view(run_id)

    def compare_runs(self, left_run_id: str, right_run_id: str):
        return self.ui.compare_runs(left_run_id, right_run_id)

    def load_legacy_history(self):
        return load_legacy_benchmark_rows(self.config)


class TrainingService:
    def __init__(self, base_config_manager: ConfigManager):
        self.base_config_manager = base_config_manager
        self.base_config = base_config_manager.config

    def preview_universe(
        self,
        mode: str,
        single_ticker_symbol: str | None = None,
        predefined_group_name: str | None = None,
    ):
        return resolve_training_universe(
            self.base_config,
            mode=mode,
            single_ticker_symbol=single_ticker_symbol,
            predefined_group_name=predefined_group_name,
        )

    def list_registered_profiles(self) -> list[dict]:
        return list_model_profiles(self.base_config)

    def get_runtime_status(self, config: dict | None = None):
        target_config = config or self.base_config
        return resolve_execution_context(target_config, purpose="train").to_display_dict()

    def train(
        self,
        mode: str,
        years: int,
        use_optuna: bool,
        single_ticker_symbol: str | None = None,
        predefined_group_name: str | None = None,
    ):
        base_config_path = self.base_config.get("_meta", {}).get("config_path")
        return start_training(
            config_path=base_config_path,
            years=years,
            use_optuna=use_optuna,
            continue_training=False,
            training_universe_mode=mode,
            single_ticker_symbol=single_ticker_symbol,
            predefined_group_name=predefined_group_name,
        )

    @staticmethod
    def format_profile_label(profile_entry: dict) -> str:
        label = str(profile_entry.get("label") or profile_entry.get("model_name") or "modelo")
        model_name = str(profile_entry.get("model_name") or "")
        return f"{label} [{model_name}]"

    @staticmethod
    def normalize_profile_path(profile_path: str | None) -> str | None:
        if not profile_path:
            return None
        return str(Path(profile_path))

