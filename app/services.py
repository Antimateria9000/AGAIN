from __future__ import annotations

from datetime import datetime

import pandas as pd
import torch

from app.benchmark_utils import run_benchmark
from scripts.data_fetcher import DataFetcher
from scripts.prediction_engine import generate_predictions, load_data_and_model, preprocess_data
from scripts.runtime_config import ConfigManager


class ForecastService:
    def __init__(self, config: dict, years: int):
        self.config = config
        self.years = years
        self.max_prediction_length = config["model"]["max_prediction_length"]
        self.fetcher = DataFetcher(ConfigManager(), years)

    def fetch_stock_data(self, ticker, start_date, end_date):
        return self.fetcher.fetch_stock_data(ticker, start_date, end_date)

    def predict(self, ticker, start_date, end_date):
        new_data = self.fetch_stock_data(ticker, start_date, end_date)
        if new_data.empty:
            raise ValueError(f"No hay datos para {ticker}")
        new_data["Date"] = pd.to_datetime(new_data["Date"]).dt.tz_localize(None)
        _, dataset, normalizers, model = load_data_and_model(self.config, ticker, raw_data=new_data)
        ticker_data, original_close = preprocess_data(self.config, new_data, ticker, normalizers)
        with torch.no_grad():
            median, lower_bound, upper_bound = generate_predictions(self.config, dataset, model, ticker_data)
        return ticker_data, original_close, median, lower_bound, upper_bound

    def predict_historical(self, ticker, start_date, end_date):
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
            median, lower_bound, upper_bound = generate_predictions(self.config, dataset, model, ticker_data)

        historical_close = full_data.set_index("Date")["Close"]
        return ticker_data, original_close, median, lower_bound, upper_bound, historical_close


class BenchmarkService:
    def __init__(self, config: dict, years: int):
        self.config = config
        self.years = years

    def run(self, benchmark_tickers, historical_period_days: int):
        return run_benchmark(
            config=self.config,
            benchmark_tickers=benchmark_tickers,
            years=self.years,
            historical_period_days=historical_period_days,
            run_timestamp=datetime.now().replace(tzinfo=None),
        )

