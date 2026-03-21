from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging
from pathlib import Path
import tempfile

import pandas as pd
import yaml
import yfinance as yf

from .runtime_config import ConfigManager

logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, config_manager: ConfigManager, years: int):
        self.config = config_manager.config
        self.years = years
        self.tickers_file = Path(self.config["data"]["tickers_file"])
        self.raw_data_path = Path(self.config["data"]["raw_data_path"])
        self.yfinance_cache_dir = Path(
            self.config.get("paths", {}).get(
                "yfinance_cache_dir",
                Path(tempfile.gettempdir()) / "predictor_bursatil_tft" / "yfinance_cache",
            )
        )
        self.extra_days = 50
        self.max_workers = min(8, max(1, int(self.config["training"].get("num_workers", 4))))
        self._configure_yfinance_cache()

    def _configure_yfinance_cache(self) -> None:
        self.yfinance_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            yf.set_tz_cache_location(str(self.yfinance_cache_dir))
        except Exception as exc:
            logger.warning("No se ha podido fijar la cache de yfinance en %s: %s", self.yfinance_cache_dir, exc)

    def _load_tickers(self, region: str | None = None) -> list[str]:
        with open(self.tickers_file, "r", encoding="utf-8") as handle:
            tickers_config = yaml.safe_load(handle) or {}
        regions = tickers_config.get("tickers", {})
        if region:
            return list(regions.get(region, {}).keys())
        all_tickers: list[str] = []
        for region_tickers in regions.values():
            all_tickers.extend(region_tickers.keys())
        return list(dict.fromkeys(all_tickers))

    @staticmethod
    def _normalize_date(value: datetime) -> datetime:
        if hasattr(value, "tzinfo") and value.tzinfo is not None:
            return value.replace(tzinfo=None)
        return value

    def fetch_stock_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            start_date = self._normalize_date(start_date)
            end_date = self._normalize_date(end_date)
            adjusted_start_date = start_date - timedelta(days=self.extra_days)

            stock = yf.Ticker(ticker)
            info = stock.info or {}
            first_trade_date = info.get("firstTradeDateEpochUtc")
            if first_trade_date:
                first_trade_timestamp = pd.to_datetime(first_trade_date / 1000, unit="s", utc=True).tz_localize(None)
                if first_trade_timestamp > adjusted_start_date:
                    adjusted_start_date = first_trade_timestamp

            df = stock.history(start=adjusted_start_date, end=end_date, repair=True, auto_adjust=True)
            if df.empty:
                logger.warning("No hay datos para %s", ticker)
                return pd.DataFrame()

            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            df.columns = [col.replace(" ", "_") for col in df.columns]
            sector = info.get("sector", "Unknown")
            if sector not in self.config["model"]["sectors"]:
                sector = "Unknown"
            df["Ticker"] = ticker
            df["Sector"] = sector
            df = df[df["Date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Sector"]
            return df[required_cols]
        except Exception as exc:
            logger.error("Error al descargar %s: %s", ticker, exc)
            return pd.DataFrame()

    def fetch_many_stocks(self, tickers: list[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self.fetch_stock_data, ticker, start_date, end_date): ticker
                for ticker in tickers
            }
            for future in as_completed(future_map):
                ticker = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error("Error al resolver %s: %s", ticker, exc)
                    result = pd.DataFrame()
                if result.empty:
                    logger.warning("Se omite %s por falta de datos", ticker)
                    continue
                results.append(result)

        if not results:
            return pd.DataFrame()

        combined = pd.concat(results, ignore_index=True)
        combined["Sector"] = pd.Categorical(
            combined["Sector"],
            categories=self.config["model"]["sectors"],
            ordered=False,
        )
        return combined

    def fetch_global_stocks(self, region: str | None = None) -> pd.DataFrame:
        end_date = datetime.now().replace(tzinfo=None)
        start_date = end_date - timedelta(days=self.years * 365)
        tickers = self.config.get("data", {}).get("tickers") or self._load_tickers(region)
        combined = self.fetch_many_stocks(tickers, start_date, end_date)
        if combined.empty:
            logger.error("No se ha podido descargar ningun ticker")
            return combined
        self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(self.raw_data_path, index=False)
        logger.info("Datos guardados en %s", self.raw_data_path)
        return combined

    def fetch_stock_data_sync(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        return self.fetch_stock_data(ticker, start_date, end_date)


if __name__ == "__main__":
    config_manager = ConfigManager()
    fetcher = DataFetcher(config_manager, years=3)
    fetcher.fetch_global_stocks()
