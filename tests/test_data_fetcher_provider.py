import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock

import pandas as pd
from yfinance.exceptions import YFRateLimitError

from scripts.data_fetcher import DataFetcher
from scripts.runtime_config import ConfigManager
from scripts.utils.yfinance_provider import DownloadAttempt, FetchMetadata, FetchResult, YFinanceProvider


def build_metadata(symbol: str, backend_used: str | None = None) -> FetchMetadata:
    return FetchMetadata(
        provider_name="test-provider",
        provider_version="1.0.0",
        source="test",
        request_id="req-test",
        requested_symbol=symbol,
        resolved_symbol=symbol,
        requested_interval="1d",
        resolved_interval="1d",
        requested_start="2024-01-01T00:00:00",
        requested_end="2024-01-10T00:00:00",
        effective_start="2024-01-01T00:00:00",
        effective_end="2024-01-10T00:00:00",
        actual_start=None,
        actual_end=None,
        extracted_at_utc="2024-01-10T00:00:00+00:00",
        auto_adjust=True,
        actions=False,
        repair=True,
        backend_used=backend_used,
    )


class DataFetcherProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "config").mkdir(parents=True, exist_ok=True)
        (self.root / "models" / "normalizers").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "train").mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)

        config_text = f"""
model_name: smoke_model
model:
  max_prediction_length: 5
  min_encoder_length: 10
  max_encoder_length: 20
  hidden_size: 8
  attention_head_size: 1
  dropout: 0.1
  learning_rate: 0.001
  lstm_layers: 1
  use_quantile_loss: true
  quantiles: [0.1, 0.5, 0.9]
  embedding_sizes:
    Sector: [3, 2]
    Day_of_Week: [7, 4]
    Month: [12, 6]
  sectors: [Technology, Unknown]
  tuning:
    min_hidden_size: 8
    max_hidden_size: 8
    min_hidden_continuous_size: 4
    max_hidden_continuous_size: 4
    min_lstm_layers: 1
    max_lstm_layers: 1
    min_attention_head_size: 1
    max_attention_head_size: 1
    min_dropout: 0.1
    max_dropout: 0.1
    min_learning_rate: 0.001
    max_learning_rate: 0.001
training:
  seed: 42
  max_epochs: 1
  optuna_trials: 1
  num_workers: 2
  prefetch_factor: 2
  early_stopping_patience: 1
  reduce_lr_patience: 1
  reduce_lr_factor: 0.5
  weight_decay: 0.0
  batch_size: 8
  auto_batch_size: false
  max_vram_usage: 0.8
training_universe:
  minimum_group_tickers: 2
prediction:
  years: 1
  batch_size: 8
validation:
  debug: false
  enable_detailed_validation: false
  max_validation_batches_to_log: 0
  save_plots: false
  max_plots_per_epoch: 0
data:
  valid_regions: [usa]
  raw_data_path: {self.root / "data" / "stock_data.csv"}
  processed_data_path: {self.root / "data" / "train" / "processed_dataset.pt"}
  tickers_file: {self.root / "config" / "tickers.yaml"}
  benchmark_tickers_file: {self.root / "config" / "benchmark_tickers.yaml"}
  train_processed_df_path: {self.root / "data" / "train" / "train_processed_df.parquet"}
  val_processed_df_path: {self.root / "data" / "train" / "val_processed_df.parquet"}
data_fetch:
  session_backend: requests
  max_workers: 1
  retries: 2
  timeout: 1.0
  min_delay: 0.0
  auto_reset_cookie_cache: true
  trust_env_proxies: false
  use_yfinance_sector_lookup: false
  yfinance_cache_dir: {self.root / "data" / "cache" / "yfinance"}
paths:
  data_dir: {self.root / "data"}
  models_dir: {self.root / "models"}
  normalizers_dir: {self.root / "models" / "normalizers"}
  config_dir: {self.root / "config"}
  runtime_profiles_dir: {self.root / "config" / "runtime_profiles"}
  training_universes_path: {self.root / "config" / "training_universes.yaml"}
  model_registry_path: {self.root / "config" / "model_registry.yaml"}
  logs_dir: {self.root / "logs"}
  benchmark_history_db_path: {self.root / "data" / "benchmarks_history.sqlite"}
artifacts:
  require_hash_validation: true
"""
        self.config_path = self.root / "config" / "config.yaml"
        self.config_path.write_text(config_text, encoding="utf-8")
        (self.root / "config" / "tickers.yaml").write_text("tickers:\n  usa:\n    AAPL: Apple\n    MSFT: Microsoft\n", encoding="utf-8")
        (self.root / "config" / "benchmark_tickers.yaml").write_text("tickers:\n  usa:\n    AAPL: Apple\n", encoding="utf-8")
        (self.root / "config" / "training_universes.yaml").write_text(
            "groups:\n  smoke_group:\n    label: Grupo smoke\n    enabled: true\n    tickers:\n      - AAPL\n      - MSFT\n",
            encoding="utf-8",
        )
        self.config_manager = ConfigManager(str(self.config_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _history_frame() -> pd.DataFrame:
        index = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Adj Close": [100.5, 101.5, 102.5],
                "Volume": [1_000_000, 1_100_000, 1_200_000],
                "Dividends": [0.0, 0.0, 0.0],
                "Stock Splits": [0.0, 0.0, 0.0],
            },
            index=index,
        )

    def test_provider_cambia_al_backend_download_si_ticker_falla(self):
        with mock.patch("scripts.utils.yfinance_provider.yf.Ticker") as ticker_cls, mock.patch(
            "scripts.utils.yfinance_provider.yf.download"
        ) as download_fn:
            ticker_cls.return_value.history.side_effect = RuntimeError("fallo en ticker.history")
            download_fn.return_value = self._history_frame()

            provider = YFinanceProvider(max_workers=1, retries=1, min_delay=0.0, timeout=1.0)
            result = provider.get_history_bundle(
                symbols="AAPL",
                start=pd.Timestamp("2024-01-01"),
                end=pd.Timestamp("2024-01-10"),
                interval="1d",
                auto_adjust=True,
                actions=False,
            )

        self.assertIsInstance(result, FetchResult)
        self.assertFalse(result.data.empty)
        self.assertEqual(result.metadata.backend_used, "download")
        self.assertEqual(list(result.data.columns), ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"])
        self.assertEqual(result.data.attrs["provider_metadata"]["backend_used"], "download")

    def test_provider_usa_requests_como_backend_seguro_por_defecto(self):
        provider = YFinanceProvider(max_workers=1, retries=1, min_delay=0.0, timeout=1.0)

        self.assertEqual(provider.session_backend, "requests")

    def test_fetch_stock_data_usa_cache_local_si_falla_la_descarga(self):
        cached = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1_000_000, 1_100_000, 1_200_000],
                "Ticker": ["AAPL", "AAPL", "AAPL"],
                "Sector": ["Technology", "Technology", "Technology"],
            }
        )
        raw_path = Path(self.config_manager.config["data"]["raw_data_path"])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(raw_path, index=False)

        fetcher = DataFetcher(self.config_manager, years=1)
        with mock.patch.object(fetcher.provider, "get_history_bundle", side_effect=RuntimeError("rate limit")):
            result = fetcher.fetch_stock_data(
                "AAPL",
                pd.Timestamp("2024-01-02").to_pydatetime(),
                pd.Timestamp("2024-01-10").to_pydatetime(),
            )

        self.assertFalse(result.empty)
        self.assertEqual(list(result.columns), ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Sector"])
        self.assertTrue((result["Ticker"] == "AAPL").all())
        self.assertTrue((result["Sector"] == "Technology").all())

    def test_fetch_many_stocks_mantiene_el_contrato_actual(self):
        fetcher = DataFetcher(self.config_manager, years=1)
        frame_aapl = self._history_frame()
        frame_msft = self._history_frame()
        bundle = {
            "AAPL": FetchResult(symbol="AAPL", data=frame_aapl, metadata=build_metadata("AAPL", backend_used="ticker")),
            "MSFT": FetchResult(symbol="MSFT", data=frame_msft, metadata=build_metadata("MSFT", backend_used="download")),
        }

        with mock.patch.object(fetcher.provider, "get_history_bundle", return_value=bundle), mock.patch.object(
            fetcher, "_resolve_sector", return_value="Technology"
        ):
            result = fetcher.fetch_many_stocks(
                ["AAPL", "MSFT"],
                pd.Timestamp("2024-01-02").to_pydatetime(),
                pd.Timestamp("2024-01-10").to_pydatetime(),
            )

        self.assertFalse(result.empty)
        self.assertEqual(list(result.columns), ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Sector"])
        self.assertEqual(set(result["Ticker"].unique()), {"AAPL", "MSFT"})
        self.assertTrue((result["Sector"] == "Technology").all())

    def test_provider_reinicia_cache_si_detecta_rate_limit(self):
        provider = YFinanceProvider(max_workers=1, retries=1, min_delay=0.0, timeout=1.0, auto_reset_cookie_cache=True)
        frame = self._history_frame().reset_index().rename(columns={"index": "Date"})

        calls = {"count": 0}

        def fake_download_range(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise YFRateLimitError()
            return provider._normalize_raw_history(frame.set_index("Date"), "AAPL", "1d"), False, 1

        with mock.patch.object(provider, "_download_range", side_effect=fake_download_range), mock.patch.object(
            provider, "_reset_cookie_runtime"
        ) as reset_mock:
            result = provider.get_history_bundle(
                symbols="AAPL",
                start=pd.Timestamp("2024-01-01"),
                end=pd.Timestamp("2024-01-10"),
                interval="1d",
                auto_adjust=True,
                actions=False,
            )

        self.assertFalse(result.data.empty)
        self.assertEqual(result.metadata.backend_used, "download")
        self.assertTrue(any("cookie/crumb" in warning for warning in result.metadata.warnings))
        reset_mock.assert_called_once()

    def test_provider_sanea_la_cache_corrupta_antes_de_descargar(self):
        provider = YFinanceProvider(max_workers=1, retries=1, min_delay=0.0, timeout=1.0, auto_reset_cookie_cache=True)
        frame = self._history_frame().reset_index().rename(columns={"index": "Date"})

        with mock.patch.object(
            provider,
            "_inspect_cookie_cache_issues",
            return_value=["La entrada persistente 'basic' es de tipo str"],
        ), mock.patch.object(provider, "_reset_cookie_runtime") as reset_mock, mock.patch.object(
            provider,
            "_download_range",
            return_value=(provider._normalize_raw_history(frame.set_index("Date"), "AAPL", "1d"), False, 1),
        ):
            result = provider.get_history_bundle(
                symbols="AAPL",
                start=pd.Timestamp("2024-01-01"),
                end=pd.Timestamp("2024-01-10"),
                interval="1d",
                auto_adjust=True,
                actions=False,
            )

        self.assertFalse(result.data.empty)
        self.assertEqual(result.metadata.backend_used, "ticker")
        reset_mock.assert_called_once()

    def test_fetch_many_stocks_reporta_el_detalle_real_del_provider(self):
        fetcher = DataFetcher(self.config_manager, years=1)
        failure_result = FetchResult(
            symbol="AAPL",
            data=pd.DataFrame(),
            metadata=build_metadata("AAPL", backend_used=None),
        )
        failure_result.metadata.attempts.append(
            DownloadAttempt(
                attempt_number=1,
                backend="ticker",
                interval="1d",
                start="2024-01-01T00:00:00",
                end="2024-01-10T00:00:00",
                duration_seconds=0.01,
                success=False,
                rows=0,
                error="AttributeError(\"'str' object has no attribute 'name'\")",
            )
        )

        with mock.patch.object(fetcher.provider, "get_history_bundle", return_value={"AAPL": failure_result}), mock.patch.object(
            fetcher, "_fallback_from_local_raw_data", return_value=fetcher._empty_output_frame()
        ):
            combined, report = fetcher.fetch_many_stocks_with_report(
                ["AAPL"],
                pd.Timestamp("2024-01-02").to_pydatetime(),
                pd.Timestamp("2024-01-10").to_pydatetime(),
            )

        self.assertTrue(combined.empty)
        self.assertEqual(report.discarded_tickers, ["AAPL"])
        self.assertIn("AttributeError", report.discarded_details["AAPL"])

    def test_fetch_stock_data_no_consulta_info_si_el_lookup_de_sector_esta_desactivado(self):
        fetcher = DataFetcher(self.config_manager, years=1)
        frame = self._history_frame()
        bundle = FetchResult(symbol="AAPL", data=frame, metadata=build_metadata("AAPL", backend_used="ticker"))

        with mock.patch.object(fetcher.provider, "get_history_bundle", return_value=bundle), mock.patch(
            "scripts.data_fetcher.yf.Ticker"
        ) as ticker_cls:
            result = fetcher.fetch_stock_data(
                "AAPL",
                pd.Timestamp("2024-01-02").to_pydatetime(),
                pd.Timestamp("2024-01-10").to_pydatetime(),
            )

        self.assertFalse(result.empty)
        self.assertTrue((result["Sector"] == "Unknown").all())
        ticker_cls.assert_not_called()

    def test_provider_no_hereda_proxies_del_entorno_por_defecto(self):
        provider = YFinanceProvider(max_workers=1, retries=1, min_delay=0.0, timeout=1.0, trust_env_proxies=False)
        session = provider._build_session()

        self.assertTrue(hasattr(session, "trust_env"))
        self.assertFalse(session.trust_env)
        if hasattr(session, "proxies"):
            self.assertEqual(getattr(session, "proxies"), {})

    def test_provider_limpia_proxies_del_entorno_durante_la_descarga(self):
        provider = YFinanceProvider(max_workers=1, retries=1, min_delay=0.0, timeout=1.0, trust_env_proxies=False)
        original_proxy = os.environ.get("HTTP_PROXY")
        os.environ["HTTP_PROXY"] = "http://127.0.0.1:9"

        def fake_download(**kwargs):
            self.assertNotIn("HTTP_PROXY", os.environ)
            return self._history_frame()

        try:
            with mock.patch("scripts.utils.yfinance_provider.yf.download", side_effect=fake_download):
                result = provider.get_history_bundle(
                    symbols="AAPL",
                    start=pd.Timestamp("2024-01-01"),
                    end=pd.Timestamp("2024-01-10"),
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                )
        finally:
            if original_proxy is None:
                os.environ.pop("HTTP_PROXY", None)
            else:
                os.environ["HTTP_PROXY"] = original_proxy

        self.assertFalse(result.data.empty)


if __name__ == "__main__":
    unittest.main()
