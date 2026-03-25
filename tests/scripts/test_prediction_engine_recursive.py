import os
import tempfile
import unittest
from pathlib import Path


try:
    import numpy as np
    import pandas as pd

    from scripts.preprocessor import DataPreprocessor
    from scripts.prediction_engine import generate_predictions, load_data_and_model, preprocess_data
    from scripts.runtime_config import ConfigManager
    from scripts.train import train_model

    DEPS_AVAILABLE = True
except Exception:
    DEPS_AVAILABLE = False


@unittest.skipUnless(DEPS_AVAILABLE, "Dependencias de runtime no instaladas")
class RecursivePredictionEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "config").mkdir(parents=True, exist_ok=True)
        (self.root / "models" / "normalizers").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "train").mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)

        config_text = f"""
model_name: recursive_smoke
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
  num_workers: 0
  prefetch_factor: 2
  early_stopping_patience: 1
  reduce_lr_patience: 1
  reduce_lr_factor: 0.5
  weight_decay: 0.0
  batch_size: 8
  auto_batch_size: false
  max_vram_usage: 0.8
training_universe:
  minimum_group_tickers: 1
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
        (self.root / "config" / "tickers.yaml").write_text("tickers:\n  usa:\n    AAPL: Apple\n", encoding="utf-8")
        (self.root / "config" / "benchmark_tickers.yaml").write_text("tickers:\n  usa:\n    AAPL: Apple\n", encoding="utf-8")
        (self.root / "config" / "training_universes.yaml").write_text("groups: {}\n", encoding="utf-8")
        os.environ["PREDICTOR_CONFIG_PATH"] = str(self.config_path)
        self.config = ConfigManager(str(self.config_path)).config

    def tearDown(self):
        os.environ.pop("PREDICTOR_CONFIG_PATH", None)
        self.temp_dir.cleanup()

    def _build_synthetic_df(self):
        dates = pd.bdate_range("2023-01-02", periods=120)
        rows = []
        price = 100.0
        for index, date in enumerate(dates):
            price = price * 1.001
            rows.append(
                {
                    "Date": date,
                    "Open": price * 0.99,
                    "High": price * 1.01,
                    "Low": price * 0.98,
                    "Close": price,
                    "Volume": 1_000_000 + index * 1000,
                    "Ticker": "AAPL",
                    "Sector": "Technology",
                }
            )
        return pd.DataFrame(rows)

    def test_recursive_forecast_genera_fechas_estrictamente_futuras_y_horizonte_correcto(self):
        df = self._build_synthetic_df()
        dataset = DataPreprocessor(self.config).process_data(mode="train", df=df)
        model = train_model(dataset, self.config, use_optuna=False, continue_training=False)
        self.assertIsNotNone(model)

        ticker = "AAPL"
        _, dataset_loaded, normalizers, model_loaded = load_data_and_model(self.config, ticker, raw_data=df.copy())
        processed_df, _ = preprocess_data(self.config, df.copy(), ticker, normalizers)
        median, lower_bound, upper_bound, details = generate_predictions(
            self.config,
            dataset_loaded,
            model_loaded,
            processed_df,
            return_details=True,
            raw_ticker_data=df.copy(),
        )

        forecast_dates = pd.to_datetime(details["forecast_dates"])
        last_history_date = pd.Timestamp(df["Date"].max())

        self.assertEqual(details["forecast_mode"], "recursive_one_step")
        self.assertEqual(len(forecast_dates), self.config["model"]["max_prediction_length"])
        self.assertTrue((forecast_dates > last_history_date).all())
        self.assertTrue(forecast_dates.is_monotonic_increasing)
        self.assertEqual(len(forecast_dates.unique()), len(forecast_dates))
        self.assertTrue(np.isfinite(median).all())
        self.assertTrue(np.isfinite(lower_bound).all())
        self.assertTrue(np.isfinite(upper_bound).all())
        self.assertTrue((lower_bound <= median).all())
        self.assertTrue((median <= upper_bound).all())


if __name__ == "__main__":
    unittest.main()
