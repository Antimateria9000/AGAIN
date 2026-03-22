import os
import pickle
import tempfile
import unittest
from pathlib import Path


try:
    import numpy as np
    import pandas as pd

    from scripts.prediction_engine import generate_predictions, load_data_and_model, preprocess_data
    from scripts.preprocessor import DataPreprocessor
    from scripts.runtime_config import ConfigManager
    from scripts.train import train_model
    from scripts.utils.artifact_utils import write_checksum
    from scripts.utils.data_schema import NUMERIC_FEATURES, TARGET_COLUMN, build_schema_hash
    from scripts.utils.model_readiness import assess_model_readiness
    from scripts.utils.prediction_utils import price_path_to_step_returns

    DEPS_AVAILABLE = True
except Exception:
    DEPS_AVAILABLE = False


@unittest.skipUnless(DEPS_AVAILABLE, "Dependencias de runtime no instaladas")
class TargetNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "config").mkdir(parents=True, exist_ok=True)
        (self.root / "models" / "normalizers").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "train").mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)

        config_text = f"""
model_name: target_norm_smoke
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
        (self.root / "config" / "tickers.yaml").write_text(
            "tickers:\n  usa:\n    AAPL: Apple\n    MSFT: Microsoft\n",
            encoding="utf-8",
        )
        (self.root / "config" / "benchmark_tickers.yaml").write_text(
            "tickers:\n  usa:\n    AAPL: Apple\n",
            encoding="utf-8",
        )
        (self.root / "config" / "training_universes.yaml").write_text(
            "groups:\n  smoke_group:\n    label: Grupo smoke\n    enabled: true\n    tickers:\n      - AAPL\n      - MSFT\n",
            encoding="utf-8",
        )
        os.environ["PREDICTOR_CONFIG_PATH"] = str(self.config_path)
        self.config = ConfigManager(str(self.config_path)).config

    def tearDown(self):
        os.environ.pop("PREDICTOR_CONFIG_PATH", None)
        self.temp_dir.cleanup()

    def _build_synthetic_df(self, tickers=None):
        tickers = tickers or [("AAPL", 0.0), ("MSFT", 3.0)]
        dates = pd.bdate_range("2023-01-02", periods=120)
        rows = []
        for ticker, offset in tickers:
            price = 100.0 + offset
            for index, date in enumerate(dates):
                price = price * (1.0 + 0.001 + (0.0002 if ticker == "MSFT" else 0.0))
                rows.append(
                    {
                        "Date": date,
                        "Open": price * 0.99,
                        "High": price * 1.01,
                        "Low": price * 0.98,
                        "Close": price,
                        "Volume": 1_000_000 + index * 1000,
                        "Ticker": ticker,
                        "Sector": "Technology",
                    }
                )
        return pd.DataFrame(rows)

    def _prepare_raw_and_saved_train_frames(self, df):
        preprocessor = DataPreprocessor(self.config)
        shared_df, _, numeric_features = preprocessor._apply_shared_transformations(df.copy(), mode="train")
        raw_train_df, _ = preprocessor._split_with_gap(shared_df)
        preprocessor.process_data(mode="train", df=df.copy())
        saved_train_df = pd.read_parquet(self.config["data"]["train_processed_df_path"])
        raw_train_df = raw_train_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        saved_train_df = saved_train_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        return numeric_features, raw_train_df, saved_train_df

    def test_target_no_se_normaliza_manualmente_en_train_df(self):
        numeric_features, raw_train_df, saved_train_df = self._prepare_raw_and_saved_train_frames(self._build_synthetic_df())

        self.assertNotIn(TARGET_COLUMN, numeric_features)
        np.testing.assert_allclose(
            saved_train_df[TARGET_COLUMN].to_numpy(dtype=float),
            raw_train_df[TARGET_COLUMN].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_features_numericas_no_target_siguen_normalizadas(self):
        _, raw_train_df, saved_train_df = self._prepare_raw_and_saved_train_frames(self._build_synthetic_df())

        self.assertFalse(
            np.allclose(
                saved_train_df["Close"].to_numpy(dtype=float),
                raw_train_df["Close"].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-6,
            )
        )
        self.assertFalse(
            np.allclose(
                saved_train_df["Volume"].to_numpy(dtype=float),
                raw_train_df["Volume"].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-6,
            )
        )

    def test_timeseries_dataset_recibe_target_crudo_y_no_lo_duplica_en_reals(self):
        df = self._build_synthetic_df(tickers=[("AAPL", 0.0)])
        preprocessor = DataPreprocessor(self.config)
        train_dataset, _ = preprocessor.process_data(mode="train", df=df.copy())
        saved_train_df = pd.read_parquet(self.config["data"]["train_processed_df_path"]).sort_values("time_idx").reset_index(drop=True)

        x, y = train_dataset[0]
        encoder_length = self.config["model"]["max_encoder_length"]
        prediction_length = self.config["model"]["max_prediction_length"]

        self.assertNotIn(TARGET_COLUMN, train_dataset.reals)
        np.testing.assert_allclose(
            x["encoder_target"].numpy(),
            saved_train_df[TARGET_COLUMN].iloc[:encoder_length].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            y[0].numpy(),
            saved_train_df[TARGET_COLUMN].iloc[encoder_length:encoder_length + prediction_length].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_generate_predictions_devuelve_retornos_coherentes_con_la_trayectoria(self):
        df = self._build_synthetic_df()
        dataset = DataPreprocessor(self.config).process_data(mode="train", df=df)
        model = train_model(dataset, self.config, use_optuna=False, continue_training=False)
        self.assertIsNotNone(model)

        ticker = "AAPL"
        raw_slice = df[df["Ticker"] == ticker].copy()
        _, dataset_loaded, normalizers, model_loaded = load_data_and_model(self.config, ticker, raw_data=raw_slice)
        processed_df, _ = preprocess_data(self.config, raw_slice, ticker, normalizers)
        median, lower_bound, upper_bound, details = generate_predictions(
            self.config,
            dataset_loaded,
            model_loaded,
            processed_df,
            return_details=True,
        )

        recovered_median_returns = price_path_to_step_returns(details["last_close_denorm"], median)
        recovered_lower_returns = price_path_to_step_returns(details["last_close_denorm"], lower_bound)
        recovered_upper_returns = price_path_to_step_returns(details["last_close_denorm"], upper_bound)

        np.testing.assert_allclose(recovered_median_returns, details["relative_returns_median"], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(recovered_lower_returns, details["relative_returns_lower"], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(recovered_upper_returns, details["relative_returns_upper"], rtol=1e-5, atol=1e-5)

    def test_artefactos_legacy_incompatibles_exigen_regeneracion(self):
        df = self._build_synthetic_df()
        dataset = DataPreprocessor(self.config).process_data(mode="train", df=df)
        model = train_model(dataset, self.config, use_optuna=False, continue_training=False)
        self.assertIsNotNone(model)

        normalizers_path = Path(self.config["paths"]["normalizers_dir"]) / f"{self.config['model_name']}_normalizers.pkl"
        with open(normalizers_path, "rb") as handle:
            payload = pickle.load(handle)

        legacy_numeric_features = [*NUMERIC_FEATURES, TARGET_COLUMN]
        payload["metadata"]["numeric_features"] = legacy_numeric_features
        payload["metadata"]["schema_hash"] = build_schema_hash(self.config, legacy_numeric_features)

        with open(normalizers_path, "wb") as handle:
            pickle.dump(payload, handle)
        write_checksum(normalizers_path)

        report = assess_model_readiness(self.config)
        self.assertFalse(report.ready)
        self.assertTrue(any("normalizadores" in issue.lower() and "esquema activo" in issue.lower() for issue in report.issues))


if __name__ == "__main__":
    unittest.main()
