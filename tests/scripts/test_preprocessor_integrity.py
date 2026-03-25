import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock


try:
    import pandas as pd

    import scripts.preprocessor as preprocessor_module
    from scripts.runtime_config import ConfigManager
    from scripts.utils.data_schema import NUMERIC_FEATURES, TARGET_COLUMN

    DataPreprocessor = preprocessor_module.DataPreprocessor
    OriginalTorchNormalizer = preprocessor_module.TorchNormalizer
    DEPS_AVAILABLE = True
except Exception:
    DEPS_AVAILABLE = False


@unittest.skipUnless(DEPS_AVAILABLE, "Dependencias de runtime no instaladas")
class PreprocessorIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "config").mkdir(parents=True, exist_ok=True)
        (self.root / "models" / "normalizers").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "train").mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)

        config_text = f"""
model_name: preprocessor_smoke
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

    @staticmethod
    def _build_synthetic_df(periods: int = 120) -> pd.DataFrame:
        dates = pd.bdate_range("2023-01-02", periods=periods)
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

    def test_process_data_lanza_un_error_semantico_si_split_con_gap_queda_vacio(self):
        preprocessor = DataPreprocessor(self.config)
        short_frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "Ticker": ["AAPL", "AAPL", "AAPL"],
            }
        )

        with mock.patch.object(preprocessor, "_apply_shared_transformations", return_value=(short_frame, None, [])):
            with self.assertRaisesRegex(ValueError, "Split train/val vacio"):
                preprocessor.process_data(mode="train", df=short_frame)

    def test_metadata_de_normalizadores_refleja_las_features_que_realmente_se_guardan(self):
        FailingFeatureNormalizer.created = 0
        preprocessor = DataPreprocessor(self.config)
        df = self._build_synthetic_df()

        with mock.patch.object(preprocessor_module, "TorchNormalizer", FailingFeatureNormalizer):
            preprocessor.process_data(mode="train", df=df)

        normalizers_path = Path(self.config["paths"]["normalizers_dir"]) / f"{self.config['model_name']}_normalizers.pkl"
        with open(normalizers_path, "rb") as handle:
            payload = pickle.load(handle)

        failed_feature = NUMERIC_FEATURES[0]
        self.assertNotIn(failed_feature, payload["normalizers"])
        self.assertNotIn(failed_feature, payload["metadata"]["numeric_features"])
        self.assertNotIn(failed_feature, payload["metadata"]["normalizer_keys"])
        self.assertIn(TARGET_COLUMN, payload["metadata"]["normalizer_keys"])


if DEPS_AVAILABLE:
    class FailingFeatureNormalizer(OriginalTorchNormalizer):
        created = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.instance_id = FailingFeatureNormalizer.created
            FailingFeatureNormalizer.created += 1

        def fit_transform(self, values, *args, **kwargs):
            if self.instance_id == 1:
                raise RuntimeError("fallo forzado en la feature")
            return super().fit_transform(values, *args, **kwargs)

        def transform(self, values, *args, **kwargs):
            if self.instance_id == 1:
                raise RuntimeError("fallo forzado en la feature")
            return super().transform(values, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()
