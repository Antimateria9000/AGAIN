import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from scripts.runtime_config import ConfigManager


class RuntimeConfigTests(unittest.TestCase):
    def test_save_normalizers_propaga_errores_de_persistencia(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "models" / "normalizers").mkdir(parents=True, exist_ok=True)
            (root / "data" / "train").mkdir(parents=True, exist_ok=True)
            (root / "logs").mkdir(parents=True, exist_ok=True)

            config_payload = {
                "model_name": "runtime_test",
                "model": {
                    "max_prediction_length": 5,
                    "min_encoder_length": 10,
                    "max_encoder_length": 20,
                    "hidden_size": 8,
                    "attention_head_size": 1,
                    "dropout": 0.1,
                    "learning_rate": 0.001,
                    "lstm_layers": 1,
                    "use_quantile_loss": True,
                    "quantiles": [0.1, 0.5, 0.9],
                    "embedding_sizes": {"Sector": [3, 2], "Day_of_Week": [7, 4], "Month": [12, 6]},
                    "sectors": ["Financials", "Unknown"],
                    "tuning": {
                        "min_hidden_size": 8,
                        "max_hidden_size": 8,
                        "min_hidden_continuous_size": 4,
                        "max_hidden_continuous_size": 4,
                        "min_lstm_layers": 1,
                        "max_lstm_layers": 1,
                        "min_attention_head_size": 1,
                        "max_attention_head_size": 1,
                        "min_dropout": 0.1,
                        "max_dropout": 0.1,
                        "min_learning_rate": 0.001,
                        "max_learning_rate": 0.001,
                    },
                },
                "training": {
                    "seed": 42,
                    "max_epochs": 1,
                    "optuna_trials": 1,
                    "num_workers": 0,
                    "prefetch_factor": 2,
                    "early_stopping_patience": 1,
                    "reduce_lr_patience": 1,
                    "reduce_lr_factor": 0.5,
                    "weight_decay": 0.0,
                    "batch_size": 8,
                    "auto_batch_size": False,
                    "max_vram_usage": 0.8,
                },
                "training_universe": {"minimum_group_tickers": 1},
                "prediction": {"years": 1, "batch_size": 8},
                "validation": {
                    "debug": False,
                    "enable_detailed_validation": False,
                    "max_validation_batches_to_log": 0,
                    "save_plots": False,
                    "max_plots_per_epoch": 0,
                },
                "data": {
                    "raw_data_path": str(root / "data" / "stock_data.csv"),
                    "processed_data_path": str(root / "data" / "train" / "processed_dataset.pt"),
                    "tickers_file": str(root / "config" / "tickers.yaml"),
                    "benchmark_tickers_file": str(root / "config" / "benchmark_tickers.yaml"),
                    "train_processed_df_path": str(root / "data" / "train" / "train_processed_df.parquet"),
                    "val_processed_df_path": str(root / "data" / "train" / "val_processed_df.parquet"),
                },
                "paths": {
                    "data_dir": str(root / "data"),
                    "models_dir": str(root / "models"),
                    "normalizers_dir": str(root / "models" / "normalizers"),
                    "config_dir": str(root / "config"),
                    "runtime_profiles_dir": str(root / "config" / "runtime_profiles"),
                    "training_universes_path": str(root / "config" / "training_universes.yaml"),
                    "model_registry_path": str(root / "config" / "model_registry.yaml"),
                    "logs_dir": str(root / "logs"),
                    "benchmark_history_db_path": str(root / "data" / "benchmarks_history.sqlite"),
                },
                "artifacts": {"require_hash_validation": True},
            }
            config_path = root / "config" / "config.yaml"
            config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
            manager = ConfigManager(str(config_path))

            with mock.patch("scripts.runtime_config.pickle.dump", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    manager.save_normalizers("runtime_test", {"Close": object()}, metadata={"schema_hash": "abc"})


if __name__ == "__main__":
    unittest.main()
