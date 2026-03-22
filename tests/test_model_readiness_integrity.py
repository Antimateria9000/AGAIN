import pickle
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from scripts.utils.artifact_utils import write_checksum, write_metadata
from scripts.utils.data_schema import build_artifact_metadata
from scripts.utils.model_readiness import assess_model_readiness


class ModelReadinessIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "config").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "train").mkdir(parents=True, exist_ok=True)
        (self.root / "models" / "normalizers").mkdir(parents=True, exist_ok=True)

        config_payload = {
            "model_name": "Gen6_1",
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
                "accelerator": "auto",
                "precision": "auto",
            },
            "training_universe": {
                "minimum_group_tickers": 2,
            },
            "prediction": {"years": 3, "batch_size": 8},
            "validation": {
                "debug": False,
                "enable_detailed_validation": False,
                "max_validation_batches_to_log": 0,
                "save_plots": False,
                "max_plots_per_epoch": 0,
            },
            "data": {
                "valid_regions": ["global"],
                "raw_data_path": str(self.root / "data" / "stock_data.csv"),
                "processed_data_path": str(self.root / "data" / "train" / "processed_dataset.pt"),
                "tickers_file": str(self.root / "config" / "tickers.yaml"),
                "benchmark_tickers_file": str(self.root / "config" / "benchmark_tickers.yaml"),
                "train_processed_df_path": str(self.root / "data" / "train" / "train_processed_df.parquet"),
                "val_processed_df_path": str(self.root / "data" / "train" / "val_processed_df.parquet"),
            },
            "paths": {
                "data_dir": str(self.root / "data"),
                "models_dir": str(self.root / "models"),
                "normalizers_dir": str(self.root / "models" / "normalizers"),
                "config_dir": str(self.root / "config"),
                "runtime_profiles_dir": str(self.root / "config" / "runtime_profiles"),
                "training_universes_path": str(self.root / "config" / "training_universes.yaml"),
                "model_registry_path": str(self.root / "config" / "model_registry.yaml"),
                "logs_dir": str(self.root / "logs"),
                "benchmark_history_db_path": str(self.root / "data" / "benchmarks_history.sqlite"),
            },
            "artifacts": {"require_hash_validation": True},
        }
        self.config_path = self.root / "config" / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
        self.config = config_payload

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_artifacts(self, training_run_metadata: dict) -> None:
        config = dict(self.config)
        config["training_run"] = training_run_metadata
        metadata = build_artifact_metadata(config)

        checkpoint_path = Path(config["paths"]["models_dir"]) / f"{config['model_name']}.pth"
        payload = {"state_dict": {}, "hyperparams": {}, "metadata": metadata}
        torch.save(payload, checkpoint_path)
        write_metadata(checkpoint_path, metadata)
        write_checksum(checkpoint_path)

        normalizers_path = Path(config["paths"]["normalizers_dir"]) / f"{config['model_name']}_normalizers.pkl"
        with open(normalizers_path, "wb") as handle:
            pickle.dump({"normalizers": {}, "metadata": metadata}, handle)
        write_checksum(normalizers_path)

        dataset_path = Path(config["data"]["processed_data_path"])
        dataset_path.write_bytes(b"dummy-dataset")
        write_metadata(dataset_path, metadata)
        write_checksum(dataset_path)

    def test_model_readiness_rejects_semantically_invalid_universe(self):
        self._write_artifacts(
            {
                "mode": "predefined_group",
                "requested_tickers": ["AAPL", "MSFT"],
                "universe_integrity": {
                    "decision": "ABORT",
                    "training_allowed": False,
                    "decision_reasons": ["Falta el ticker ancla"],
                },
            }
        )

        report = assess_model_readiness(self.config)

        self.assertFalse(report.ready)
        self.assertTrue(any("semanticamente invalido" in issue for issue in report.issues))

    def test_model_readiness_accepts_degraded_allowed_with_warning_summary(self):
        self._write_artifacts(
            {
                "mode": "predefined_group",
                "requested_tickers": ["AAPL", "MSFT"],
                "universe_integrity": {
                    "decision": "DEGRADED_ALLOWED",
                    "training_allowed": True,
                    "degraded": True,
                    "summary": "decision=DEGRADED_ALLOWED | requested=2 | usable=2",
                },
            }
        )

        report = assess_model_readiness(self.config)

        self.assertTrue(report.ready)
        self.assertIn("degradado", report.summary.lower())


if __name__ == "__main__":
    unittest.main()
