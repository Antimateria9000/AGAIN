import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml

from scripts.data_fetcher import FetchUniverseReport
from scripts.runtime_config import ConfigManager
from scripts.utils.model_registry import get_active_profile_path, list_model_profiles, register_model_profile
from scripts.utils.training_universe import (
    build_runtime_training_config,
    build_training_model_name,
    resolve_training_universe,
    save_runtime_profile,
)
from start_training import start_training


class TrainingUniverseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "config").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "train").mkdir(parents=True, exist_ok=True)
        (self.root / "models" / "normalizers").mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)

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
                "embedding_sizes": {
                    "Sector": [3, 2],
                    "Day_of_Week": [7, 4],
                    "Month": [12, 6],
                },
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
            "training_universe": {
                "minimum_group_tickers": 2,
            },
            "prediction": {
                "years": 3,
                "batch_size": 8,
            },
            "validation": {
                "debug": False,
                "enable_detailed_validation": False,
                "max_validation_batches_to_log": 0,
                "save_plots": False,
                "max_plots_per_epoch": 0,
            },
            "data": {
                "valid_regions": ["global", "all"],
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
            "artifacts": {
                "require_hash_validation": True,
            },
        }

        (self.root / "config" / "config.yaml").write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
        (self.root / "config" / "tickers.yaml").write_text("tickers:\n  global:\n    AAPL: Apple\n", encoding="utf-8")
        (self.root / "config" / "benchmark_tickers.yaml").write_text("tickers:\n  global:\n    AAPL: Apple\n", encoding="utf-8")
        (self.root / "config" / "training_universes.yaml").write_text(
            "groups:\n"
            "  bbva_peer_banks:\n"
            "    label: BBVA + peer set bancario curado\n"
            "    description: Grupo bancario\n"
            "    notes: Curado\n"
            "    enabled: true\n"
            "    tickers:\n"
            "      - BBVA.MC\n"
            "      - SAN.MC\n",
            encoding="utf-8",
        )
        self.config = ConfigManager(str(self.root / "config" / "config.yaml")).config

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_resolve_single_ticker_universe(self):
        universe = resolve_training_universe(self.config, mode="single_ticker", single_ticker_symbol="bbva.mc")
        self.assertEqual(universe.mode, "single_ticker")
        self.assertEqual(universe.single_ticker_symbol, "BBVA.MC")
        self.assertEqual(universe.tickers, ["BBVA.MC"])

    def test_resolve_predefined_group_universe(self):
        universe = resolve_training_universe(self.config, mode="predefined_group", predefined_group_name="bbva_peer_banks")
        self.assertEqual(universe.mode, "predefined_group")
        self.assertEqual(universe.predefined_group_name, "bbva_peer_banks")
        self.assertEqual(universe.tickers, ["BBVA.MC", "SAN.MC"])

    def test_error_if_group_does_not_exist(self):
        with self.assertRaises(ValueError):
            resolve_training_universe(self.config, mode="predefined_group", predefined_group_name="grupo_inexistente")

    def test_error_if_single_ticker_is_empty_or_invalid(self):
        with self.assertRaises(ValueError):
            resolve_training_universe(self.config, mode="single_ticker", single_ticker_symbol="")
        with self.assertRaises(ValueError):
            resolve_training_universe(self.config, mode="single_ticker", single_ticker_symbol="BAD TICKER")

    def test_generated_model_name_and_paths_do_not_collide(self):
        single_universe = resolve_training_universe(self.config, mode="single_ticker", single_ticker_symbol="BBVA.MC")
        group_universe = resolve_training_universe(self.config, mode="predefined_group", predefined_group_name="bbva_peer_banks")

        single_model_name = build_training_model_name(self.config["model_name"], single_universe)
        group_model_name = build_training_model_name(self.config["model_name"], group_universe)
        self.assertNotEqual(single_model_name, group_model_name)
        self.assertEqual(single_model_name, "Gen6_1__single__BBVA_MC")
        self.assertEqual(group_model_name, "Gen6_1__group__bbva_peer_banks")

        single_config = build_runtime_training_config(self.config, single_universe, years=3)
        group_config = build_runtime_training_config(self.config, group_universe, years=3)
        self.assertNotEqual(single_config["data"]["processed_data_path"], group_config["data"]["processed_data_path"])
        self.assertNotEqual(single_config["paths"]["model_save_path"], group_config["paths"]["model_save_path"])

    def test_runtime_profile_and_registry_integration(self):
        universe = resolve_training_universe(self.config, mode="single_ticker", single_ticker_symbol="BBVA.MC")
        runtime_config = build_runtime_training_config(self.config, universe, years=3)
        profile_path = save_runtime_profile(runtime_config)
        self.assertTrue(profile_path.exists())

        register_model_profile(
            self.config,
            {
                "model_name": runtime_config["model_name"],
                "profile_path": str(profile_path),
                "label": runtime_config["training_run"]["label"],
                "description": runtime_config["training_run"]["description"],
                "notes": runtime_config["training_run"]["notes"],
                "universe_mode": runtime_config["training_run"]["mode"],
                "single_ticker_symbol": runtime_config["training_run"]["single_ticker_symbol"],
                "predefined_group_name": runtime_config["training_run"]["predefined_group_name"],
                "requested_tickers": runtime_config["training_run"]["requested_tickers"],
                "downloaded_tickers": [],
                "discarded_tickers": [],
                "final_tickers_used": [],
                "created_at": runtime_config["training_run"]["trained_at"],
                "last_trained_at": runtime_config["training_run"]["trained_at"],
                "prediction_horizon": runtime_config["training_run"]["prediction_horizon"],
                "years": runtime_config["training_run"]["years"],
            },
        )

        self.assertEqual(get_active_profile_path(self.config), str(profile_path))
        profiles = list_model_profiles(self.config)
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["model_name"], runtime_config["model_name"])

    @mock.patch("start_training.train_model")
    @mock.patch("start_training.DataPreprocessor")
    @mock.patch("start_training.DataFetcher")
    def test_start_training_integration_uses_the_selected_universe(self, fetcher_cls, preprocessor_cls, train_model_mock):
        sample_df = pd.DataFrame(
            [
                {
                    "Date": pd.Timestamp("2024-01-02"),
                    "Open": 10.0,
                    "High": 11.0,
                    "Low": 9.0,
                    "Close": 10.5,
                    "Volume": 1000,
                    "Ticker": "BBVA.MC",
                    "Sector": "Financials",
                }
            ]
        )
        fetcher_cls.return_value.fetch_training_universe.return_value = (
            sample_df,
            FetchUniverseReport(
                requested_tickers=["BBVA.MC"],
                successful_tickers=["BBVA.MC"],
                discarded_tickers=[],
                discarded_details={},
            ),
        )
        preprocessor_cls.return_value.process_data.return_value = ("train_dataset", "val_dataset")
        train_model_mock.return_value = object()

        result = start_training(
            config_path=str(self.root / "config" / "config.yaml"),
            years=3,
            use_optuna=False,
            continue_training=False,
            training_universe_mode="single_ticker",
            single_ticker_symbol="BBVA.MC",
        )

        fetcher_cls.return_value.fetch_training_universe.assert_called_once_with(["BBVA.MC"])
        preprocessor_cls.return_value.process_data.assert_called_once()
        train_model_mock.assert_called_once()
        self.assertEqual(result.universe_mode, "single_ticker")
        self.assertEqual(result.requested_tickers, ["BBVA.MC"])
        self.assertEqual(result.downloaded_tickers, ["BBVA.MC"])
        self.assertEqual(result.model_name, "Gen6_1__single__BBVA_MC")
        self.assertTrue(str(result.profile_path).endswith("Gen6_1__single__BBVA_MC.yaml"))
        self.assertEqual(get_active_profile_path(self.config), str(result.profile_path))
