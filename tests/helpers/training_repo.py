from __future__ import annotations

import json
from pathlib import Path

import yaml


def _minimal_config_payload(*, model_name: str) -> dict:
    return {
        "model_name": model_name,
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
            "raw_data_path": "data/stock_data.csv",
            "processed_data_path": "data/train/processed_dataset.pt",
            "tickers_file": "config/tickers.yaml",
            "benchmark_tickers_file": "config/benchmark_tickers.yaml",
            "train_processed_df_path": "data/train/train_processed_df.parquet",
            "val_processed_df_path": "data/train/val_processed_df.parquet",
        },
        "paths": {
            "data_dir": "data",
            "models_dir": "models",
            "normalizers_dir": "models/normalizers",
            "config_dir": "config",
            "runtime_profiles_dir": "config/runtime_profiles",
            "training_universes_path": "config/training_universes.yaml",
            "model_registry_path": "config/model_registry.yaml",
            "logs_dir": "logs/default",
            "benchmark_history_db_path": "data/benchmarks_history.sqlite",
        },
        "artifacts": {"require_hash_validation": True},
    }


def write_base_repo(root: Path, *, model_name: str = "Gen6_1") -> Path:
    (root / "config" / "runtime_profiles").mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "models" / "normalizers").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[build-system]\nrequires=[]\n", encoding="utf-8")
    (root / "config" / "tickers.yaml").write_text("tickers: {}\n", encoding="utf-8")
    (root / "config" / "benchmark_tickers.yaml").write_text("tickers: {}\n", encoding="utf-8")
    (root / "config" / "training_universes.yaml").write_text("groups: {}\n", encoding="utf-8")
    config_path = root / "config" / "config.yaml"
    config_path.write_text(yaml.safe_dump(_minimal_config_payload(model_name=model_name), sort_keys=False), encoding="utf-8")
    return config_path


def write_runtime_profile(
    root: Path,
    *,
    model_name: str,
    label: str = "Ticker unico: AAA",
    ticker: str = "AAA",
    trained_at: str = "2026-03-20T12:00:00",
) -> Path:
    slug = model_name.replace("__", "_")
    payload = _minimal_config_payload(model_name=model_name)
    payload["base_model_name"] = "Gen6_1"
    payload["data"]["raw_data_path"] = f"data/training_universes/{slug}/stock_data.csv"
    payload["data"]["processed_data_path"] = f"data/training_universes/{slug}/processed_dataset.pt"
    payload["data"]["train_processed_df_path"] = f"data/training_universes/{slug}/train_processed_df.parquet"
    payload["data"]["val_processed_df_path"] = f"data/training_universes/{slug}/val_processed_df.parquet"
    payload["paths"]["logs_dir"] = f"logs/{slug}"
    payload["training_run"] = {
        "mode": "single_ticker",
        "single_ticker_symbol": ticker,
        "predefined_group_name": None,
        "requested_tickers": [ticker],
        "downloaded_tickers": [ticker],
        "discarded_tickers": [],
        "discarded_details": {},
        "final_tickers_used": [ticker],
        "dropped_after_preprocessing": [],
        "anchor_ticker": ticker,
        "universe_integrity": {},
        "download_universe_integrity": {},
        "preprocessed_universe_integrity": {},
        "universe_integrity_report_path": f"data/training_universes/{slug}/stock_data__integrity_report.json",
        "raw_data_staging_path": f"data/training_universes/{slug}/stock_data__staging.csv",
        "raw_data_promoted": True,
        "trained_at": trained_at,
        "years": 3,
        "prediction_horizon": 5,
        "label": label,
        "description": "",
        "notes": "",
    }
    profile_path = root / "config" / "runtime_profiles" / f"{model_name}.yaml"
    profile_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return profile_path


def seed_legacy_training_artifacts(root: Path, *, model_name: str) -> dict[str, Path]:
    slug = model_name.replace("__", "_")
    market_dir = root / "data" / "training_universes" / slug
    models_dir = root / "models"
    normalizers_dir = models_dir / "normalizers"
    logs_dir = root / "logs" / slug
    optuna_dir = models_dir / "optuna" / model_name

    market_dir.mkdir(parents=True, exist_ok=True)
    normalizers_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    optuna_dir.mkdir(parents=True, exist_ok=True)

    (market_dir / "stock_data.csv").write_text("Date,Open,High,Low,Close,Volume,Ticker,Sector\n2024-01-02,1,2,0.5,1.5,100,AAA,Financials\n", encoding="utf-8")
    (market_dir / "stock_data__staging.csv").write_text("Date,Open,High,Low,Close,Volume,Ticker,Sector\n", encoding="utf-8")
    (market_dir / "stock_data__integrity_report.json").write_text(json.dumps({"decision": "CONTINUE_CLEAN"}), encoding="utf-8")
    (market_dir / "processed_dataset.pt").write_bytes(b"dataset")
    (market_dir / "train_processed_df.parquet").write_bytes(b"train")
    (market_dir / "val_processed_df.parquet").write_bytes(b"val")
    (models_dir / f"{model_name}.pth").write_bytes(b"checkpoint")
    (models_dir / f"{model_name}.pth.sha256").write_text("deadbeef  checkpoint\n", encoding="utf-8")
    (models_dir / f"{model_name}.pth.meta.json").write_text(json.dumps({"model_name": model_name}), encoding="utf-8")
    (normalizers_dir / f"{model_name}_normalizers.pkl").write_bytes(b"normalizers")
    (normalizers_dir / f"{model_name}_normalizers.pkl.sha256").write_text("deadbeef  normalizers\n", encoding="utf-8")
    (logs_dir / "metrics.csv").write_text("epoch,loss\n0,1.0\n", encoding="utf-8")
    (optuna_dir / "study.txt").write_text("optuna\n", encoding="utf-8")

    return {
        "market_dir": market_dir,
        "models_dir": models_dir,
        "normalizers_dir": normalizers_dir,
        "logs_dir": logs_dir,
        "optuna_dir": optuna_dir,
    }


def write_model_registry(root: Path, *, model_name: str) -> Path:
    registry_path = root / "config" / "model_registry.yaml"
    payload = {
        "active_profile_path": f"config/runtime_profiles/{model_name}.yaml",
        "profiles": {
            model_name: {
                "model_name": model_name,
                "base_model_name": "Gen6_1",
                "profile_path": f"config/runtime_profiles/{model_name}.yaml",
                "label": f"Perfil {model_name}",
                "description": "",
                "notes": "",
                "created_at": "2026-03-20T12:00:00",
                "last_trained_at": "2026-03-20T12:00:00",
            }
        },
    }
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return registry_path
