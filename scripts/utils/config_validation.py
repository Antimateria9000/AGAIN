from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
from typing import Any, Dict


def _require_path(config: Dict[str, Any], dotted_key: str):
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Falta la clave obligatoria '{dotted_key}' en config.yaml")
        value = value[key]
    return value


def validate_config_schema(config: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = [
        "model_name",
        "model.max_prediction_length",
        "model.min_encoder_length",
        "model.max_encoder_length",
        "model.hidden_size",
        "model.lstm_layers",
        "model.attention_head_size",
        "model.dropout",
        "model.learning_rate",
        "model.embedding_sizes",
        "model.sectors",
        "model.tuning",
        "training.seed",
        "training.max_epochs",
        "training.optuna_trials",
        "training.num_workers",
        "training.prefetch_factor",
        "training.early_stopping_patience",
        "training.reduce_lr_patience",
        "training.reduce_lr_factor",
        "training.weight_decay",
        "prediction.years",
        "prediction.batch_size",
        "validation.debug",
        "validation.enable_detailed_validation",
        "validation.max_validation_batches_to_log",
        "validation.save_plots",
        "validation.max_plots_per_epoch",
        "data.raw_data_path",
        "data.processed_data_path",
        "data.train_processed_df_path",
        "data.val_processed_df_path",
        "data.tickers_file",
        "data.benchmark_tickers_file",
        "paths.data_dir",
        "paths.models_dir",
        "paths.normalizers_dir",
        "paths.config_dir",
        "paths.logs_dir",
    ]
    for key in required_keys:
        _require_path(config, key)

    if "tuning" in config.get("training", {}):
        raise ValueError("La configuracion de Optuna debe estar en 'model.tuning' y no en 'training.tuning'")

    for dotted_key in ("data.train_processed_df_path", "data.val_processed_df_path"):
        parquet_path = Path(_require_path(config, dotted_key))
        if parquet_path.suffix.lower() != ".parquet":
            raise ValueError(f"La ruta '{dotted_key}' debe terminar en '.parquet'")

    processed_dataset_path = Path(_require_path(config, "data.processed_data_path"))
    if processed_dataset_path.suffix.lower() != ".pt":
        raise ValueError("La ruta 'data.processed_data_path' debe terminar en '.pt'")

    return config


def resolve_tuning_config(config: Dict[str, Any]) -> Dict[str, Any]:
    validate_config_schema(config)
    return config["model"]["tuning"]


def apply_runtime_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = deepcopy(config)
    model_name = resolved.get("model_name", "modelo")

    resolved.setdefault("training", {})
    resolved["training"].setdefault("batch_size_candidates", [256, 192, 128, 96, 64, 48, 32, 24, 16, 8, 4, 2, 1])
    resolved["training"].setdefault("accelerator", "auto")
    resolved["training"].setdefault("precision", "auto")

    resolved.setdefault("prediction", {})
    resolved["prediction"].setdefault("future_dates_mode", "approximate_business_days")

    resolved.setdefault("paths", {})
    resolved["paths"].setdefault("benchmark_history_db_path", "data/benchmarks_history.sqlite")
    resolved["paths"].setdefault("model_save_path", str(Path(resolved["paths"]["models_dir"]) / f"{model_name}.pth"))
    resolved["paths"].setdefault("yfinance_cache_dir", str(Path(tempfile.gettempdir()) / "predictor_bursatil_tft" / "yfinance_cache"))

    resolved.setdefault("artifacts", {})
    resolved["artifacts"].setdefault("require_hash_validation", True)

    return resolved
