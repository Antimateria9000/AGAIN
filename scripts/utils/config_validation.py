from __future__ import annotations

from copy import deepcopy
from pathlib import Path
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
        "training_universe.minimum_group_tickers",
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
        "paths.runtime_profiles_dir",
        "paths.training_universes_path",
        "paths.model_registry_path",
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

    session_backend = str(config.get("data_fetch", {}).get("session_backend", "requests")).strip().lower()
    if session_backend not in {"requests", "curl_cffi"}:
        raise ValueError("La clave 'data_fetch.session_backend' debe ser 'requests' o 'curl_cffi'")

    training_universe = dict(config.get("training_universe") or {})
    coverage_ratio = float(training_universe.get("minimum_group_coverage_ratio", 1.0))
    if not 0.0 <= coverage_ratio <= 1.0:
        raise ValueError("La clave 'training_universe.minimum_group_coverage_ratio' debe estar en [0, 1]")

    fallback_ratio = float(training_universe.get("maximum_fallback_ratio", 0.0))
    if fallback_ratio < 0.0 or fallback_ratio > 1.0:
        raise ValueError("La clave 'training_universe.maximum_fallback_ratio' debe estar en [0, 1]")

    backtesting_runtime = dict(config.get("backtesting_runtime") or {})
    if backtesting_runtime:
        accelerator = str(backtesting_runtime.get("accelerator", "auto")).strip().lower()
        if accelerator not in {"auto", "cpu", "gpu", "cuda"}:
            raise ValueError("La clave 'backtesting_runtime.accelerator' debe ser auto, cpu o gpu")

        execution_backend = str(backtesting_runtime.get("execution_backend", "gpu_full")).strip().lower()
        if execution_backend not in {"cpu_reference", "gpu_full"}:
            raise ValueError("La clave 'backtesting_runtime.execution_backend' debe ser cpu_reference o gpu_full")

        inference_backend = str(backtesting_runtime.get("inference_backend", "gpu_batched")).strip().lower()
        if inference_backend not in {"legacy_per_timestamp", "gpu_batched"}:
            raise ValueError(
                "La clave 'backtesting_runtime.inference_backend' debe ser legacy_per_timestamp o gpu_batched"
            )

        parity_check_sample_windows = int(backtesting_runtime.get("parity_check_sample_windows", 0))
        if parity_check_sample_windows < 0:
            raise ValueError("La clave 'backtesting_runtime.parity_check_sample_windows' no puede ser negativa")

    return config


def _coerce_numeric_if_possible(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            if any(marker in text.lower() for marker in (".", "e")):
                return float(text)
            return int(text)
        except ValueError:
            return value
    return value


def resolve_tuning_config(config: Dict[str, Any]) -> Dict[str, Any]:
    validate_config_schema(config)
    raw_tuning = config["model"]["tuning"]
    return {key: _coerce_numeric_if_possible(value) for key, value in raw_tuning.items()}


def apply_runtime_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = deepcopy(config)
    model_name = resolved.get("model_name", "modelo")

    resolved.setdefault("training", {})
    resolved["training"].setdefault("batch_size_candidates", [256, 192, 128, 96, 64, 48, 32, 24, 16, 8, 4, 2, 1])
    resolved["training"].setdefault("accelerator", "auto")
    resolved["training"].setdefault("precision", "auto")

    resolved.setdefault("prediction", {})
    resolved["prediction"].setdefault("future_dates_mode", "approximate_business_days")
    resolved["prediction"].setdefault("accelerator", resolved["training"].get("accelerator", "auto"))
    resolved["prediction"].setdefault("precision", resolved["training"].get("precision", "auto"))

    resolved.setdefault("backtesting_runtime", {})
    resolved["backtesting_runtime"].setdefault("accelerator", resolved["prediction"].get("accelerator", "auto"))
    resolved["backtesting_runtime"].setdefault("precision", resolved["prediction"].get("precision", "auto"))
    resolved["backtesting_runtime"].setdefault("execution_backend", "gpu_full")
    resolved["backtesting_runtime"].setdefault("inference_backend", "gpu_batched")
    resolved["backtesting_runtime"].setdefault("allow_cpu_fallback_live", True)
    resolved["backtesting_runtime"].setdefault("allow_cpu_fallback_frozen", False)
    resolved["backtesting_runtime"].setdefault("parity_check_sample_windows", 0)
    resolved["backtesting_runtime"].setdefault("emit_runtime_trace", True)

    resolved.setdefault("paths", {})
    resolved["paths"].setdefault("artifacts_dir", "artifacts")
    resolved["paths"].setdefault("training_artifacts_dir", str(Path(resolved["paths"]["artifacts_dir"]) / "training"))
    resolved["paths"].setdefault("benchmark_storage_dir", str(Path(resolved["paths"]["artifacts_dir"]) / "benchmarks"))
    resolved["paths"].setdefault("backtest_storage_dir", str(Path(resolved["paths"]["artifacts_dir"]) / "backtests" / "econ"))
    resolved["paths"].setdefault("cache_dir", "var/cache")
    resolved["paths"].setdefault("tmp_dir", "var/tmp")
    resolved["paths"].setdefault("logs_root_dir", "var/logs")
    resolved["paths"].setdefault("python_cache_dir", str(Path(resolved["paths"]["cache_dir"]) / "python"))
    resolved["paths"].setdefault("matplotlib_cache_dir", str(Path(resolved["paths"]["cache_dir"]) / "matplotlib"))
    resolved["paths"].setdefault("pytest_tmp_dir", str(Path(resolved["paths"]["tmp_dir"]) / "pytest"))
    resolved["paths"].setdefault("training_catalog_path", str(Path(resolved["paths"]["training_artifacts_dir"]) / "catalog.yaml"))
    resolved["paths"].setdefault("benchmark_history_db_path", "data/benchmarks_history.sqlite")
    resolved["paths"].setdefault("model_save_path", str(Path(resolved["paths"]["models_dir"]) / f"{model_name}.pth"))
    resolved["paths"].setdefault("runtime_profiles_dir", str(Path(resolved["paths"]["config_dir"]) / "runtime_profiles"))
    resolved["paths"].setdefault("training_universes_path", str(Path(resolved["paths"]["config_dir"]) / "training_universes.yaml"))
    resolved["paths"].setdefault("model_registry_path", str(Path(resolved["paths"]["config_dir"]) / "model_registry.yaml"))
    resolved["paths"].setdefault("optuna_dir", str(Path(resolved["paths"]["models_dir"]) / "optuna" / model_name))

    resolved.setdefault("artifacts", {})
    resolved["artifacts"].setdefault("require_hash_validation", True)

    resolved.setdefault("training_universe", {})
    resolved["training_universe"].setdefault("minimum_group_tickers", 2)
    resolved["training_universe"].setdefault(
        "minimum_group_tickers_abs",
        int(resolved["training_universe"].get("minimum_group_tickers", 2)),
    )
    resolved["training_universe"].setdefault("minimum_group_coverage_ratio", 1.0)
    resolved["training_universe"].setdefault(
        "minimum_rows_per_ticker",
        int(resolved["model"]["max_encoder_length"]) + int(resolved["model"]["max_prediction_length"]) + 1,
    )
    resolved["training_universe"].setdefault(
        "minimum_common_overlap_days",
        int(resolved["model"]["max_prediction_length"]) + 1,
    )
    resolved["training_universe"].setdefault("abort_if_anchor_missing", True)
    resolved["training_universe"].setdefault("abort_if_too_many_fallback_tickers", True)
    resolved["training_universe"].setdefault("abort_on_cached_backfill", False)
    resolved["training_universe"].setdefault("allow_degraded_universe", False)
    resolved["training_universe"].setdefault("maximum_fallback_tickers", 0)
    resolved["training_universe"].setdefault("maximum_fallback_ratio", 0.0)

    resolved.setdefault("data_fetch", {})
    resolved["data_fetch"].setdefault("max_workers", 1)
    resolved["data_fetch"].setdefault("retries", 4)
    resolved["data_fetch"].setdefault("timeout", 10.0)
    resolved["data_fetch"].setdefault("min_delay", 0.35)
    resolved["data_fetch"].setdefault("max_intraday_lookback_days", 60)
    resolved["data_fetch"].setdefault("allow_partial_intraday", False)
    resolved["data_fetch"].setdefault("repair", True)
    resolved["data_fetch"].setdefault("auto_reset_cookie_cache", True)
    resolved["data_fetch"].setdefault("trust_env_proxies", False)
    resolved["data_fetch"].setdefault("use_yfinance_sector_lookup", False)
    resolved["data_fetch"].setdefault("session_backend", "requests")
    resolved["data_fetch"].setdefault("yfinance_cache_dir", str(Path(resolved["paths"]["cache_dir"]) / "yfinance"))
    resolved["data_fetch"].setdefault("rate_limit_cooldown_seconds", 8.0)
    resolved["data_fetch"].setdefault("rate_limit_circuit_breaker_threshold", 2)
    resolved["data_fetch"].setdefault("rate_limit_circuit_breaker_seconds", 30.0)

    return resolved
