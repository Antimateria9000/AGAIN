import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer

from scripts.data_fetcher import DataFetcher
from scripts.config_manager import ConfigManager
from scripts.model import build_model
from scripts.preprocessor import DataPreprocessor
from scripts.utils.artifact_utils import ensure_relative_to, load_trusted_torch_artifact, read_metadata, verify_checksum
from scripts.utils.data_schema import (
    KNOWN_CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    STATIC_CATEGORICALS,
    TARGET_COLUMN,
    metadata_matches_active_schema,
)
from scripts.utils.device_utils import get_inference_autocast_context, log_runtime_context, resolve_execution_context
from scripts.utils.lightning_compat import LIGHTNING_LOGGER_NAME
from scripts.utils.prediction_utils import (
    accumulate_quantile_price_paths,
    denormalize_logged_close,
    estimate_future_business_dates,
)
from scripts.utils.repo_layout import resolve_repo_path

logger = logging.getLogger(__name__)
logging.getLogger(LIGHTNING_LOGGER_NAME).setLevel(logging.ERROR)


def _resolve_dataset_base_dir(config: dict, dataset_path: Path) -> Path:
    candidates = [
        resolve_repo_path(config, config["paths"]["artifacts_dir"]),
        resolve_repo_path(config, config["paths"]["data_dir"]),
    ]
    for candidate in candidates:
        try:
            ensure_relative_to(dataset_path, candidate)
            return candidate
        except ValueError:
            continue
    raise ValueError(f"La ruta de dataset {dataset_path} no cae dentro de roots permitidos")


def _validate_schema_metadata(config: dict, metadata: dict | None, artifact_name: str):
    if metadata is None:
        logger.warning(f"El artefacto {artifact_name} no tiene metadatos de esquema. No se puede validar por completo.")
        return
    if not metadata_matches_active_schema(config, metadata):
        raise ValueError(f"El artefacto {artifact_name} no es compatible con la configuracion actual")


def _supports_nan_slot(embedding_sizes: dict | None, feature: str, base_cardinality: int) -> bool:
    if not embedding_sizes or feature not in embedding_sizes:
        return True
    try:
        configured_cardinality = int(embedding_sizes[feature][0])
    except (TypeError, ValueError, IndexError):
        return True
    return configured_cardinality > base_cardinality


def _build_inference_categorical_encoders(config: dict, checkpoint_hyperparams: dict | None = None) -> dict:
    embedding_sizes = (checkpoint_hyperparams or {}).get("embedding_sizes") if checkpoint_hyperparams else None
    return {
        "Sector": NaNLabelEncoder(add_nan=_supports_nan_slot(embedding_sizes, "Sector", len(config["model"]["sectors"]))),
        "Day_of_Week": NaNLabelEncoder(add_nan=_supports_nan_slot(embedding_sizes, "Day_of_Week", 7)),
        "Month": NaNLabelEncoder(add_nan=_supports_nan_slot(embedding_sizes, "Month", 12)),
    }


def _build_reference_dataset_from_input(
    config: dict,
    new_data: pd.DataFrame,
    normalizers: dict,
    ticker: str,
    checkpoint_hyperparams: dict | None = None,
) -> TimeSeriesDataSet:
    logger.warning(
        "No se ha podido usar el artefacto %s. Se reconstruira un TimeSeriesDataSet de referencia en memoria.",
        config["data"]["processed_data_path"],
    )
    processed_df, _ = preprocess_data(config, new_data.copy(), ticker, normalizers)
    available_numeric_features = [feature for feature in NUMERIC_FEATURES if feature in processed_df.columns]

    return TimeSeriesDataSet(
        processed_df,
        time_idx="time_idx",
        target=TARGET_COLUMN,
        group_ids=["group_id"],
        min_encoder_length=config["model"]["min_encoder_length"],
        max_encoder_length=config["model"]["max_encoder_length"],
        max_prediction_length=config["model"]["max_prediction_length"],
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_known_categoricals=KNOWN_CATEGORICAL_FEATURES,
        time_varying_unknown_reals=available_numeric_features,
        target_normalizer=normalizers.get(TARGET_COLUMN, TorchNormalizer()),
        allow_missing_timesteps=True,
        add_encoder_length=False,
        categorical_encoders=_build_inference_categorical_encoders(config, checkpoint_hyperparams),
    )


def _extract_prediction_array(predictions) -> np.ndarray:
    pred_array = predictions.output if hasattr(predictions, "output") else predictions
    if isinstance(pred_array, torch.Tensor):
        pred_array = pred_array.detach().float().cpu().numpy()
    pred_array = np.asarray(pred_array, dtype=float)
    if pred_array.ndim != 3:
        raise ValueError(f"Forma de prediccion no soportada: {pred_array.shape}")
    if pred_array.shape[0] < 1 or pred_array.shape[-1] < 3:
        raise ValueError(f"Prediccion cuantílica incompleta: {pred_array.shape}")
    if not np.all(np.isfinite(pred_array)):
        raise ValueError("La salida cuantílica contiene NaN o Inf")
    return pred_array


def _log_prediction_quantile_summary(pred_array: np.ndarray, context: str) -> None:
    lower = pred_array[..., 0]
    median = pred_array[..., 1]
    upper = pred_array[..., 2]
    logger.info(
        "%s | pred_array shape=%s | q10[min=%.6f mean=%.6f max=%.6f] | q50[min=%.6f mean=%.6f max=%.6f] | q90[min=%.6f mean=%.6f max=%.6f]",
        context,
        pred_array.shape,
        float(np.min(lower)),
        float(np.mean(lower)),
        float(np.max(lower)),
        float(np.min(median)),
        float(np.mean(median)),
        float(np.max(median)),
        float(np.min(upper)),
        float(np.mean(upper)),
        float(np.max(upper)),
    )


def _sanitize_quantile_order(lower, median, upper, context: str) -> tuple[float, float, float]:
    lower_value = float(lower)
    median_value = float(median)
    upper_value = float(upper)
    ordered = sorted([lower_value, median_value, upper_value])
    if ordered != [lower_value, median_value, upper_value]:
        logger.warning(
            "%s | Cruce cuantílico detectado (q10=%.6f, q50=%.6f, q90=%.6f). Se reordena localmente.",
            context,
            lower_value,
            median_value,
            upper_value,
        )
    return ordered[0], ordered[1], ordered[2]


def _normalize_raw_history_frame(raw_ticker_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw_ticker_data is None or raw_ticker_data.empty:
        raise ValueError("raw_ticker_data es obligatorio para el forecast recursivo")
    history = raw_ticker_data.copy()
    history["Date"] = pd.to_datetime(history["Date"]).dt.tz_localize(None)
    if "Ticker" not in history.columns:
        history["Ticker"] = ticker
    history = history[history["Ticker"].astype(str) == str(ticker)].copy()
    if history.empty:
        raise ValueError(f"No hay historial bruto para {ticker}")
    if "Sector" not in history.columns:
        history["Sector"] = "Unknown"
    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Sector"]
    missing_columns = [column for column in required_columns if column not in history.columns]
    if missing_columns:
        raise ValueError(f"El historial bruto no contiene columnas obligatorias: {missing_columns}")
    history = history.sort_values("Date").drop_duplicates(subset=["Ticker", "Date"], keep="last").reset_index(drop=True)
    return history


def _build_next_step_input(working_history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    last_row = working_history.sort_values("Date").iloc[-1].copy()
    next_date = pd.Timestamp(estimate_future_business_dates(last_row["Date"], 1)[0]).tz_localize(None)
    last_close = max(float(last_row["Close"]), 1e-12)
    last_volume = max(float(last_row["Volume"]), 0.0)
    placeholder_row = last_row.copy()
    placeholder_row["Date"] = next_date
    placeholder_row["Open"] = last_close
    placeholder_row["High"] = last_close
    placeholder_row["Low"] = last_close
    placeholder_row["Close"] = last_close
    placeholder_row["Volume"] = last_volume
    extended_history = pd.concat([working_history, pd.DataFrame([placeholder_row])], ignore_index=True)
    return extended_history, next_date


def _append_predicted_step(working_history: pd.DataFrame, next_date: pd.Timestamp, predicted_close: float) -> pd.DataFrame:
    if not np.isfinite(predicted_close) or predicted_close <= 0.0:
        raise ValueError(f"Precio previsto no valido para la extension recursiva: {predicted_close}")
    last_row = working_history.sort_values("Date").iloc[-1].copy()
    next_row = last_row.copy()
    next_row["Date"] = pd.Timestamp(next_date).tz_localize(None)
    next_row["Open"] = predicted_close
    next_row["High"] = predicted_close
    next_row["Low"] = predicted_close
    next_row["Close"] = predicted_close
    next_row["Volume"] = max(float(last_row["Volume"]), 0.0)
    extended_history = pd.concat([working_history, pd.DataFrame([next_row])], ignore_index=True)
    return extended_history.sort_values("Date").reset_index(drop=True)


def _validate_forecast_timeline(last_observed_date, forecast_dates, horizon: int) -> None:
    if len(forecast_dates) != int(horizon):
        raise ValueError(f"El horizonte generado no coincide con el configurado: {len(forecast_dates)} != {horizon}")
    if not forecast_dates:
        raise ValueError("No se han generado fechas futuras")
    normalized_dates = pd.to_datetime(pd.Index(forecast_dates)).tz_localize(None)
    if not normalized_dates.is_monotonic_increasing:
        raise ValueError("Las fechas del forecast no son estrictamente crecientes")
    if normalized_dates.has_duplicates:
        raise ValueError("El forecast contiene fechas duplicadas")
    last_timestamp = pd.Timestamp(last_observed_date).tz_localize(None)
    if normalized_dates[0] <= last_timestamp:
        raise ValueError("La primera fecha prevista no es estrictamente posterior a la ultima observada")


def _validate_forecast_arrays(median, lower, upper, horizon: int) -> None:
    median_np = np.asarray(median, dtype=float)
    lower_np = np.asarray(lower, dtype=float)
    upper_np = np.asarray(upper, dtype=float)
    if not (len(median_np) == len(lower_np) == len(upper_np) == int(horizon)):
        raise ValueError("Las trayectorias previstas no tienen la longitud esperada")
    if not (np.all(np.isfinite(median_np)) and np.all(np.isfinite(lower_np)) and np.all(np.isfinite(upper_np))):
        raise ValueError("Las trayectorias previstas contienen NaN o Inf")
    if np.any(median_np <= 0.0) or np.any(lower_np <= 0.0) or np.any(upper_np <= 0.0):
        raise ValueError("Las trayectorias previstas contienen precios no positivos")
    if np.any(lower_np > median_np) or np.any(median_np > upper_np):
        raise ValueError("Las trayectorias previstas violan lower <= median <= upper")


def load_data_and_model(
    config,
    ticker,
    temp_raw_data_path=None,
    historical_mode=False,
    trim_days=0,
    years=3,
    raw_data: pd.DataFrame | None = None,
    runtime_purpose: str = "predict",
):
    start_time = time.time()
    runtime = resolve_execution_context(config, purpose=runtime_purpose)
    device = runtime.torch_device
    log_runtime_context(logger, "Carga de inferencia", runtime)
    config_path = config.get("_meta", {}).get("config_path")

    if raw_data is None:
        fetcher = DataFetcher(ConfigManager(config_path), years)
        start_date = pd.Timestamp(datetime.now()).tz_localize(None) - pd.Timedelta(days=years * 365 + trim_days)
        new_data = fetcher.fetch_stock_data(ticker, start_date, datetime.now().replace(tzinfo=None))
        if new_data.empty:
            raise ValueError(f"No se han podido descargar datos para {ticker}")
    else:
        new_data = raw_data.copy()

    new_data['Date'] = pd.to_datetime(new_data['Date']).dt.tz_localize(None)
    if 'Ticker' not in new_data.columns:
        new_data['Ticker'] = ticker
    if temp_raw_data_path:
        new_data.to_csv(temp_raw_data_path, index=False)

    config_manager = ConfigManager(config_path)
    model_name = config['model_name']
    normalizers = config_manager.load_normalizers(model_name, required=True)
    normalizers_metadata = config_manager.get_last_normalizers_metadata()
    _validate_schema_metadata(config, normalizers_metadata, f"{model_name}_normalizers.pkl")

    model_path = resolve_repo_path(config, config['paths']['models_dir']) / f"{model_name}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo {model_path}")

    ensure_relative_to(model_path, resolve_repo_path(config, config['paths']['models_dir']))
    verify_checksum(model_path, required=config['artifacts']['require_hash_validation'])
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    checkpoint_metadata = checkpoint.get('metadata')
    _validate_schema_metadata(config, checkpoint_metadata, str(model_path))
    hyperparams = checkpoint['hyperparams']
    if 'hidden_continuous_size' not in hyperparams:
        hyperparams['hidden_continuous_size'] = config['model']['hidden_size'] // 2

    dataset_path = resolve_repo_path(config, config['data']['processed_data_path'])
    ensure_relative_to(dataset_path, _resolve_dataset_base_dir(config, dataset_path))
    try:
        verify_checksum(dataset_path, required=config['artifacts']['require_hash_validation'])
        dataset_metadata = read_metadata(dataset_path)
        _validate_schema_metadata(config, dataset_metadata, str(dataset_path))
        dataset = load_trusted_torch_artifact(dataset_path, trusted_types=[TimeSeriesDataSet])
        if not isinstance(dataset, TimeSeriesDataSet):
            raise TypeError(f"El artefacto {dataset_path} no contiene un TimeSeriesDataSet valido")
    except (FileNotFoundError, ValueError, TypeError) as exc:
        logger.warning("No se usara el dataset persistido %s: %s", dataset_path, exc)
        dataset = _build_reference_dataset_from_input(config, new_data, normalizers, ticker, checkpoint_hyperparams=hyperparams)

    model = build_model(dataset, config, hyperparams=hyperparams)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    logger.info(f"load_data_and_model completado en {time.time() - start_time:.3f}s")
    return new_data, dataset, normalizers, model


def preprocess_data(config, ticker_data, ticker, normalizers, historical_mode=False, trim_days=0):
    preprocessor = DataPreprocessor(config)
    return preprocessor.process_data(
        mode='predict',
        df=ticker_data,
        normalizers=normalizers,
        ticker=ticker,
        historical_mode=historical_mode,
        trim_days=trim_days,
    )


def _recompute_future_features(config: dict, raw_history: pd.DataFrame, ticker: str, normalizers: dict) -> pd.DataFrame:
    processed_df, _ = preprocess_data(config, raw_history.copy(), ticker, normalizers)
    return processed_df


def _predict_quantiles_from_processed_frame(
    config: dict,
    dataset: TimeSeriesDataSet,
    model,
    processed_df: pd.DataFrame,
    runtime,
    context: str,
    prediction_length: int | None = None,
) -> np.ndarray:
    ticker_data = processed_df.copy()
    for cat_col in ("Day_of_Week", "Month"):
        if cat_col in ticker_data.columns:
            ticker_data[cat_col] = ticker_data[cat_col].astype(str)

    update_kwargs = {}
    if prediction_length is not None:
        update_kwargs["min_prediction_length"] = int(prediction_length)
        update_kwargs["max_prediction_length"] = int(prediction_length)

    ticker_dataset = TimeSeriesDataSet.from_dataset(
        dataset,
        ticker_data,
        stop_randomization=True,
        predict=True,
        **update_kwargs,
    )
    dataloader = ticker_dataset.to_dataloader(
        train=False,
        batch_size=config["prediction"]["batch_size"],
        num_workers=0,
        pin_memory=runtime.pin_memory,
        persistent_workers=False,
    )
    log_runtime_context(
        logger,
        context,
        runtime,
        batch_size=int(config["prediction"]["batch_size"]),
    )
    autocast_context = get_inference_autocast_context(runtime)
    with torch.inference_mode(), autocast_context:
        predictions = model.predict(dataloader, mode="quantiles", return_x=True, trainer_kwargs={"logger": False})

    pred_array = _extract_prediction_array(predictions)
    _log_prediction_quantile_summary(pred_array, context)
    return pred_array


def _build_batched_group_id(ticker: str, batch_index: int) -> str:
    return f"{ticker}__backtest_batch_{batch_index:06d}"


def _predict_quantiles_from_processed_frames(
    config: dict,
    dataset: TimeSeriesDataSet,
    model,
    processed_frames: list[pd.DataFrame],
    runtime,
    context: str,
    prediction_length: int | None = None,
) -> np.ndarray:
    if not processed_frames:
        raise ValueError("Se requiere al menos un dataframe procesado para inferencia batched")
    combined = pd.concat(processed_frames, ignore_index=True)
    return _predict_quantiles_from_processed_frame(
        config,
        dataset,
        model,
        combined,
        runtime,
        context,
        prediction_length=prediction_length,
    )


def _generate_legacy_dataset_predictions(
    config: dict,
    dataset: TimeSeriesDataSet,
    model,
    ticker_data: pd.DataFrame,
    return_details: bool = False,
    runtime=None,
):
    runtime = runtime or resolve_execution_context(config, purpose="predict")
    pred_array = _predict_quantiles_from_processed_frame(
        config,
        dataset,
        model,
        ticker_data,
        runtime,
        context="Inferencia TFT legacy",
        prediction_length=None,
    )

    target_normalizer = dataset.target_normalizer
    config_manager = ConfigManager(config.get("_meta", {}).get("config_path"))
    normalizers = config_manager.load_normalizers(config["model_name"], required=True)
    close_normalizer = normalizers.get("Close", target_normalizer)
    last_close_denorm = denormalize_logged_close(close_normalizer, float(ticker_data['Close'].iloc[-1]))

    relative_returns_lower = pred_array[0, :, 0]
    relative_returns_median = pred_array[0, :, 1]
    relative_returns_upper = pred_array[0, :, 2]
    median, lower_bound, upper_bound = accumulate_quantile_price_paths(
        last_close_denorm,
        relative_returns_median,
        relative_returns_lower,
        relative_returns_upper,
    )
    if return_details:
        return median, lower_bound, upper_bound, {
            "forecast_mode": "legacy_dataset_predict",
            "forecast_dates": list(estimate_future_business_dates(ticker_data["Date"].iloc[-1], len(median)).to_pydatetime()),
            "relative_returns_lower": relative_returns_lower,
            "relative_returns_median": relative_returns_median,
            "relative_returns_upper": relative_returns_upper,
            "predicted_returns_lower": relative_returns_lower,
            "predicted_returns_median": relative_returns_median,
            "predicted_returns_upper": relative_returns_upper,
            "forecast_close_median": np.asarray(median, dtype=float),
            "forecast_close_lower": np.asarray(lower_bound, dtype=float),
            "forecast_close_upper": np.asarray(upper_bound, dtype=float),
            "last_observed_date": pd.Timestamp(ticker_data["Date"].iloc[-1]).to_pydatetime(),
            "last_observed_close": float(last_close_denorm),
            "last_close_denorm": float(last_close_denorm),
        }
    return median, lower_bound, upper_bound


def generate_recursive_forecast(
    config: dict,
    dataset: TimeSeriesDataSet,
    model,
    ticker_data: pd.DataFrame,
    raw_ticker_data: pd.DataFrame,
    return_details: bool = False,
    runtime=None,
    horizon: int | None = None,
):
    runtime = runtime or resolve_execution_context(config, purpose="predict")
    config_manager = ConfigManager(config.get("_meta", {}).get("config_path"))
    normalizers = config_manager.load_normalizers(config["model_name"], required=True)
    horizon = int(horizon if horizon is not None else config["model"]["max_prediction_length"])
    ticker = str(ticker_data["Ticker"].iloc[-1]) if "Ticker" in ticker_data.columns else str(raw_ticker_data["Ticker"].iloc[-1])

    working_history = _normalize_raw_history_frame(raw_ticker_data, ticker)
    last_observed_date = pd.Timestamp(working_history["Date"].iloc[-1]).to_pydatetime()
    last_observed_close = float(working_history["Close"].iloc[-1])

    forecast_dates = []
    relative_returns_lower = []
    relative_returns_median = []
    relative_returns_upper = []
    forecast_close_median = []
    forecast_close_lower = []
    forecast_close_upper = []

    logger.info(
        "Modo de forecast: recursive_one_step | ultima_fecha_observada=%s | ultimo_close_observado=%.6f | horizonte=%s",
        last_observed_date,
        last_observed_close,
        horizon,
    )

    for step_index in range(horizon):
        placeholder_history, next_date = _build_next_step_input(working_history)
        processed_df = _recompute_future_features(config, placeholder_history, ticker, normalizers)
        processed_last_date = pd.Timestamp(processed_df["Date"].iloc[-1]).tz_localize(None)
        if processed_last_date != pd.Timestamp(next_date).tz_localize(None):
            raise ValueError(
                f"La ultima fila procesada ({processed_last_date}) no coincide con la fecha futura esperada ({next_date})"
            )

        pred_array = _predict_quantiles_from_processed_frame(
            config,
            dataset,
            model,
            processed_df,
            runtime,
            context=f"Inferencia TFT recursiva paso {step_index + 1}/{horizon}",
            prediction_length=1,
        )
        if pred_array.shape[1] != 1:
            raise ValueError(f"El forecast recursivo esperaba exactamente 1 paso y obtuvo {pred_array.shape[1]}")

        step_lower, step_median, step_upper = _sanitize_quantile_order(
            pred_array[0, 0, 0],
            pred_array[0, 0, 1],
            pred_array[0, 0, 2],
            context=f"Paso {step_index + 1}/{horizon}",
        )
        current_last_close = float(working_history["Close"].iloc[-1])
        next_median_path, next_lower_path, next_upper_path = accumulate_quantile_price_paths(
            current_last_close,
            np.asarray([step_median], dtype=float),
            np.asarray([step_lower], dtype=float),
            np.asarray([step_upper], dtype=float),
        )

        forecast_dates.append(pd.Timestamp(next_date).to_pydatetime())
        relative_returns_lower.append(step_lower)
        relative_returns_median.append(step_median)
        relative_returns_upper.append(step_upper)
        forecast_close_median.append(float(next_median_path[-1]))
        forecast_close_lower.append(float(next_lower_path[-1]))
        forecast_close_upper.append(float(next_upper_path[-1]))

        working_history = _append_predicted_step(working_history, next_date, float(next_median_path[-1]))

    forecast_close_median_np = np.asarray(forecast_close_median, dtype=float)
    forecast_close_lower_np = np.asarray(forecast_close_lower, dtype=float)
    forecast_close_upper_np = np.asarray(forecast_close_upper, dtype=float)
    relative_returns_lower_np = np.asarray(relative_returns_lower, dtype=float)
    relative_returns_median_np = np.asarray(relative_returns_median, dtype=float)
    relative_returns_upper_np = np.asarray(relative_returns_upper, dtype=float)

    _validate_forecast_timeline(last_observed_date, forecast_dates, horizon)
    _validate_forecast_arrays(forecast_close_median_np, forecast_close_lower_np, forecast_close_upper_np, horizon)

    logger.info(
        "Forecast completado | modo=recursive_one_step | primera_fecha=%s | ultima_fecha=%s | retorno_mediano[min=%.6f mean=%.6f max=%.6f] | close_inicial=%.6f | close_final=%.6f | ancho_banda_final=%.6f",
        forecast_dates[0],
        forecast_dates[-1],
        float(np.min(relative_returns_median_np)),
        float(np.mean(relative_returns_median_np)),
        float(np.max(relative_returns_median_np)),
        last_observed_close,
        float(forecast_close_median_np[-1]),
        float(forecast_close_upper_np[-1] - forecast_close_lower_np[-1]),
    )

    if return_details:
        return forecast_close_median_np, forecast_close_lower_np, forecast_close_upper_np, {
            "forecast_mode": "recursive_one_step",
            "forecast_dates": list(forecast_dates),
            "relative_returns_lower": relative_returns_lower_np,
            "relative_returns_median": relative_returns_median_np,
            "relative_returns_upper": relative_returns_upper_np,
            "predicted_returns_lower": relative_returns_lower_np,
            "predicted_returns_median": relative_returns_median_np,
            "predicted_returns_upper": relative_returns_upper_np,
            "forecast_close_median": forecast_close_median_np,
            "forecast_close_lower": forecast_close_lower_np,
            "forecast_close_upper": forecast_close_upper_np,
            "last_observed_date": last_observed_date,
            "last_observed_close": float(last_observed_close),
            "last_close_denorm": float(last_observed_close),
        }
    return forecast_close_median_np, forecast_close_lower_np, forecast_close_upper_np


def generate_recursive_forecasts_batch(
    config: dict,
    dataset: TimeSeriesDataSet,
    model,
    raw_histories: list[pd.DataFrame],
    *,
    ticker: str,
    horizon: int,
    runtime=None,
) -> tuple[dict[str, object], ...]:
    if not raw_histories:
        return ()

    runtime = runtime or resolve_execution_context(config, purpose="predict")
    config_manager = ConfigManager(config.get("_meta", {}).get("config_path"))
    normalizers = config_manager.load_normalizers(config["model_name"], required=True)
    working_histories = [_normalize_raw_history_frame(raw_history, ticker) for raw_history in raw_histories]
    states: list[dict[str, object]] = []
    for history in working_histories:
        states.append(
            {
                "last_observed_date": pd.Timestamp(history["Date"].iloc[-1]).to_pydatetime(),
                "last_observed_close": float(history["Close"].iloc[-1]),
                "forecast_dates": [],
                "relative_returns_lower": [],
                "relative_returns_median": [],
                "relative_returns_upper": [],
                "forecast_close_median": [],
                "forecast_close_lower": [],
                "forecast_close_upper": [],
            }
        )

    for step_index in range(int(horizon)):
        processed_frames: list[pd.DataFrame] = []
        next_dates: list[pd.Timestamp] = []
        for batch_index, history in enumerate(working_histories):
            placeholder_history, next_date = _build_next_step_input(history)
            processed_df = _recompute_future_features(config, placeholder_history, ticker, normalizers)
            processed_last_date = pd.Timestamp(processed_df["Date"].iloc[-1]).tz_localize(None)
            expected_next_date = pd.Timestamp(next_date).tz_localize(None)
            if processed_last_date != expected_next_date:
                raise ValueError(
                    f"La ultima fila procesada ({processed_last_date}) no coincide con la fecha futura esperada ({expected_next_date})"
                )
            batch_df = processed_df.copy()
            batch_df["group_id"] = _build_batched_group_id(ticker, batch_index)
            batch_df["time_idx"] = batch_df.groupby("group_id").cumcount()
            processed_frames.append(batch_df)
            next_dates.append(expected_next_date)

        pred_array = _predict_quantiles_from_processed_frames(
            config,
            dataset,
            model,
            processed_frames,
            runtime,
            context=f"Inferencia TFT recursiva batched paso {step_index + 1}/{horizon} | ticker={ticker} | batch={len(processed_frames)}",
            prediction_length=1,
        )
        if pred_array.shape[0] != len(working_histories) or pred_array.shape[1] != 1:
            raise ValueError(
                f"El forecast recursivo batched esperaba forma ({len(working_histories)}, 1, 3) y obtuvo {pred_array.shape}"
            )

        for batch_index, history in enumerate(working_histories):
            state = states[batch_index]
            step_lower, step_median, step_upper = _sanitize_quantile_order(
                pred_array[batch_index, 0, 0],
                pred_array[batch_index, 0, 1],
                pred_array[batch_index, 0, 2],
                context=f"Batch {batch_index + 1}/{len(working_histories)} paso {step_index + 1}/{horizon}",
            )
            current_last_close = float(history["Close"].iloc[-1])
            next_median_path, next_lower_path, next_upper_path = accumulate_quantile_price_paths(
                current_last_close,
                np.asarray([step_median], dtype=float),
                np.asarray([step_lower], dtype=float),
                np.asarray([step_upper], dtype=float),
            )
            cast_state = state
            cast_state["forecast_dates"].append(pd.Timestamp(next_dates[batch_index]).to_pydatetime())
            cast_state["relative_returns_lower"].append(step_lower)
            cast_state["relative_returns_median"].append(step_median)
            cast_state["relative_returns_upper"].append(step_upper)
            cast_state["forecast_close_median"].append(float(next_median_path[-1]))
            cast_state["forecast_close_lower"].append(float(next_lower_path[-1]))
            cast_state["forecast_close_upper"].append(float(next_upper_path[-1]))
            working_histories[batch_index] = _append_predicted_step(
                history,
                next_dates[batch_index],
                float(next_median_path[-1]),
            )

    results: list[dict[str, object]] = []
    for state in states:
        forecast_close_median_np = np.asarray(state["forecast_close_median"], dtype=float)
        forecast_close_lower_np = np.asarray(state["forecast_close_lower"], dtype=float)
        forecast_close_upper_np = np.asarray(state["forecast_close_upper"], dtype=float)
        relative_returns_lower_np = np.asarray(state["relative_returns_lower"], dtype=float)
        relative_returns_median_np = np.asarray(state["relative_returns_median"], dtype=float)
        relative_returns_upper_np = np.asarray(state["relative_returns_upper"], dtype=float)
        _validate_forecast_timeline(state["last_observed_date"], state["forecast_dates"], int(horizon))
        _validate_forecast_arrays(
            forecast_close_median_np,
            forecast_close_lower_np,
            forecast_close_upper_np,
            int(horizon),
        )
        results.append(
            {
                "forecast_mode": "recursive_one_step",
                "forecast_dates": list(state["forecast_dates"]),
                "relative_returns_lower": relative_returns_lower_np,
                "relative_returns_median": relative_returns_median_np,
                "relative_returns_upper": relative_returns_upper_np,
                "predicted_returns_lower": relative_returns_lower_np,
                "predicted_returns_median": relative_returns_median_np,
                "predicted_returns_upper": relative_returns_upper_np,
                "forecast_close_median": forecast_close_median_np,
                "forecast_close_lower": forecast_close_lower_np,
                "forecast_close_upper": forecast_close_upper_np,
                "last_observed_date": state["last_observed_date"],
                "last_observed_close": float(state["last_observed_close"]),
                "last_close_denorm": float(state["last_observed_close"]),
            }
        )
    return tuple(results)


def generate_predictions(
    config,
    dataset,
    model,
    ticker_data,
    return_details: bool = False,
    raw_ticker_data: pd.DataFrame | None = None,
    forecast_mode: str = "recursive_one_step",
    runtime=None,
    runtime_purpose: str = "predict",
    forecast_horizon: int | None = None,
):
    start_time = time.time()
    runtime = runtime or resolve_execution_context(config, purpose=runtime_purpose)
    model = model.to(runtime.torch_device)

    if forecast_mode == "recursive_one_step":
        if raw_ticker_data is None:
            logger.warning(
                "Se solicito forecast recursivo pero no se proporciono raw_ticker_data. Se usara el modo legacy de compatibilidad."
            )
            result = _generate_legacy_dataset_predictions(
                config,
                dataset,
                model,
                ticker_data,
                return_details=return_details,
                runtime=runtime,
            )
        else:
            result = generate_recursive_forecast(
                config,
                dataset,
                model,
                ticker_data,
                raw_ticker_data=raw_ticker_data,
                return_details=return_details,
                runtime=runtime,
                horizon=forecast_horizon,
            )
    elif forecast_mode == "legacy_dataset_predict":
        logger.warning("Se esta usando el modo legacy_dataset_predict. Solo deberia usarse para depuracion o compatibilidad.")
        result = _generate_legacy_dataset_predictions(
            config,
            dataset,
            model,
            ticker_data,
            return_details=return_details,
            runtime=runtime,
        )
    else:
        raise ValueError(f"Modo de forecast no soportado: {forecast_mode}")

    logger.info("generate_predictions completado en %.3fs", time.time() - start_time)
    return result
