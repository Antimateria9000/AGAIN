import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from scripts.data_fetcher import DataFetcher
from scripts.model import build_model
from scripts.preprocessor import DataPreprocessor
from scripts.runtime_config import ConfigManager
from scripts.utils.data_schema import build_schema_hash
from scripts.utils.prediction_utils import (
    accumulate_quantile_price_paths,
    denormalize_logged_close,
    inverse_transform_if_available,
)

logger = logging.getLogger(__name__)
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)


def _load_artifact_metadata(metadata_path: Path) -> dict | None:
    if not metadata_path.exists():
        return None
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _validate_schema_metadata(config: dict, metadata: dict | None, artifact_name: str):
    if metadata is None:
        logger.warning(f"El artefacto {artifact_name} no tiene metadatos de esquema. No se puede validar por completo.")
        return
    expected_numeric = metadata.get('numeric_features')
    expected_categoricals = metadata.get('known_categoricals')
    expected_hash = build_schema_hash(config, expected_numeric, expected_categoricals)
    if metadata.get('schema_hash') != expected_hash:
        raise ValueError(f"El artefacto {artifact_name} no es compatible con la configuracion actual")


async def load_data_and_model_async(
    config,
    ticker,
    temp_raw_data_path=None,
    historical_mode=False,
    trim_days=0,
    years=3,
    raw_data: pd.DataFrame | None = None,
):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Inferencia en dispositivo: {device}")

    if raw_data is None:
        async with aiohttp.ClientSession() as session:
            fetcher = DataFetcher(ConfigManager(), years)
            start_date = pd.Timestamp(datetime.now(), tz='UTC') - pd.Timedelta(days=years * 365 + trim_days)
            new_data = await fetcher.fetch_stock_data(ticker, start_date, datetime.now(), session)
        if new_data.empty:
            raise ValueError(f"No se han podido descargar datos para {ticker}")
    else:
        new_data = raw_data.copy()

    new_data['Date'] = pd.to_datetime(new_data['Date']).dt.tz_localize(None)
    if 'Ticker' not in new_data.columns:
        new_data['Ticker'] = ticker
    if temp_raw_data_path:
        new_data.to_csv(temp_raw_data_path, index=False)

    dataset_path = Path(config['data']['processed_data_path'])
    dataset_metadata = _load_artifact_metadata(Path(f"{dataset_path}.meta.json"))
    _validate_schema_metadata(config, dataset_metadata, str(dataset_path))
    dataset = TimeSeriesDataSet.load(dataset_path)

    config_manager = ConfigManager()
    model_name = config['model_name']
    normalizers = config_manager.load_normalizers(model_name)
    normalizers_metadata = config_manager.get_last_normalizers_metadata()
    _validate_schema_metadata(config, normalizers_metadata, f"{model_name}_normalizers.pkl")

    model_path = Path(config['paths']['models_dir']) / f"{model_name}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    checkpoint_metadata = checkpoint.get('metadata')
    _validate_schema_metadata(config, checkpoint_metadata, str(model_path))
    hyperparams = checkpoint['hyperparams']
    if 'hidden_continuous_size' not in hyperparams:
        hyperparams['hidden_continuous_size'] = config['model']['hidden_size'] // 2

    model = build_model(dataset, config, hyperparams=hyperparams)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    logger.info(f"load_data_and_model_async completado en {time.time() - start_time:.3f}s")
    return new_data, dataset, normalizers, model


def load_data_and_model(config, ticker, temp_raw_data_path=None, historical_mode=False, trim_days=0, years=3, raw_data=None):
    result = asyncio.get_event_loop().run_until_complete(
        load_data_and_model_async(
            config,
            ticker,
            temp_raw_data_path=temp_raw_data_path,
            historical_mode=historical_mode,
            trim_days=trim_days,
            years=years,
            raw_data=raw_data,
        )
    )
    return result


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


def generate_predictions(config, dataset, model, ticker_data, return_details: bool = False):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for cat_col in ('Day_of_Week', 'Month'):
        if cat_col in ticker_data.columns:
            ticker_data[cat_col] = ticker_data[cat_col].astype(str)

    ticker_dataset = TimeSeriesDataSet.from_dataset(
        dataset,
        ticker_data,
        stop_randomization=True,
        predict=True,
    )
    dataloader = ticker_dataset.to_dataloader(
        train=False,
        batch_size=config['prediction']['batch_size'],
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    with torch.inference_mode(), torch.amp.autocast(
        device_type='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=torch.float32,
    ):
        predictions = model.predict(dataloader, mode='quantiles', return_x=True, trainer_kwargs={'logger': False})

    pred_array = predictions.output
    if isinstance(pred_array, torch.Tensor):
        pred_array = pred_array.detach().cpu().numpy()

    target_normalizer = dataset.target_normalizer
    pred_array = inverse_transform_if_available(target_normalizer, torch.from_numpy(pred_array))

    config_manager = ConfigManager()
    normalizers = config_manager.load_normalizers(config['model_name'])
    close_normalizer = normalizers.get('Close', target_normalizer)
    last_close_denorm = denormalize_logged_close(close_normalizer, float(ticker_data['Close'].iloc[-1]))

    if len(pred_array.shape) != 3:
        raise ValueError(f"Forma de prediccion no soportada: {pred_array.shape}")

    relative_returns_lower = pred_array[0, :, 0]
    relative_returns_median = pred_array[0, :, 1]
    relative_returns_upper = pred_array[0, :, 2]
    median, lower_bound, upper_bound = accumulate_quantile_price_paths(
        last_close_denorm,
        relative_returns_median,
        relative_returns_lower,
        relative_returns_upper,
    )

    logger.info(f"generate_predictions completado en {time.time() - start_time:.3f}s")
    if return_details:
        return median, lower_bound, upper_bound, {
            'relative_returns_lower': relative_returns_lower,
            'relative_returns_median': relative_returns_median,
            'relative_returns_upper': relative_returns_upper,
            'last_close_denorm': last_close_denorm,
        }
    return median, lower_bound, upper_bound
