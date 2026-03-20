from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ALL_SECTORS = [
    "Technology",
    "Healthcare",
    "Financials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Industrials",
    "Materials",
    "Communication Services",
    "Real Estate",
    "Unknown",
]


def debug_dataset(dataset_path: str = "data/train/processed_dataset.pt"):
    dataset_path_obj = Path(dataset_path)
    logger.info("Cargando dataset desde %s", dataset_path_obj)
    dataset = TimeSeriesDataSet.load(dataset_path_obj)
    logger.info("Dataset cargado correctamente")
    logger.info("Parametros del dataset: %s", dataset.get_parameters())

    attributes_to_check = [
        "reals",
        "categoricals",
        "static_categoricals",
        "static_reals",
        "time_varying_known_reals",
        "time_varying_known_categoricals",
        "time_varying_unknown_reals",
        "target",
        "group_ids",
        "time_idx",
    ]
    for attribute in attributes_to_check:
        if hasattr(dataset, attribute):
            logger.info("Atributo %s: %s", attribute, getattr(dataset, attribute))

    if hasattr(dataset, "target_normalizer"):
        logger.info("Target normalizer: %s", dataset.target_normalizer)

    if hasattr(dataset, "static_categoricals") and dataset.static_categoricals:
        for category in dataset.static_categoricals:
            encoder = dataset.categorical_encoders.get(category)
            classes = getattr(encoder, "classes_", []) if encoder is not None else []
            logger.info("Categorias para %s: %s", category, classes)
            if category == "Sector":
                missing = set(ALL_SECTORS) - set(classes)
                if missing:
                    logger.warning("Sectores ausentes en encoder: %s", sorted(missing))

    dataloader = dataset.to_dataloader(train=False, batch_size=32, num_workers=0)
    x, y = next(iter(dataloader))
    logger.info("Batch ejemplo: claves=%s target_shape=%s", list(x.keys()), y[0].shape)
    for key, value in x.items():
        if isinstance(value, torch.Tensor):
            logger.info("Tensor %s: shape=%s dtype=%s", key, value.shape, value.dtype)
            if torch.isnan(value).any() or torch.isinf(value).any():
                logger.warning("Tensor %s contiene NaN o Inf", key)
    if "static_categoricals" in x and "Sector" in dataset.static_categoricals:
        sector_values = x["static_categoricals"][:, dataset.static_categoricals.index("Sector")]
        logger.info("Valores unicos de Sector en batch: %s", np.unique(sector_values))


if __name__ == "__main__":
    debug_dataset()
