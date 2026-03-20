from __future__ import annotations

import logging

import torch
from pytorch_forecasting import TimeSeriesDataSet

logger = logging.getLogger(__name__)


try:
    from lightning.pytorch.tuner.tuning import Tuner
except Exception:  # pragma: no cover - compatibilidad con entornos antiguos
    try:
        from pytorch_lightning.tuner.tuning import Tuner
    except Exception:  # pragma: no cover
        Tuner = None



def _candidate_batch_sizes(config: dict) -> list[int]:
    candidates = [int(value) for value in config["training"].get("batch_size_candidates", []) if int(value) > 0]
    if not candidates:
        candidates = [int(config["training"]["batch_size"])]
    return sorted(dict.fromkeys(candidates), reverse=True)



def estimate_batch_size(model, dataset: TimeSeriesDataSet, config: dict) -> int:
    configured_batch_size = int(config["training"]["batch_size"])
    if not config["training"].get("auto_batch_size", False):
        logger.info("Auto-batch desactivado. Se usa %s", configured_batch_size)
        return configured_batch_size

    dataset_size = max(1, len(dataset))
    candidates = _candidate_batch_sizes(config)
    usable_candidates = [candidate for candidate in candidates if candidate <= dataset_size]
    if not usable_candidates:
        usable_candidates = [1]

    if not torch.cuda.is_available() or Tuner is None:
        selected = min(configured_batch_size, usable_candidates[0]) if usable_candidates else 1
        logger.info("Se usa batch size conservador %s por falta de GPU o Tuner", selected)
        return max(1, selected)

    free_memory, total_memory = torch.cuda.mem_get_info()
    usage_limit = float(config["training"].get("max_vram_usage", 0.8))
    allowed_memory = total_memory * usage_limit
    available_ratio = min(1.0, max(0.1, free_memory / max(1, allowed_memory)))

    for candidate in usable_candidates:
        if candidate <= int(configured_batch_size * available_ratio) or candidate <= configured_batch_size:
            logger.info("Batch size seleccionado desde candidatos: %s", candidate)
            return max(1, candidate)

    return max(1, min(configured_batch_size, usable_candidates[-1]))
