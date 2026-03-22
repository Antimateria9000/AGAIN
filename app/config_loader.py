from __future__ import annotations

import logging
from pathlib import Path

import yaml

from scripts.utils.training_universe import list_enabled_training_groups

logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_default_config_path(config: dict) -> str:
    return str(Path(config.get("_meta", {}).get("config_path", "config/config.yaml")))


def load_tickers_and_names(config: dict) -> dict[str, str]:
    try:
        tickers_config = _load_yaml(config["data"]["tickers_file"])
        ticker_dict: dict[str, str] = {}
        for region in tickers_config.get("tickers", {}).values():
            for ticker, name in region.items():
                ticker_dict[ticker] = name
        return ticker_dict
    except Exception as exc:
        logger.error("Error al cargar tickers y nombres: %s", exc)
        return {}


def load_benchmark_tickers(config: dict) -> list[str]:
    try:
        tickers_config = _load_yaml(config["data"]["benchmark_tickers_file"])
        all_tickers: list[str] = []
        for region in tickers_config.get("tickers", {}).values():
            all_tickers.extend(list(region.keys()))
        return list(dict.fromkeys(all_tickers))
    except KeyError as exc:
        logger.error("Falta una clave de configuracion: %s", exc)
        raise ValueError(f"Configuracion invalida: falta la clave {exc} en config.yaml") from exc
    except Exception as exc:
        logger.error("Error al cargar benchmark_tickers.yaml: %s", exc)
        raise


def load_training_groups(config: dict) -> list[dict[str, object]]:
    try:
        groups = list_enabled_training_groups(config)
        return [
            {
                "name": group.name,
                "label": group.label,
                "tickers": list(group.tickers),
                "description": group.description,
                "notes": group.notes,
                "enabled": group.enabled,
            }
            for group in groups
        ]
    except Exception as exc:
        logger.error("Error al cargar training_universes.yaml: %s", exc)
        raise
