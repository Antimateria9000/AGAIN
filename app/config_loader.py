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


def _normalize_ticker_list(values: object) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        ticker = str(value or "").strip().upper()
        if not ticker or ticker in seen:
            continue
        normalized.append(ticker)
        seen.add(ticker)
    return normalized


def _format_ticker_label(ticker: str, names_catalog: dict[str, str]) -> str:
    display_name = str(names_catalog.get(ticker) or "").strip()
    if not display_name:
        return ticker
    return f"{ticker} - {display_name}"


def load_active_inference_universe(config: dict) -> dict[str, object]:
    names_catalog = load_tickers_and_names(config)
    training_run = dict(config.get("training_run") or {})
    training_universe = dict(config.get("training_universe") or {})
    warnings: list[str] = []

    try:
        training_groups = {group["name"]: group for group in load_training_groups(config)}
    except Exception as exc:
        logger.warning("No se pudo cargar training_universes.yaml para inferencia: %s", exc)
        training_groups = {}

    group_name = str(
        training_run.get("predefined_group_name")
        or training_universe.get("predefined_group_name")
        or ""
    ).strip()
    group_payload = training_groups.get(group_name) or {}

    universe_tickers = _normalize_ticker_list(training_run.get("final_tickers_used"))
    source = "training_run.final_tickers_used"
    if not universe_tickers:
        universe_tickers = _normalize_ticker_list(training_run.get("requested_tickers"))
        source = "training_run.requested_tickers"
    if not universe_tickers:
        universe_tickers = _normalize_ticker_list(group_payload.get("tickers"))
        source = "training_universes"
    if not universe_tickers:
        universe_tickers = _normalize_ticker_list(training_universe.get("tickers"))
        source = "training_universe"
    if not universe_tickers:
        universe_tickers = _normalize_ticker_list(config.get("data", {}).get("tickers"))
        source = "data.tickers"

    anchor_ticker = str(
        training_run.get("anchor_ticker")
        or training_universe.get("anchor_ticker")
        or group_payload.get("anchor_ticker")
        or training_run.get("single_ticker_symbol")
        or training_universe.get("single_ticker_symbol")
        or ""
    ).strip().upper()

    if anchor_ticker and universe_tickers and anchor_ticker not in universe_tickers:
        warnings.append(
            "El anchor_ticker activo no esta dentro del universo resuelto; se usa el primer ticker disponible."
        )
        anchor_ticker = universe_tickers[0]
    elif not anchor_ticker and universe_tickers:
        warnings.append(
            "El perfil activo no declara anchor_ticker; se usa el primer ticker del universo resuelto."
        )
        anchor_ticker = universe_tickers[0]
    elif anchor_ticker and not universe_tickers:
        universe_tickers = [anchor_ticker]

    peer_tickers = [ticker for ticker in universe_tickers if ticker != anchor_ticker]
    all_known_tickers = {
        ticker: _format_ticker_label(ticker, names_catalog)
        for ticker in universe_tickers
    }
    peer_labels = {ticker: all_known_tickers[ticker] for ticker in peer_tickers}

    return {
        "anchor_ticker": anchor_ticker or None,
        "anchor_label": all_known_tickers.get(anchor_ticker, anchor_ticker) if anchor_ticker else None,
        "peer_tickers": peer_tickers,
        "peer_labels": peer_labels,
        "all_known_tickers": all_known_tickers,
        "universe_tickers": universe_tickers,
        "source": source,
        "warnings": warnings,
    }


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
                "anchor_ticker": group.anchor_ticker,
                "description": group.description,
                "notes": group.notes,
                "enabled": group.enabled,
            }
            for group in groups
        ]
    except Exception as exc:
        logger.error("Error al cargar training_universes.yaml: %s", exc)
        raise
