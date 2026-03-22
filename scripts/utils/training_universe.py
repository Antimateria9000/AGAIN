from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

VALID_TICKER_PATTERN = re.compile(r"^[A-Z0-9.^=_-]+$")


@dataclass(frozen=True)
class TrainingUniverseSelection:
    mode: str
    single_ticker_symbol: str | None = None
    predefined_group_name: str | None = None


@dataclass(frozen=True)
class PredefinedTrainingGroup:
    name: str
    label: str
    tickers: list[str]
    description: str = ""
    notes: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class ResolvedTrainingUniverse:
    mode: str
    label: str
    tickers: list[str]
    single_ticker_symbol: str | None = None
    predefined_group_name: str | None = None
    description: str = ""
    notes: str = ""

    @property
    def slug(self) -> str:
        if self.mode == "single_ticker" and self.single_ticker_symbol:
            return sanitize_path_component(self.single_ticker_symbol)
        if self.predefined_group_name:
            return sanitize_path_component(self.predefined_group_name)
        raise ValueError("No se ha podido construir el slug del universo de entrenamiento")

    @property
    def model_suffix(self) -> str:
        if self.mode == "single_ticker":
            return f"single__{self.slug}"
        return f"group__{self.slug}"


def sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "valor"


def normalize_ticker_symbol(symbol: str) -> str:
    candidate = str(symbol or "").strip().upper()
    if not candidate:
        raise ValueError("El ticker no puede estar vacio")
    if " " in candidate:
        raise ValueError("El ticker no puede contener espacios")
    if not VALID_TICKER_PATTERN.fullmatch(candidate):
        raise ValueError(f"El ticker '{symbol}' tiene un formato no valido")
    return candidate


def load_training_groups(config: dict) -> dict[str, PredefinedTrainingGroup]:
    groups_path = Path(config["paths"]["training_universes_path"])
    with open(groups_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    groups: dict[str, PredefinedTrainingGroup] = {}
    for group_name, group_payload in (payload.get("groups") or {}).items():
        tickers = [normalize_ticker_symbol(ticker) for ticker in group_payload.get("tickers", [])]
        groups[group_name] = PredefinedTrainingGroup(
            name=group_name,
            label=str(group_payload.get("label") or group_name),
            description=str(group_payload.get("description") or ""),
            notes=str(group_payload.get("notes") or ""),
            enabled=bool(group_payload.get("enabled", True)),
            tickers=list(dict.fromkeys(tickers)),
        )
    return groups


def list_enabled_training_groups(config: dict) -> list[PredefinedTrainingGroup]:
    return [group for group in load_training_groups(config).values() if group.enabled]


def resolve_training_universe(
    config: dict,
    mode: str,
    single_ticker_symbol: str | None = None,
    predefined_group_name: str | None = None,
) -> ResolvedTrainingUniverse:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"single_ticker", "predefined_group"}:
        raise ValueError("El modo de entrenamiento debe ser 'single_ticker' o 'predefined_group'")

    if normalized_mode == "single_ticker":
        ticker = normalize_ticker_symbol(single_ticker_symbol or "")
        return ResolvedTrainingUniverse(
            mode="single_ticker",
            label=f"Ticker unico: {ticker}",
            tickers=[ticker],
            single_ticker_symbol=ticker,
        )

    groups = load_training_groups(config)
    if not predefined_group_name:
        raise ValueError("Debe indicarse un grupo predefinido")
    if predefined_group_name not in groups:
        raise ValueError(f"El grupo predefinido '{predefined_group_name}' no existe")

    selected_group = groups[predefined_group_name]
    if not selected_group.enabled:
        raise ValueError(f"El grupo predefinido '{predefined_group_name}' esta deshabilitado")
    if not selected_group.tickers:
        raise ValueError(f"El grupo predefinido '{predefined_group_name}' no tiene tickers")

    return ResolvedTrainingUniverse(
        mode="predefined_group",
        label=selected_group.label,
        tickers=selected_group.tickers,
        predefined_group_name=selected_group.name,
        description=selected_group.description,
        notes=selected_group.notes,
    )


def build_training_model_name(base_model_name: str, universe: ResolvedTrainingUniverse) -> str:
    return f"{sanitize_path_component(base_model_name)}__{universe.model_suffix}"


def _build_runtime_training_metadata(
    universe: ResolvedTrainingUniverse,
    years: int,
    trained_at: str | None = None,
) -> dict[str, Any]:
    return {
        "mode": universe.mode,
        "single_ticker_symbol": universe.single_ticker_symbol,
        "predefined_group_name": universe.predefined_group_name,
        "requested_tickers": list(universe.tickers),
        "downloaded_tickers": [],
        "discarded_tickers": [],
        "discarded_details": {},
        "final_tickers_used": [],
        "dropped_after_preprocessing": [],
        "trained_at": trained_at or datetime.now().replace(microsecond=0).isoformat(),
        "years": int(years),
        "prediction_horizon": None,
        "label": universe.label,
        "description": universe.description,
        "notes": universe.notes,
    }


def build_runtime_training_config(base_config: dict, universe: ResolvedTrainingUniverse, years: int) -> dict:
    config = deepcopy(base_config)
    derived_model_name = build_training_model_name(config["model_name"], universe)

    data_dir = Path(config["paths"]["data_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    logs_dir = Path(config["paths"]["logs_dir"])

    universe_data_dir = data_dir / "training_universes" / sanitize_path_component(derived_model_name)

    config["base_model_name"] = base_config["model_name"]
    config["model_name"] = derived_model_name
    config["paths"]["model_save_path"] = str(models_dir / f"{derived_model_name}.pth")
    config["paths"]["logs_dir"] = str(logs_dir / sanitize_path_component(derived_model_name))
    config["data"]["raw_data_path"] = str(universe_data_dir / "stock_data.csv")
    config["data"]["processed_data_path"] = str(universe_data_dir / "processed_dataset.pt")
    config["data"]["train_processed_df_path"] = str(universe_data_dir / "train_processed_df.parquet")
    config["data"]["val_processed_df_path"] = str(universe_data_dir / "val_processed_df.parquet")
    config["data"]["years"] = int(years)
    config["data"]["tickers"] = list(universe.tickers)
    config["prediction"]["years"] = int(years)
    config.setdefault("training_universe", {})
    config["training_universe"].update({
        "mode": universe.mode,
        "single_ticker_symbol": universe.single_ticker_symbol,
        "predefined_group_name": universe.predefined_group_name,
        "label": universe.label,
        "description": universe.description,
        "notes": universe.notes,
    })
    config["training_run"] = _build_runtime_training_metadata(universe, years)
    config["training_run"]["prediction_horizon"] = int(config["model"]["max_prediction_length"])
    return config


def build_runtime_profile_path(config: dict) -> Path:
    runtime_profiles_dir = Path(config["paths"]["runtime_profiles_dir"])
    runtime_profiles_dir.mkdir(parents=True, exist_ok=True)
    return runtime_profiles_dir / f"{config['model_name']}.yaml"


def save_runtime_profile(config: dict) -> Path:
    profile_path = build_runtime_profile_path(config)
    serializable = deepcopy(config)
    serializable.pop("_meta", None)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=False)
    return profile_path


def build_training_signature(config: dict) -> str:
    payload = deepcopy(config)
    payload.pop("_meta", None)
    serialized = yaml.safe_dump(payload, sort_keys=True, allow_unicode=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def resolved_universe_to_dict(universe: ResolvedTrainingUniverse) -> dict[str, Any]:
    return asdict(universe)
