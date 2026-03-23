import hashlib
import json
from copy import deepcopy
from typing import Any, Dict, Iterable, List

TARGET_COLUMN = "Relative_Returns"
NUMERIC_FEATURES = [
    "Close",
    "Volume",
    "MA10",
    "MA50",
    "RSI",
    "MACD",
    "ROC",
    "VWAP",
    "Momentum_20d",
    "Close_to_MA_ratio",
    "Close_to_BB_upper",
]
REQUIRED_NUMERIC_COLUMNS = [*NUMERIC_FEATURES, TARGET_COLUMN]
REQUIRED_NORMALIZER_KEYS = [*NUMERIC_FEATURES, TARGET_COLUMN]
KNOWN_CATEGORICAL_FEATURES = ["Day_of_Week", "Month"]
STATIC_CATEGORICALS = ["Sector"]
DAY_OF_WEEK_CATEGORIES = [str(i) for i in range(7)]
MONTH_CATEGORIES = [str(i) for i in range(1, 13)]
LOG_FEATURES = ["Close", "Volume", "MA10", "MA50", "VWAP"]
SCHEMA_VERSION = 2


def resolve_embedding_sizes(config: Dict[str, Any]) -> Dict[str, List[int]]:
    configured = dict(config["model"]["embedding_sizes"])
    sectors = list(config["model"]["sectors"])
    minimum_cardinalities = {
        "Sector": len(sectors) + 1,
        "Day_of_Week": len(DAY_OF_WEEK_CATEGORIES) + 1,
        "Month": len(MONTH_CATEGORIES) + 1,
    }

    resolved: Dict[str, List[int]] = {}
    for feature, size_pair in configured.items():
        cardinality, embedding_dim = int(size_pair[0]), int(size_pair[1])
        resolved[feature] = [max(cardinality, minimum_cardinalities.get(feature, cardinality)), embedding_dim]
    return resolved


def build_schema_payload(
    config: Dict[str, Any],
    numeric_features: Iterable[str] | None = None,
    categorical_features: Iterable[str] | None = None,
) -> Dict[str, Any]:
    numeric = list(numeric_features or NUMERIC_FEATURES)
    categoricals = list(categorical_features or KNOWN_CATEGORICAL_FEATURES)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "target": TARGET_COLUMN,
        "numeric_features": numeric,
        "required_numeric_columns": list(REQUIRED_NUMERIC_COLUMNS),
        "required_normalizer_keys": list(REQUIRED_NORMALIZER_KEYS),
        "known_categoricals": categoricals,
        "static_categoricals": list(STATIC_CATEGORICALS),
        "day_of_week_categories": list(DAY_OF_WEEK_CATEGORIES),
        "month_categories": list(MONTH_CATEGORIES),
        "log_features": list(LOG_FEATURES),
        "target_manually_normalized": False,
        "embedding_sizes": resolve_embedding_sizes(config),
        "sectors": list(config["model"]["sectors"]),
        "max_encoder_length": config["model"]["max_encoder_length"],
        "min_encoder_length": config["model"]["min_encoder_length"],
        "max_prediction_length": config["model"]["max_prediction_length"],
    }
    return payload


def build_schema_hash(
    config: Dict[str, Any],
    numeric_features: Iterable[str] | None = None,
    categorical_features: Iterable[str] | None = None,
) -> str:
    payload = build_schema_payload(config, numeric_features, categorical_features)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_config_hash(config: Dict[str, Any]) -> str:
    serializable = deepcopy(config)
    serializable.pop("_meta", None)
    serialized = json.dumps(serializable, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_artifact_metadata(
    config: Dict[str, Any],
    numeric_features: Iterable[str] | None = None,
    categorical_features: Iterable[str] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = build_schema_payload(config, numeric_features, categorical_features)
    metadata["model_name"] = config["model_name"]
    metadata["base_model_name"] = config.get("base_model_name", config["model_name"])
    metadata["schema_hash"] = build_schema_hash(config, numeric_features, categorical_features)
    metadata["config_hash"] = build_config_hash(config)
    if "training_universe" in config:
        metadata["training_universe"] = deepcopy(config["training_universe"])
    if "training_run" in config:
        metadata["training_run"] = deepcopy(config["training_run"])
    if extra:
        metadata.update(extra)
    return metadata


def metadata_matches_active_schema(config: Dict[str, Any], metadata: Dict[str, Any] | None) -> bool:
    if not metadata:
        return False
    return metadata.get("schema_hash") == build_schema_hash(config) and metadata.get("config_hash") == build_config_hash(config)


def normalize_feature_list(features: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(feature) for feature in features))
