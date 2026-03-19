import hashlib
import json
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
    TARGET_COLUMN,
]
KNOWN_CATEGORICAL_FEATURES = ["Day_of_Week", "Month"]
STATIC_CATEGORICALS = ["Sector"]
DAY_OF_WEEK_CATEGORIES = [str(i) for i in range(7)]
LOG_FEATURES = ["Close", "Volume", "MA10", "MA50", "VWAP"]
SCHEMA_VERSION = 1


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
        "known_categoricals": categoricals,
        "static_categoricals": list(STATIC_CATEGORICALS),
        "day_of_week_categories": list(DAY_OF_WEEK_CATEGORIES),
        "log_features": list(LOG_FEATURES),
        "embedding_sizes": config["model"]["embedding_sizes"],
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


def build_artifact_metadata(
    config: Dict[str, Any],
    numeric_features: Iterable[str] | None = None,
    categorical_features: Iterable[str] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = build_schema_payload(config, numeric_features, categorical_features)
    metadata["model_name"] = config["model_name"]
    metadata["schema_hash"] = build_schema_hash(config, numeric_features, categorical_features)
    if extra:
        metadata.update(extra)
    return metadata


def normalize_feature_list(features: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(feature) for feature in features))
