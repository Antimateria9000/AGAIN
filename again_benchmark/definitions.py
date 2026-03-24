from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from again_benchmark.contracts import BenchmarkDefinition, DEFAULT_METRICS
from again_benchmark.validation import validate_definition


def _load_benchmark_tickers(config: dict) -> tuple[str, ...]:
    tickers_path = Path(str(config["data"]["benchmark_tickers_file"]))
    payload = yaml.safe_load(tickers_path.read_text(encoding="utf-8")) or {}
    tickers = []
    for region in (payload.get("tickers") or {}).values():
        tickers.extend(str(ticker) for ticker in region.keys())
    ordered = tuple(dict.fromkeys(tickers))
    if not ordered:
        raise ValueError("benchmark_tickers.yaml no contiene tickers")
    return ordered


def build_default_definition(config: dict, *, benchmark_version: int = 1) -> BenchmarkDefinition:
    tickers = _load_benchmark_tickers(config)
    fingerprint_payload = json.dumps(
        {
            "tickers": list(tickers),
            "horizon": int(config["model"]["max_prediction_length"]),
            "lookback_years": int(config["prediction"]["years"]),
            "metrics": list(DEFAULT_METRICS),
            "benchmark_version": benchmark_version,
        },
        sort_keys=True,
    )
    definition_id = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:16]
    definition = BenchmarkDefinition(
        benchmark_id="again_official_benchmark",
        benchmark_version=benchmark_version,
        definition_id=definition_id,
        label="Benchmark oficial reproducible AGAIN",
        tickers=tickers,
        horizon=int(config["model"]["max_prediction_length"]),
        metrics=DEFAULT_METRICS,
        lookback_years=int(config["prediction"]["years"]),
        historical_display_days=365,
        notes="Definicion generada desde config/config.yaml y benchmark_tickers.yaml",
    )
    validate_definition(definition)
    return definition
