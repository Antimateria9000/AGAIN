from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any

from again_econ.config import BacktestConfig
from again_econ.contracts import MarketFrame, RunManifest, WalkforwardWindow


def _normalize_for_hash(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_for_hash(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _normalize_for_hash(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    return value


def _fingerprint_payload(value: Any) -> str:
    payload = json.dumps(_normalize_for_hash(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_run_manifest(
    config: BacktestConfig,
    market_frame: MarketFrame,
    windows: tuple[WalkforwardWindow, ...],
    adapter_name: str,
    input_payload: Any,
    input_reference: str | None = None,
) -> RunManifest:
    config_fingerprint = _fingerprint_payload(config)
    market_fingerprint = _fingerprint_payload(market_frame)
    input_fingerprint = _fingerprint_payload(input_payload)
    run_id = hashlib.sha256(
        f"{config_fingerprint}:{market_fingerprint}:{input_fingerprint}:{adapter_name}".encode("utf-8")
    ).hexdigest()[:16]
    return RunManifest(
        run_id=run_id,
        label=config.label,
        adapter_name=adapter_name,
        config_fingerprint=config_fingerprint,
        market_fingerprint=market_fingerprint,
        input_fingerprint=input_fingerprint,
        window_count=len(windows),
        input_reference=input_reference,
    )
