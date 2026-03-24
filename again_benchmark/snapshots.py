from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path

import pandas as pd

from again_benchmark.contracts import BenchmarkDefinition, BenchmarkSnapshotManifest, SplitPolicy
from again_benchmark.errors import BenchmarkValidationError

REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume", "Ticker")


def normalize_market_data_frame(market_data: pd.DataFrame) -> pd.DataFrame:
    frame = market_data.copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise BenchmarkValidationError(f"Faltan columnas obligatorias en market_data: {missing}")
    frame["Date"] = pd.to_datetime(frame["Date"], utc=True).dt.tz_localize(None)
    if "Sector" not in frame.columns:
        frame["Sector"] = "Unknown"
    frame = frame.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    if frame.empty:
        raise BenchmarkValidationError("El snapshot no puede materializarse con market_data vacio")
    return frame


def fingerprint_market_data(frame: pd.DataFrame) -> str:
    normalized = frame.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    payload = normalized.to_json(orient="records", date_format="iso")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_snapshot_manifest(
    *,
    definition: BenchmarkDefinition,
    created_at: datetime,
    as_of_timestamp: datetime,
    frame: pd.DataFrame,
    data_path: Path,
    data_sha256: str,
) -> BenchmarkSnapshotManifest:
    effective_universe = tuple(str(value) for value in sorted(frame["Ticker"].dropna().astype(str).unique()))
    return BenchmarkSnapshotManifest(
        snapshot_id=hashlib.sha256(
            f"{definition.definition_id}:{as_of_timestamp.isoformat()}:{fingerprint_market_data(frame)}".encode("utf-8")
        ).hexdigest()[:16],
        benchmark_id=definition.benchmark_id,
        benchmark_version=definition.benchmark_version,
        definition_id=definition.definition_id,
        created_at=created_at,
        as_of_timestamp=as_of_timestamp,
        tickers=definition.tickers,
        effective_universe=effective_universe,
        horizon=definition.horizon,
        metrics=definition.metrics,
        split_policy=SplitPolicy(definition.split_policy),
        row_count=int(len(frame)),
        data_min_timestamp=pd.Timestamp(frame["Date"].min()).to_pydatetime(),
        data_max_timestamp=pd.Timestamp(frame["Date"].max()).to_pydatetime(),
        data_path=str(data_path),
        data_sha256=data_sha256,
    )
