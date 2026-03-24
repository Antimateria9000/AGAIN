from __future__ import annotations

from datetime import datetime
import hashlib
import json
from typing import Protocol

import pandas as pd

from again_benchmark.contracts import (
    BenchmarkDefinition,
    BenchmarkMode,
    BenchmarkRunBundle,
    BenchmarkRunManifest,
    BenchmarkTickerResult,
)
from again_benchmark.errors import BenchmarkExecutionError
from again_benchmark.metrics import summarize_results
from again_benchmark.reports import build_plot_payload
from again_benchmark.snapshots import build_snapshot_manifest, normalize_market_data_frame
from again_benchmark.storage import BenchmarkStorage, compute_sha256
from again_benchmark.validation import validate_definition, validate_run_manifest, validate_snapshot_manifest


class BenchmarkAdapterProtocol(Protocol):
    adapter_name: str

    def fetch_market_data(self, definition: BenchmarkDefinition, as_of_timestamp: datetime) -> pd.DataFrame: ...

    def evaluate_ticker(
        self,
        definition: BenchmarkDefinition,
        market_data: pd.DataFrame,
        ticker: str,
        split_date: datetime,
    ) -> BenchmarkTickerResult: ...

    def get_model_metadata(self) -> dict[str, str | None]: ...

    def get_runtime_metadata(self) -> dict[str, str | None]: ...

    def get_config_fingerprint(self) -> str | None: ...

    def reset(self) -> None: ...


def _resolve_common_split_date(market_data: pd.DataFrame, definition: BenchmarkDefinition) -> tuple[datetime, tuple[str, ...], tuple[str, ...]]:
    candidates = []
    effective_universe = []
    failed_tickers = []
    for ticker in definition.tickers:
        ticker_frame = market_data[market_data["Ticker"] == ticker].copy()
        unique_dates = pd.Series(pd.to_datetime(ticker_frame["Date"]).sort_values().unique())
        if len(unique_dates) <= definition.horizon:
            failed_tickers.append(ticker)
            continue
        effective_universe.append(ticker)
        candidates.append(pd.Timestamp(unique_dates.iloc[-(definition.horizon + 1)]).to_pydatetime())
    if not candidates:
        raise BenchmarkExecutionError("No hay suficiente historial para construir un split comun reproducible")
    return min(candidates), tuple(effective_universe), tuple(failed_tickers)


def _build_run_id(
    *,
    definition: BenchmarkDefinition,
    mode: BenchmarkMode,
    split_date: datetime,
    snapshot_id: str | None,
    config_fingerprint: str | None,
    model_sha256: str | None,
    normalizers_sha256: str | None,
    dataset_sha256: str | None,
) -> str:
    payload = json.dumps(
        {
            "definition_id": definition.definition_id,
            "benchmark_version": definition.benchmark_version,
            "mode": mode.value,
            "split_date": split_date.isoformat(),
            "snapshot_id": snapshot_id,
            "config_fingerprint": config_fingerprint,
            "model_sha256": model_sha256,
            "normalizers_sha256": normalizers_sha256,
            "dataset_sha256": dataset_sha256,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _resolve_run_created_at(mode: BenchmarkMode, as_of_timestamp: datetime, snapshot_created_at: datetime | None) -> datetime:
    if mode == BenchmarkMode.FROZEN and snapshot_created_at is not None:
        return snapshot_created_at
    return as_of_timestamp


def _build_run_manifest(
    *,
    definition: BenchmarkDefinition,
    mode: BenchmarkMode,
    as_of_timestamp: datetime,
    split_date: datetime,
    effective_universe: tuple[str, ...],
    snapshot_id: str | None,
    snapshot_sha256: str | None,
    snapshot_created_at: datetime | None,
    adapter: BenchmarkAdapterProtocol,
) -> BenchmarkRunManifest:
    model_metadata = adapter.get_model_metadata()
    runtime_metadata = adapter.get_runtime_metadata()
    run_id = _build_run_id(
        definition=definition,
        mode=mode,
        split_date=split_date,
        snapshot_id=snapshot_id,
        config_fingerprint=adapter.get_config_fingerprint(),
        model_sha256=model_metadata.get("model_sha256"),
        normalizers_sha256=model_metadata.get("normalizers_sha256"),
        dataset_sha256=model_metadata.get("dataset_sha256"),
    )
    manifest = BenchmarkRunManifest(
        run_id=run_id,
        benchmark_id=definition.benchmark_id,
        benchmark_version=definition.benchmark_version,
        definition_id=definition.definition_id,
        mode=mode,
        created_at=_resolve_run_created_at(mode, as_of_timestamp, snapshot_created_at),
        as_of_timestamp=as_of_timestamp,
        split_date=split_date,
        tickers=definition.tickers,
        effective_universe=effective_universe,
        horizon=definition.horizon,
        metrics=definition.metrics,
        model_name=model_metadata.get("model_name") or "unknown_model",
        profile_path=model_metadata.get("profile_path"),
        config_fingerprint=adapter.get_config_fingerprint(),
        model_sha256=model_metadata.get("model_sha256"),
        normalizers_sha256=model_metadata.get("normalizers_sha256"),
        dataset_sha256=model_metadata.get("dataset_sha256"),
        snapshot_id=snapshot_id,
        snapshot_sha256=snapshot_sha256,
        code_commit_sha=model_metadata.get("code_commit_sha"),
        python_version=runtime_metadata.get("python_version"),
        torch_version=runtime_metadata.get("torch_version"),
        backend=runtime_metadata.get("backend"),
        device=runtime_metadata.get("device"),
    )
    validate_run_manifest(manifest, definition=definition)
    return manifest


class BenchmarkRunner:
    def __init__(self, storage: BenchmarkStorage, adapter: BenchmarkAdapterProtocol):
        self.storage = storage
        self.adapter = adapter

    def create_frozen_snapshot(
        self,
        definition: BenchmarkDefinition,
        *,
        as_of_timestamp: datetime,
        market_data: pd.DataFrame | None = None,
    ):
        validate_definition(definition)
        self.storage.write_definition(definition)
        frame = normalize_market_data_frame(
            market_data if market_data is not None else self.adapter.fetch_market_data(definition, as_of_timestamp)
        )
        provisional_manifest = build_snapshot_manifest(
            definition=definition,
            created_at=as_of_timestamp,
            as_of_timestamp=as_of_timestamp,
            frame=frame,
            data_path=self.storage.snapshot_dir("pending") / "market_data.parquet",
            data_sha256="pending",
        )
        target_dir = self.storage.snapshot_dir(provisional_manifest.snapshot_id)
        data_path = target_dir / "market_data.parquet"
        target_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(data_path, index=False)
        data_sha256 = compute_sha256(data_path)
        manifest = build_snapshot_manifest(
            definition=definition,
            created_at=as_of_timestamp,
            as_of_timestamp=as_of_timestamp,
            frame=frame,
            data_path=data_path,
            data_sha256=data_sha256,
        )
        validate_snapshot_manifest(manifest, definition=definition)
        self.storage.write_snapshot(frame, manifest)
        return manifest

    def run_frozen_from_snapshot(self, definition: BenchmarkDefinition, snapshot_id: str) -> BenchmarkRunBundle:
        validate_definition(definition)
        self.storage.write_definition(definition)
        market_data, snapshot_manifest = self.storage.load_snapshot(snapshot_id)
        validate_snapshot_manifest(snapshot_manifest, definition=definition)
        split_date, effective_universe, failed_tickers = _resolve_common_split_date(market_data, definition)
        self.adapter.reset()
        try:
            ticker_results = tuple(
                self.adapter.evaluate_ticker(definition, market_data, ticker, split_date)
                for ticker in effective_universe
            )
        finally:
            self.adapter.reset()
        manifest = _build_run_manifest(
            definition=definition,
            mode=BenchmarkMode.FROZEN,
            as_of_timestamp=snapshot_manifest.as_of_timestamp,
            split_date=split_date,
            effective_universe=effective_universe,
            snapshot_id=snapshot_manifest.snapshot_id,
            snapshot_sha256=snapshot_manifest.data_sha256,
            snapshot_created_at=snapshot_manifest.created_at,
            adapter=self.adapter,
        )
        if self.storage.run_dir(manifest.run_id).exists():
            return self.storage.load_run_bundle(manifest.run_id)
        summary = summarize_results(
            run_id=manifest.run_id,
            benchmark_id=definition.benchmark_id,
            benchmark_version=definition.benchmark_version,
            mode=BenchmarkMode.FROZEN,
            requested_tickers=definition.tickers,
            effective_universe=effective_universe,
            ticker_results=ticker_results,
            failed_tickers=failed_tickers,
        )
        bundle = BenchmarkRunBundle(
            manifest=manifest,
            summary=summary,
            ticker_results=ticker_results,
            plot_payload=build_plot_payload(ticker_results),
        )
        self.storage.write_run_bundle(bundle)
        return bundle

    def run_live(self, definition: BenchmarkDefinition, *, as_of_timestamp: datetime) -> BenchmarkRunBundle:
        validate_definition(definition)
        self.storage.write_definition(definition)
        market_data = normalize_market_data_frame(self.adapter.fetch_market_data(definition, as_of_timestamp))
        split_date, effective_universe, failed_tickers = _resolve_common_split_date(market_data, definition)
        self.adapter.reset()
        try:
            ticker_results = tuple(
                self.adapter.evaluate_ticker(definition, market_data, ticker, split_date)
                for ticker in effective_universe
            )
        finally:
            self.adapter.reset()
        manifest = _build_run_manifest(
            definition=definition,
            mode=BenchmarkMode.LIVE,
            as_of_timestamp=as_of_timestamp,
            split_date=split_date,
            effective_universe=effective_universe,
            snapshot_id=None,
            snapshot_sha256=None,
            snapshot_created_at=None,
            adapter=self.adapter,
        )
        if self.storage.run_dir(manifest.run_id).exists():
            return self.storage.load_run_bundle(manifest.run_id)
        summary = summarize_results(
            run_id=manifest.run_id,
            benchmark_id=definition.benchmark_id,
            benchmark_version=definition.benchmark_version,
            mode=BenchmarkMode.LIVE,
            requested_tickers=definition.tickers,
            effective_universe=effective_universe,
            ticker_results=ticker_results,
            failed_tickers=failed_tickers,
        )
        bundle = BenchmarkRunBundle(
            manifest=manifest,
            summary=summary,
            ticker_results=ticker_results,
            plot_payload=build_plot_payload(ticker_results),
        )
        self.storage.write_run_bundle(bundle)
        return bundle

    def rerun_from_run_id(self, run_id: str) -> BenchmarkRunBundle:
        existing = self.storage.load_run_bundle(run_id)
        if existing.manifest.mode != BenchmarkMode.FROZEN:
            raise BenchmarkExecutionError("Solo las corridas frozen pueden reejecutarse exactamente desde run_id")
        definition = self.storage.load_definition(existing.manifest.definition_id)
        return self.run_frozen_from_snapshot(definition, existing.manifest.snapshot_id or "")
