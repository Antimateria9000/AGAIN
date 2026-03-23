from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from again_econ.contracts import (
    BundleProvenanceMode,
    ForecastRecord,
    InputBundle,
    MarketFrame,
    ScheduledSignal,
    SignalRecord,
    WalkforwardWindow,
    WindowProvenance,
)
from again_econ.errors import ContractValidationError, TemporalIntegrityError


def validate_market_frame(market_frame: MarketFrame) -> None:
    by_instrument: dict[str, list] = defaultdict(list)
    for bar in market_frame.bars:
        by_instrument[bar.instrument_id].append(bar)
    for instrument_id, bars in by_instrument.items():
        ordered = sorted(bars, key=lambda item: item.timestamp)
        for previous, current in zip(ordered, ordered[1:]):
            if current.timestamp <= previous.timestamp:
                raise TemporalIntegrityError(
                    f"Las barras de {instrument_id} deben estar en orden temporal estrictamente creciente"
                )


def validate_window_provenance(provenance: WindowProvenance) -> None:
    if not (provenance.train_end < provenance.test_start <= provenance.test_end):
        raise TemporalIntegrityError("WindowProvenance invalida: train_end < test_start <= test_end es obligatorio")


def validate_forecasts(forecasts: Iterable[ForecastRecord]) -> None:
    seen: set[tuple[str, object]] = set()
    for record in forecasts:
        key = (record.instrument_id, record.decision_timestamp)
        if key in seen:
            raise ContractValidationError("No puede haber dos forecasts para el mismo instrumento y decision_timestamp")
        seen.add(key)
        if record.available_at > record.decision_timestamp:
            raise TemporalIntegrityError("available_at no puede ser posterior a decision_timestamp")
        if record.reference_value is not None and record.reference_value <= 0.0:
            raise ContractValidationError("reference_value debe ser > 0 cuando existe")
        if record.provenance is not None:
            validate_window_provenance(record.provenance)


def validate_signals(signals: Iterable[SignalRecord]) -> None:
    seen: set[tuple[str, object]] = set()
    for record in signals:
        key = (record.instrument_id, record.decision_timestamp)
        if key in seen:
            raise ContractValidationError("No puede haber dos signals para el mismo instrumento y decision_timestamp")
        seen.add(key)
        if record.available_at > record.decision_timestamp:
            raise TemporalIntegrityError("available_at no puede ser posterior a decision_timestamp")
        if record.provenance is not None:
            validate_window_provenance(record.provenance)


def validate_input_bundle(bundle: InputBundle) -> None:
    if bundle.bundle_version == 1:
        if bundle.provenance_mode != BundleProvenanceMode.LEGACY_DEGRADED:
            raise ContractValidationError("Los bundles legacy deben declararse como legacy_degraded")
        if bundle.provenance is not None:
            raise ContractValidationError("bundle_version=1 no debe incluir provenance estructurada")
        return
    if bundle.bundle_version == 2:
        if bundle.provenance_mode != BundleProvenanceMode.STRICT_V2:
            raise ContractValidationError("bundle_version=2 requiere provenance_mode strict_v2")
        if bundle.provenance is None:
            raise ContractValidationError("bundle_version=2 requiere provenance de bundle")
        if bundle.provenance.window is not None:
            validate_window_provenance(bundle.provenance.window)
        records = bundle.forecasts or bundle.signals
        if bundle.provenance.window is None and any(record.provenance is None for record in records):
            raise ContractValidationError(
                "bundle_version=2 requiere provenance de ventana en el bundle o en cada record"
            )
        return
    raise ContractValidationError(f"bundle_version no soportada: {bundle.bundle_version}")


def validate_walkforward_windows(windows: Iterable[WalkforwardWindow]) -> None:
    ordered = sorted(windows, key=lambda item: item.index)
    previous_test_end = None
    for window in ordered:
        if not (window.train_start <= window.train_end < window.test_start <= window.test_end):
            raise TemporalIntegrityError("Cada ventana walk-forward debe cumplir train_end < test_start <= test_end")
        if previous_test_end is not None and window.test_start <= previous_test_end:
            raise TemporalIntegrityError("Las ventanas de test no deben solaparse")
        previous_test_end = window.test_end


def validate_scheduled_signals(scheduled_signals: Iterable[ScheduledSignal]) -> None:
    seen: set[tuple[str, object]] = set()
    for scheduled in scheduled_signals:
        key = (scheduled.signal.instrument_id, scheduled.execution_timestamp)
        if key in seen:
            raise TemporalIntegrityError(
                "No puede haber dos señales programadas para el mismo instrumento y execution_timestamp"
            )
        seen.add(key)
        if scheduled.execution_timestamp <= scheduled.signal.decision_timestamp:
            raise TemporalIntegrityError("La ejecucion debe ocurrir estrictamente despues de la decision")


def validate_record_matches_window(record: ForecastRecord | SignalRecord, window: WalkforwardWindow) -> None:
    provenance = record.provenance
    if provenance is None:
        if not (window.test_start <= record.decision_timestamp <= window.test_end):
            raise TemporalIntegrityError("El record no cae en el rango temporal permitido de la ventana")
        return
    if provenance.window_index != window.index:
        raise TemporalIntegrityError("window_index del record no coincide con la ventana walk-forward")
    if provenance.train_end != window.train_end or provenance.test_start != window.test_start or provenance.test_end != window.test_end:
        raise TemporalIntegrityError("La provenance temporal del record no coincide con la ventana activa")
    if not (window.test_start <= record.decision_timestamp <= window.test_end):
        raise TemporalIntegrityError("El decision_timestamp del record no es coherente con su provenance")


def ensure_market_timestamp_exists(market_frame: MarketFrame, instrument_id: str, timestamp) -> None:
    if market_frame.bar_for(instrument_id, timestamp) is None:
        raise ContractValidationError(f"No existe barra para {instrument_id} en {timestamp!s}")
