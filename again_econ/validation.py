from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from again_econ.contracts import (
    BundleProvenanceMode,
    ForecastRecord,
    InputBundle,
    MarketFrame,
    PortfolioSnapshot,
    ProviderWindowPayload,
    ScheduledSignal,
    SignalRecord,
    WalkforwardWindow,
    WindowClosePolicy,
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


def _validate_record_temporal_semantics(record: ForecastRecord | SignalRecord) -> None:
    if record.observed_at > record.decision_timestamp:
        raise TemporalIntegrityError("observed_at no puede ser posterior a decision_timestamp")
    if record.available_at < record.observed_at:
        raise TemporalIntegrityError("available_at no puede ser anterior a observed_at")


def validate_window_provenance(provenance: WindowProvenance) -> None:  # type: ignore[no-redef]
    if provenance.train_start is not None and provenance.train_start > provenance.train_end:
        raise TemporalIntegrityError("WindowProvenance invalida: train_start no puede ser posterior a train_end")
    if not (provenance.train_end < provenance.test_start <= provenance.test_end):
        raise TemporalIntegrityError("WindowProvenance invalida: train_end < test_start <= test_end es obligatorio")
    if provenance.lookahead_bars <= 0:
        raise TemporalIntegrityError("WindowProvenance.lookahead_bars debe ser > 0")
    if provenance.execution_lag_bars <= 0:
        raise TemporalIntegrityError("WindowProvenance.execution_lag_bars debe ser > 0")


def validate_forecasts(forecasts: Iterable[ForecastRecord]) -> None:  # type: ignore[no-redef]
    seen: set[tuple[str, object]] = set()
    for record in forecasts:
        key = (record.instrument_id, record.decision_timestamp)
        if key in seen:
            raise ContractValidationError("No puede haber dos forecasts para el mismo instrumento y decision_timestamp")
        seen.add(key)
        _validate_record_temporal_semantics(record)
        if record.reference_value is not None and record.reference_value <= 0.0:
            raise ContractValidationError("reference_value debe ser > 0 cuando existe")
        if record.provenance is not None:
            validate_window_provenance(record.provenance)


def validate_signals(signals: Iterable[SignalRecord]) -> None:  # type: ignore[no-redef]
    seen: set[tuple[str, object]] = set()
    for record in signals:
        key = (record.instrument_id, record.decision_timestamp)
        if key in seen:
            raise ContractValidationError("No puede haber dos signals para el mismo instrumento y decision_timestamp")
        seen.add(key)
        _validate_record_temporal_semantics(record)
        if record.provenance is not None:
            validate_window_provenance(record.provenance)


def validate_provider_window_payload(payload: ProviderWindowPayload, window: WalkforwardWindow) -> None:
    if payload.window_index != window.index:
        raise TemporalIntegrityError("El payload del provider no coincide con la window solicitada")
    for record in payload.forecasts or payload.signals:
        validate_record_matches_window(record, window)


def validate_walkforward_windows(windows: Iterable[WalkforwardWindow]) -> None:  # type: ignore[no-redef]
    ordered = sorted(windows, key=lambda item: item.index)
    previous_test_end = None
    for window in ordered:
        if not (window.train_start <= window.train_end < window.test_start <= window.test_end):
            raise TemporalIntegrityError("Cada ventana walk-forward debe cumplir train_end < test_start <= test_end")
        if previous_test_end is not None and window.test_start <= previous_test_end:
            raise TemporalIntegrityError("Las ventanas de test no deben solaparse")
        if window.lookahead_bars <= 0 or window.execution_lag_bars <= 0:
            raise TemporalIntegrityError("Las ventanas deben declarar lookahead_bars y execution_lag_bars validos")
        previous_test_end = window.test_end


def validate_scheduled_signals(scheduled_signals: Iterable[ScheduledSignal]) -> None:  # type: ignore[no-redef]
    seen: set[tuple[str, object]] = set()
    for scheduled in scheduled_signals:
        key = (scheduled.signal.instrument_id, scheduled.execution_timestamp)
        if key in seen:
            raise TemporalIntegrityError(
                "No puede haber dos senales programadas para el mismo instrumento y execution_timestamp"
            )
        seen.add(key)
        operational_timestamp = max(scheduled.signal.decision_timestamp, scheduled.signal.available_at)
        if scheduled.execution_timestamp <= operational_timestamp:
            raise TemporalIntegrityError("La ejecucion debe ocurrir estrictamente despues del timestamp operativo")


def validate_record_matches_window(record: ForecastRecord | SignalRecord, window: WalkforwardWindow) -> None:  # type: ignore[no-redef]
    provenance = record.provenance
    if provenance is None:
        if not (window.test_start <= record.decision_timestamp <= window.test_end):
            raise TemporalIntegrityError("El record no cae en el rango temporal permitido de la ventana")
        return
    if provenance.window_index != window.index:
        raise TemporalIntegrityError("window_index del record no coincide con la ventana walk-forward")
    if provenance.train_start is not None and provenance.train_start != window.train_start:
        raise TemporalIntegrityError("train_start del record no coincide con la ventana walk-forward")
    if provenance.train_end != window.train_end or provenance.test_start != window.test_start or provenance.test_end != window.test_end:
        raise TemporalIntegrityError("La provenance temporal del record no coincide con la ventana activa")
    if provenance.lookahead_bars != window.lookahead_bars:
        raise TemporalIntegrityError("lookahead_bars del record no coincide con la ventana activa")
    if provenance.execution_lag_bars != window.execution_lag_bars:
        raise TemporalIntegrityError("execution_lag_bars del record no coincide con la ventana activa")
    if provenance.close_policy != window.close_policy:
        raise TemporalIntegrityError("close_policy del record no coincide con la ventana activa")
    if not (window.test_start <= record.decision_timestamp <= window.test_end):
        raise TemporalIntegrityError("El decision_timestamp del record no es coherente con su provenance")


def validate_snapshot_invariants(
    snapshots: tuple[PortfolioSnapshot, ...],
    *,
    close_policy: WindowClosePolicy,
    tolerance: float = 1e-9,
) -> None:
    previous_timestamp = None
    for snapshot in snapshots:
        if snapshot.cash < -tolerance:
            raise TemporalIntegrityError("El ledger produjo cash negativo en un snapshot")
        if abs((snapshot.cash + snapshot.market_value) - snapshot.total_equity) > tolerance:
            raise TemporalIntegrityError("El snapshot viola la identidad equity = cash + market_value")
        if previous_timestamp is not None and snapshot.timestamp <= previous_timestamp:
            raise TemporalIntegrityError("Los snapshots deben quedar en orden temporal estrictamente creciente")
        previous_timestamp = snapshot.timestamp
    if close_policy == WindowClosePolicy.ADMINISTRATIVE_CLOSE_ON_LAST_BAR and snapshots and snapshots[-1].open_positions != 0:
        raise TemporalIntegrityError("La ventana debe terminar sin posiciones abiertas tras el cierre administrativo")
