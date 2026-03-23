from __future__ import annotations

from again_econ.config import SignalConfig
from again_econ.contracts import ForecastRecord, PositionTarget, SignalRecord, TargetKind
from again_econ.errors import ContractValidationError


def _resolve_forecast_score(record: ForecastRecord) -> float:
    if record.target_kind == TargetKind.PRICE:
        if record.reference_value is None:
            raise ContractValidationError("Los forecasts de precio requieren reference_value para traducir a senal")
        return (record.value - record.reference_value) / record.reference_value
    if record.target_kind in {TargetKind.RETURN, TargetKind.SCORE, TargetKind.DIRECTION}:
        return float(record.value)
    raise ContractValidationError(f"TargetKind no soportado para traduccion a senal: {record.target_kind.value}")


def translate_forecasts_to_signals(
    forecasts: tuple[ForecastRecord, ...],
    config: SignalConfig,
) -> tuple[SignalRecord, ...]:
    signals = []
    for record in sorted(forecasts, key=lambda item: (item.decision_timestamp, item.instrument_id)):
        score = _resolve_forecast_score(record)
        if record.target_kind == TargetKind.DIRECTION:
            target_state = PositionTarget.LONG if score > 0.0 else PositionTarget.FLAT
        else:
            target_state = PositionTarget.LONG if score > config.long_threshold else PositionTarget.FLAT
        signals.append(
            SignalRecord(
                instrument_id=record.instrument_id,
                decision_timestamp=record.decision_timestamp,
                available_at=record.available_at,
                target_state=target_state,
                score=score,
                metadata={"target_kind": record.target_kind.value, **record.metadata},
            )
        )
    return tuple(signals)
