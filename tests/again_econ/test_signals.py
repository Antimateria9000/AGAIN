from datetime import datetime

from again_econ.config import SignalConfig
from again_econ.contracts import ForecastRecord, PositionTarget, TargetKind
from again_econ.signals import translate_forecasts_to_signals


def test_translate_forecasts_to_long_only_signals_for_supported_targets():
    decision_timestamp = datetime(2024, 1, 1)
    forecasts = (
        ForecastRecord(
            instrument_id="AAA",
            decision_timestamp=decision_timestamp,
            available_at=decision_timestamp,
            target_kind=TargetKind.PRICE,
            value=103.0,
            reference_value=100.0,
        ),
        ForecastRecord(
            instrument_id="BBB",
            decision_timestamp=decision_timestamp,
            available_at=decision_timestamp,
            target_kind=TargetKind.RETURN,
            value=-0.01,
        ),
        ForecastRecord(
            instrument_id="CCC",
            decision_timestamp=decision_timestamp,
            available_at=decision_timestamp,
            target_kind=TargetKind.DIRECTION,
            value=1.0,
        ),
        ForecastRecord(
            instrument_id="DDD",
            decision_timestamp=decision_timestamp,
            available_at=decision_timestamp,
            target_kind=TargetKind.SCORE,
            value=0.02,
        ),
    )

    signals = translate_forecasts_to_signals(forecasts, SignalConfig(long_threshold=0.01))

    assert signals[0].target_state == PositionTarget.LONG
    assert signals[1].target_state == PositionTarget.FLAT
    assert signals[2].target_state == PositionTarget.LONG
    assert signals[3].target_state == PositionTarget.LONG
