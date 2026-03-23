from datetime import datetime

import pytest

from again_econ.contracts import ForecastRecord, MarketBar, MarketFrame, TargetKind
from again_econ.errors import TemporalIntegrityError
from again_econ.validation import validate_forecasts, validate_market_frame


def test_validate_market_frame_rejects_duplicate_or_non_increasing_timestamps():
    timestamp = datetime(2024, 1, 1)
    market = MarketFrame(
        bars=(
            MarketBar("AAA", timestamp, 10.0, 11.0, 9.0, 10.0, 100.0),
            MarketBar("AAA", timestamp, 10.5, 11.5, 9.5, 10.5, 100.0),
        )
    )

    with pytest.raises(TemporalIntegrityError):
        validate_market_frame(market)


def test_validate_forecasts_rejects_future_available_at():
    record = ForecastRecord(
        instrument_id="AAA",
        decision_timestamp=datetime(2024, 1, 1),
        available_at=datetime(2024, 1, 2),
        target_kind=TargetKind.RETURN,
        value=0.01,
    )

    with pytest.raises(TemporalIntegrityError):
        validate_forecasts((record,))
