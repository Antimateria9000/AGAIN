from datetime import datetime

import pytest

from again_econ.contracts import (
    BundleProvenance,
    BundleProvenanceMode,
    ForecastRecord,
    InputBundle,
    MarketBar,
    MarketFrame,
    PositionTarget,
    SignalRecord,
    TargetKind,
    WalkforwardWindow,
    WindowProvenance,
)
from again_econ.errors import ContractValidationError, TemporalIntegrityError
from again_econ.validation import (
    validate_forecasts,
    validate_input_bundle,
    validate_market_frame,
    validate_record_matches_window,
    validate_signals,
)
from tests.again_econ_test_utils import aware_utc_datetime


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


def test_validate_input_bundle_requires_provenance_coverage_for_v2():
    decision_timestamp = datetime(2024, 1, 3)
    bundle = InputBundle(
        adapter_name="bundle",
        bundle_version=2,
        provenance_mode=BundleProvenanceMode.STRICT_V2,
        forecasts=(
            ForecastRecord(
                instrument_id="AAA",
                decision_timestamp=decision_timestamp,
                available_at=decision_timestamp,
                target_kind=TargetKind.RETURN,
                value=0.01,
            ),
        ),
        provenance=BundleProvenance(
            generated_at=decision_timestamp,
            model_run_id="model-1",
            data_fingerprint="data-1",
            code_fingerprint="code-1",
        ),
    )

    with pytest.raises(ContractValidationError):
        validate_input_bundle(bundle)


def test_validate_record_matches_window_rejects_incoherent_provenance():
    window = WalkforwardWindow(
        index=0,
        train_start=datetime(2024, 1, 1),
        train_end=datetime(2024, 1, 2),
        test_start=datetime(2024, 1, 3),
        test_end=datetime(2024, 1, 5),
    )
    record = ForecastRecord(
        instrument_id="AAA",
        decision_timestamp=datetime(2024, 1, 4),
        available_at=datetime(2024, 1, 4),
        target_kind=TargetKind.RETURN,
        value=0.02,
        provenance=WindowProvenance(
            window_index=0,
            train_end=datetime(2024, 1, 1),
            test_start=datetime(2024, 1, 3),
            test_end=datetime(2024, 1, 5),
        ),
    )

    with pytest.raises(TemporalIntegrityError):
        validate_record_matches_window(record, window)


def test_timestamp_policy_rejects_mixed_naive_and_aware_timestamps():
    with pytest.raises(ContractValidationError):
        ForecastRecord(
            instrument_id="AAA",
            decision_timestamp=datetime(2024, 1, 1),
            available_at=aware_utc_datetime(2024, 1, 1),
            target_kind=TargetKind.RETURN,
            value=0.01,
        )


def test_validate_signals_rejects_duplicate_records():
    decision_timestamp = datetime(2024, 1, 1)
    signals = (
        SignalRecord(
            instrument_id="AAA",
            decision_timestamp=decision_timestamp,
            available_at=decision_timestamp,
            target_state=PositionTarget.LONG,
        ),
        SignalRecord(
            instrument_id="AAA",
            decision_timestamp=decision_timestamp,
            available_at=decision_timestamp,
            target_state=PositionTarget.FLAT,
        ),
    )

    with pytest.raises(ContractValidationError):
        validate_signals(signals)
