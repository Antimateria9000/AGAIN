from __future__ import annotations

from datetime import timedelta
import json

import pytest

from again_econ.config import BacktestConfig, ExecutionConfig, SignalConfig, WalkforwardConfig
from again_econ.contracts import (
    BundleProvenanceMode,
    DiscardReason,
    ExecutionReason,
    InputSourceKind,
    ProviderDataKind,
    ProviderIdentity,
    ProviderWindowPayload,
    PositionTarget,
    SignalRecord,
)
from again_econ.fingerprints import fingerprint_payload
from again_econ.providers import SignalProvider
from again_econ.runner import run_backtest_from_bundle, run_backtest_with_provider

from tests.again_econ_test_utils import build_single_symbol_market


def test_backtest_integration_runs_end_to_end_from_legacy_forecast_bundle(tmp_path):
    market = build_single_symbol_market([10, 10, 10, 11, 12], [10, 10, 10, 11, 12])
    timestamps = market.timestamps()
    bundle_path = tmp_path / "again_forecasts.json"
    bundle_path.write_text(
        json.dumps(
            {
                "bundle_version": 1,
                "payload_type": "forecast_records",
                "records": [
                    {
                        "instrument_id": "AAA",
                        "decision_timestamp": timestamps[2].isoformat(),
                        "available_at": timestamps[2].isoformat(),
                        "target_kind": "return",
                        "value": 0.05,
                    },
                    {
                        "instrument_id": "AAA",
                        "decision_timestamp": timestamps[3].isoformat(),
                        "available_at": timestamps[3].isoformat(),
                        "target_kind": "return",
                        "value": -0.01,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = BacktestConfig(
        walkforward=WalkforwardConfig(train_size=2, test_size=3, step_size=3),
        signal=SignalConfig(long_threshold=0.0),
        execution=ExecutionConfig(initial_cash=100.0, allocation_fraction=1.0, allow_fractional_shares=False),
        label="integration_test",
    )

    result = run_backtest_from_bundle(market, config, bundle_path)

    assert result.manifest.adapter_name == "again_json_bundle"
    assert result.manifest.provenance_mode == BundleProvenanceMode.LEGACY_DEGRADED
    assert len(result.windows) == 1
    assert result.summary_metrics.trade_count == 1
    assert result.windows[0].trades[0].net_pnl == 9.0
    assert result.summary_metrics.total_return == pytest.approx(0.09)


def test_backtest_integration_accepts_v2_signal_bundle_with_provenance(tmp_path):
    market = build_single_symbol_market([10, 10, 10, 11, 12], [10, 10, 10, 11, 12])
    timestamps = market.timestamps()
    bundle_path = tmp_path / "again_signals_v2.json"
    bundle_path.write_text(
        json.dumps(
            {
                "bundle_version": 2,
                "payload_type": "signal_records",
                "provenance": {
                    "generated_at": "2024-01-10T00:00:00+00:00",
                    "model_run_id": "model-run-1",
                    "data_fingerprint": "data-123",
                    "code_fingerprint": "code-456",
                    "window": {
                        "window_index": 0,
                        "train_end": timestamps[1].isoformat(),
                        "test_start": timestamps[2].isoformat(),
                        "test_end": timestamps[4].isoformat(),
                    },
                },
                "records": [
                    {
                        "instrument_id": "AAA",
                        "decision_timestamp": timestamps[2].isoformat(),
                        "available_at": timestamps[2].isoformat(),
                        "target_state": "long",
                    },
                    {
                        "instrument_id": "AAA",
                        "decision_timestamp": timestamps[3].isoformat(),
                        "available_at": timestamps[3].isoformat(),
                        "target_state": "flat",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = BacktestConfig(
        walkforward=WalkforwardConfig(train_size=2, test_size=3, step_size=3),
        signal=SignalConfig(long_threshold=0.0),
        execution=ExecutionConfig(initial_cash=100.0, allocation_fraction=1.0, allow_fractional_shares=False),
        label="integration_v2_signal_test",
    )

    result = run_backtest_from_bundle(market, config, bundle_path)

    assert result.manifest.bundle_version == 2
    assert result.manifest.provenance_mode == BundleProvenanceMode.STRICT_V2
    assert len(result.windows[0].trades) == 1
    assert result.windows[0].trades[0].exit_reason == ExecutionReason.SIGNAL


def test_signal_whose_next_open_falls_outside_window_is_discarded_and_not_carried(tmp_path):
    market = build_single_symbol_market([10, 10, 10, 11, 12, 13, 14, 15], [10, 10, 10, 11, 12, 13, 14, 15])
    timestamps = market.timestamps()
    bundle_path = tmp_path / "again_signals.json"
    bundle_path.write_text(
        json.dumps(
            {
                "bundle_version": 1,
                "payload_type": "signal_records",
                "records": [
                    {
                        "instrument_id": "AAA",
                        "decision_timestamp": timestamps[4].isoformat(),
                        "available_at": timestamps[4].isoformat(),
                        "target_state": "long",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = BacktestConfig(
        walkforward=WalkforwardConfig(train_size=2, test_size=3, step_size=3),
        execution=ExecutionConfig(initial_cash=100.0, allocation_fraction=1.0, allow_fractional_shares=False),
        label="window_discard_test",
    )

    result = run_backtest_from_bundle(market, config, bundle_path)

    assert len(result.windows) == 2
    assert len(result.windows[0].discarded_signals) == 1
    assert result.windows[0].discarded_signals[0].reason == DiscardReason.NEXT_OPEN_OUTSIDE_WINDOW
    assert result.windows[1].fills == ()
    assert result.summary_metrics.trade_count == 0


def test_signal_at_last_timestamp_is_explicitly_discarded_when_no_next_open_exists(tmp_path):
    market = build_single_symbol_market([10, 10, 10, 11, 12], [10, 10, 10, 11, 12])
    timestamps = market.timestamps()
    bundle_path = tmp_path / "again_last_timestamp_signal.json"
    bundle_path.write_text(
        json.dumps(
            {
                "bundle_version": 1,
                "payload_type": "signal_records",
                "records": [
                    {
                        "instrument_id": "AAA",
                        "decision_timestamp": timestamps[4].isoformat(),
                        "available_at": timestamps[4].isoformat(),
                        "target_state": "long",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = BacktestConfig(
        walkforward=WalkforwardConfig(train_size=2, test_size=3, step_size=3),
        execution=ExecutionConfig(initial_cash=100.0, allocation_fraction=1.0, allow_fractional_shares=False),
        label="last_timestamp_discard_test",
    )

    result = run_backtest_from_bundle(market, config, bundle_path)

    assert result.windows[0].discarded_signals[0].reason == DiscardReason.NO_NEXT_OPEN_AVAILABLE
    assert result.windows[0].discarded_signals[0].execution_timestamp is None
    assert result.summary_metrics.total_return == pytest.approx(0.0)


class MockWindowSignalProvider(SignalProvider):
    def __init__(self) -> None:
        self.calls: list[int] = []
        self._identity = ProviderIdentity(
            name="mock_window_signal_provider",
            version="v1",
            source_kind=InputSourceKind.WINDOW_PROVIDER_SIGNALS,
            data_kind=ProviderDataKind.SIGNAL,
        )

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def get_window_payload(self, window, market_frame):
        del market_frame
        self.calls.append(window.index)
        signals = (
            SignalRecord(
                instrument_id="AAA",
                decision_timestamp=window.test_start,
                available_at=window.test_start,
                target_state=PositionTarget.LONG,
                score=0.8,
                provenance=window.to_provenance(),
            ),
            SignalRecord(
                instrument_id="AAA",
                decision_timestamp=window.test_start + timedelta(days=1),
                available_at=window.test_start + timedelta(days=1),
                target_state=PositionTarget.FLAT,
                score=-0.2,
                provenance=window.to_provenance(),
            ),
        )
        return ProviderWindowPayload(
            window_index=window.index,
            provider=self.identity,
            payload_kind=ProviderDataKind.SIGNAL,
            signals=signals,
            payload_fingerprint=fingerprint_payload((window.index, signals)),
        )


def test_backtest_integration_runs_per_window_from_provider_pipeline():
    market = build_single_symbol_market([10, 10, 10, 11, 12, 12, 13, 14], [10, 10, 10, 11, 12, 12, 13, 14])
    provider = MockWindowSignalProvider()
    config = BacktestConfig(
        walkforward=WalkforwardConfig(train_size=2, test_size=3, step_size=3),
        execution=ExecutionConfig(initial_cash=100.0, allocation_fraction=1.0, allow_fractional_shares=False),
        label="provider_pipeline_test",
    )

    result = run_backtest_with_provider(market, config, provider)

    assert provider.calls == [0, 1]
    assert result.manifest.provider.name == "mock_window_signal_provider"
    assert result.manifest.window_count == 2
    assert len(result.windows) == 2
    assert result.summary_metrics.trade_count == 2
    assert result.windows[0].manifest is not None
    assert result.windows[0].manifest.provider.name == "mock_window_signal_provider"
