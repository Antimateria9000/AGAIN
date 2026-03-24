from __future__ import annotations

from pathlib import Path

from again_econ.adapters.again_bundle import AgainBundleAdapter
from again_econ.config import BacktestConfig
from again_econ.contracts import (
    BacktestResult,
    DiscardReason,
    DiscardedSignal,
    ForecastRecord,
    InputSourceKind,
    MarketFrame,
    ProviderDataKind,
    ProviderIdentity,
    SignalRecord,
    WindowManifest,
    WindowResult,
)
from again_econ.errors import ContractValidationError, TemporalIntegrityError
from again_econ.execution import SignalScheduler, run_window_execution
from again_econ.fingerprints import fingerprint_payload
from again_econ.manifest import build_run_manifest
from again_econ.metrics import compute_window_average_metrics, summarize_global_oos_metrics, summarize_metrics
from again_econ.providers import (
    ForecastProvider,
    SignalProvider,
    StaticForecastProvider,
    StaticSignalProvider,
    provider_from_bundle,
)
from again_econ.signals import translate_forecasts_to_signals
from again_econ.validation import (
    validate_forecasts,
    validate_input_bundle,
    validate_market_frame,
    validate_provider_window_payload,
    validate_signals,
)
from again_econ.walkforward import build_walkforward_windows, slice_test_market


def _resolve_direct_provider(
    *,
    forecasts: tuple[ForecastRecord, ...] | None,
    signals: tuple[SignalRecord, ...] | None,
    config: BacktestConfig,
    adapter_name: str,
    input_reference: str | None,
) -> ForecastProvider | SignalProvider:
    resolved_forecasts = forecasts or ()
    resolved_signals = signals or ()
    if resolved_forecasts and resolved_signals:
        raise ContractValidationError("Una corrida economica debe partir de forecasts o de signals, no de ambos a la vez")
    if not resolved_forecasts and not resolved_signals:
        raise ContractValidationError("La corrida economica requiere forecasts o signals")
    if resolved_forecasts:
        return StaticForecastProvider(
            tuple(resolved_forecasts),
            identity=ProviderIdentity(
                name=adapter_name or config.provider.name,
                version=config.provider.version,
                source_kind=InputSourceKind.DIRECT_FORECASTS,
                data_kind=ProviderDataKind.FORECAST,
            ),
            input_reference=input_reference,
        )
    return StaticSignalProvider(
        tuple(resolved_signals),
        identity=ProviderIdentity(
            name=adapter_name or config.provider.name,
            version=config.provider.version,
            source_kind=InputSourceKind.DIRECT_SIGNALS,
            data_kind=ProviderDataKind.SIGNAL,
        ),
        input_reference=input_reference,
    )


def _schedule_window_signals(signals: tuple[SignalRecord, ...], market_frame: MarketFrame, window):
    scheduler = SignalScheduler(market_frame, execution_lag_bars=window.execution_lag_bars)
    scheduled = []
    discarded = []
    ordered_signals = sorted(signals, key=lambda item: (item.decision_timestamp, item.instrument_id))
    for signal in ordered_signals:
        try:
            scheduled_signal = scheduler.schedule_signal_next_open(signal)
        except TemporalIntegrityError:
            discarded.append(
                DiscardedSignal(
                    instrument_id=signal.instrument_id,
                    decision_timestamp=signal.decision_timestamp,
                    execution_timestamp=None,
                    available_at=signal.available_at,
                    window_index=window.index,
                    reason=DiscardReason.NO_NEXT_OPEN_AVAILABLE,
                )
            )
            continue
        if scheduled_signal.execution_timestamp < window.test_start or scheduled_signal.execution_timestamp > window.test_end:
            discarded.append(
                DiscardedSignal(
                    instrument_id=signal.instrument_id,
                    decision_timestamp=signal.decision_timestamp,
                    execution_timestamp=scheduled_signal.execution_timestamp,
                    available_at=signal.available_at,
                    window_index=window.index,
                    reason=DiscardReason.NEXT_OPEN_OUTSIDE_WINDOW,
                )
            )
            continue
        scheduled.append(scheduled_signal)
    return tuple(scheduled), tuple(discarded)


def _build_window_manifest(window, payload, translated_signals, scheduled_signals, execution_result, discarded_signals) -> WindowManifest:
    discarded_reason_counts: dict[str, int] = {}
    for discarded in discarded_signals:
        key = discarded.reason.value
        discarded_reason_counts[key] = discarded_reason_counts.get(key, 0) + 1
    input_records = payload.forecasts or payload.signals
    payload_fingerprint = payload.payload_fingerprint or fingerprint_payload(
        {
            "provider": payload.provider,
            "window_index": window.index,
            "records": input_records,
        }
    )
    return WindowManifest(
        window_index=window.index,
        train_start=window.train_start,
        train_end=window.train_end,
        test_start=window.test_start,
        test_end=window.test_end,
        lookahead_bars=window.lookahead_bars,
        execution_lag_bars=window.execution_lag_bars,
        close_policy=window.close_policy,
        provider=payload.provider,
        payload_kind=payload.payload_kind,
        input_record_count=len(input_records),
        translated_signal_count=len(translated_signals),
        scheduled_signal_count=len(scheduled_signals),
        fill_count=len(execution_result.fills),
        trade_count=len(execution_result.trades),
        discarded_signal_count=len(discarded_signals),
        payload_fingerprint=payload_fingerprint,
        discarded_reason_counts=discarded_reason_counts,
        input_reference=payload.input_reference,
        artifact_references=payload.artifact_references,
    )


def run_backtest_with_provider(
    market_frame: MarketFrame,
    config: BacktestConfig,
    provider: ForecastProvider | SignalProvider,
) -> BacktestResult:
    validate_market_frame(market_frame)
    windows = build_walkforward_windows(market_frame, config.walkforward)

    window_results = []
    window_manifests = []
    input_fingerprints = []
    adapter_name = None
    bundle_version = None
    provenance_mode = None
    input_reference = None
    artifact_references = []

    for window in windows:
        payload = provider.get_window_payload(window, market_frame)
        validate_provider_window_payload(payload, window)
        if payload.payload_kind == ProviderDataKind.FORECAST:
            validate_forecasts(payload.forecasts)
            translated_signals = translate_forecasts_to_signals(payload.forecasts, config.signal)
        else:
            translated_signals = payload.signals
        validate_signals(translated_signals)
        scheduled_signals, discarded_signals = _schedule_window_signals(translated_signals, market_frame, window)
        test_market = slice_test_market(market_frame, window)
        execution_result = run_window_execution(
            test_market,
            scheduled_signals,
            config.execution,
            window_index=window.index,
            close_policy=window.close_policy,
        )
        winning_trades = sum(1 for trade in execution_result.trades if trade.net_pnl > 0.0)
        metrics = summarize_metrics(
            execution_result.snapshots,
            initial_equity=config.execution.initial_cash,
            trade_count=len(execution_result.trades),
            winning_trades=winning_trades,
            bars_per_year=config.execution.bars_per_year,
        )
        window_manifest = _build_window_manifest(
            window,
            payload,
            translated_signals,
            scheduled_signals,
            execution_result,
            discarded_signals,
        )
        window_results.append(
            WindowResult(
                window=window,
                fills=execution_result.fills,
                trades=execution_result.trades,
                snapshots=execution_result.snapshots,
                discarded_signals=discarded_signals,
                metrics=metrics,
                manifest=window_manifest,
            )
        )
        window_manifests.append(window_manifest)
        input_fingerprints.append(window_manifest.payload_fingerprint)
        adapter_name = payload.provider.name if payload.provider.source_kind == InputSourceKind.ADAPTED_BUNDLE else adapter_name
        bundle_version = payload.bundle_version if payload.bundle_version is not None else bundle_version
        provenance_mode = payload.provenance_mode if payload.provenance_mode is not None else provenance_mode
        input_reference = payload.input_reference or input_reference
        if payload.artifact_references:
            artifact_references.extend(payload.artifact_references)

    window_results_tuple = tuple(window_results)
    oos_curve, summary_metrics = summarize_global_oos_metrics(
        window_results_tuple,
        initial_equity=config.execution.initial_cash,
        bars_per_year=config.execution.bars_per_year,
    )
    manifest = build_run_manifest(
        config=config,
        market_frame=market_frame,
        windows=windows,
        provider=provider.identity,
        window_manifests=tuple(window_manifests),
        input_fingerprint=fingerprint_payload(input_fingerprints),
        adapter_name=adapter_name,
        bundle_version=bundle_version,
        provenance_mode=provenance_mode,
        input_reference=input_reference,
        artifact_references=tuple(
            {
                (artifact.artifact_type, artifact.locator, artifact.fingerprint, artifact.detail): artifact
                for artifact in artifact_references
            }.values()
        ),
    )
    return BacktestResult(
        manifest=manifest,
        windows=window_results_tuple,
        oos_curve=oos_curve,
        summary_metrics=summary_metrics,
        window_average_metrics=compute_window_average_metrics(window_results_tuple),
    )


def run_backtest(
    market_frame: MarketFrame,
    config: BacktestConfig,
    *,
    forecasts: tuple[ForecastRecord, ...] | None = None,
    signals: tuple[SignalRecord, ...] | None = None,
    adapter_name: str = "direct_input",
    input_reference: str | None = None,
) -> BacktestResult:
    provider = _resolve_direct_provider(
        forecasts=forecasts,
        signals=signals,
        config=config,
        adapter_name=adapter_name,
        input_reference=input_reference,
    )
    return run_backtest_with_provider(market_frame, config, provider)


def run_backtest_from_bundle(
    market_frame: MarketFrame,
    config: BacktestConfig,
    bundle_path: str | Path,
) -> BacktestResult:
    adapter = AgainBundleAdapter()
    loaded = adapter.load(bundle_path)
    validate_input_bundle(loaded)
    provider = provider_from_bundle(loaded)
    return run_backtest_with_provider(market_frame, config, provider)
