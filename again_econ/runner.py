from __future__ import annotations

from pathlib import Path

from again_econ.adapters.again_bundle import AgainBundleAdapter
from again_econ.config import BacktestConfig
from again_econ.contracts import (
    BacktestResult,
    BundleProvenanceMode,
    DiscardReason,
    DiscardedSignal,
    ForecastRecord,
    InputBundle,
    MarketFrame,
    SignalRecord,
    WindowResult,
)
from again_econ.errors import ContractValidationError, TemporalIntegrityError
from again_econ.execution import run_window_execution, schedule_signal_next_open
from again_econ.manifest import build_run_manifest
from again_econ.metrics import compute_window_average_metrics, summarize_global_oos_metrics, summarize_metrics
from again_econ.signals import translate_forecasts_to_signals
from again_econ.validation import (
    validate_forecasts,
    validate_input_bundle,
    validate_market_frame,
    validate_record_matches_window,
    validate_signals,
)
from again_econ.walkforward import build_walkforward_windows, slice_test_market


def _resolve_input_bundle(
    forecasts: tuple[ForecastRecord, ...] | None,
    signals: tuple[SignalRecord, ...] | None,
    adapter_name: str,
    input_reference: str | None = None,
) -> InputBundle:
    resolved_forecasts = forecasts or ()
    resolved_signals = signals or ()
    if resolved_forecasts and resolved_signals:
        raise ContractValidationError("Una corrida economica debe partir de forecasts o de signals, no de ambos a la vez")
    if not resolved_forecasts and not resolved_signals:
        raise ContractValidationError("La corrida economica requiere forecasts o signals")
    return InputBundle(
        adapter_name=adapter_name,
        bundle_version=1,
        provenance_mode=BundleProvenanceMode.LEGACY_DEGRADED,
        forecasts=tuple(resolved_forecasts),
        signals=tuple(resolved_signals),
        source_path=input_reference,
    )


def _validate_record_provenance_against_windows(
    records: tuple[ForecastRecord, ...] | tuple[SignalRecord, ...],
    windows,
) -> None:
    windows_by_index = {window.index: window for window in windows}
    for record in records:
        if record.provenance is None:
            continue
        window = windows_by_index.get(record.provenance.window_index)
        if window is None:
            raise ContractValidationError(
                f"El record de {record.instrument_id} referencia una window_index inexistente: {record.provenance.window_index}"
            )
        validate_record_matches_window(record, window)


def _signal_belongs_to_window(signal: SignalRecord, window) -> bool:
    if signal.provenance is not None:
        return signal.provenance.window_index == window.index
    return window.test_start <= signal.decision_timestamp <= window.test_end


def _prepare_window_signals(
    signals: tuple[SignalRecord, ...],
    market_frame: MarketFrame,
    window,
) -> tuple[tuple, tuple[DiscardedSignal, ...]]:
    scheduled = []
    discarded = []
    ordered_signals = sorted(signals, key=lambda item: (item.decision_timestamp, item.instrument_id))
    for signal in ordered_signals:
        if not _signal_belongs_to_window(signal, window):
            continue
        try:
            scheduled_signal = schedule_signal_next_open(signal, market_frame)
        except TemporalIntegrityError:
            discarded.append(
                DiscardedSignal(
                    instrument_id=signal.instrument_id,
                    decision_timestamp=signal.decision_timestamp,
                    execution_timestamp=None,
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
                    window_index=window.index,
                    reason=DiscardReason.NEXT_OPEN_OUTSIDE_WINDOW,
                )
            )
            continue
        scheduled.append(scheduled_signal)
    return tuple(scheduled), tuple(discarded)


def _run_backtest_with_bundle(
    market_frame: MarketFrame,
    config: BacktestConfig,
    bundle: InputBundle,
) -> BacktestResult:
    validate_market_frame(market_frame)
    validate_input_bundle(bundle)

    if bundle.forecasts:
        validate_forecasts(bundle.forecasts)
        resolved_signals = translate_forecasts_to_signals(bundle.forecasts, config.signal)
    else:
        resolved_signals = bundle.signals
    validate_signals(resolved_signals)

    windows = build_walkforward_windows(market_frame, config.walkforward)
    _validate_record_provenance_against_windows(resolved_signals, windows)

    window_results = []
    for window in windows:
        test_market = slice_test_market(market_frame, window)
        scheduled_signals, discarded_signals = _prepare_window_signals(resolved_signals, market_frame, window)
        execution_result = run_window_execution(
            test_market,
            scheduled_signals,
            config.execution,
            window_index=window.index,
        )
        winning_trades = sum(1 for trade in execution_result.trades if trade.net_pnl > 0.0)
        metrics = summarize_metrics(
            execution_result.snapshots,
            initial_equity=config.execution.initial_cash,
            trade_count=len(execution_result.trades),
            winning_trades=winning_trades,
            bars_per_year=config.execution.bars_per_year,
        )
        window_results.append(
            WindowResult(
                window=window,
                fills=execution_result.fills,
                trades=execution_result.trades,
                snapshots=execution_result.snapshots,
                discarded_signals=discarded_signals,
                metrics=metrics,
            )
        )

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
        bundle=bundle,
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
    adapter_name: str = "direct",
    input_reference: str | None = None,
) -> BacktestResult:
    bundle = _resolve_input_bundle(forecasts, signals, adapter_name=adapter_name, input_reference=input_reference)
    return _run_backtest_with_bundle(market_frame, config, bundle)


def run_backtest_from_bundle(
    market_frame: MarketFrame,
    config: BacktestConfig,
    bundle_path: str | Path,
) -> BacktestResult:
    adapter = AgainBundleAdapter()
    loaded = adapter.load(bundle_path)
    return _run_backtest_with_bundle(market_frame, config, loaded)
