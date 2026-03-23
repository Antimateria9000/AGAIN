from __future__ import annotations

from pathlib import Path

from again_econ.adapters.again_bundle import AgainBundleAdapter
from again_econ.config import BacktestConfig
from again_econ.contracts import BacktestResult, ForecastRecord, InputBundle, MarketFrame, SignalRecord, WindowResult
from again_econ.errors import ContractValidationError
from again_econ.execution import run_window_execution, schedule_signals_next_open
from again_econ.manifest import build_run_manifest
from again_econ.metrics import aggregate_window_metrics, summarize_metrics
from again_econ.signals import translate_forecasts_to_signals
from again_econ.validation import validate_forecasts, validate_market_frame, validate_signals
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
        forecasts=tuple(resolved_forecasts),
        signals=tuple(resolved_signals),
        source_path=input_reference,
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
    validate_market_frame(market_frame)
    bundle = _resolve_input_bundle(forecasts, signals, adapter_name=adapter_name, input_reference=input_reference)
    if bundle.forecasts:
        validate_forecasts(bundle.forecasts)
        resolved_signals = translate_forecasts_to_signals(bundle.forecasts, config.signal)
    else:
        resolved_signals = bundle.signals
    validate_signals(resolved_signals)

    windows = build_walkforward_windows(market_frame, config.walkforward)
    scheduled_signals = schedule_signals_next_open(resolved_signals, market_frame)

    window_results = []
    for window in windows:
        test_market = slice_test_market(market_frame, window)
        window_scheduled_signals = tuple(
            scheduled for scheduled in scheduled_signals if window.test_start <= scheduled.execution_timestamp <= window.test_end
        )
        execution_result = run_window_execution(test_market, window_scheduled_signals, config.execution)
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
                metrics=metrics,
            )
        )

    summary_metrics = aggregate_window_metrics(tuple(window_results))
    input_payload = bundle.forecasts if bundle.forecasts else resolved_signals
    manifest = build_run_manifest(
        config=config,
        market_frame=market_frame,
        windows=windows,
        adapter_name=bundle.adapter_name,
        input_payload=input_payload,
        input_reference=bundle.source_path,
    )
    return BacktestResult(manifest=manifest, windows=tuple(window_results), summary_metrics=summary_metrics)


def run_backtest_from_bundle(
    market_frame: MarketFrame,
    config: BacktestConfig,
    bundle_path: str | Path,
) -> BacktestResult:
    adapter = AgainBundleAdapter()
    loaded = adapter.load(bundle_path)
    return run_backtest(
        market_frame=market_frame,
        config=config,
        forecasts=loaded.forecasts or None,
        signals=loaded.signals or None,
        adapter_name=loaded.adapter_name,
        input_reference=loaded.source_path,
    )
