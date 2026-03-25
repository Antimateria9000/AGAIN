from __future__ import annotations

import pytest
import torch

from again_econ.config import ExecutionConfig, WalkforwardConfig
from again_econ.contracts import CapitalCompetitionPolicy, PositionTarget, ScheduledSignal, SignalRecord, WindowResult, WalkforwardWindow
from again_econ.execution import run_window_execution
from again_econ.gpu.execution import run_window_execution_tensor, schedule_window_signals_tensor
from again_econ.gpu.metrics import summarize_global_oos_metrics_tensor, summarize_metrics_tensor
from again_econ.metrics import summarize_global_oos_metrics, summarize_metrics
from again_econ.parity import compare_execution_results, compare_metric_bundles, compare_scheduling_outputs
from again_econ.runner import _schedule_window_signals_cpu
from again_econ.walkforward import build_walkforward_windows
from tests.helpers.again_econ import build_multi_symbol_market, build_single_symbol_market


def test_tensor_scheduler_matches_cpu_for_temporal_invariants():
    market = build_single_symbol_market([10, 10, 10, 11, 12, 13, 14, 15], [10, 10, 10, 11, 12, 13, 14, 15])
    window = build_walkforward_windows(market, WalkforwardConfig(train_size=2, test_size=3, step_size=3))[0]
    timestamps = market.timestamps()
    signals = (
        SignalRecord(
            instrument_id="AAA",
            decision_timestamp=timestamps[2],
            available_at=timestamps[3],
            target_state=PositionTarget.LONG,
        ),
        SignalRecord(
            instrument_id="AAA",
            decision_timestamp=timestamps[4],
            available_at=timestamps[4],
            target_state=PositionTarget.LONG,
        ),
    )

    cpu_scheduled, cpu_discarded = _schedule_window_signals_cpu(signals, market, window)
    tensor_scheduled, tensor_discarded, trace = schedule_window_signals_tensor(
        signals,
        market,
        window,
        device=torch.device("cpu"),
    )

    assert trace["scheduler_backend"] == "tensor"
    assert compare_scheduling_outputs(cpu_scheduled, tensor_scheduled, cpu_discarded, tensor_discarded) == ()


def test_tensor_execution_matches_cpu_for_fills_trades_and_metrics():
    market = build_multi_symbol_market(
        {
            "AAA": [50, 60],
            "BBB": [50, 60],
        },
        symbol_to_closes={
            "AAA": [50, 60],
            "BBB": [50, 66],
        },
    )
    timestamps = market.timestamps()
    scheduled_signals = (
        ScheduledSignal(
            signal=SignalRecord(
                instrument_id="AAA",
                decision_timestamp=timestamps[0],
                available_at=timestamps[0],
                target_state=PositionTarget.LONG,
                score=0.10,
            ),
            execution_timestamp=timestamps[1],
        ),
        ScheduledSignal(
            signal=SignalRecord(
                instrument_id="BBB",
                decision_timestamp=timestamps[0],
                available_at=timestamps[0],
                target_state=PositionTarget.LONG,
                score=0.90,
            ),
            execution_timestamp=timestamps[1],
        ),
    )
    config = ExecutionConfig(
        initial_cash=100.0,
        allocation_fraction=1.0,
        allow_fractional_shares=False,
        capital_competition_policy=CapitalCompetitionPolicy.SCORE_DESC,
    )

    cpu_result = run_window_execution(market, scheduled_signals, config, window_index=0)
    tensor_result = run_window_execution_tensor(
        market,
        scheduled_signals,
        config,
        window_index=0,
        device=torch.device("cpu"),
    )

    assert compare_execution_results(cpu_result, tensor_result) == ()

    cpu_metrics = summarize_metrics(
        cpu_result.snapshots,
        initial_equity=config.initial_cash,
        trade_count=len(cpu_result.trades),
        winning_trades=sum(1 for trade in cpu_result.trades if trade.net_pnl > 0.0),
        bars_per_year=config.bars_per_year,
    )
    tensor_metrics = summarize_metrics_tensor(
        tensor_result.snapshots,
        initial_equity=config.initial_cash,
        trade_count=len(tensor_result.trades),
        winning_trades=sum(1 for trade in tensor_result.trades if trade.net_pnl > 0.0),
        bars_per_year=config.bars_per_year,
        device=torch.device("cpu"),
    )
    assert compare_metric_bundles(cpu_metrics, tensor_metrics) == ()


def test_tensor_global_oos_metrics_match_cpu_chain_linking():
    market_one = build_single_symbol_market([10, 11, 12], [10, 12, 13])
    market_two = build_single_symbol_market([20, 21, 22], [20, 23, 24], start=market_one.timestamps()[-1].replace(day=10))
    timestamps_one = market_one.timestamps()
    timestamps_two = market_two.timestamps()
    config = ExecutionConfig(initial_cash=100.0, allocation_fraction=1.0, allow_fractional_shares=False)

    execution_one = run_window_execution(
        market_one,
        (
            ScheduledSignal(
                signal=SignalRecord(
                    instrument_id="AAA",
                    decision_timestamp=timestamps_one[0],
                    available_at=timestamps_one[0],
                    target_state=PositionTarget.LONG,
                ),
                execution_timestamp=timestamps_one[1],
            ),
        ),
        config,
        window_index=0,
    )
    execution_two = run_window_execution(
        market_two,
        (
            ScheduledSignal(
                signal=SignalRecord(
                    instrument_id="AAA",
                    decision_timestamp=timestamps_two[0],
                    available_at=timestamps_two[0],
                    target_state=PositionTarget.LONG,
                ),
                execution_timestamp=timestamps_two[1],
            ),
        ),
        config,
        window_index=1,
    )

    window_one = build_walkforward_windows(market_one, WalkforwardConfig(train_size=1, test_size=2, step_size=2))[0]
    window_two_base = build_walkforward_windows(market_two, WalkforwardConfig(train_size=1, test_size=2, step_size=2))[0]
    window_two = WalkforwardWindow(
        index=1,
        train_start=window_two_base.train_start,
        train_end=window_two_base.train_end,
        test_start=window_two_base.test_start,
        test_end=window_two_base.test_end,
        lookahead_bars=window_two_base.lookahead_bars,
        execution_lag_bars=window_two_base.execution_lag_bars,
        close_policy=window_two_base.close_policy,
    )
    result_one = WindowResult(
        window=window_one,
        fills=execution_one.fills,
        trades=execution_one.trades,
        snapshots=execution_one.snapshots,
        discarded_signals=(),
        metrics=summarize_metrics(
            execution_one.snapshots,
            initial_equity=config.initial_cash,
            trade_count=len(execution_one.trades),
            winning_trades=sum(1 for trade in execution_one.trades if trade.net_pnl > 0.0),
            bars_per_year=config.bars_per_year,
        ),
    )
    result_two = WindowResult(
        window=window_two,
        fills=execution_two.fills,
        trades=execution_two.trades,
        snapshots=execution_two.snapshots,
        discarded_signals=(),
        metrics=summarize_metrics(
            execution_two.snapshots,
            initial_equity=config.initial_cash,
            trade_count=len(execution_two.trades),
            winning_trades=sum(1 for trade in execution_two.trades if trade.net_pnl > 0.0),
            bars_per_year=config.bars_per_year,
        ),
    )

    cpu_curve, cpu_metrics = summarize_global_oos_metrics(
        (result_one, result_two),
        initial_equity=config.initial_cash,
        bars_per_year=config.bars_per_year,
    )
    tensor_curve, tensor_metrics = summarize_global_oos_metrics_tensor(
        (result_one, result_two),
        initial_equity=config.initial_cash,
        bars_per_year=config.bars_per_year,
        device=torch.device("cpu"),
    )

    assert len(cpu_curve) == len(tensor_curve)
    assert compare_metric_bundles(cpu_metrics, tensor_metrics) == ()
    for cpu_point, tensor_point in zip(cpu_curve, tensor_curve, strict=True):
        assert cpu_point.timestamp == tensor_point.timestamp
        assert cpu_point.window_index == tensor_point.window_index
        assert tensor_point.equity == pytest.approx(cpu_point.equity)
