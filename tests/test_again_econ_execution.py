from again_econ.config import ExecutionConfig
from again_econ.contracts import OrderSide, PositionTarget, ScheduledSignal, SignalRecord
from again_econ.execution import apply_slippage, calculate_fee, run_window_execution, schedule_signal_next_open

from tests.again_econ_test_utils import build_single_symbol_market


def test_schedule_signal_uses_next_open_after_close_decision():
    market = build_single_symbol_market([10, 11, 12])
    decision_timestamp = market.timestamps()[0]
    signal = SignalRecord(
        instrument_id="AAA",
        decision_timestamp=decision_timestamp,
        available_at=decision_timestamp,
        target_state=PositionTarget.LONG,
    )

    scheduled = schedule_signal_next_open(signal, market)

    assert scheduled.execution_timestamp == market.timestamps()[1]


def test_costs_and_slippage_are_applied_deterministically():
    config = ExecutionConfig(slippage_bps=100.0, commission_rate=0.01, commission_per_order=2.0)

    assert apply_slippage(100.0, OrderSide.BUY, config) == 101.0
    assert apply_slippage(100.0, OrderSide.SELL, config) == 100.0 / 1.01
    assert calculate_fee(1_000.0, config) == 12.0


def test_execution_engine_builds_trade_records_and_snapshots():
    market = build_single_symbol_market([10, 11, 13, 12], [10, 12, 13, 12])
    timestamps = market.timestamps()
    scheduled_signals = (
        ScheduledSignal(
            signal=SignalRecord(
                instrument_id="AAA",
                decision_timestamp=timestamps[0],
                available_at=timestamps[0],
                target_state=PositionTarget.LONG,
            ),
            execution_timestamp=timestamps[1],
        ),
        ScheduledSignal(
            signal=SignalRecord(
                instrument_id="AAA",
                decision_timestamp=timestamps[2],
                available_at=timestamps[2],
                target_state=PositionTarget.FLAT,
            ),
            execution_timestamp=timestamps[3],
        ),
    )

    result = run_window_execution(
        market,
        scheduled_signals,
        ExecutionConfig(initial_cash=100.0, allocation_fraction=1.0, allow_fractional_shares=False),
    )

    assert len(result.fills) == 2
    assert len(result.trades) == 1
    assert len(result.snapshots) == 4
    assert result.trades[0].quantity == 9.0
    assert result.trades[0].net_pnl == 9.0
    assert result.snapshots[-1].total_equity == 109.0
