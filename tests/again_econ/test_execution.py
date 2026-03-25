from again_econ.config import ExecutionConfig
from again_econ.contracts import CapitalCompetitionPolicy, ExecutionReason, OrderSide, PositionTarget, ScheduledSignal, SignalRecord
from again_econ.execution import apply_slippage, calculate_fee, run_window_execution, schedule_signal_next_open

from tests.helpers.again_econ import build_multi_symbol_market, build_single_symbol_market


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


def test_schedule_signal_respects_real_available_at_timestamp():
    market = build_single_symbol_market([10, 11, 12, 13])
    timestamps = market.timestamps()
    signal = SignalRecord(
        instrument_id="AAA",
        decision_timestamp=timestamps[0],
        available_at=timestamps[1],
        target_state=PositionTarget.LONG,
    )

    scheduled = schedule_signal_next_open(signal, market)

    assert scheduled.execution_timestamp == timestamps[2]


def test_costs_and_slippage_are_applied_deterministically():
    config = ExecutionConfig(slippage_bps=100.0, commission_rate=0.01, commission_per_order=2.0)

    assert apply_slippage(100.0, OrderSide.BUY, config) == 101.0
    assert apply_slippage(100.0, OrderSide.SELL, config) == 100.0 / 1.01
    assert calculate_fee(1_000.0, config) == 12.0


def test_execution_engine_forces_window_close_and_keeps_ledger_consistent():
    market = build_single_symbol_market([10, 11, 12], [10, 12, 13])
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
    )

    result = run_window_execution(
        market,
        scheduled_signals,
        ExecutionConfig(initial_cash=100.0, allocation_fraction=1.0, allow_fractional_shares=False),
        window_index=0,
    )

    assert len(result.fills) == 2
    assert len(result.trades) == 1
    assert len(result.snapshots) == 3
    assert result.trades[0].quantity == 9.0
    assert result.trades[0].net_pnl == 18.0
    assert result.trades[0].exit_reason == ExecutionReason.WINDOW_CLOSE
    assert result.fills[0].reason == ExecutionReason.SIGNAL
    assert result.fills[1].reason == ExecutionReason.WINDOW_CLOSE
    assert result.snapshots[-1].open_positions == 0
    assert result.snapshots[-1].total_equity == 118.0


def test_execution_engine_applies_explicit_instrument_order_capital_competition_policy():
    market = build_multi_symbol_market(
        {
            "AAA": [50, 60],
            "BBB": [50, 60],
        },
        symbol_to_closes={
            "AAA": [50, 65],
            "BBB": [50, 60],
        },
    )
    timestamps = market.timestamps()
    scheduled_signals = (
        ScheduledSignal(
            signal=SignalRecord(
                instrument_id="BBB",
                decision_timestamp=timestamps[0],
                available_at=timestamps[0],
                target_state=PositionTarget.LONG,
            ),
            execution_timestamp=timestamps[1],
        ),
        ScheduledSignal(
            signal=SignalRecord(
                instrument_id="AAA",
                decision_timestamp=timestamps[0],
                available_at=timestamps[0],
                target_state=PositionTarget.LONG,
            ),
            execution_timestamp=timestamps[1],
        ),
    )

    result = run_window_execution(
        market,
        scheduled_signals,
        ExecutionConfig(
            initial_cash=100.0,
            allocation_fraction=1.0,
            allow_fractional_shares=False,
            capital_competition_policy=CapitalCompetitionPolicy.INSTRUMENT_ASC,
        ),
        window_index=0,
    )

    assert [fill.instrument_id for fill in result.fills] == ["AAA", "AAA"]
    assert len(result.trades) == 1


def test_execution_engine_can_rank_competing_signals_by_score():
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

    result = run_window_execution(
        market,
        scheduled_signals,
        ExecutionConfig(
            initial_cash=100.0,
            allocation_fraction=1.0,
            allow_fractional_shares=False,
            capital_competition_policy=CapitalCompetitionPolicy.SCORE_DESC,
        ),
        window_index=0,
    )

    assert [fill.instrument_id for fill in result.fills] == ["BBB", "BBB"]
    assert len(result.trades) == 1
