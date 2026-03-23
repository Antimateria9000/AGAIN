from datetime import datetime

import pytest

from again_econ.contracts import ExecutionReason, MetricBundle, PortfolioSnapshot, TradeRecord, WalkforwardWindow, WindowResult
from again_econ.metrics import compute_window_average_metrics, summarize_global_oos_metrics, summarize_metrics


def test_metric_summary_reports_core_economic_values():
    snapshots = (
        PortfolioSnapshot(timestamp=datetime(2024, 1, 1), cash=100.0, market_value=0.0, total_equity=100.0, realized_pnl=0.0, unrealized_pnl=0.0, open_positions=0, window_index=0),
        PortfolioSnapshot(timestamp=datetime(2024, 1, 2), cash=50.0, market_value=60.0, total_equity=110.0, realized_pnl=0.0, unrealized_pnl=10.0, open_positions=1, window_index=0),
        PortfolioSnapshot(timestamp=datetime(2024, 1, 3), cash=105.0, market_value=0.0, total_equity=105.0, realized_pnl=5.0, unrealized_pnl=0.0, open_positions=0, window_index=0),
    )

    metrics = summarize_metrics(
        snapshots=snapshots,
        initial_equity=100.0,
        trade_count=1,
        winning_trades=1,
        bars_per_year=252,
    )

    assert isinstance(metrics, MetricBundle)
    assert metrics.total_return == pytest.approx(0.05)
    assert metrics.trade_count == 1
    assert metrics.win_rate == pytest.approx(1.0)
    assert metrics.max_drawdown < 0.0
    assert metrics.exposure_ratio > 0.0


def test_global_oos_summary_uses_chain_linked_curve_instead_of_window_means():
    initial_equity = 100.0
    window_0 = WalkforwardWindow(
        index=0,
        train_start=datetime(2024, 1, 1),
        train_end=datetime(2024, 1, 2),
        test_start=datetime(2024, 1, 3),
        test_end=datetime(2024, 1, 4),
    )
    window_1 = WalkforwardWindow(
        index=1,
        train_start=datetime(2024, 1, 3),
        train_end=datetime(2024, 1, 4),
        test_start=datetime(2024, 1, 5),
        test_end=datetime(2024, 1, 6),
    )
    snapshots_0 = (
        PortfolioSnapshot(datetime(2024, 1, 3), 100.0, 10.0, 110.0, 0.0, 10.0, 1, 0),
        PortfolioSnapshot(datetime(2024, 1, 4), 105.0, 0.0, 105.0, 5.0, 0.0, 0, 0),
    )
    snapshots_1 = (
        PortfolioSnapshot(datetime(2024, 1, 5), 90.0, 0.0, 90.0, -10.0, 0.0, 0, 1),
        PortfolioSnapshot(datetime(2024, 1, 6), 120.0, 0.0, 120.0, 20.0, 0.0, 0, 1),
    )
    trade = TradeRecord(
        instrument_id="AAA",
        entry_timestamp=datetime(2024, 1, 3),
        exit_timestamp=datetime(2024, 1, 4),
        entry_price=10.0,
        exit_price=10.5,
        quantity=1.0,
        entry_fee=0.0,
        exit_fee=0.0,
        gross_pnl=0.5,
        net_pnl=0.5,
        exit_reason=ExecutionReason.SIGNAL,
    )
    result_0 = WindowResult(
        window=window_0,
        fills=(),
        trades=(trade,),
        snapshots=snapshots_0,
        discarded_signals=(),
        metrics=summarize_metrics(snapshots_0, initial_equity, trade_count=1, winning_trades=1, bars_per_year=252),
    )
    result_1 = WindowResult(
        window=window_1,
        fills=(),
        trades=(trade,),
        snapshots=snapshots_1,
        discarded_signals=(),
        metrics=summarize_metrics(snapshots_1, initial_equity, trade_count=1, winning_trades=1, bars_per_year=252),
    )

    oos_curve, summary_metrics = summarize_global_oos_metrics((result_0, result_1), initial_equity=initial_equity, bars_per_year=252)
    window_average_metrics = compute_window_average_metrics((result_0, result_1))

    assert oos_curve[-1].equity == pytest.approx(126.0)
    assert summary_metrics.total_return == pytest.approx(0.26)
    assert window_average_metrics.total_return == pytest.approx(0.125)
    assert summary_metrics.total_return != pytest.approx(window_average_metrics.total_return)
