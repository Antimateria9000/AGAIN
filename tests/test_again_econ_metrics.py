import pytest

from again_econ.contracts import MetricBundle, PortfolioSnapshot
from again_econ.metrics import summarize_metrics


def test_metric_summary_reports_core_economic_values():
    snapshots = (
        PortfolioSnapshot(timestamp=None, cash=100.0, market_value=0.0, total_equity=100.0, realized_pnl=0.0, unrealized_pnl=0.0, open_positions=0),
        PortfolioSnapshot(timestamp=None, cash=50.0, market_value=60.0, total_equity=110.0, realized_pnl=0.0, unrealized_pnl=10.0, open_positions=1),
        PortfolioSnapshot(timestamp=None, cash=105.0, market_value=0.0, total_equity=105.0, realized_pnl=5.0, unrealized_pnl=0.0, open_positions=0),
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
