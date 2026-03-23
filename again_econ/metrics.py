from __future__ import annotations

import math
from statistics import mean, pstdev

from again_econ.contracts import GlobalOOSPoint, MetricBundle, PortfolioSnapshot, WindowResult


def _zero_metric_bundle(*, trade_count: int = 0) -> MetricBundle:
    return MetricBundle(
        total_return=0.0,
        annualized_return=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        trade_count=trade_count,
        win_rate=0.0,
        exposure_ratio=0.0,
    )


def _summarize_equity_curve(
    *,
    equity_curve: list[float],
    exposures: list[float],
    trade_count: int,
    winning_trades: int,
    bars_per_year: int,
) -> MetricBundle:
    if len(equity_curve) <= 1:
        return _zero_metric_bundle(trade_count=trade_count)

    initial_equity = equity_curve[0]
    final_equity = equity_curve[-1]
    total_return = (final_equity / initial_equity) - 1.0 if initial_equity > 0.0 else 0.0
    periods = max(len(equity_curve) - 1, 1)
    annualized_return = ((final_equity / initial_equity) ** (bars_per_year / periods) - 1.0) if initial_equity > 0.0 else 0.0

    running_peak = equity_curve[0]
    drawdowns = []
    returns = []
    for previous, current in zip(equity_curve, equity_curve[1:]):
        running_peak = max(running_peak, current)
        drawdowns.append((current / running_peak) - 1.0 if running_peak > 0.0 else 0.0)
        if previous > 0.0:
            returns.append((current / previous) - 1.0)

    sharpe_ratio = 0.0
    if returns:
        volatility = pstdev(returns)
        if volatility > 0.0:
            sharpe_ratio = (mean(returns) / volatility) * math.sqrt(bars_per_year)

    return MetricBundle(
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=min(drawdowns, default=0.0),
        sharpe_ratio=sharpe_ratio,
        trade_count=trade_count,
        win_rate=(winning_trades / trade_count) if trade_count else 0.0,
        exposure_ratio=mean(exposures) if exposures else 0.0,
    )


def summarize_metrics(
    snapshots: tuple[PortfolioSnapshot, ...],
    initial_equity: float,
    trade_count: int,
    winning_trades: int,
    bars_per_year: int,
) -> MetricBundle:
    if not snapshots:
        return _zero_metric_bundle(trade_count=trade_count)
    equity_curve = [initial_equity, *[snapshot.total_equity for snapshot in snapshots]]
    exposures = [
        snapshot.market_value / snapshot.total_equity
        for snapshot in snapshots
        if snapshot.total_equity > 0.0
    ]
    return _summarize_equity_curve(
        equity_curve=equity_curve,
        exposures=exposures,
        trade_count=trade_count,
        winning_trades=winning_trades,
        bars_per_year=bars_per_year,
    )


def build_global_oos_curve(window_results: tuple[WindowResult, ...], initial_equity: float) -> tuple[GlobalOOSPoint, ...]:
    """Chain-link independent windows into a single OOS equity curve.

    Each walk-forward window is economically independent and starts from the same
    `initial_equity`. To obtain a global OOS summary without averaging window-level
    metrics, we rebalance the equity path of every window onto the ending capital
    of the previous one. This preserves the within-window return path while
    avoiding artificial carry-over of positions or cash between windows.
    """

    chained_points: list[GlobalOOSPoint] = []
    base_equity = float(initial_equity)
    for result in sorted(window_results, key=lambda item: item.window.index):
        if not result.snapshots:
            continue
        for snapshot in result.snapshots:
            relative_equity = snapshot.total_equity / initial_equity if initial_equity > 0.0 else 1.0
            chained_points.append(
                GlobalOOSPoint(
                    timestamp=snapshot.timestamp,
                    window_index=result.window.index,
                    equity=base_equity * relative_equity,
                )
            )
        base_equity = chained_points[-1].equity
    return tuple(chained_points)


def summarize_global_oos_metrics(
    window_results: tuple[WindowResult, ...],
    *,
    initial_equity: float,
    bars_per_year: int,
) -> tuple[tuple[GlobalOOSPoint, ...], MetricBundle]:
    oos_curve = build_global_oos_curve(window_results, initial_equity)
    if not oos_curve:
        return oos_curve, _zero_metric_bundle()

    all_snapshots = [
        snapshot
        for result in sorted(window_results, key=lambda item: item.window.index)
        for snapshot in result.snapshots
    ]
    trade_count = sum(len(result.trades) for result in window_results)
    winning_trades = sum(1 for result in window_results for trade in result.trades if trade.net_pnl > 0.0)
    exposures = [
        snapshot.market_value / snapshot.total_equity
        for snapshot in all_snapshots
        if snapshot.total_equity > 0.0
    ]
    equity_curve = [initial_equity, *[point.equity for point in oos_curve]]
    metrics = _summarize_equity_curve(
        equity_curve=equity_curve,
        exposures=exposures,
        trade_count=trade_count,
        winning_trades=winning_trades,
        bars_per_year=bars_per_year,
    )
    return oos_curve, metrics


def compute_window_average_metrics(window_results: tuple[WindowResult, ...]) -> MetricBundle:
    if not window_results:
        return _zero_metric_bundle()
    metrics = [result.metrics for result in window_results]
    return MetricBundle(
        total_return=mean(metric.total_return for metric in metrics),
        annualized_return=mean(metric.annualized_return for metric in metrics),
        max_drawdown=mean(metric.max_drawdown for metric in metrics),
        sharpe_ratio=mean(metric.sharpe_ratio for metric in metrics),
        trade_count=sum(metric.trade_count for metric in metrics),
        win_rate=mean(metric.win_rate for metric in metrics),
        exposure_ratio=mean(metric.exposure_ratio for metric in metrics),
    )


def aggregate_window_metrics(window_results: tuple[WindowResult, ...]) -> MetricBundle:
    """Auxiliary inspection metric: simple average across window summaries.

    This is kept only as a secondary diagnostic. It is not the principal economic
    summary of the backtest and must not be interpreted as the global OOS result.
    """

    return compute_window_average_metrics(window_results)
