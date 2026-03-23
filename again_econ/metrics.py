from __future__ import annotations

import math
from statistics import mean, pstdev

from again_econ.contracts import MetricBundle, PortfolioSnapshot, WindowResult


def summarize_metrics(
    snapshots: tuple[PortfolioSnapshot, ...],
    initial_equity: float,
    trade_count: int,
    winning_trades: int,
    bars_per_year: int,
) -> MetricBundle:
    if not snapshots:
        return MetricBundle(
            total_return=0.0,
            annualized_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            trade_count=trade_count,
            win_rate=0.0,
            exposure_ratio=0.0,
        )

    equity_curve = [initial_equity, *[snapshot.total_equity for snapshot in snapshots]]
    final_equity = equity_curve[-1]
    total_return = (final_equity / initial_equity) - 1.0 if initial_equity > 0.0 else 0.0
    periods = max(len(equity_curve) - 1, 1)
    annualized_return = ((final_equity / initial_equity) ** (bars_per_year / periods) - 1.0) if initial_equity > 0.0 else 0.0

    running_peak = equity_curve[0]
    drawdowns = []
    for equity in equity_curve:
        running_peak = max(running_peak, equity)
        drawdowns.append((equity / running_peak) - 1.0 if running_peak > 0.0 else 0.0)
    max_drawdown = min(drawdowns)

    returns = []
    for previous, current in zip(equity_curve, equity_curve[1:]):
        if previous > 0.0:
            returns.append((current / previous) - 1.0)
    sharpe_ratio = 0.0
    if returns:
        volatility = pstdev(returns)
        if volatility > 0.0:
            sharpe_ratio = (mean(returns) / volatility) * math.sqrt(bars_per_year)

    exposures = []
    for snapshot in snapshots:
        if snapshot.total_equity > 0.0:
            exposures.append(snapshot.market_value / snapshot.total_equity)
    exposure_ratio = mean(exposures) if exposures else 0.0
    win_rate = (winning_trades / trade_count) if trade_count else 0.0

    return MetricBundle(
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        trade_count=trade_count,
        win_rate=win_rate,
        exposure_ratio=exposure_ratio,
    )


def aggregate_window_metrics(window_results: tuple[WindowResult, ...]) -> MetricBundle:
    if not window_results:
        return MetricBundle(
            total_return=0.0,
            annualized_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            trade_count=0,
            win_rate=0.0,
            exposure_ratio=0.0,
        )
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
