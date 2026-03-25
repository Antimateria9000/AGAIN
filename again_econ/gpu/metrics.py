from __future__ import annotations

import math

import torch

from again_econ.contracts import GlobalOOSPoint, MetricBundle, WindowResult


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


def _summarize_equity_curve_tensor(
    *,
    equity_curve: torch.Tensor,
    exposures: torch.Tensor,
    trade_count: int,
    winning_trades: int,
    bars_per_year: int,
) -> MetricBundle:
    if equity_curve.numel() <= 1:
        return _zero_metric_bundle(trade_count=trade_count)

    initial_equity = float(equity_curve[0].item())
    final_equity = float(equity_curve[-1].item())
    total_return = (final_equity / initial_equity) - 1.0 if initial_equity > 0.0 else 0.0
    periods = max(int(equity_curve.numel()) - 1, 1)
    annualized_return = ((final_equity / initial_equity) ** (bars_per_year / periods) - 1.0) if initial_equity > 0.0 else 0.0

    running_peak = torch.cummax(equity_curve[1:], dim=0).values
    drawdowns = torch.where(running_peak > 0.0, (equity_curve[1:] / running_peak) - 1.0, torch.zeros_like(running_peak))
    returns = torch.where(equity_curve[:-1] > 0.0, (equity_curve[1:] / equity_curve[:-1]) - 1.0, torch.zeros_like(equity_curve[1:]))
    sharpe_ratio = 0.0
    if returns.numel() > 0:
        volatility = float(torch.std(returns, unbiased=False).item())
        if volatility > 0.0:
            sharpe_ratio = float(torch.mean(returns).item()) / volatility * math.sqrt(bars_per_year)

    exposure_ratio = float(torch.mean(exposures).item()) if exposures.numel() > 0 else 0.0
    return MetricBundle(
        total_return=total_return,
        annualized_return=annualized_return,
        max_drawdown=float(torch.min(drawdowns).item()) if drawdowns.numel() > 0 else 0.0,
        sharpe_ratio=sharpe_ratio,
        trade_count=trade_count,
        win_rate=(winning_trades / trade_count) if trade_count else 0.0,
        exposure_ratio=exposure_ratio,
    )


def summarize_metrics_tensor(
    snapshots,
    *,
    initial_equity: float,
    trade_count: int,
    winning_trades: int,
    bars_per_year: int,
    device: torch.device,
) -> MetricBundle:
    if not snapshots:
        return _zero_metric_bundle(trade_count=trade_count)
    equity_curve = torch.tensor(
        [float(initial_equity), *[float(snapshot.total_equity) for snapshot in snapshots]],
        dtype=torch.float64,
        device=device,
    )
    exposures = torch.tensor(
        [
            float(snapshot.market_value) / float(snapshot.total_equity)
            for snapshot in snapshots
            if float(snapshot.total_equity) > 0.0
        ],
        dtype=torch.float64,
        device=device,
    )
    return _summarize_equity_curve_tensor(
        equity_curve=equity_curve,
        exposures=exposures,
        trade_count=trade_count,
        winning_trades=winning_trades,
        bars_per_year=bars_per_year,
    )


def summarize_global_oos_metrics_tensor(
    window_results: tuple[WindowResult, ...],
    *,
    initial_equity: float,
    bars_per_year: int,
    device: torch.device,
) -> tuple[tuple[GlobalOOSPoint, ...], MetricBundle]:
    if not window_results:
        return (), _zero_metric_bundle()

    oos_points: list[GlobalOOSPoint] = []
    base_equity = torch.tensor(float(initial_equity), dtype=torch.float64, device=device)
    for result in sorted(window_results, key=lambda item: item.window.index):
        if not result.snapshots:
            continue
        relative_equities = torch.tensor(
            [float(snapshot.total_equity) / float(initial_equity) for snapshot in result.snapshots],
            dtype=torch.float64,
            device=device,
        )
        chained_equities = base_equity * relative_equities
        chained_values = chained_equities.detach().cpu().tolist()
        for snapshot, chained_equity in zip(result.snapshots, chained_values, strict=True):
            oos_points.append(
                GlobalOOSPoint(
                    timestamp=snapshot.timestamp,
                    window_index=result.window.index,
                    equity=float(chained_equity),
                )
            )
        base_equity = chained_equities[-1]

    if not oos_points:
        return (), _zero_metric_bundle()

    trade_count = sum(len(result.trades) for result in window_results)
    winning_trades = sum(1 for result in window_results for trade in result.trades if trade.net_pnl > 0.0)
    exposures = torch.tensor(
        [
            float(snapshot.market_value) / float(snapshot.total_equity)
            for result in sorted(window_results, key=lambda item: item.window.index)
            for snapshot in result.snapshots
            if float(snapshot.total_equity) > 0.0
        ],
        dtype=torch.float64,
        device=device,
    )
    equity_curve = torch.tensor(
        [float(initial_equity), *[float(point.equity) for point in oos_points]],
        dtype=torch.float64,
        device=device,
    )
    metrics = _summarize_equity_curve_tensor(
        equity_curve=equity_curve,
        exposures=exposures,
        trade_count=trade_count,
        winning_trades=winning_trades,
        bars_per_year=bars_per_year,
    )
    return tuple(oos_points), metrics
