from __future__ import annotations

from again_econ.config import WalkforwardConfig
from again_econ.contracts import MarketFrame, WalkforwardWindow
from again_econ.errors import BacktestConfigurationError
from again_econ.validation import validate_walkforward_windows


def materialize_walkforward_windows(market_frame: MarketFrame, config: WalkforwardConfig) -> tuple[WalkforwardWindow, ...]:
    timestamps = market_frame.timestamps()
    minimum_points = config.train_size + config.test_size
    if len(timestamps) < minimum_points:
        raise BacktestConfigurationError(
            f"No hay suficientes timestamps para walk-forward: {len(timestamps)} < {minimum_points}"
        )
    windows = []
    last_start = len(timestamps) - minimum_points
    for start_idx in range(0, last_start + 1, config.step_size):
        train_start = timestamps[start_idx]
        train_end = timestamps[start_idx + config.train_size - 1]
        test_start = timestamps[start_idx + config.train_size]
        test_end = timestamps[start_idx + config.train_size + config.test_size - 1]
        windows.append(
            WalkforwardWindow(
                index=len(windows),
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                lookahead_bars=config.lookahead_bars,
                execution_lag_bars=config.execution_lag_bars,
                close_policy=config.close_policy,
            )
        )
    validate_walkforward_windows(windows)
    return tuple(windows)


def build_walkforward_windows(market_frame: MarketFrame, config: WalkforwardConfig) -> tuple[WalkforwardWindow, ...]:
    return materialize_walkforward_windows(market_frame, config)


def slice_test_market(market_frame: MarketFrame, window: WalkforwardWindow) -> MarketFrame:
    return market_frame.slice_between(window.test_start, window.test_end)
