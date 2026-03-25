from again_econ.config import WalkforwardConfig
from again_econ.walkforward import build_walkforward_windows

from tests.helpers.again_econ import build_single_symbol_market


def test_walkforward_splitter_creates_non_overlapping_test_windows():
    market = build_single_symbol_market([10, 11, 12, 13, 14, 15, 16, 17])

    windows = build_walkforward_windows(
        market,
        WalkforwardConfig(train_size=4, test_size=2, step_size=2, lookahead_bars=2, execution_lag_bars=1),
    )

    assert len(windows) == 2
    assert windows[0].train_start == market.timestamps()[0]
    assert windows[0].train_end == market.timestamps()[3]
    assert windows[0].test_start == market.timestamps()[4]
    assert windows[0].test_end == market.timestamps()[5]
    assert windows[1].train_start == market.timestamps()[2]
    assert windows[1].test_start == market.timestamps()[6]
    assert windows[0].test_end < windows[1].test_start
    assert windows[0].lookahead_bars == 2
    assert windows[0].execution_lag_bars == 1
