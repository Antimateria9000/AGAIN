import inspect

import app.app as app_module


def test_app_exposes_backtesting_section_and_render_helper():
    source = inspect.getsource(app_module.main)

    assert "Backtesting" in source
    assert hasattr(app_module, "_render_backtesting_view")
    assert hasattr(app_module, "_get_backtest_service")
