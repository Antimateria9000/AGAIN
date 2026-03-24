import importlib
import sys


def test_app_services_import_is_lazy_for_prediction_and_benchmark_modules():
    sys.modules.pop("app.services", None)
    sys.modules.pop("scripts.prediction_engine", None)
    sys.modules.pop("again_benchmark.adapters.again_inference", None)

    importlib.import_module("app.services")

    assert "scripts.prediction_engine" not in sys.modules
    assert "again_benchmark.adapters.again_inference" not in sys.modules


def test_again_benchmark_adapters_package_does_not_eagerly_import_again_inference():
    sys.modules.pop("again_benchmark.adapters", None)
    sys.modules.pop("again_benchmark.adapters.again_inference", None)

    adapters_module = importlib.import_module("again_benchmark.adapters")

    assert "again_benchmark.adapters.again_inference" not in sys.modules
    assert hasattr(adapters_module, "load_legacy_benchmark_rows")


def test_backtest_service_import_is_lazy_for_prediction_engine():
    sys.modules.pop("app.backtest_service", None)
    sys.modules.pop("scripts.prediction_engine", None)

    importlib.import_module("app.backtest_service")

    assert "scripts.prediction_engine" not in sys.modules
