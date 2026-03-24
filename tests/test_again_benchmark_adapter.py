import pandas as pd

from again_benchmark.adapters.again_inference import AgainInferenceAdapter


class _DummyConfigManager:
    def __init__(self):
        self.config = {
            "prediction": {"years": 3},
            "_meta": {"config_path": "config/config.yaml"},
            "model_name": "dummy_model",
            "paths": {
                "models_dir": "models",
                "normalizers_dir": "models/normalizers",
            },
            "data": {"processed_data_path": "data/train/processed_dataset.pt"},
        }


def test_again_inference_adapter_keeps_sessions_isolated_per_ticker(monkeypatch):
    calls = []

    def fake_load_data_and_model(config, ticker, historical_mode, years, raw_data):
        calls.append(ticker)
        return None, f"dataset-{ticker}", f"normalizers-{ticker}", f"model-{ticker}"

    monkeypatch.setattr(
        AgainInferenceAdapter,
        "_load_prediction_api",
        staticmethod(lambda: (fake_load_data_and_model, None, None)),
    )
    adapter = AgainInferenceAdapter(_DummyConfigManager())
    observed = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "Close": [100.0]})

    first = adapter._ensure_session("AAA", observed)
    second = adapter._ensure_session("BBB", observed)

    assert first != second
    assert calls == ["AAA", "BBB"]
