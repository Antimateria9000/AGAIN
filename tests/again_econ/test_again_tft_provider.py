from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from again_econ.adapters.again_tft_provider import AgainTFTForecastProvider, AgainTFTPredictionAPI
from again_econ.contracts import ArtifactReference, ProviderDataKind, TargetKind, WalkforwardWindow
from tests.helpers.again_econ import build_single_symbol_market


def test_again_tft_provider_builds_window_payload_with_explicit_temporal_semantics():
    market_data = pd.DataFrame(
        [
            {"Date": "2024-01-01", "Open": 10, "High": 11, "Low": 9, "Close": 10, "Volume": 1000, "Ticker": "AAA", "Sector": "Unknown"},
            {"Date": "2024-01-02", "Open": 10, "High": 11, "Low": 9, "Close": 10, "Volume": 1000, "Ticker": "AAA", "Sector": "Unknown"},
            {"Date": "2024-01-03", "Open": 11, "High": 12, "Low": 10, "Close": 11, "Volume": 1000, "Ticker": "AAA", "Sector": "Unknown"},
            {"Date": "2024-01-04", "Open": 12, "High": 13, "Low": 11, "Close": 12, "Volume": 1000, "Ticker": "AAA", "Sector": "Unknown"},
        ]
    )
    prediction_api = AgainTFTPredictionAPI(
        load_data_and_model=lambda config, ticker, raw_data=None: (raw_data, "dataset", {"Close": object()}, "model"),
        preprocess_data=lambda config, raw_data, ticker, normalizers: (raw_data.copy(), None),
        generate_predictions=lambda config, dataset, model, processed, **kwargs: (
            None,
            None,
            None,
            {"forecast_close_median": [float(kwargs["raw_ticker_data"]["Close"].iloc[-1]) * 1.05]},
        ),
    )
    provider = AgainTFTForecastProvider(
        again_config={
            "model_name": "TestModel",
            "paths": {"models_dir": "models", "normalizers_dir": "models/normalizers"},
            "data": {"processed_data_path": "data/train/processed_dataset.pt"},
            "_meta": {"config_path": "config/config.yaml"},
        },
        market_data=market_data,
        provider_mode="exploratory_live",
        methodology_label="exploratory_global_model_replay",
        prediction_api=prediction_api,
        artifact_references=(ArtifactReference(artifact_type="test", locator="memory://artifact"),),
    )
    market = build_single_symbol_market([10, 10, 11, 12])
    timestamps = market.timestamps()
    window = WalkforwardWindow(
        index=0,
        train_start=timestamps[0],
        train_end=timestamps[1],
        test_start=timestamps[2],
        test_end=timestamps[3],
        lookahead_bars=1,
        execution_lag_bars=1,
    )

    payload = provider.get_window_payload(window, market)

    assert payload.payload_kind == ProviderDataKind.FORECAST
    assert payload.provider.name == "again_tft_exploratory_live"
    assert payload.metadata["provider_mode"] == "exploratory_live"
    assert payload.metadata["methodology_label"] == "exploratory_global_model_replay"
    assert payload.metadata["skipped_count"] == 0
    assert len(payload.forecasts) == 2
    first = payload.forecasts[0]
    assert first.target_kind == TargetKind.PRICE
    assert first.available_at == first.decision_timestamp
    assert first.observed_at == first.decision_timestamp
    assert first.provenance is not None
    assert first.provenance.window_index == 0
    assert first.reference_value == 11.0
    assert first.value == pytest.approx(11.55)


def test_again_tft_provider_caches_runtime_per_ticker_instead_of_globally():
    market_data = pd.DataFrame(
        [
            {"Date": "2024-01-01", "Open": 10, "High": 11, "Low": 9, "Close": 10, "Volume": 1000, "Ticker": "AAA", "Sector": "Unknown"},
            {"Date": "2024-01-02", "Open": 10, "High": 11, "Low": 9, "Close": 10, "Volume": 1000, "Ticker": "AAA", "Sector": "Unknown"},
            {"Date": "2024-01-03", "Open": 11, "High": 12, "Low": 10, "Close": 11, "Volume": 1000, "Ticker": "AAA", "Sector": "Unknown"},
            {"Date": "2024-01-04", "Open": 12, "High": 13, "Low": 11, "Close": 12, "Volume": 1000, "Ticker": "AAA", "Sector": "Unknown"},
            {"Date": "2024-01-01", "Open": 20, "High": 21, "Low": 19, "Close": 20, "Volume": 1000, "Ticker": "BBB", "Sector": "Unknown"},
            {"Date": "2024-01-02", "Open": 20, "High": 21, "Low": 19, "Close": 20, "Volume": 1000, "Ticker": "BBB", "Sector": "Unknown"},
            {"Date": "2024-01-03", "Open": 21, "High": 22, "Low": 20, "Close": 21, "Volume": 1000, "Ticker": "BBB", "Sector": "Unknown"},
            {"Date": "2024-01-04", "Open": 22, "High": 23, "Low": 21, "Close": 22, "Volume": 1000, "Ticker": "BBB", "Sector": "Unknown"},
        ]
    )
    load_calls: list[str] = []

    def load_data_and_model(config, ticker, raw_data=None):
        load_calls.append(ticker)
        return raw_data, f"dataset-{ticker}", {"Close": object()}, f"model-{ticker}"

    prediction_api = AgainTFTPredictionAPI(
        load_data_and_model=load_data_and_model,
        preprocess_data=lambda config, raw_data, ticker, normalizers: (raw_data.copy(), None),
        generate_predictions=lambda config, dataset, model, processed, **kwargs: (
            None,
            None,
            None,
            {"forecast_close_median": [float(kwargs["raw_ticker_data"]["Close"].iloc[-1]) * 1.01]},
        ),
    )
    provider = AgainTFTForecastProvider(
        again_config={
            "model_name": "TestModel",
            "paths": {"models_dir": "models", "normalizers_dir": "models/normalizers"},
            "data": {"processed_data_path": "data/train/processed_dataset.pt"},
            "_meta": {"config_path": "config/config.yaml"},
        },
        market_data=market_data,
        prediction_api=prediction_api,
        artifact_references=(ArtifactReference(artifact_type="test", locator="memory://artifact"),),
    )
    market = build_single_symbol_market([10, 10, 11, 12])
    timestamps = market.timestamps()
    window = WalkforwardWindow(
        index=0,
        train_start=timestamps[0],
        train_end=timestamps[1],
        test_start=timestamps[2],
        test_end=timestamps[3],
        lookahead_bars=1,
        execution_lag_bars=1,
    )

    provider.get_window_payload(window, market)

    assert load_calls.count("AAA") == 1
    assert load_calls.count("BBB") == 1
