from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from app.backtest_service import BacktestService
from app.backtest_market_builder import MarketDataQualityReport
from scripts.runtime_config import ConfigManager
from scripts.utils.universe_integrity import UniverseIntegrityReport, UniverseTickerIntegrity
from tests.helpers.again_econ import build_single_symbol_market
from tests.helpers.again_econ_ui import build_result, market_to_frame


class FakeStorage:
    def __init__(self, root):
        self.root = root


class FakeUIAdapter:
    def __init__(self, storage):
        self.storage = storage
        self.persist_calls = []

    def persist_result(self, result, **kwargs):
        self.persist_calls.append((result, kwargs))
        return {"manifest": {"run_id": result.manifest.run_id}, "summary": kwargs}


class FakeBuilder:
    last_call = None

    @classmethod
    def from_config_manager(cls, config_manager, years):
        del config_manager, years
        return cls()

    def build(self, tickers, start_date, end_date, *, allow_local_fallback):
        type(self).last_call = {
            "tickers": list(tickers),
            "start_date": start_date,
            "end_date": end_date,
            "allow_local_fallback": allow_local_fallback,
        }
        market = build_single_symbol_market([10, 10, 10, 11, 12], [10, 10, 10, 11, 12], start=datetime(2024, 1, 1))
        frame = market_to_frame(market)
        report = UniverseIntegrityReport(
            requested_tickers=["AAA"],
            successful_tickers=["AAA"],
            discarded_tickers=[],
            discarded_details={},
            ticker_integrity={
                "AAA": UniverseTickerIntegrity(
                    ticker="AAA",
                    final_status="ok",
                    source="fresh_network",
                    rows_obtained=5,
                    trainable=True,
                    freshness="fresh",
                )
            },
            fresh_tickers=["AAA"],
            fallback_tickers=[],
            decision="CONTINUE_CLEAN",
            training_allowed=True,
            degraded=False,
        )
        return SimpleNamespace(
            market_frame=market,
            frame=frame,
            requested_tickers=("AAA",),
            effective_tickers=("AAA",),
            discarded_tickers=(),
            integrity_report=report,
            source_summary={"fresh_network": 1},
            provenance_by_ticker={"AAA": {"source": "fresh_network"}},
            quality_report=MarketDataQualityReport(input_rows=5, output_rows=5),
            warnings=(),
        )


class FakeProvider:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


def test_backtest_service_resolves_frozen_mode_and_passes_clean_contracts(monkeypatch):
    import again_econ.storage as storage_module
    import again_econ.ui_adapter as ui_adapter_module

    monkeypatch.setattr(storage_module, "BacktestStorage", FakeStorage)
    monkeypatch.setattr(ui_adapter_module, "BacktestUIAdapter", FakeUIAdapter)
    service = BacktestService(ConfigManager("config/config.yaml"), years=3)
    service.get_model_readiness = lambda: SimpleNamespace(ready=True, summary="ok", issues=[])
    service._market_builder_cls = FakeBuilder
    service._provider_cls = FakeProvider
    sample_result, _ = build_result(label="service_test", exit_index=3)
    captured = {}

    def fake_run(market_frame, config, provider):
        captured["market_frame"] = market_frame
        captured["config"] = config
        captured["provider"] = provider
        return sample_result

    service._run_backtest_with_provider = fake_run

    response = service.run_backtest(
        mode="official_frozen",
        tickers=["AAA"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 2, 1),
        walkforward_overrides={"train_size": 120},
        policy_overrides={"capital_competition_policy": "instrument_asc"},
    )

    assert FakeBuilder.last_call["allow_local_fallback"] is False
    assert captured["config"].walkforward.train_size == 120
    assert captured["config"].execution.capital_competition_policy.value == "instrument_asc"
    assert FakeProvider.last_kwargs["provider_mode"] == "official_frozen"
    assert FakeProvider.last_kwargs["methodology_label"] == "frozen_global_model_replay"
    assert response["summary"]["preset_name"] == "strict_frozen"


def test_backtest_service_exposes_mode_defaults(monkeypatch):
    import again_econ.storage as storage_module
    import again_econ.ui_adapter as ui_adapter_module

    monkeypatch.setattr(storage_module, "BacktestStorage", FakeStorage)
    monkeypatch.setattr(ui_adapter_module, "BacktestUIAdapter", FakeUIAdapter)
    service = BacktestService(ConfigManager("config/config.yaml"), years=3)

    defaults = service.get_preset_defaults("exploratory_live")

    assert defaults["mode"]["label"] == "exploratory_live"
    assert defaults["market"]["allow_local_fallback"] is True
