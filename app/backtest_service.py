from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.runtime_config import ConfigManager
from scripts.utils.device_utils import resolve_execution_context
from scripts.utils.model_readiness import assess_model_readiness


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class BacktestService:
    PRESET_FILE_BY_NAME = {
        "default": "default.yaml",
        "exploratory": "exploratory.yaml",
        "strict_frozen": "strict_frozen.yaml",
    }

    def __init__(self, config_manager: ConfigManager, years: int):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.years = years

        from app.backtest_market_builder import MarketFrameBuilder
        from again_econ import BacktestConfig, run_backtest_with_provider
        from again_econ.adapters.again_tft_provider import AgainTFTForecastProvider
        from again_econ.storage import BacktestStorage
        from again_econ.ui_adapter import BacktestUIAdapter

        self._market_builder_cls = MarketFrameBuilder
        self._backtest_config_cls = BacktestConfig
        self._provider_cls = AgainTFTForecastProvider
        self._run_backtest_with_provider = run_backtest_with_provider

        self.presets_dir = Path(self.config["paths"]["config_dir"]) / "backtests"
        default_payload = self._load_yaml(self.presets_dir / self.PRESET_FILE_BY_NAME["default"])
        storage_root_value = str((default_payload.get("storage") or {}).get("root") or "data/backtests_econ")
        storage_root = Path(storage_root_value)
        self.storage = BacktestStorage(storage_root)
        self.ui = BacktestUIAdapter(self.storage)

    def get_model_readiness(self):
        return assess_model_readiness(self.config)

    def get_runtime_status(self):
        return resolve_execution_context(self.config, purpose="predict").to_display_dict()

    def list_presets(self) -> list[dict[str, Any]]:
        presets = []
        for preset_name in ("exploratory", "strict_frozen"):
            payload = self._load_preset_payload(preset_name)
            mode_section = payload.get("mode") or {}
            market_section = payload.get("market") or {}
            presets.append(
                {
                    "preset_name": preset_name,
                    "mode": str(mode_section.get("label") or preset_name),
                    "methodology_label": str(mode_section.get("methodology_label") or preset_name),
                    "methodology_note": str(mode_section.get("methodology_note") or ""),
                    "allow_local_fallback": bool(market_section.get("allow_local_fallback", False)),
                }
            )
        return presets

    def get_preset_defaults(self, mode: str) -> dict[str, Any]:
        return deepcopy(self._load_preset_payload(self._resolve_preset_name(mode)))

    def run_backtest(
        self,
        *,
        mode: str,
        tickers: list[str],
        start_date: datetime,
        end_date: datetime,
        walkforward_overrides: dict[str, Any] | None = None,
        execution_overrides: dict[str, Any] | None = None,
        signal_overrides: dict[str, Any] | None = None,
        policy_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        readiness = self.get_model_readiness()
        if not readiness.ready:
            details = " | ".join(readiness.issues)
            raise RuntimeError(f"{readiness.summary} Detalles: {details}")

        preset_name = self._resolve_preset_name(mode)
        payload = self._load_preset_payload(preset_name)
        payload = self._apply_overrides(
            payload,
            walkforward_overrides=walkforward_overrides,
            execution_overrides=execution_overrides,
            signal_overrides=signal_overrides,
            policy_overrides=policy_overrides,
        )
        self._validate_against_active_model(payload)
        backtest_config = self._backtest_config_cls.from_mapping(payload)

        market_section = payload.get("market") or {}
        builder = self._market_builder_cls.from_config_manager(self.config_manager, self.years)
        build_result = builder.build(
            tickers,
            start_date,
            end_date,
            allow_local_fallback=bool(market_section.get("allow_local_fallback", False)),
        )

        mode_section = payload.get("mode") or {}
        provider = self._provider_cls(
            again_config=self.config,
            market_data=build_result.frame,
            provider_version=str((payload.get("provider") or {}).get("version", "v1")),
            provider_mode=str(mode_section.get("label") or preset_name),
            methodology_label=str(mode_section.get("methodology_label") or "global_model_replay"),
            forecast_mode=str((payload.get("provider_runtime") or {}).get("forecast_mode") or "recursive_one_step"),
            input_reference=f"{self.config.get('_meta', {}).get('config_path', '')}::{preset_name}",
        )
        result = self._run_backtest_with_provider(
            build_result.market_frame,
            backtest_config,
            provider,
        )

        market_context = {
            "requested_tickers": list(build_result.requested_tickers),
            "effective_tickers": list(build_result.effective_tickers),
            "discarded_tickers": list(build_result.discarded_tickers),
            "source_summary": dict(build_result.source_summary),
            "provenance_by_ticker": build_result.provenance_by_ticker,
            "integrity_report": build_result.integrity_report.to_dict(),
            "warnings": list(build_result.warnings),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "methodology_note": str(mode_section.get("methodology_note") or ""),
        }
        return self.ui.persist_result(
            result,
            mode=str(mode_section.get("label") or preset_name),
            preset_name=preset_name,
            methodology_label=str(mode_section.get("methodology_label") or preset_name),
            model_name=str(self.config["model_name"]),
            config_reference=str(self.config.get("_meta", {}).get("config_path") or ""),
            requested_universe=build_result.requested_tickers,
            effective_universe=build_result.effective_tickers,
            market_data=build_result.frame,
            market_context=market_context,
            warnings=build_result.warnings,
        )

    def list_runs(
        self,
        *,
        mode: str | None = None,
        preset_name: str | None = None,
        model_name: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.ui.list_runs(
            mode=mode,
            preset_name=preset_name,
            model_name=model_name,
            run_id=run_id,
        )

    def load_run_view(self, run_id: str) -> dict[str, Any]:
        return self.ui.load_run_view(run_id)

    def compare_runs(self, left_run_id: str, right_run_id: str) -> dict[str, Any]:
        return self.ui.compare_runs(left_run_id, right_run_id)

    def _apply_overrides(
        self,
        payload: dict[str, Any],
        *,
        walkforward_overrides: dict[str, Any] | None,
        execution_overrides: dict[str, Any] | None,
        signal_overrides: dict[str, Any] | None,
        policy_overrides: dict[str, Any] | None,
    ) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        if walkforward_overrides:
            overrides["walkforward"] = dict(walkforward_overrides)
        if execution_overrides:
            overrides["execution"] = dict(execution_overrides)
        if signal_overrides:
            overrides["signal"] = dict(signal_overrides)
        if policy_overrides:
            execution_section = dict(overrides.get("execution") or {})
            execution_section.update(policy_overrides)
            overrides["execution"] = execution_section
        return _deep_merge(payload, overrides)

    def _validate_against_active_model(self, payload: dict[str, Any]) -> None:
        lookahead_bars = int((payload.get("walkforward") or {}).get("lookahead_bars", 1))
        max_prediction_length = int(self.config["model"]["max_prediction_length"])
        if lookahead_bars > max_prediction_length:
            raise ValueError(
                f"lookahead_bars={lookahead_bars} excede el max_prediction_length del modelo activo ({max_prediction_length})"
            )

    def _load_preset_payload(self, preset_name: str) -> dict[str, Any]:
        payload = self._load_yaml(self.presets_dir / self.PRESET_FILE_BY_NAME["default"])
        if preset_name != "default":
            payload = _deep_merge(payload, self._load_yaml(self.presets_dir / self.PRESET_FILE_BY_NAME[preset_name]))
        return payload

    def _resolve_preset_name(self, mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized in {"exploratory", "exploratory_live"}:
            return "exploratory"
        if normalized in {"official_frozen", "strict_frozen", "frozen"}:
            return "strict_frozen"
        raise ValueError(f"Modo de backtesting no soportado: {mode}")

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
