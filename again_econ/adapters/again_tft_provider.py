from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from again_econ.contracts import (
    ArtifactReference,
    ForecastRecord,
    InputSourceKind,
    MarketFrame,
    ProviderDataKind,
    ProviderIdentity,
    ProviderWindowPayload,
    TargetKind,
    WalkforwardWindow,
)
from again_econ.fingerprints import fingerprint_payload
from again_econ.providers import ForecastProvider


def _read_sha256_sidecar(path: Path) -> str | None:
    checksum_path = Path(f"{path}.sha256")
    if not checksum_path.exists():
        return None
    checksum = checksum_path.read_text(encoding="utf-8").strip()
    return checksum or None


@dataclass(frozen=True)
class AgainTFTPredictionAPI:
    load_data_and_model: Callable[..., tuple[Any, Any, dict, Any]]
    preprocess_data: Callable[..., tuple[pd.DataFrame, Any]]
    generate_predictions: Callable[..., Any]


def load_default_prediction_api() -> AgainTFTPredictionAPI:
    from scripts.prediction_engine import generate_predictions, load_data_and_model, preprocess_data

    return AgainTFTPredictionAPI(
        load_data_and_model=load_data_and_model,
        preprocess_data=preprocess_data,
        generate_predictions=generate_predictions,
    )


class AgainTFTForecastProvider(ForecastProvider):
    def __init__(
        self,
        *,
        again_config: dict,
        market_data: pd.DataFrame,
        provider_version: str = "v1",
        provider_mode: str = "exploratory_live",
        methodology_label: str = "global_model_replay",
        forecast_mode: str = "recursive_one_step",
        input_reference: str | None = None,
        prediction_api: AgainTFTPredictionAPI | None = None,
        artifact_references: tuple[ArtifactReference, ...] = (),
    ) -> None:
        self._again_config = again_config
        self._market_data = self._normalize_market_data(market_data)
        self._provider_mode = str(provider_mode)
        self._methodology_label = str(methodology_label)
        self._forecast_mode = str(forecast_mode)
        self._input_reference = input_reference
        self._prediction_api = prediction_api
        self._runtime_cache: tuple[Any, dict, Any] | None = None
        self._artifact_references = artifact_references or self._build_default_artifact_references()
        self._identity = ProviderIdentity(
            name=f"again_tft_{self._provider_mode}",
            version=str(provider_version),
            source_kind=InputSourceKind.WINDOW_PROVIDER_FORECASTS,
            data_kind=ProviderDataKind.FORECAST,
        )
        self._history_by_ticker = {
            ticker: ticker_frame.copy()
            for ticker, ticker_frame in self._market_data.groupby("Ticker", sort=True)
        }

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def get_window_payload(self, window: WalkforwardWindow, market_frame: MarketFrame) -> ProviderWindowPayload:
        del market_frame
        forecasts: list[ForecastRecord] = []
        skipped: list[dict[str, str]] = []
        for ticker, ticker_history in sorted(self._history_by_ticker.items()):
            decision_rows = ticker_history[
                (ticker_history["Date"] >= pd.Timestamp(window.test_start))
                & (ticker_history["Date"] <= pd.Timestamp(window.test_end))
            ]
            for decision_ts in decision_rows["Date"].tolist():
                try:
                    record = self._build_forecast_record(window, ticker, pd.Timestamp(decision_ts))
                except Exception as exc:
                    skipped.append(
                        {
                            "instrument_id": ticker,
                            "decision_timestamp": pd.Timestamp(decision_ts).isoformat(),
                            "error": str(exc),
                        }
                    )
                    continue
                if record is not None:
                    forecasts.append(record)

        ordered = tuple(sorted(forecasts, key=lambda item: (item.decision_timestamp, item.instrument_id)))
        payload_metadata = {
            "provider_mode": self._provider_mode,
            "methodology_label": self._methodology_label,
            "forecast_mode": self._forecast_mode,
            "skipped_count": len(skipped),
            "skipped_examples": skipped[:10],
        }
        return ProviderWindowPayload(
            window_index=window.index,
            provider=self.identity,
            payload_kind=ProviderDataKind.FORECAST,
            forecasts=ordered,
            input_reference=self._input_reference,
            payload_fingerprint=fingerprint_payload(
                {
                    "provider": self.identity,
                    "window": window,
                    "records": ordered,
                }
            ),
            artifact_references=self._artifact_references,
            metadata=payload_metadata,
        )

    def _build_forecast_record(
        self,
        window: WalkforwardWindow,
        ticker: str,
        decision_ts: pd.Timestamp,
    ) -> ForecastRecord | None:
        raw_history = self._history_by_ticker[ticker]
        raw_history = raw_history[raw_history["Date"] <= decision_ts].copy()
        if raw_history.empty:
            return None

        dataset, normalizers, model = self._ensure_runtime(ticker, raw_history)
        processed, _ = self._prediction_api_or_default().preprocess_data(
            self._again_config,
            raw_history.copy(),
            ticker,
            normalizers,
        )
        _, _, _, details = self._prediction_api_or_default().generate_predictions(
            self._again_config,
            dataset,
            model,
            processed,
            return_details=True,
            raw_ticker_data=raw_history,
            forecast_mode=self._forecast_mode,
        )
        lookahead_index = window.lookahead_bars - 1
        forecast_close = details.get("forecast_close_median")
        if forecast_close is None or len(forecast_close) <= lookahead_index:
            raise ValueError(
                f"El forecast no cubre lookahead_bars={window.lookahead_bars} para {ticker} en {decision_ts.date()}"
            )

        reference_value = float(raw_history["Close"].iloc[-1])
        predicted_value = float(forecast_close[lookahead_index])
        predicted_return = (predicted_value - reference_value) / reference_value
        decision_timestamp = decision_ts.to_pydatetime()
        return ForecastRecord(
            instrument_id=ticker,
            decision_timestamp=decision_timestamp,
            available_at=decision_timestamp,
            observed_at=decision_timestamp,
            target_kind=TargetKind.PRICE,
            value=predicted_value,
            reference_value=reference_value,
            score=predicted_return,
            provenance=window.to_provenance(),
            metadata={
                "provider_mode": self._provider_mode,
                "methodology_label": self._methodology_label,
                "forecast_mode": self._forecast_mode,
                "forecast_horizon": int(len(forecast_close)),
                "lookahead_bars": int(window.lookahead_bars),
                "model_name": str(self._again_config.get("model_name", "")),
                "predicted_return": predicted_return,
            },
        )

    def _ensure_runtime(self, ticker: str, raw_history: pd.DataFrame) -> tuple[Any, dict, Any]:
        if self._runtime_cache is None:
            _, dataset, normalizers, model = self._prediction_api_or_default().load_data_and_model(
                self._again_config,
                ticker,
                raw_data=raw_history.copy(),
            )
            self._runtime_cache = (dataset, normalizers, model)
        return self._runtime_cache

    def _prediction_api_or_default(self) -> AgainTFTPredictionAPI:
        if self._prediction_api is None:
            self._prediction_api = load_default_prediction_api()
        return self._prediction_api

    @staticmethod
    def _normalize_market_data(market_data: pd.DataFrame) -> pd.DataFrame:
        required = {"Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Sector"}
        missing = sorted(required.difference(market_data.columns))
        if missing:
            raise ValueError(f"El provider AGAIN TFT requiere columnas de mercado completas: {missing}")
        normalized = market_data.copy()
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce").dt.tz_localize(None)
        normalized["Ticker"] = normalized["Ticker"].astype(str).str.strip().str.upper()
        normalized = normalized.dropna(subset=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]).copy()
        return normalized.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    def _build_default_artifact_references(self) -> tuple[ArtifactReference, ...]:
        references: list[ArtifactReference] = []
        model_name = str(self._again_config.get("model_name", ""))
        models_dir = Path(self._again_config["paths"]["models_dir"])
        normalizers_dir = Path(self._again_config["paths"]["normalizers_dir"])
        dataset_path = Path(self._again_config["data"]["processed_data_path"])
        config_path = self._again_config.get("_meta", {}).get("config_path")

        artifacts = [
            ("model_checkpoint", models_dir / f"{model_name}.pth"),
            ("normalizers", normalizers_dir / f"{model_name}_normalizers.pkl"),
            ("processed_dataset", dataset_path),
        ]
        for artifact_type, path in artifacts:
            if Path(path).exists():
                references.append(
                    ArtifactReference(
                        artifact_type=artifact_type,
                        locator=str(path),
                        fingerprint=_read_sha256_sidecar(Path(path)),
                    )
                )
        if config_path:
            references.append(
                ArtifactReference(
                    artifact_type="again_config",
                    locator=str(config_path),
                )
            )
        return tuple(references)
