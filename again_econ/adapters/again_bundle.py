from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from again_econ.contracts import (
    ArtifactReference,
    BundleProvenance,
    BundleProvenanceMode,
    ForecastRecord,
    InputBundle,
    InputSourceKind,
    PositionTarget,
    ProviderDataKind,
    ProviderIdentity,
    SignalRecord,
    TargetKind,
    WindowProvenance,
)
from again_econ.errors import AdapterError


class AgainBundleAdapter:
    """Stable JSON boundary between AGAIN outputs and the economic module."""

    adapter_name = "again_json_bundle"

    def load(self, bundle_path: str | Path) -> InputBundle:
        path = Path(bundle_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = int(payload.get("bundle_version", 1))
        payload_type = str(payload.get("payload_type", "")).strip()
        records = payload.get("records") or []
        provenance = self._parse_bundle_provenance(payload.get("provenance")) if version == 2 else None
        artifact_references = (
            ArtifactReference(
                artifact_type="input_bundle",
                locator=str(path),
            ),
        )

        if payload_type == "forecast_records":
            forecasts = tuple(self._parse_forecast_record(record, bundle_provenance=provenance) for record in records)
            return InputBundle(
                adapter_name=self.adapter_name,
                bundle_version=version,
                provenance_mode=self._resolve_provenance_mode(version),
                forecasts=forecasts,
                provenance=provenance,
                source_path=str(path),
                source_kind=InputSourceKind.ADAPTED_BUNDLE,
                provider_identity=ProviderIdentity(
                    name=self.adapter_name,
                    version=f"bundle_v{version}",
                    source_kind=InputSourceKind.ADAPTED_BUNDLE,
                    data_kind=ProviderDataKind.FORECAST,
                ),
                artifact_references=artifact_references,
            )
        if payload_type == "signal_records":
            signals = tuple(self._parse_signal_record(record, bundle_provenance=provenance) for record in records)
            return InputBundle(
                adapter_name=self.adapter_name,
                bundle_version=version,
                provenance_mode=self._resolve_provenance_mode(version),
                signals=signals,
                provenance=provenance,
                source_path=str(path),
                source_kind=InputSourceKind.ADAPTED_BUNDLE,
                provider_identity=ProviderIdentity(
                    name=self.adapter_name,
                    version=f"bundle_v{version}",
                    source_kind=InputSourceKind.ADAPTED_BUNDLE,
                    data_kind=ProviderDataKind.SIGNAL,
                ),
                artifact_references=artifact_references,
            )
        raise AdapterError(f"payload_type no soportado: {payload_type}")

    @staticmethod
    def _resolve_provenance_mode(version: int) -> BundleProvenanceMode:
        if version == 1:
            return BundleProvenanceMode.LEGACY_DEGRADED
        if version == 2:
            return BundleProvenanceMode.STRICT_V2
        raise AdapterError(f"bundle_version no soportada: {version}")

    @staticmethod
    def _parse_window_provenance(payload: dict | None) -> WindowProvenance | None:
        if payload is None:
            return None
        try:
            return WindowProvenance(
                window_index=int(payload["window_index"]),
                train_start=datetime.fromisoformat(payload["train_start"]) if payload.get("train_start") is not None else None,
                train_end=datetime.fromisoformat(payload["train_end"]),
                test_start=datetime.fromisoformat(payload["test_start"]),
                test_end=datetime.fromisoformat(payload["test_end"]),
                lookahead_bars=int(payload.get("lookahead_bars", 1)),
                execution_lag_bars=int(payload.get("execution_lag_bars", 1)),
            )
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"Metadata de ventana invalida: {exc}") from exc

    @classmethod
    def _parse_bundle_provenance(cls, payload: dict | None) -> BundleProvenance | None:
        if payload is None:
            return None
        try:
            return BundleProvenance(
                generated_at=datetime.fromisoformat(payload["generated_at"]),
                model_run_id=str(payload["model_run_id"]),
                data_fingerprint=str(payload["data_fingerprint"]),
                code_fingerprint=str(payload["code_fingerprint"]),
                window=cls._parse_window_provenance(payload.get("window")),
            )
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"Provenance de bundle invalida: {exc}") from exc

    @classmethod
    def _resolve_record_provenance(
        cls,
        payload: dict,
        *,
        bundle_provenance: BundleProvenance | None,
    ) -> WindowProvenance | None:
        record_provenance = cls._parse_window_provenance(payload.get("provenance"))
        if record_provenance is not None:
            return record_provenance
        if bundle_provenance is not None and bundle_provenance.window is not None:
            return bundle_provenance.window
        return None

    @classmethod
    def _parse_forecast_record(
        cls,
        payload: dict,
        *,
        bundle_provenance: BundleProvenance | None,
    ) -> ForecastRecord:
        try:
            return ForecastRecord(
                instrument_id=str(payload["instrument_id"]),
                decision_timestamp=datetime.fromisoformat(payload["decision_timestamp"]),
                available_at=datetime.fromisoformat(payload["available_at"]),
                target_kind=TargetKind(str(payload["target_kind"])),
                value=float(payload["value"]),
                reference_value=float(payload["reference_value"]) if payload.get("reference_value") is not None else None,
                score=float(payload["score"]) if payload.get("score") is not None else None,
                provenance=cls._resolve_record_provenance(payload, bundle_provenance=bundle_provenance),
                metadata=dict(payload.get("metadata") or {}),
                observed_at=datetime.fromisoformat(payload["observed_at"]) if payload.get("observed_at") is not None else None,
            )
        except Exception as exc:  # noqa: BLE001 - the adapter must annotate malformed bundles
            raise AdapterError(f"Forecast bundle invalido: {exc}") from exc

    @classmethod
    def _parse_signal_record(
        cls,
        payload: dict,
        *,
        bundle_provenance: BundleProvenance | None,
    ) -> SignalRecord:
        try:
            return SignalRecord(
                instrument_id=str(payload["instrument_id"]),
                decision_timestamp=datetime.fromisoformat(payload["decision_timestamp"]),
                available_at=datetime.fromisoformat(payload["available_at"]),
                target_state=PositionTarget(str(payload["target_state"])),
                score=float(payload["score"]) if payload.get("score") is not None else None,
                provenance=cls._resolve_record_provenance(payload, bundle_provenance=bundle_provenance),
                metadata=dict(payload.get("metadata") or {}),
                observed_at=datetime.fromisoformat(payload["observed_at"]) if payload.get("observed_at") is not None else None,
            )
        except Exception as exc:  # noqa: BLE001 - the adapter must annotate malformed bundles
            raise AdapterError(f"Signal bundle invalido: {exc}") from exc
