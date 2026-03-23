from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from again_econ.contracts import ForecastRecord, InputBundle, PositionTarget, SignalRecord, TargetKind
from again_econ.errors import AdapterError


class AgainBundleAdapter:
    """Stable JSON boundary between AGAIN outputs and the economic module."""

    adapter_name = "again_json_bundle"

    def load(self, bundle_path: str | Path) -> InputBundle:
        path = Path(bundle_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = int(payload.get("bundle_version", 0))
        if version != 1:
            raise AdapterError(f"bundle_version no soportada: {version}")
        payload_type = str(payload.get("payload_type", "")).strip()
        records = payload.get("records") or []
        if payload_type == "forecast_records":
            forecasts = tuple(self._parse_forecast_record(record) for record in records)
            return InputBundle(adapter_name=self.adapter_name, forecasts=forecasts, source_path=str(path))
        if payload_type == "signal_records":
            signals = tuple(self._parse_signal_record(record) for record in records)
            return InputBundle(adapter_name=self.adapter_name, signals=signals, source_path=str(path))
        raise AdapterError(f"payload_type no soportado: {payload_type}")

    @staticmethod
    def _parse_forecast_record(payload: dict) -> ForecastRecord:
        try:
            return ForecastRecord(
                instrument_id=str(payload["instrument_id"]),
                decision_timestamp=datetime.fromisoformat(payload["decision_timestamp"]),
                available_at=datetime.fromisoformat(payload["available_at"]),
                target_kind=TargetKind(str(payload["target_kind"])),
                value=float(payload["value"]),
                reference_value=float(payload["reference_value"]) if payload.get("reference_value") is not None else None,
                score=float(payload["score"]) if payload.get("score") is not None else None,
                metadata=dict(payload.get("metadata") or {}),
            )
        except Exception as exc:  # noqa: BLE001 - the adapter must annotate malformed bundles
            raise AdapterError(f"Forecast bundle invalido: {exc}") from exc

    @staticmethod
    def _parse_signal_record(payload: dict) -> SignalRecord:
        try:
            return SignalRecord(
                instrument_id=str(payload["instrument_id"]),
                decision_timestamp=datetime.fromisoformat(payload["decision_timestamp"]),
                available_at=datetime.fromisoformat(payload["available_at"]),
                target_state=PositionTarget(str(payload["target_state"])),
                score=float(payload["score"]) if payload.get("score") is not None else None,
                metadata=dict(payload.get("metadata") or {}),
            )
        except Exception as exc:  # noqa: BLE001 - the adapter must annotate malformed bundles
            raise AdapterError(f"Signal bundle invalido: {exc}") from exc
