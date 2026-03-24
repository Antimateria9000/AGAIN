from __future__ import annotations

from abc import ABC, abstractmethod

from again_econ.contracts import (
    ArtifactReference,
    ForecastRecord,
    InputBundle,
    InputSourceKind,
    MarketFrame,
    ProviderDataKind,
    ProviderIdentity,
    ProviderWindowPayload,
    SignalRecord,
    WalkforwardWindow,
)
from again_econ.fingerprints import fingerprint_payload


def _record_belongs_to_window(record: ForecastRecord | SignalRecord, window: WalkforwardWindow) -> bool:
    if record.provenance is not None:
        return record.provenance.window_index == window.index
    return window.test_start <= record.decision_timestamp <= window.test_end


class ForecastProvider(ABC):
    @property
    @abstractmethod
    def identity(self) -> ProviderIdentity:
        raise NotImplementedError

    @abstractmethod
    def get_window_payload(self, window: WalkforwardWindow, market_frame: MarketFrame) -> ProviderWindowPayload:
        raise NotImplementedError


class SignalProvider(ABC):
    @property
    @abstractmethod
    def identity(self) -> ProviderIdentity:
        raise NotImplementedError

    @abstractmethod
    def get_window_payload(self, window: WalkforwardWindow, market_frame: MarketFrame) -> ProviderWindowPayload:
        raise NotImplementedError


class StaticForecastProvider(ForecastProvider):
    def __init__(
        self,
        forecasts: tuple[ForecastRecord, ...],
        *,
        identity: ProviderIdentity,
        input_reference: str | None = None,
        artifact_references: tuple[ArtifactReference, ...] = (),
    ) -> None:
        self._forecasts = tuple(forecasts)
        self._identity = identity
        self._input_reference = input_reference
        self._artifact_references = tuple(artifact_references)

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def get_window_payload(self, window: WalkforwardWindow, market_frame: MarketFrame) -> ProviderWindowPayload:
        del market_frame
        records = tuple(record for record in self._forecasts if _record_belongs_to_window(record, window))
        return ProviderWindowPayload(
            window_index=window.index,
            provider=self.identity,
            payload_kind=ProviderDataKind.FORECAST,
            forecasts=records,
            input_reference=self._input_reference,
            payload_fingerprint=fingerprint_payload(
                {
                    "provider": self.identity,
                    "window_index": window.index,
                    "records": records,
                }
            ),
            artifact_references=self._artifact_references,
        )


class StaticSignalProvider(SignalProvider):
    def __init__(
        self,
        signals: tuple[SignalRecord, ...],
        *,
        identity: ProviderIdentity,
        input_reference: str | None = None,
        artifact_references: tuple[ArtifactReference, ...] = (),
    ) -> None:
        self._signals = tuple(signals)
        self._identity = identity
        self._input_reference = input_reference
        self._artifact_references = tuple(artifact_references)

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def get_window_payload(self, window: WalkforwardWindow, market_frame: MarketFrame) -> ProviderWindowPayload:
        del market_frame
        records = tuple(record for record in self._signals if _record_belongs_to_window(record, window))
        return ProviderWindowPayload(
            window_index=window.index,
            provider=self.identity,
            payload_kind=ProviderDataKind.SIGNAL,
            signals=records,
            input_reference=self._input_reference,
            payload_fingerprint=fingerprint_payload(
                {
                    "provider": self.identity,
                    "window_index": window.index,
                    "records": records,
                }
            ),
            artifact_references=self._artifact_references,
        )


class BundleForecastProvider(StaticForecastProvider):
    def __init__(self, bundle: InputBundle) -> None:
        identity = bundle.provider_identity or ProviderIdentity(
            name=bundle.adapter_name,
            version=f"bundle_v{bundle.bundle_version}",
            source_kind=InputSourceKind.ADAPTED_BUNDLE,
            data_kind=ProviderDataKind.FORECAST,
        )
        artifact_references = bundle.artifact_references
        if bundle.source_path is not None and not artifact_references:
            artifact_references = (
                ArtifactReference(
                    artifact_type="input_bundle",
                    locator=bundle.source_path,
                ),
            )
        super().__init__(
            bundle.forecasts,
            identity=identity,
            input_reference=bundle.source_path,
            artifact_references=artifact_references,
        )
        self._bundle = bundle

    def get_window_payload(self, window: WalkforwardWindow, market_frame: MarketFrame) -> ProviderWindowPayload:
        payload = super().get_window_payload(window, market_frame)
        return ProviderWindowPayload(
            window_index=payload.window_index,
            provider=payload.provider,
            payload_kind=payload.payload_kind,
            forecasts=payload.forecasts,
            input_reference=payload.input_reference,
            payload_fingerprint=payload.payload_fingerprint,
            bundle_version=self._bundle.bundle_version,
            provenance_mode=self._bundle.provenance_mode,
            artifact_references=payload.artifact_references,
            metadata=self._bundle.metadata,
        )


class BundleSignalProvider(StaticSignalProvider):
    def __init__(self, bundle: InputBundle) -> None:
        identity = bundle.provider_identity or ProviderIdentity(
            name=bundle.adapter_name,
            version=f"bundle_v{bundle.bundle_version}",
            source_kind=InputSourceKind.ADAPTED_BUNDLE,
            data_kind=ProviderDataKind.SIGNAL,
        )
        artifact_references = bundle.artifact_references
        if bundle.source_path is not None and not artifact_references:
            artifact_references = (
                ArtifactReference(
                    artifact_type="input_bundle",
                    locator=bundle.source_path,
                ),
            )
        super().__init__(
            bundle.signals,
            identity=identity,
            input_reference=bundle.source_path,
            artifact_references=artifact_references,
        )
        self._bundle = bundle

    def get_window_payload(self, window: WalkforwardWindow, market_frame: MarketFrame) -> ProviderWindowPayload:
        payload = super().get_window_payload(window, market_frame)
        return ProviderWindowPayload(
            window_index=payload.window_index,
            provider=payload.provider,
            payload_kind=payload.payload_kind,
            signals=payload.signals,
            input_reference=payload.input_reference,
            payload_fingerprint=payload.payload_fingerprint,
            bundle_version=self._bundle.bundle_version,
            provenance_mode=self._bundle.provenance_mode,
            artifact_references=payload.artifact_references,
            metadata=self._bundle.metadata,
        )


def provider_from_bundle(bundle: InputBundle) -> ForecastProvider | SignalProvider:
    if bundle.forecasts:
        return BundleForecastProvider(bundle)
    return BundleSignalProvider(bundle)
