from __future__ import annotations

import hashlib
import sys

from again_econ.config import BacktestConfig
from again_econ.contracts import PolicyIdentity, ProviderIdentity, RunManifest, WindowManifest
from again_econ.fingerprints import fingerprint_payload
from again_econ.signals import resolve_signal_policy_identity


def build_run_manifest(
    *,
    config: BacktestConfig,
    market_frame,
    windows,
    provider: ProviderIdentity,
    window_manifests: tuple[WindowManifest, ...],
    input_fingerprint: str,
    adapter_name: str | None = None,
    bundle_version: int | None = None,
    provenance_mode=None,
    input_reference: str | None = None,
    artifact_references=(),
) -> RunManifest:
    config_fingerprint = fingerprint_payload(config)
    market_fingerprint = fingerprint_payload(market_frame)
    window_plan_fingerprint = fingerprint_payload(windows)
    discarded_reason_counts: dict[str, int] = {}
    discarded_signal_count = 0
    for window_manifest in window_manifests:
        discarded_signal_count += window_manifest.discarded_signal_count
        for reason, count in window_manifest.discarded_reason_counts.items():
            discarded_reason_counts[reason] = discarded_reason_counts.get(reason, 0) + count
    run_id = hashlib.sha256(
        (
            f"{config_fingerprint}:{market_fingerprint}:{window_plan_fingerprint}:{input_fingerprint}:"
            f"{provider.name}:{provider.version}:{config.execution.capital_competition_policy.value}:"
            f"{config.execution.capital_competition_policy_version}:{config.signal.translation_policy_name}:"
            f"{config.signal.translation_policy_version}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    return RunManifest(
        run_id=run_id,
        label=config.label,
        provider=provider,
        signal_policy=resolve_signal_policy_identity(config.signal),
        scheduling_policy=PolicyIdentity(
            name=config.execution.scheduling_policy.value,
            version=config.execution.scheduling_policy_version,
        ),
        sizing_policy=PolicyIdentity(
            name=config.execution.sizing_policy.value,
            version=config.execution.sizing_policy_version,
        ),
        capital_competition_policy=PolicyIdentity(
            name=config.execution.capital_competition_policy.value,
            version=config.execution.capital_competition_policy_version,
        ),
        artifact_policy=PolicyIdentity(
            name=config.manifest.artifact_policy_name,
            version=config.manifest.artifact_policy_version,
        ),
        config_fingerprint=config_fingerprint,
        market_fingerprint=market_fingerprint,
        window_plan_fingerprint=window_plan_fingerprint,
        input_fingerprint=input_fingerprint,
        window_count=len(windows),
        adapter_name=adapter_name,
        bundle_version=bundle_version,
        provenance_mode=provenance_mode,
        input_reference=input_reference,
        seed=config.manifest.seed,
        command=config.manifest.command,
        python_version=sys.version.split()[0],
        code_commit_sha=config.manifest.code_commit_sha,
        discarded_signal_count=discarded_signal_count,
        discarded_reason_counts=discarded_reason_counts,
        artifact_references=tuple(artifact_references),
        windows=tuple(window_manifests),
    )
