from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from again_econ.contracts import ArtifactReference
from scripts.utils.device_utils import ExecutionContext, build_runtime_log_payload, resolve_execution_context

FROZEN_BACKTEST_MODES = {"official_frozen", "strict_frozen", "frozen"}


def _normalize_mode(mode: str | None) -> str:
    return str(mode or "exploratory_live").strip().lower()


def _normalize_backend(value: str | None, *, default: str, allowed: set[str]) -> str:
    normalized = str(value or default).strip().lower()
    return normalized if normalized in allowed else default


@dataclass(frozen=True)
class BacktestRuntimeProfile:
    mode: str
    execution_context: ExecutionContext
    requested_execution_backend: str
    resolved_execution_backend: str
    requested_inference_backend: str
    resolved_inference_backend: str
    allow_cpu_fallback: bool
    parity_check_sample_windows: int
    emit_runtime_trace: bool
    fallback_reason: str | None = None

    @property
    def uses_gpu_execution(self) -> bool:
        return self.resolved_execution_backend == "gpu_full" and self.execution_context.uses_cuda

    @property
    def uses_batched_inference(self) -> bool:
        return self.resolved_inference_backend in {"gpu_batched", "cpu_batched"}

    @property
    def used_fallback(self) -> bool:
        return bool(self.fallback_reason)

    def to_trace_payload(self) -> dict[str, Any]:
        payload = build_runtime_log_payload(self.execution_context)
        payload.update(
            {
                "mode": self.mode,
                "requested_execution_backend": self.requested_execution_backend,
                "execution_backend": self.resolved_execution_backend,
                "requested_inference_backend": self.requested_inference_backend,
                "inference_backend": self.resolved_inference_backend,
                "allow_cpu_fallback": self.allow_cpu_fallback,
                "parity_check_sample_windows": self.parity_check_sample_windows,
                "emit_runtime_trace": self.emit_runtime_trace,
            }
        )
        if self.fallback_reason:
            payload["fallback_reason"] = self.fallback_reason
        return payload

    def to_artifact_reference(self) -> ArtifactReference:
        locator = f"runtime://backtest/{self.mode}/{self.resolved_execution_backend}"
        return ArtifactReference(
            artifact_type="backtest_runtime",
            locator=locator,
            detail=json.dumps(self.to_trace_payload(), sort_keys=True),
        )


def resolve_backtest_runtime_profile(
    config: dict,
    *,
    mode: str,
    explicit_profile: BacktestRuntimeProfile | None = None,
) -> BacktestRuntimeProfile:
    if explicit_profile is not None:
        return explicit_profile

    runtime_section = dict(config.get("backtesting_runtime") or {})
    runtime = resolve_execution_context(config, purpose="backtest")
    normalized_mode = _normalize_mode(mode)
    allow_cpu_fallback = bool(
        runtime_section.get(
            "allow_cpu_fallback_frozen" if normalized_mode in FROZEN_BACKTEST_MODES else "allow_cpu_fallback_live",
            normalized_mode not in FROZEN_BACKTEST_MODES,
        )
    )
    requested_execution_backend = _normalize_backend(
        runtime_section.get("execution_backend"),
        default="gpu_full",
        allowed={"cpu_reference", "gpu_full"},
    )
    requested_inference_backend = _normalize_backend(
        runtime_section.get("inference_backend"),
        default="gpu_batched",
        allowed={"legacy_per_timestamp", "gpu_batched"},
    )
    parity_check_sample_windows = max(int(runtime_section.get("parity_check_sample_windows", 0)), 0)
    emit_runtime_trace = bool(runtime_section.get("emit_runtime_trace", True))

    resolved_execution_backend = requested_execution_backend
    resolved_inference_backend = requested_inference_backend
    fallback_parts: list[str] = []
    gpu_required = requested_execution_backend == "gpu_full" or requested_inference_backend == "gpu_batched"

    if not runtime.uses_cuda:
        if requested_execution_backend == "gpu_full":
            resolved_execution_backend = "cpu_reference"
            fallback_parts.append("execution_backend=gpu_full->cpu_reference")
        if requested_inference_backend == "gpu_batched":
            resolved_inference_backend = "cpu_batched"
            fallback_parts.append("inference_backend=gpu_batched->cpu_batched")
        if runtime.fallback_reason:
            fallback_parts.append(runtime.fallback_reason)
        if gpu_required and not allow_cpu_fallback:
            reason = " | ".join(part for part in fallback_parts if part)
            raise RuntimeError(
                f"El modo de backtesting {mode} requiere backend GPU y CUDA no esta disponible. {reason}".strip()
            )

    fallback_reason = " | ".join(part for part in fallback_parts if part) or None
    return BacktestRuntimeProfile(
        mode=normalized_mode,
        execution_context=runtime,
        requested_execution_backend=requested_execution_backend,
        resolved_execution_backend=resolved_execution_backend,
        requested_inference_backend=requested_inference_backend,
        resolved_inference_backend=resolved_inference_backend,
        allow_cpu_fallback=allow_cpu_fallback,
        parity_check_sample_windows=parity_check_sample_windows,
        emit_runtime_trace=emit_runtime_trace,
        fallback_reason=fallback_reason,
    )
