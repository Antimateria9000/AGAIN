from __future__ import annotations

import logging
import shutil
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareStatus:
    python_version: str
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    device_count: int
    gpu_names: tuple[str, ...]
    nvidia_smi_available: bool
    nvidia_smi_path: str | None
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionContext:
    purpose: str
    requested_accelerator: str
    requested_precision: str
    accelerator: str
    precision: str
    device_type: str
    device_string: str
    pin_memory: bool
    non_blocking: bool
    hardware: HardwareStatus
    fallback_reason: str | None

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device_string)

    @property
    def uses_cuda(self) -> bool:
        return self.accelerator == "gpu"

    @property
    def primary_gpu_name(self) -> str:
        return self.hardware.gpu_names[0] if self.hardware.gpu_names else "No CUDA"

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "backend": "GPU" if self.uses_cuda else "CPU",
            "device_string": self.device_string,
            "cuda_available": self.hardware.cuda_available,
            "device_count": self.hardware.device_count,
            "gpu_name": self.primary_gpu_name,
            "torch_version": self.hardware.torch_version,
            "torch_cuda_version": self.hardware.torch_cuda_version or "None",
            "python_version": self.hardware.python_version,
            "precision": self.precision,
            "fallback_reason": self.fallback_reason,
            "nvidia_smi_available": self.hardware.nvidia_smi_available,
        }


def _detect_nvidia_smi() -> tuple[bool, str | None]:
    nvidia_smi_path = shutil.which("nvidia-smi")
    return nvidia_smi_path is not None, nvidia_smi_path


def _query_gpu_names_from_torch(device_count: int) -> tuple[str, ...]:
    names: list[str] = []
    for index in range(device_count):
        try:
            names.append(str(torch.cuda.get_device_name(index)))
        except Exception:
            names.append(f"GPU {index}")
    return tuple(names)


def _build_fallback_reason(
    torch_cuda_version: str | None,
    cuda_available: bool,
    device_count: int,
    nvidia_smi_available: bool,
) -> str | None:
    if cuda_available and device_count > 0:
        return None
    if torch_cuda_version is None:
        if nvidia_smi_available:
            return (
                "La instalacion actual de torch es CPU-only. "
                "El sistema si tiene GPU NVIDIA, pero este entorno virtual no incluye soporte CUDA."
            )
        return "La instalacion actual de torch es CPU-only y no expone soporte CUDA."
    if device_count <= 0:
        return "Torch incluye soporte CUDA, pero no detecta ninguna GPU utilizable."
    return "Torch incluye soporte CUDA, pero cuda.is_available() es False en este entorno."


@lru_cache(maxsize=1)
def detect_hardware_status() -> HardwareStatus:
    nvidia_smi_available, nvidia_smi_path = _detect_nvidia_smi()
    torch_cuda_version = torch.version.cuda
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else int(torch.cuda.device_count())
    gpu_names = _query_gpu_names_from_torch(device_count) if device_count > 0 else tuple()
    fallback_reason = _build_fallback_reason(torch_cuda_version, cuda_available, device_count, nvidia_smi_available)
    return HardwareStatus(
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        torch_cuda_version=torch_cuda_version,
        cuda_available=cuda_available,
        device_count=device_count,
        gpu_names=gpu_names,
        nvidia_smi_available=nvidia_smi_available,
        nvidia_smi_path=nvidia_smi_path,
        fallback_reason=fallback_reason,
    )


def clear_hardware_status_cache():
    detect_hardware_status.cache_clear()


def _read_requested_runtime(config: dict, purpose: str) -> tuple[str, str]:
    training_config = config.get("training", {})
    if purpose == "train":
        section = training_config
    elif purpose == "backtest":
        section = config.get("backtesting_runtime", {})
    else:
        section = config.get("prediction", {})
    requested_accelerator = str(section.get("accelerator", training_config.get("accelerator", "auto"))).lower()
    requested_precision = str(section.get("precision", training_config.get("precision", "auto"))).lower()
    return requested_accelerator, requested_precision


def _cuda_bf16_supported() -> bool:
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def _resolve_precision(requested_precision: str, accelerator: str) -> str:
    if requested_precision == "auto":
        if accelerator != "gpu":
            return "32-true"
        return "bf16-mixed" if _cuda_bf16_supported() else "32-true"
    if accelerator != "gpu" and requested_precision in {"16-mixed", "16-true", "bf16-mixed"}:
        return "32-true"
    return requested_precision


def resolve_execution_context(config: dict, purpose: str) -> ExecutionContext:
    hardware = detect_hardware_status()
    requested_accelerator, requested_precision = _read_requested_runtime(config, purpose)

    if requested_accelerator in {"gpu", "cuda"}:
        accelerator = "gpu" if hardware.cuda_available else "cpu"
        fallback_reason = None if accelerator == "gpu" else (
            "Se solicito GPU, pero CUDA no esta disponible. " + (hardware.fallback_reason or "")
        ).strip()
    elif requested_accelerator == "auto":
        accelerator = "gpu" if hardware.cuda_available else "cpu"
        fallback_reason = hardware.fallback_reason if accelerator == "cpu" else None
    else:
        accelerator = "cpu"
        fallback_reason = "La configuracion fuerza el uso de CPU." if requested_accelerator == "cpu" else hardware.fallback_reason

    precision = _resolve_precision(requested_precision, accelerator)
    device_type = "cuda" if accelerator == "gpu" else "cpu"
    device_string = "cuda:0" if accelerator == "gpu" else "cpu"

    return ExecutionContext(
        purpose=purpose,
        requested_accelerator=requested_accelerator,
        requested_precision=requested_precision,
        accelerator=accelerator,
        precision=precision,
        device_type=device_type,
        device_string=device_string,
        pin_memory=accelerator == "gpu",
        non_blocking=accelerator == "gpu",
        hardware=hardware,
        fallback_reason=fallback_reason,
    )


def build_runtime_log_payload(
    runtime: ExecutionContext,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
) -> dict[str, Any]:
    payload = {
        "python_version": runtime.hardware.python_version,
        "torch_version": runtime.hardware.torch_version,
        "torch_cuda_version": runtime.hardware.torch_cuda_version,
        "cuda_available": runtime.hardware.cuda_available,
        "device_count": runtime.hardware.device_count,
        "gpu_name": runtime.primary_gpu_name,
        "accelerator": runtime.accelerator,
        "device": runtime.device_string,
        "precision": runtime.precision,
    }
    if batch_size is not None:
        payload["batch_size"] = int(batch_size)
    if num_workers is not None:
        payload["num_workers"] = int(num_workers)
    if prefetch_factor is not None:
        payload["prefetch_factor"] = int(prefetch_factor)
    if runtime.fallback_reason:
        payload["fallback_reason"] = runtime.fallback_reason
    return payload


def log_runtime_context(
    logger_instance: logging.Logger,
    context_name: str,
    runtime: ExecutionContext,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
):
    logger_instance.info(
        "%s | runtime=%s",
        context_name,
        build_runtime_log_payload(
            runtime,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        ),
    )


def get_inference_autocast_context(runtime: ExecutionContext):
    if not runtime.uses_cuda:
        return nullcontext()
    if runtime.precision == "bf16-mixed":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if runtime.precision == "16-mixed":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()
