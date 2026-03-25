from __future__ import annotations

from unittest import mock

import pytest

from again_econ.runtime import resolve_backtest_runtime_profile
from scripts.utils.device_utils import ExecutionContext, HardwareStatus


def _cpu_execution_context() -> ExecutionContext:
    hardware = HardwareStatus(
        python_version="3.11.9",
        torch_version="2.7.0+cpu",
        torch_cuda_version=None,
        cuda_available=False,
        device_count=0,
        gpu_names=tuple(),
        nvidia_smi_available=False,
        nvidia_smi_path=None,
        fallback_reason="Torch CPU-only",
    )
    return ExecutionContext(
        purpose="backtest",
        requested_accelerator="gpu",
        requested_precision="auto",
        accelerator="cpu",
        precision="32-true",
        device_type="cpu",
        device_string="cpu",
        pin_memory=False,
        non_blocking=False,
        hardware=hardware,
        fallback_reason="Se solicito GPU, pero CUDA no esta disponible. Torch CPU-only",
    )


@mock.patch("again_econ.runtime.resolve_execution_context")
def test_backtest_runtime_profile_allows_explicit_live_cpu_fallback(resolve_execution_context_mock):
    resolve_execution_context_mock.return_value = _cpu_execution_context()
    profile = resolve_backtest_runtime_profile(
        {
            "backtesting_runtime": {
                "accelerator": "gpu",
                "execution_backend": "gpu_full",
                "inference_backend": "gpu_batched",
                "allow_cpu_fallback_live": True,
                "allow_cpu_fallback_frozen": False,
            }
        },
        mode="exploratory_live",
    )

    assert profile.resolved_execution_backend == "cpu_reference"
    assert profile.resolved_inference_backend == "cpu_batched"
    assert profile.used_fallback is True


@mock.patch("again_econ.runtime.resolve_execution_context")
def test_backtest_runtime_profile_aborts_in_frozen_mode_when_cpu_fallback_is_forbidden(resolve_execution_context_mock):
    resolve_execution_context_mock.return_value = _cpu_execution_context()
    with pytest.raises(RuntimeError):
        resolve_backtest_runtime_profile(
            {
                "backtesting_runtime": {
                    "accelerator": "gpu",
                    "execution_backend": "gpu_full",
                    "inference_backend": "gpu_batched",
                    "allow_cpu_fallback_live": True,
                    "allow_cpu_fallback_frozen": False,
                }
            },
            mode="official_frozen",
        )
