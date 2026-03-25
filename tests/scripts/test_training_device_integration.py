import unittest
from unittest import mock

from scripts.train import _create_trainer
from scripts.utils.device_utils import ExecutionContext, HardwareStatus


class TrainingDeviceIntegrationTests(unittest.TestCase):
    def _config(self):
        return {
            "training": {
                "max_epochs": 1,
                "early_stopping_patience": 1,
                "accelerator": "auto",
                "precision": "auto",
            },
            "paths": {
                "logs_dir": "logs",
            },
        }

    def _runtime(self, accelerator: str, precision: str):
        hardware = HardwareStatus(
            python_version="3.11.9",
            torch_version="2.7.0+cpu" if accelerator == "cpu" else "2.7.0+cu128",
            torch_cuda_version=None if accelerator == "cpu" else "12.8",
            cuda_available=accelerator == "gpu",
            device_count=0 if accelerator == "cpu" else 1,
            gpu_names=tuple() if accelerator == "cpu" else ("NVIDIA RTX Mock",),
            nvidia_smi_available=accelerator == "gpu",
            nvidia_smi_path="nvidia-smi" if accelerator == "gpu" else None,
            fallback_reason="Torch CPU-only" if accelerator == "cpu" else None,
        )
        return ExecutionContext(
            purpose="train",
            requested_accelerator="auto",
            requested_precision="auto",
            accelerator=accelerator,
            precision=precision,
            device_type="cpu" if accelerator == "cpu" else "cuda",
            device_string="cpu" if accelerator == "cpu" else "cuda:0",
            pin_memory=accelerator == "gpu",
            non_blocking=accelerator == "gpu",
            hardware=hardware,
            fallback_reason=hardware.fallback_reason,
        )

    @mock.patch("scripts.train.resolve_execution_context")
    @mock.patch("scripts.train.pl.Trainer")
    def test_trainer_recibe_gpu_si_cuda_esta_disponible(self, trainer_mock, runtime_mock):
        runtime_mock.return_value = self._runtime("gpu", "16-mixed")
        _create_trainer(self._config(), checkpoint_path=mock.Mock())
        _, kwargs = trainer_mock.call_args
        self.assertEqual(kwargs["accelerator"], "gpu")
        self.assertEqual(kwargs["precision"], "16-mixed")

    @mock.patch("scripts.train.resolve_execution_context")
    @mock.patch("scripts.train.pl.Trainer")
    def test_trainer_hace_fallback_a_cpu_si_no_hay_cuda(self, trainer_mock, runtime_mock):
        runtime_mock.return_value = self._runtime("cpu", "32-true")
        _create_trainer(self._config(), checkpoint_path=mock.Mock())
        _, kwargs = trainer_mock.call_args
        self.assertEqual(kwargs["accelerator"], "cpu")
        self.assertEqual(kwargs["precision"], "32-true")


if __name__ == "__main__":
    unittest.main()
