import unittest
from unittest import mock

from scripts.utils import device_utils


class DeviceUtilsTests(unittest.TestCase):
    def setUp(self):
        device_utils.clear_hardware_status_cache()

    def tearDown(self):
        device_utils.clear_hardware_status_cache()

    def _base_config(self):
        return {
            "training": {
                "accelerator": "auto",
                "precision": "auto",
                "batch_size": 32,
                "num_workers": 2,
                "prefetch_factor": 2,
            },
            "prediction": {
                "batch_size": 32,
            },
        }

    @mock.patch("scripts.utils.device_utils._detect_nvidia_smi", return_value=(False, None))
    @mock.patch("scripts.utils.device_utils.torch.cuda.device_count", return_value=0)
    @mock.patch("scripts.utils.device_utils.torch.cuda.is_available", return_value=False)
    @mock.patch("scripts.utils.device_utils.torch.version.cuda", None)
    @mock.patch("scripts.utils.device_utils.torch.__version__", "2.7.0+cpu")
    def test_resuelve_cpu_si_cuda_no_esta_disponible(self, *_mocks):
        runtime = device_utils.resolve_execution_context(self._base_config(), purpose="train")
        self.assertEqual(runtime.accelerator, "cpu")
        self.assertEqual(runtime.device_string, "cpu")
        self.assertEqual(runtime.precision, "32-true")
        self.assertIn("CPU-only", runtime.fallback_reason)

    @mock.patch("scripts.utils.device_utils._detect_nvidia_smi", return_value=(True, "nvidia-smi"))
    @mock.patch("scripts.utils.device_utils._cuda_bf16_supported", return_value=True)
    @mock.patch("scripts.utils.device_utils.torch.cuda.get_device_name", return_value="NVIDIA RTX Mock")
    @mock.patch("scripts.utils.device_utils.torch.cuda.device_count", return_value=1)
    @mock.patch("scripts.utils.device_utils.torch.cuda.is_available", return_value=True)
    @mock.patch("scripts.utils.device_utils.torch.version.cuda", "12.8")
    @mock.patch("scripts.utils.device_utils.torch.__version__", "2.7.0+cu128")
    def test_resuelve_gpu_si_cuda_esta_disponible(self, *_mocks):
        runtime = device_utils.resolve_execution_context(self._base_config(), purpose="train")
        self.assertEqual(runtime.accelerator, "gpu")
        self.assertEqual(runtime.device_string, "cuda:0")
        self.assertEqual(runtime.precision, "bf16-mixed")
        self.assertEqual(runtime.primary_gpu_name, "NVIDIA RTX Mock")
        self.assertIsNone(runtime.fallback_reason)

    def test_payload_para_ui_contiene_los_campos_clave(self):
        with mock.patch("scripts.utils.device_utils.detect_hardware_status") as detect:
            detect.return_value = device_utils.HardwareStatus(
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
            runtime = device_utils.resolve_execution_context(self._base_config(), purpose="predict")
            payload = runtime.to_display_dict()
            self.assertEqual(payload["backend"], "CPU")
            self.assertIn("torch_version", payload)
            self.assertIn("torch_cuda_version", payload)
            self.assertIn("fallback_reason", payload)

    @mock.patch("scripts.utils.device_utils.detect_hardware_status")
    @mock.patch("scripts.utils.device_utils._cuda_bf16_supported", return_value=False)
    def test_auto_precision_en_gpu_sin_bf16_caiga_a_32_true(self, _bf16_mock, detect_mock):
        hardware = device_utils.HardwareStatus(
            python_version="3.11.9",
            torch_version="2.7.0+cu128",
            torch_cuda_version="12.8",
            cuda_available=True,
            device_count=1,
            gpu_names=("GPU mock",),
            nvidia_smi_available=True,
            nvidia_smi_path="nvidia-smi",
            fallback_reason=None,
        )
        detect_mock.return_value = hardware
        runtime = device_utils.resolve_execution_context(self._base_config(), purpose="train")
        self.assertEqual(runtime.accelerator, "gpu")
        self.assertEqual(runtime.precision, "32-true")


if __name__ == "__main__":
    unittest.main()
