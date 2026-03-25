import tempfile
import unittest
from pathlib import Path

import torch

from scripts.utils.artifact_utils import load_trusted_torch_artifact


class TrustedArtifact:
    def __init__(self, value: int):
        self.value = value


class UntrustedArtifact:
    def __init__(self, value: int):
        self.value = value


class ArtifactUtilsTests(unittest.TestCase):
    def test_load_trusted_torch_artifact_rechaza_tipos_fuera_de_la_whitelist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "artifact.pt"
            torch.save(UntrustedArtifact(7), path)

            with self.assertRaises(ValueError):
                load_trusted_torch_artifact(path, trusted_types=[TrustedArtifact])

    def test_load_trusted_torch_artifact_acepta_tipos_confiables(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "artifact.pt"
            torch.save(TrustedArtifact(3), path)

            loaded = load_trusted_torch_artifact(path, trusted_types=[TrustedArtifact])

            self.assertIsInstance(loaded, TrustedArtifact)
            self.assertEqual(loaded.value, 3)


if __name__ == "__main__":
    unittest.main()
