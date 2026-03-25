import copy
import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.utils.artifact_utils import write_checksum
from scripts.utils.data_schema import REQUIRED_NORMALIZER_KEYS, build_artifact_metadata
from scripts.utils.transfer_weights import transfer_weights


class DummyModel:
    def __init__(self):
        self._state_dict = {"encoder.weight": torch.zeros(1)}
        self.hparams = {"hidden_size": 8}

    def state_dict(self):
        return dict(self._state_dict)

    def load_state_dict(self, state_dict):
        self._state_dict = dict(state_dict)


class TransferWeightsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "models").mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts" / "normalizers").mkdir(parents=True, exist_ok=True)

        self.config = {
            "model_name": "new_model",
            "base_model_name": "base_model",
            "model": {
                "max_prediction_length": 5,
                "min_encoder_length": 10,
                "max_encoder_length": 20,
                "embedding_sizes": {"Sector": [3, 2], "Day_of_Week": [7, 4], "Month": [12, 6]},
                "sectors": ["Financials", "Unknown"],
            },
            "paths": {
                "models_dir": str(self.root / "models"),
                "normalizers_dir": str(self.root / "artifacts" / "normalizers"),
            },
            "artifacts": {
                "require_hash_validation": True,
            },
            "training_run": {
                "mode": "single_ticker",
                "requested_tickers": ["BBVA.MC"],
                "downloaded_tickers": ["BBVA.MC"],
                "final_tickers_used": ["BBVA.MC"],
            },
        }

        self.old_config = copy.deepcopy(self.config)
        self.old_config["model_name"] = "old_model"
        self.old_config["base_model_name"] = "old_base_model"
        self.old_config["training_run"] = {
            "mode": "predefined_group",
            "requested_tickers": ["BBVA.MC", "SAN.MC"],
            "downloaded_tickers": ["BBVA.MC", "SAN.MC"],
            "final_tickers_used": ["BBVA.MC", "SAN.MC"],
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_old_checkpoint(self) -> Path:
        old_checkpoint_path = Path(self.config["paths"]["models_dir"]) / "old_model.pth"
        payload = {
            "state_dict": {"encoder.weight": torch.ones(1)},
            "hyperparams": {"hidden_size": 8},
            "metadata": build_artifact_metadata(self.old_config),
        }
        torch.save(payload, old_checkpoint_path)
        write_checksum(old_checkpoint_path)
        return old_checkpoint_path

    def _write_destination_normalizers(self) -> Path:
        destination_path = Path(self.config["paths"]["normalizers_dir"]) / "new_model_normalizers.pkl"
        metadata = build_artifact_metadata(
            self.config,
            extra={"normalizer_keys": list(REQUIRED_NORMALIZER_KEYS)},
        )
        payload = {
            "normalizers": {key: {"dummy": key} for key in REQUIRED_NORMALIZER_KEYS},
            "metadata": metadata,
        }
        with open(destination_path, "wb") as handle:
            pickle.dump(payload, handle)
        write_checksum(destination_path)
        return destination_path

    def test_transfer_weights_usa_normalizers_del_destino_y_reescribe_metadata(self):
        old_checkpoint_path = self._write_old_checkpoint()
        destination_normalizers_path = self._write_destination_normalizers()
        model = DummyModel()

        transfer_weights(
            old_checkpoint_path=str(old_checkpoint_path),
            new_model=model,
            config=self.config,
            normalizers_path=destination_normalizers_path,
        )

        saved_checkpoint_path = Path(self.config["paths"]["models_dir"]) / "new_model.pth"
        saved_checkpoint = torch.load(saved_checkpoint_path, map_location="cpu", weights_only=False)
        self.assertTrue(torch.equal(model.state_dict()["encoder.weight"], torch.ones(1)))
        self.assertEqual(saved_checkpoint["metadata"]["model_name"], "new_model")
        self.assertEqual(saved_checkpoint["metadata"]["base_model_name"], "base_model")
        self.assertEqual(saved_checkpoint["metadata"]["transfer_learning"]["source_checkpoint"], "old_model.pth")
        self.assertEqual(saved_checkpoint["metadata"]["transfer_learning"]["source_model_name"], "old_model")


if __name__ == "__main__":
    unittest.main()
