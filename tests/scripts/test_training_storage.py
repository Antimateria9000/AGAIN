import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.runtime_config import ConfigManager
from scripts.utils.training_catalog import load_training_catalog
from scripts.utils.training_storage import sync_training_storage
from tests.helpers.training_repo import seed_legacy_training_artifacts, write_base_repo, write_model_registry, write_runtime_profile


class TrainingStorageSyncTests(unittest.TestCase):
    def test_sync_training_storage_migrates_runtime_profiles_and_catalogs_runs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            model_name = "Gen6_1__single__AAA"
            config_path = write_base_repo(root)
            write_runtime_profile(root, model_name=model_name, ticker="AAA")
            seed_legacy_training_artifacts(root, model_name=model_name)
            write_model_registry(root, model_name=model_name)

            manager = ConfigManager(str(config_path))
            result = sync_training_storage(manager.config)

            self.assertEqual(result["profile_count"], 1)
            migrated_profile = root / "config" / "runtime_profiles" / f"{model_name}.yaml"
            migrated_payload = yaml.safe_load(migrated_profile.read_text(encoding="utf-8"))
            self.assertEqual(
                Path(migrated_payload["paths"]["models_dir"]),
                Path("artifacts/training/Gen6_1_single_AAA/active/checkpoints"),
            )
            self.assertEqual(
                Path(migrated_payload["data"]["processed_data_path"]),
                Path("artifacts/training/Gen6_1_single_AAA/active/dataset/processed_dataset.pt"),
            )
            self.assertTrue((root / "artifacts" / "training" / "Gen6_1_single_AAA" / "active" / "market" / "stock_data.csv").exists())
            self.assertTrue((root / "artifacts" / "training" / "Gen6_1_single_AAA" / "active" / "checkpoints" / f"{model_name}.pth").exists())
            self.assertTrue((root / "var" / "logs" / "training" / "Gen6_1_single_AAA" / "latest" / "metrics.csv").exists())

            catalog = load_training_catalog(manager.config)
            self.assertIn(model_name, catalog["profiles"])
            active_run_id = catalog["profiles"][model_name]["active_run_id"]
            self.assertIn(active_run_id, catalog["runs"])
            self.assertTrue((root / "artifacts" / "training" / "Gen6_1_single_AAA" / "runs" / active_run_id / "manifest.json").exists())
