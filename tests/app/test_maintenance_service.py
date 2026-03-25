import tempfile
import unittest
from pathlib import Path

from app.maintenance_service import RepositoryMaintenanceService
from scripts.runtime_config import ConfigManager
from scripts.utils.model_registry import load_model_registry
from scripts.utils.training_catalog import load_training_catalog
from tests.helpers.training_repo import seed_legacy_training_artifacts, write_base_repo, write_model_registry, write_runtime_profile


class RepositoryMaintenanceServiceTests(unittest.TestCase):
    def _build_service(self, root: Path, *, model_name: str) -> RepositoryMaintenanceService:
        config_path = write_base_repo(root)
        write_runtime_profile(root, model_name=model_name, ticker="AAA")
        seed_legacy_training_artifacts(root, model_name=model_name)
        write_model_registry(root, model_name=model_name)
        return RepositoryMaintenanceService(ConfigManager(str(config_path)))

    def test_purge_cache_and_temp_removes_whitelisted_paths_and_recreates_roots(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            service = self._build_service(root, model_name="Gen6_1__single__AAA")

            (root / "var" / "cache" / "demo").mkdir(parents=True, exist_ok=True)
            (root / "var" / "cache" / "demo" / "cache.txt").write_text("cache", encoding="utf-8")
            (root / "var" / "tmp" / "demo").mkdir(parents=True, exist_ok=True)
            (root / "var" / "tmp" / "demo" / "tmp.txt").write_text("tmp", encoding="utf-8")
            (root / ".matplotlib-cache").mkdir(parents=True, exist_ok=True)
            (root / ".matplotlib-cache" / "fontlist.json").write_text("{}", encoding="utf-8")
            (root / "pkg" / "__pycache__").mkdir(parents=True, exist_ok=True)
            (root / "pkg" / "__pycache__" / "module.pyc").write_bytes(b"pyc")

            preview = service.get_cache_cleanup_preview()
            self.assertTrue(any(path.endswith("var\\cache") for path in preview["candidates"]))
            self.assertTrue(any(path.endswith("var\\tmp") for path in preview["candidates"]))

            result = service.purge_cache_and_temp()

            self.assertGreaterEqual(result["deleted_count"], 2)
            self.assertTrue((root / "var" / "cache").exists())
            self.assertTrue((root / "var" / "tmp").exists())
            self.assertFalse((root / "pkg" / "__pycache__").exists())
            self.assertFalse((root / ".matplotlib-cache").exists())

    def test_delete_training_profile_requires_explicit_confirmation_for_active_profile(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            model_name = "Gen6_1__single__AAA"
            service = self._build_service(root, model_name=model_name)

            preview = service.preview_training_deletion(model_name)
            self.assertTrue(preview["active_profile"])
            self.assertTrue(any("artifacts\\training\\Gen6_1_single_AAA" in path for path in preview["existing_paths"]))

            with self.assertRaises(ValueError):
                service.delete_training_profile(model_name, allow_delete_active=False)

            result = service.delete_training_profile(model_name, allow_delete_active=True)

            self.assertTrue(result["active_profile_deleted"])
            self.assertFalse((root / "artifacts" / "training" / "Gen6_1_single_AAA").exists())
            self.assertFalse((root / "var" / "logs" / "training" / "Gen6_1_single_AAA").exists())
            registry = load_model_registry(service.config)
            self.assertNotIn(model_name, registry["profiles"])
            catalog = load_training_catalog(service.config)
            self.assertNotIn(model_name, catalog["profiles"])
