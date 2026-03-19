import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StaticRepoTests(unittest.TestCase):
    def test_config_contains_expected_runtime_paths(self):
        config_text = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
        self.assertIn("train_processed_df_path: data/train/train_processed_df.parquet", config_text)
        self.assertIn("val_processed_df_path: data/train/val_processed_df.parquet", config_text)
        self.assertIn("tickers_file: config/tickers_with_names.yaml", config_text)
        self.assertIn("seed: 42", config_text)
        self.assertNotIn("\n  processed_df_path:", config_text)

    def test_requirements_include_critical_runtime_dependencies(self):
        requirements_text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        for dependency in ("aiohttp", "matplotlib", "pyarrow", "streamlit", "yfinance", "torch", "pytorch-lightning", "pytorch-forecasting"):
            self.assertIn(dependency, requirements_text)

    def test_readme_matches_current_core_paths(self):
        readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("config/tickers_with_names.yaml", readme_text)
        self.assertIn("streamlit run app/app.py", readme_text)
        self.assertIn("Politica de artefactos", readme_text)
        self.assertIn("Reproducibilidad de resultados", readme_text)
        self.assertNotIn("config/tickers.yaml", readme_text)

    def test_logging_basicconfig_only_exists_in_entrypoints(self):
        allowed = {
            ROOT / "scripts" / "utils" / "logging_utils.py",
            ROOT / "scripts" / "debug" / "analyze_data.py",
            ROOT / "scripts" / "debug" / "debug_dataset.py",
            ROOT / "scripts" / "debug" / "feature_importance.py",
        }
        for path in ROOT.rglob("*.py"):
            if path == ROOT / "tests" / "test_static_repo.py":
                continue
            source = path.read_text(encoding="utf-8")
            if "logging.basicConfig(" in source:
                self.assertIn(path, allowed, msg=f"logging.basicConfig no deberia aparecer en {path}")
        self.assertIn("configure_logging()", (ROOT / "start_training.py").read_text(encoding="utf-8"))
        self.assertIn("configure_logging()", (ROOT / "app" / "app.py").read_text(encoding="utf-8"))

    def test_save_history_range_uses_config_path(self):
        source = (ROOT / "app" / "config_loader.py").read_text(encoding="utf-8")
        self.assertIn("config_manager.config_path", source)
        self.assertNotIn("config_manager.config_file", source)

    def test_key_python_modules_compile(self):
        files_to_check = [
            ROOT / "start_training.py",
            ROOT / "app" / "app.py",
            ROOT / "app" / "benchmark_utils.py",
            ROOT / "scripts" / "runtime_config.py",
            ROOT / "scripts" / "data_fetcher.py",
            ROOT / "scripts" / "model.py",
            ROOT / "scripts" / "preprocessor.py",
            ROOT / "scripts" / "prediction_engine.py",
            ROOT / "scripts" / "train.py",
            ROOT / "scripts" / "utils" / "logging_utils.py",
        ]
        for path in files_to_check:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")


if __name__ == "__main__":
    unittest.main()
