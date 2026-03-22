import unittest
from pathlib import Path

import yaml

from scripts.utils.config_validation import resolve_tuning_config

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".venv", ".git", ".codex_pycache", "__pycache__"}


def iter_repo_python_files():
    for path in ROOT.rglob("*.py"):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        yield path


class StaticRepoTests(unittest.TestCase):
    def test_config_contains_reproducibility_keys(self):
        config_text = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
        self.assertIn("train_processed_df_path: data/train/train_processed_df.parquet", config_text)
        self.assertIn("val_processed_df_path: data/train/val_processed_df.parquet", config_text)
        self.assertIn("tickers_file: config/tickers_with_names.yaml", config_text)
        self.assertIn("seed: 42", config_text)
        self.assertIn("benchmark_history_db_path: data/benchmarks_history.sqlite", config_text)
        self.assertIn("require_hash_validation: true", config_text)

    def test_requirements_are_pinned_and_clean(self):
        requirements_text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        for dependency in (
            "lightning==",
            "pytorch-forecasting==",
            "torch==",
            "streamlit==",
            "yfinance==",
            "pyarrow==",
        ):
            self.assertIn(dependency, requirements_text)
        for removed_dependency in ("aiohttp", "nest_asyncio", "fredapi", "python-dotenv", "pandas-datareader"):
            self.assertNotIn(removed_dependency, requirements_text)

    def test_readme_matches_current_core_paths(self):
        readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("config/tickers_with_names.yaml", readme_text)
        self.assertIn("streamlit run app/app.py", readme_text)
        self.assertIn("benchmark_store.py", readme_text)
        self.assertIn("Smoke test del pipeline", readme_text)
        self.assertNotIn("config/tickers.yaml", readme_text)

    def test_logging_basicconfig_only_exists_in_allowed_files(self):
        allowed = {
            ROOT / "scripts" / "utils" / "logging_utils.py",
            ROOT / "scripts" / "debug" / "analyze_data.py",
            ROOT / "scripts" / "debug" / "debug_dataset.py",
            ROOT / "scripts" / "debug" / "feature_importance.py",
        }
        for path in iter_repo_python_files():
            if path == ROOT / "tests" / "test_static_repo.py":
                continue
            source = path.read_text(encoding="utf-8")
            if "logging.basicConfig(" in source:
                self.assertIn(path, allowed, msg=f"logging.basicConfig no deberia aparecer en {path}")
        self.assertIn("configure_logging()", (ROOT / "start_training.py").read_text(encoding="utf-8"))
        self.assertIn("configure_logging()", (ROOT / "app" / "app.py").read_text(encoding="utf-8"))

    def test_asyncio_patches_are_removed_from_runtime_paths(self):
        runtime_files = [
            ROOT / "app" / "app.py",
            ROOT / "app" / "benchmark_utils.py",
            ROOT / "scripts" / "data_fetcher.py",
            ROOT / "scripts" / "prediction_engine.py",
        ]
        forbidden_tokens = ["nest_asyncio", "run_until_complete(", "asyncio.run("]
        for path in runtime_files:
            source = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                self.assertNotIn(token, source, msg=f"{token} no deberia aparecer en {path}")

    def test_key_python_modules_compile(self):
        files_to_check = [
            ROOT / "start_training.py",
            ROOT / "app" / "app.py",
            ROOT / "app" / "benchmark_utils.py",
            ROOT / "app" / "benchmark_store.py",
            ROOT / "app" / "services.py",
            ROOT / "scripts" / "runtime_config.py",
            ROOT / "scripts" / "data_fetcher.py",
            ROOT / "scripts" / "model.py",
            ROOT / "scripts" / "preprocessor.py",
            ROOT / "scripts" / "prediction_engine.py",
            ROOT / "scripts" / "train.py",
            ROOT / "scripts" / "utils" / "artifact_utils.py",
            ROOT / "scripts" / "utils" / "lightning_compat.py",
            ROOT / "scripts" / "utils" / "yfinance_provider.py",
        ]
        for path in files_to_check:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")

    def test_tuning_config_convierte_notacion_cientifica_a_numeros(self):
        config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
        tuning = resolve_tuning_config(config)
        self.assertIsInstance(tuning["min_learning_rate"], float)
        self.assertIsInstance(tuning["max_learning_rate"], float)
        self.assertLessEqual(tuning["min_learning_rate"], tuning["max_learning_rate"])


if __name__ == "__main__":
    unittest.main()
