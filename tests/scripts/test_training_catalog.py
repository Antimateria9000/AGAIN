import tempfile
from pathlib import Path

from scripts.runtime_config import ConfigManager
from scripts.utils.repo_layout import apply_training_profile_layout, resolve_repo_path
from scripts.utils.training_catalog import snapshot_training_run_artifacts
from tests.helpers.training_repo import write_base_repo, write_runtime_profile


def test_snapshot_training_run_artifacts_skips_copy_when_optuna_source_matches_target():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        model_name = "Gen6_1__single__AAA"
        write_base_repo(root, model_name=model_name)
        profile_path = write_runtime_profile(root, model_name=model_name, ticker="AAA")

        manager = ConfigManager(str(profile_path))
        config = manager.config
        run_id = "Gen6_1_single_AAA__20260320T120000"
        apply_training_profile_layout(config, run_id=run_id)

        active_dataset_path = resolve_repo_path(config, config["data"]["processed_data_path"])
        active_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        active_dataset_path.write_bytes(b"dataset")

        run_root = resolve_repo_path(config, config["paths"]["training_run_root"])
        optuna_dir = run_root / "optuna"
        optuna_dir.mkdir(parents=True, exist_ok=True)
        (optuna_dir / "study.txt").write_text("optuna", encoding="utf-8")

        manifest = snapshot_training_run_artifacts(config, profile_path=str(profile_path))

        assert (run_root / "dataset" / "processed_dataset.pt").read_bytes() == b"dataset"
        assert (run_root / "optuna" / "study.txt").read_text(encoding="utf-8") == "optuna"
        assert all(
            Path(entry["source"]).resolve() != Path(entry["target"]).resolve()
            for entry in manifest["snapshotted_paths"]
        )
