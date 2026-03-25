from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "valor"


def repo_root_from_config(config: dict) -> Path:
    config_path_value = config.get("_meta", {}).get("config_path")
    if config_path_value:
        candidate = Path(str(config_path_value)).resolve()
        for parent in (candidate.parent, *candidate.parents):
            if (parent / "pyproject.toml").exists() and (parent / "config").exists():
                return parent
        for parent in (candidate.parent, *candidate.parents):
            if parent.name == "config":
                return parent.parent
    return Path.cwd().resolve()


def resolve_repo_path(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root_from_config(config) / path


def serialize_repo_path(config: dict, value: str | Path) -> str:
    resolved = resolve_repo_path(config, value).resolve()
    repo_root = repo_root_from_config(config).resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def ensure_runtime_environment(config: dict) -> None:
    cache_dir = resolve_repo_path(config, config["paths"]["cache_dir"])
    tmp_dir = resolve_repo_path(config, config["paths"]["tmp_dir"])
    python_cache_dir = resolve_repo_path(config, config["paths"]["python_cache_dir"])
    matplotlib_cache_dir = resolve_repo_path(config, config["paths"]["matplotlib_cache_dir"])
    runtime_tmp_dir = tmp_dir / "runtime"
    for directory in (cache_dir, tmp_dir, python_cache_dir, matplotlib_cache_dir, runtime_tmp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    os.environ.setdefault("TMP", str(runtime_tmp_dir))
    os.environ.setdefault("TEMP", str(runtime_tmp_dir))
    os.environ.setdefault("TMPDIR", str(runtime_tmp_dir))
    if getattr(sys, "pycache_prefix", None) is None:
        sys.pycache_prefix = str(python_cache_dir)


@dataclass(frozen=True)
class TrainingProfileLayout:
    profile_id: str
    profile_root: Path
    active_root: Path
    runs_root: Path
    logs_root: Path
    active_dataset_dir: Path
    active_market_dir: Path
    active_checkpoints_dir: Path
    active_normalizers_dir: Path
    active_reports_dir: Path
    active_optuna_dir: Path

    def run_root(self, run_id: str) -> Path:
        return self.runs_root / run_id

    def run_optuna_dir(self, run_id: str) -> Path:
        return self.run_root(run_id) / "optuna"

    def run_logs_dir(self, run_id: str) -> Path:
        return self.logs_root / run_id


def build_training_profile_layout(config: dict, model_name: str | None = None) -> TrainingProfileLayout:
    resolved_model_name = str(model_name or config["model_name"])
    profile_id = sanitize_path_component(resolved_model_name)
    training_root = resolve_repo_path(config, config["paths"]["training_artifacts_dir"])
    logs_root_dir = resolve_repo_path(config, config["paths"]["logs_root_dir"])
    profile_root = training_root / profile_id
    active_root = profile_root / "active"
    return TrainingProfileLayout(
        profile_id=profile_id,
        profile_root=profile_root,
        active_root=active_root,
        runs_root=profile_root / "runs",
        logs_root=logs_root_dir / "training" / profile_id,
        active_dataset_dir=active_root / "dataset",
        active_market_dir=active_root / "market",
        active_checkpoints_dir=active_root / "checkpoints",
        active_normalizers_dir=active_root / "normalizers",
        active_reports_dir=active_root / "reports",
        active_optuna_dir=active_root / "optuna",
    )


def build_training_run_id(model_name: str, trained_at: str | None = None) -> str:
    timestamp = datetime.now().replace(microsecond=0)
    if trained_at:
        try:
            timestamp = datetime.fromisoformat(str(trained_at))
        except ValueError:
            pass
    return f"{sanitize_path_component(model_name)}__{timestamp.strftime('%Y%m%dT%H%M%S')}"


def apply_training_profile_layout(config: dict, *, run_id: str | None = None) -> dict:
    layout = build_training_profile_layout(config, config["model_name"])
    config.setdefault("paths", {})
    config.setdefault("data", {})
    config.setdefault("training_run", {})
    if config["paths"].get("models_dir") and not config["paths"].get("transfer_source_models_dir"):
        config["paths"]["transfer_source_models_dir"] = str(config["paths"]["models_dir"])
    if config["paths"].get("normalizers_dir") and not config["paths"].get("transfer_destination_normalizers_dir"):
        config["paths"]["transfer_destination_normalizers_dir"] = str(config["paths"]["normalizers_dir"])

    config["paths"]["training_profile_id"] = layout.profile_id
    config["paths"]["training_profile_root"] = serialize_repo_path(config, layout.profile_root)
    config["paths"]["training_active_root"] = serialize_repo_path(config, layout.active_root)
    config["paths"]["training_runs_root"] = serialize_repo_path(config, layout.runs_root)
    config["paths"]["training_logs_root_dir"] = serialize_repo_path(config, layout.logs_root)

    config["paths"]["models_dir"] = serialize_repo_path(config, layout.active_checkpoints_dir)
    config["paths"]["normalizers_dir"] = serialize_repo_path(config, layout.active_normalizers_dir)
    config["paths"]["model_save_path"] = serialize_repo_path(config, layout.active_checkpoints_dir / f"{config['model_name']}.pth")
    config["paths"]["optuna_dir"] = serialize_repo_path(config, layout.active_optuna_dir)
    config["paths"]["logs_dir"] = serialize_repo_path(config, layout.logs_root / "latest")

    config["data"]["raw_data_path"] = serialize_repo_path(config, layout.active_market_dir / "stock_data.csv")
    config["data"]["processed_data_path"] = serialize_repo_path(config, layout.active_dataset_dir / "processed_dataset.pt")
    config["data"]["train_processed_df_path"] = serialize_repo_path(config, layout.active_dataset_dir / "train_processed_df.parquet")
    config["data"]["val_processed_df_path"] = serialize_repo_path(config, layout.active_dataset_dir / "val_processed_df.parquet")

    training_run = dict(config.get("training_run") or {})
    training_run["profile_id"] = layout.profile_id
    training_run["profile_root"] = serialize_repo_path(config, layout.profile_root)
    training_run["active_root"] = serialize_repo_path(config, layout.active_root)
    config["training_run"] = training_run

    if run_id:
        run_root = layout.run_root(run_id)
        run_logs_dir = layout.run_logs_dir(run_id)
        config["paths"]["training_run_root"] = serialize_repo_path(config, run_root)
        config["paths"]["logs_dir"] = serialize_repo_path(config, run_logs_dir)
        config["paths"]["optuna_dir"] = serialize_repo_path(config, layout.run_optuna_dir(run_id))
        training_run["run_id"] = run_id
        training_run["run_root"] = serialize_repo_path(config, run_root)
        training_run["logs_dir"] = serialize_repo_path(config, run_logs_dir)
        config["training_run"] = training_run

    return config
