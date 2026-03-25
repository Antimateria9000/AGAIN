from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.runtime_config import ConfigManager
from scripts.utils.model_registry import load_model_registry, register_model_profile, set_active_profile_path
from scripts.utils.repo_layout import (
    apply_training_profile_layout,
    build_training_run_id,
    build_training_profile_layout,
    resolve_repo_path,
    serialize_repo_path,
)
from scripts.utils.training_catalog import register_training_run, snapshot_training_run_artifacts
from scripts.utils.training_universe import save_runtime_profile


@dataclass(frozen=True)
class LegacyTrainingArtifacts:
    profile_path: Path
    raw_data_path: Path | None
    processed_dataset_path: Path | None
    train_df_path: Path | None
    val_df_path: Path | None
    integrity_report_path: Path | None
    staging_market_path: Path | None
    models_dir: Path | None
    normalizers_dir: Path | None
    logs_dir: Path | None
    optuna_dir: Path | None


def _resolve_optional(config: dict, value: str | None) -> Path | None:
    if not value:
        return None
    return resolve_repo_path(config, value)


def _capture_legacy_artifacts(config: dict, profile_path: Path) -> LegacyTrainingArtifacts:
    training_run = dict(config.get("training_run") or {})
    return LegacyTrainingArtifacts(
        profile_path=profile_path,
        raw_data_path=_resolve_optional(config, config.get("data", {}).get("raw_data_path")),
        processed_dataset_path=_resolve_optional(config, config.get("data", {}).get("processed_data_path")),
        train_df_path=_resolve_optional(config, config.get("data", {}).get("train_processed_df_path")),
        val_df_path=_resolve_optional(config, config.get("data", {}).get("val_processed_df_path")),
        integrity_report_path=_resolve_optional(config, training_run.get("universe_integrity_report_path")),
        staging_market_path=_resolve_optional(config, training_run.get("raw_data_staging_path")),
        models_dir=_resolve_optional(config, config.get("paths", {}).get("models_dir")),
        normalizers_dir=_resolve_optional(config, config.get("paths", {}).get("normalizers_dir")),
        logs_dir=_resolve_optional(config, config.get("paths", {}).get("logs_dir")),
        optuna_dir=_resolve_optional(config, config.get("paths", {}).get("optuna_dir")),
    )


def _copy_file(source: Path | None, target: Path, copied_paths: list[str]) -> None:
    if source is None or not source.exists():
        return
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied_paths.append(str(target))


def _copy_tree(source: Path | None, target: Path, copied_paths: list[str]) -> None:
    if source is None or not source.exists():
        return
    if source.resolve() == target.resolve():
        return
    shutil.copytree(source, target, dirs_exist_ok=True)
    copied_paths.append(str(target))


def _copy_matching_files(source_dir: Path | None, target_dir: Path, patterns: list[str], copied_paths: list[str]) -> None:
    if source_dir is None or not source_dir.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    seen: set[Path] = set()
    for pattern in patterns:
        for candidate in source_dir.glob(pattern):
            if candidate in seen or not candidate.is_file():
                continue
            target = target_dir / candidate.name
            if candidate.resolve() == target.resolve():
                seen.add(candidate)
                continue
            shutil.copy2(candidate, target)
            copied_paths.append(str(target))
            seen.add(candidate)


def mirror_training_logs(config: dict, *, source_logs_dir: Path | None = None) -> Path:
    layout = build_training_profile_layout(config, config["model_name"])
    active_logs_dir = layout.logs_root / "latest"
    active_logs_dir.mkdir(parents=True, exist_ok=True)
    source_dir = source_logs_dir or resolve_repo_path(config, config["paths"]["logs_dir"])
    if source_dir.exists() and source_dir.resolve() != active_logs_dir.resolve():
        shutil.copytree(source_dir, active_logs_dir, dirs_exist_ok=True)
    return active_logs_dir


def _coerce_timestamp(value: Any, fallback: datetime) -> str:
    if value:
        try:
            return datetime.fromisoformat(str(value)).replace(microsecond=0).isoformat()
        except ValueError:
            pass
    return fallback.replace(microsecond=0).isoformat()


def build_training_profile_entry(config: dict, profile_path: Path) -> dict[str, Any]:
    training_run = dict(config.get("training_run") or {})
    layout = build_training_profile_layout(config, config["model_name"])
    return {
        "model_name": config["model_name"],
        "base_model_name": config.get("base_model_name", config["model_name"]),
        "profile_path": serialize_repo_path(config, profile_path),
        "label": training_run.get("label") or config["model_name"],
        "description": training_run.get("description", ""),
        "notes": training_run.get("notes", ""),
        "universe_mode": training_run.get("mode", "legacy_regions"),
        "single_ticker_symbol": training_run.get("single_ticker_symbol"),
        "predefined_group_name": training_run.get("predefined_group_name"),
        "requested_tickers": list(training_run.get("requested_tickers", [])),
        "downloaded_tickers": list(training_run.get("downloaded_tickers", [])),
        "discarded_tickers": list(training_run.get("discarded_tickers", [])),
        "final_tickers_used": list(training_run.get("final_tickers_used", [])),
        "created_at": training_run.get("trained_at"),
        "last_trained_at": training_run.get("trained_at"),
        "prediction_horizon": training_run.get("prediction_horizon"),
        "years": training_run.get("years"),
        "active_run_id": training_run.get("run_id"),
        "profile_root": serialize_repo_path(config, layout.profile_root),
        "active_root": serialize_repo_path(config, layout.active_root),
        "logs_root": serialize_repo_path(config, layout.logs_root),
    }


def migrate_training_profile(profile_path: str | Path, *, active_profile_path: str | None = None) -> dict[str, Any]:
    resolved_profile = Path(profile_path).resolve()
    profile_manager = ConfigManager(str(resolved_profile))
    legacy_config = deepcopy(profile_manager.config)
    legacy_artifacts = _capture_legacy_artifacts(legacy_config, resolved_profile)

    training_run = dict(legacy_config.get("training_run") or {})
    fallback_trained_at = datetime.fromtimestamp(resolved_profile.stat().st_mtime)
    trained_at = _coerce_timestamp(training_run.get("trained_at"), fallback_trained_at)
    training_run["trained_at"] = trained_at
    training_run["run_id"] = training_run.get("run_id") or build_training_run_id(legacy_config["model_name"], trained_at)
    legacy_config["training_run"] = training_run
    apply_training_profile_layout(legacy_config, run_id=str(training_run["run_id"]))

    layout = build_training_profile_layout(legacy_config, legacy_config["model_name"])
    for directory in (
        layout.active_root,
        layout.active_dataset_dir,
        layout.active_market_dir,
        layout.active_checkpoints_dir,
        layout.active_normalizers_dir,
        layout.active_reports_dir,
        layout.active_optuna_dir,
        layout.run_root(str(training_run["run_id"])),
        layout.run_logs_dir(str(training_run["run_id"])),
    ):
        directory.mkdir(parents=True, exist_ok=True)

    copied_paths: list[str] = []
    canonical_raw_path = resolve_repo_path(legacy_config, legacy_config["data"]["raw_data_path"])
    _copy_file(legacy_artifacts.raw_data_path, canonical_raw_path, copied_paths)
    if legacy_artifacts.raw_data_path and legacy_artifacts.raw_data_path.exists():
        legacy_config["training_run"]["raw_data_artifact_path"] = serialize_repo_path(legacy_config, canonical_raw_path)
    if legacy_artifacts.staging_market_path:
        target_staging = canonical_raw_path.with_name(legacy_artifacts.staging_market_path.name)
        _copy_file(legacy_artifacts.staging_market_path, target_staging, copied_paths)
        legacy_config["training_run"]["raw_data_staging_path"] = serialize_repo_path(legacy_config, target_staging)
    if legacy_artifacts.integrity_report_path:
        target_report = canonical_raw_path.with_name(legacy_artifacts.integrity_report_path.name)
        _copy_file(legacy_artifacts.integrity_report_path, target_report, copied_paths)
        legacy_config["training_run"]["universe_integrity_report_path"] = serialize_repo_path(legacy_config, target_report)

    _copy_file(legacy_artifacts.processed_dataset_path, resolve_repo_path(legacy_config, legacy_config["data"]["processed_data_path"]), copied_paths)
    _copy_file(legacy_artifacts.train_df_path, resolve_repo_path(legacy_config, legacy_config["data"]["train_processed_df_path"]), copied_paths)
    _copy_file(legacy_artifacts.val_df_path, resolve_repo_path(legacy_config, legacy_config["data"]["val_processed_df_path"]), copied_paths)

    model_name = legacy_config["model_name"]
    _copy_matching_files(
        legacy_artifacts.models_dir,
        resolve_repo_path(legacy_config, legacy_config["paths"]["models_dir"]),
        [
            f"{model_name}.pth",
            f"{model_name}.pth.*",
            f"{model_name}_best.pth",
            f"{model_name}_best.pth.*",
            f"{model_name}_last.pth",
            f"{model_name}_last.pth.*",
        ],
        copied_paths,
    )
    _copy_matching_files(
        legacy_artifacts.normalizers_dir,
        resolve_repo_path(legacy_config, legacy_config["paths"]["normalizers_dir"]),
        [
            f"{model_name}_normalizers.pkl",
            f"{model_name}_normalizers.pkl.*",
        ],
        copied_paths,
    )
    run_logs_dir = resolve_repo_path(legacy_config, legacy_config["paths"]["logs_dir"])
    _copy_tree(legacy_artifacts.logs_dir, run_logs_dir, copied_paths)
    latest_logs_dir = mirror_training_logs(legacy_config, source_logs_dir=run_logs_dir)
    if latest_logs_dir.exists():
        copied_paths.append(str(latest_logs_dir))
    _copy_tree(legacy_artifacts.optuna_dir, resolve_repo_path(legacy_config, legacy_config["paths"]["optuna_dir"]), copied_paths)

    saved_profile_path = save_runtime_profile(legacy_config)
    manifest = snapshot_training_run_artifacts(legacy_config, profile_path=str(saved_profile_path))
    active_profile_value = serialize_repo_path(legacy_config, saved_profile_path)
    register_training_run(legacy_config, manifest, active_profile_path=active_profile_value)
    register_model_profile(
        legacy_config,
        build_training_profile_entry(legacy_config, saved_profile_path),
        set_active=False,
    )
    if active_profile_path and resolve_repo_path(legacy_config, active_profile_path).resolve() == saved_profile_path.resolve():
        set_active_profile_path(legacy_config, active_profile_value)

    return {
        "model_name": model_name,
        "profile_path": active_profile_value,
        "run_id": training_run["run_id"],
        "copied_paths": copied_paths,
    }


def sync_training_storage(config: dict) -> dict[str, Any]:
    registry = load_model_registry(config)
    runtime_profiles_dir = resolve_repo_path(config, config["paths"]["runtime_profiles_dir"])
    profile_candidates: dict[str, Path] = {}

    for entry in (registry.get("profiles") or {}).values():
        profile_path = entry.get("profile_path")
        if not profile_path:
            continue
        resolved = resolve_repo_path(config, profile_path)
        profile_candidates[str(resolved)] = resolved

    if runtime_profiles_dir.exists():
        for path in runtime_profiles_dir.glob("*.yaml"):
            profile_candidates[str(path.resolve())] = path.resolve()

    migrated_profiles: list[dict[str, Any]] = []
    active_profile_path = registry.get("active_profile_path")
    for profile_path in sorted(profile_candidates.values()):
        migrated_profiles.append(
            migrate_training_profile(
                profile_path,
                active_profile_path=active_profile_path,
            )
        )

    return {
        "profile_count": len(migrated_profiles),
        "profiles": migrated_profiles,
    }
