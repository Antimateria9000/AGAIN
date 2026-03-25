from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scripts.utils.repo_layout import build_training_profile_layout, repo_root_from_config, resolve_repo_path, serialize_repo_path


CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DeletionPreview:
    model_name: str
    profile_path: str | None
    active_profile: bool
    active_run_id: str | None
    existing_paths: list[str]
    missing_paths: list[str]


def _catalog_path(config: dict) -> Path:
    return resolve_repo_path(config, config["paths"]["training_catalog_path"])


def _empty_catalog() -> dict[str, Any]:
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "profiles": {},
        "runs": {},
    }


def load_training_catalog(config: dict) -> dict[str, Any]:
    path = _catalog_path(config)
    if not path.exists():
        return _empty_catalog()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    payload.setdefault("schema_version", CATALOG_SCHEMA_VERSION)
    payload.setdefault("profiles", {})
    payload.setdefault("runs", {})
    return payload


def save_training_catalog(config: dict, catalog: dict[str, Any]) -> Path:
    path = _catalog_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(catalog, handle, sort_keys=False, allow_unicode=False)
    return path


def _copy_tree(source: Path | None, target: Path, copies: list[tuple[Path, Path]]) -> None:
    if source is None or not source.exists():
        return
    if source.resolve() == target.resolve():
        return
    shutil.copytree(source, target, dirs_exist_ok=True)
    copies.append((source, target))


def _copy_file(source: Path | None, target: Path, copies: list[tuple[Path, Path]]) -> None:
    if source is None or not source.exists():
        return
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copies.append((source, target))


def build_training_run_manifest(config: dict, *, profile_path: str | None) -> dict[str, Any]:
    layout = build_training_profile_layout(config, config["model_name"])
    run_id = str(config.get("training_run", {}).get("run_id") or "")
    if not run_id:
        raise ValueError("training_run.run_id es obligatorio para construir el manifiesto")

    run_root = resolve_repo_path(config, config["paths"]["training_run_root"])
    manifest_path = run_root / "manifest.json"
    active_checkpoint_dir = resolve_repo_path(config, config["paths"]["models_dir"])
    active_normalizers_dir = resolve_repo_path(config, config["paths"]["normalizers_dir"])
    active_dataset_path = resolve_repo_path(config, config["data"]["processed_data_path"])
    active_market_path = resolve_repo_path(config, config["data"]["raw_data_path"])
    report_path_value = config.get("training_run", {}).get("universe_integrity_report_path")
    staging_path_value = config.get("training_run", {}).get("raw_data_staging_path")
    active_report_path = resolve_repo_path(config, report_path_value) if report_path_value else None
    active_staging_path = resolve_repo_path(config, staging_path_value) if staging_path_value else None
    payload = {
        "run_id": run_id,
        "model_name": str(config["model_name"]),
        "base_model_name": str(config.get("base_model_name", config["model_name"])),
        "profile_path": str(profile_path) if profile_path else None,
        "profile_id": str(config.get("paths", {}).get("training_profile_id") or layout.profile_id),
        "created_at": str(config.get("training_run", {}).get("trained_at") or ""),
        "profile_root": str(layout.profile_root),
        "active_root": str(layout.active_root),
        "run_root": str(run_root),
        "logs_dir": str(config["paths"]["logs_dir"]),
        "training_run": deepcopy(config.get("training_run") or {}),
        "managed_paths": {
            "profile_root": str(layout.profile_root),
            "logs_root": str(layout.logs_root),
            "runtime_profile_path": str(profile_path) if profile_path else None,
            "run_root": str(run_root),
        },
        "active_artifacts": {
            "checkpoint_dir": str(active_checkpoint_dir),
            "normalizers_dir": str(active_normalizers_dir),
            "dataset_path": str(active_dataset_path),
            "market_path": str(active_market_path),
            "integrity_report_path": str(active_report_path) if active_report_path else None,
            "staging_market_path": str(active_staging_path) if active_staging_path else None,
        },
        "manifest_path": str(manifest_path),
    }
    return payload


def snapshot_training_run_artifacts(config: dict, *, profile_path: str | None) -> dict[str, Any]:
    manifest = build_training_run_manifest(config, profile_path=profile_path)
    run_root = Path(manifest["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)

    copies: list[tuple[Path, Path]] = []
    active_root = Path(manifest["active_root"])
    if active_root.exists():
        for child_name in ("dataset", "market", "checkpoints", "normalizers", "reports"):
            source_dir = active_root / child_name
            target_dir = run_root / child_name
            _copy_tree(source_dir, target_dir, copies)

    optuna_value = config["paths"].get("optuna_dir")
    optuna_dir = resolve_repo_path(config, optuna_value) if optuna_value else None
    _copy_tree(optuna_dir, run_root / "optuna", copies)

    if profile_path:
        source_profile = Path(profile_path)
        _copy_file(source_profile, run_root / "runtime_profile.yaml", copies)

    manifest["snapshotted_paths"] = [{"source": str(source), "target": str(target)} for source, target in copies]
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def register_training_run(config: dict, manifest: dict[str, Any], *, active_profile_path: str | None) -> dict[str, Any]:
    catalog = load_training_catalog(config)
    run_id = str(manifest["run_id"])
    model_name = str(manifest["model_name"])
    profile_path = active_profile_path or manifest.get("profile_path")
    profile_entry = catalog["profiles"].get(model_name) or {
        "model_name": model_name,
        "base_model_name": manifest.get("base_model_name", model_name),
        "profile_path": profile_path,
        "active_run_id": None,
        "latest_run_id": None,
        "run_ids": [],
        "label": str((manifest.get("training_run") or {}).get("label") or model_name),
        "description": str((manifest.get("training_run") or {}).get("description") or ""),
        "notes": str((manifest.get("training_run") or {}).get("notes") or ""),
        "created_at": manifest.get("created_at"),
    }
    if run_id not in profile_entry["run_ids"]:
        profile_entry["run_ids"].append(run_id)
    profile_entry["latest_run_id"] = run_id
    profile_entry["active_run_id"] = run_id
    profile_entry["profile_path"] = profile_path
    profile_entry["last_trained_at"] = manifest.get("created_at")
    profile_entry["profile_root"] = serialize_repo_path(config, manifest.get("profile_root"))
    logs_root = manifest.get("managed_paths", {}).get("logs_root")
    profile_entry["logs_root"] = serialize_repo_path(config, logs_root) if logs_root else None
    catalog["profiles"][model_name] = profile_entry
    catalog["runs"][run_id] = manifest
    save_training_catalog(config, catalog)
    return profile_entry


def list_training_profiles(config: dict, *, model_registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    catalog = load_training_catalog(config)
    registry = model_registry or {"active_profile_path": None, "profiles": {}}
    active_profile_path = registry.get("active_profile_path")
    resolved_active_profile_path = resolve_repo_path(config, active_profile_path).resolve() if active_profile_path else None
    rows: list[dict[str, Any]] = []
    for model_name, entry in catalog.get("profiles", {}).items():
        row = dict(entry)
        profile_path = entry.get("profile_path")
        resolved_profile_path = resolve_repo_path(config, profile_path).resolve() if profile_path else None
        row["is_active_profile"] = bool(resolved_active_profile_path and resolved_profile_path == resolved_active_profile_path)
        row["run_count"] = len(entry.get("run_ids", []))
        rows.append(row)
    for model_name, entry in registry.get("profiles", {}).items():
        if model_name in {row["model_name"] for row in rows}:
            continue
        layout = build_training_profile_layout(config, model_name)
        rows.append(
            {
                "model_name": model_name,
                "base_model_name": entry.get("base_model_name", model_name),
                "profile_path": entry.get("profile_path"),
                "active_run_id": entry.get("active_run_id"),
                "latest_run_id": entry.get("active_run_id"),
                "run_ids": [],
                "label": entry.get("label", model_name),
                "description": entry.get("description", ""),
                "notes": entry.get("notes", ""),
                "created_at": entry.get("created_at"),
                "last_trained_at": entry.get("last_trained_at"),
                "profile_root": serialize_repo_path(config, layout.profile_root),
                "logs_root": serialize_repo_path(config, layout.logs_root),
                "is_active_profile": bool(
                    resolved_active_profile_path
                    and entry.get("profile_path")
                    and resolve_repo_path(config, entry.get("profile_path")).resolve() == resolved_active_profile_path
                ),
                "run_count": 0,
            }
        )
    rows.sort(key=lambda item: str(item.get("last_trained_at") or item.get("created_at") or ""), reverse=True)
    return rows


def preview_profile_deletion(
    config: dict,
    *,
    model_name: str,
    profile_path: str | None,
    active_profile_path: str | None,
) -> DeletionPreview:
    layout = build_training_profile_layout(config, model_name)
    managed = [
        layout.profile_root,
        layout.logs_root,
    ]
    if profile_path:
        managed.append(resolve_repo_path(config, profile_path))

    existing_paths: list[str] = []
    missing_paths: list[str] = []
    for path in managed:
        if path.exists():
            existing_paths.append(str(path))
        else:
            missing_paths.append(str(path))

    catalog = load_training_catalog(config)
    profile_entry = catalog.get("profiles", {}).get(model_name) or {}
    resolved_profile_path = resolve_repo_path(config, profile_path).resolve() if profile_path else None
    resolved_active_profile_path = resolve_repo_path(config, active_profile_path).resolve() if active_profile_path else None
    return DeletionPreview(
        model_name=model_name,
        profile_path=profile_path,
        active_profile=bool(resolved_profile_path and resolved_active_profile_path and resolved_profile_path == resolved_active_profile_path),
        active_run_id=profile_entry.get("active_run_id"),
        existing_paths=existing_paths,
        missing_paths=missing_paths,
    )


def _ensure_within_repo(config: dict, path: Path) -> None:
    repo_root = repo_root_from_config(config)
    resolved_path = path.resolve()
    resolved_root = repo_root.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"La ruta {resolved_path} queda fuera del repositorio {resolved_root}")


def delete_training_profile(
    config: dict,
    *,
    model_name: str,
    profile_path: str | None,
    allow_delete_active: bool,
    active_profile_path: str | None,
) -> dict[str, Any]:
    preview = preview_profile_deletion(
        config,
        model_name=model_name,
        profile_path=profile_path,
        active_profile_path=active_profile_path,
    )
    if preview.active_profile and not allow_delete_active:
        raise ValueError("No se puede borrar el entrenamiento activo sin confirmacion explicita")

    deleted_paths: list[str] = []
    for path_text in preview.existing_paths:
        path = Path(path_text)
        _ensure_within_repo(config, path)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        deleted_paths.append(str(path))

    catalog = load_training_catalog(config)
    run_ids = list((catalog.get("profiles", {}).get(model_name) or {}).get("run_ids", []))
    catalog.get("profiles", {}).pop(model_name, None)
    for run_id in run_ids:
        catalog.get("runs", {}).pop(run_id, None)
    save_training_catalog(config, catalog)
    return {
        "model_name": model_name,
        "deleted_paths": deleted_paths,
        "deleted_run_ids": run_ids,
        "active_profile_deleted": preview.active_profile,
    }
