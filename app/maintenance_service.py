from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from scripts.runtime_config import ConfigManager
from scripts.utils.model_registry import load_model_registry, remove_model_profile, set_active_profile_path
from scripts.utils.repo_layout import repo_root_from_config, resolve_repo_path
from scripts.utils.training_storage import sync_training_storage
from scripts.utils.training_catalog import (
    delete_training_profile as delete_training_profile_from_catalog,
    list_training_profiles as list_catalog_training_profiles,
    preview_profile_deletion,
)


class RepositoryMaintenanceService:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.repo_root = repo_root_from_config(self.config)
        self.sync_training_storage()

    def sync_training_storage(self) -> dict[str, Any]:
        return sync_training_storage(self.config)

    def _ensure_within_repo(self, path: Path) -> None:
        resolved_path = path.resolve()
        resolved_root = self.repo_root.resolve()
        if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
            raise ValueError(f"La ruta {resolved_path} queda fuera del repositorio {resolved_root}")

    def list_training_profiles(self) -> list[dict[str, Any]]:
        registry = load_model_registry(self.config)
        return list_catalog_training_profiles(self.config, model_registry=registry)

    def preview_training_deletion(self, model_name: str) -> dict[str, Any]:
        registry = load_model_registry(self.config)
        profile_entry = registry.get("profiles", {}).get(model_name) or {}
        profile_path = profile_entry.get("profile_path")
        preview = preview_profile_deletion(
            self.config,
            model_name=model_name,
            profile_path=profile_path,
            active_profile_path=registry.get("active_profile_path"),
        )
        extra_paths = self._profile_specific_paths(profile_path)
        existing_paths = list(dict.fromkeys([*preview.existing_paths, *extra_paths["existing_paths"]]))
        missing_paths = list(dict.fromkeys([*preview.missing_paths, *extra_paths["missing_paths"]]))
        return {
            "model_name": preview.model_name,
            "profile_path": preview.profile_path,
            "active_profile": preview.active_profile,
            "active_run_id": preview.active_run_id,
            "existing_paths": existing_paths,
            "missing_paths": missing_paths,
        }

    def delete_training_profile(self, model_name: str, *, allow_delete_active: bool = False) -> dict[str, Any]:
        registry = load_model_registry(self.config)
        profile_entry = registry.get("profiles", {}).get(model_name) or {}
        profile_path = profile_entry.get("profile_path")
        preview = self.preview_training_deletion(model_name)
        result = delete_training_profile_from_catalog(
            self.config,
            model_name=model_name,
            profile_path=profile_path,
            allow_delete_active=allow_delete_active,
            active_profile_path=registry.get("active_profile_path"),
        )

        extra_paths = [Path(path_text) for path_text in preview["existing_paths"] if path_text not in result["deleted_paths"]]
        for path in extra_paths:
            if not path.exists():
                continue
            self._ensure_within_repo(path)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            result["deleted_paths"].append(str(path))

        removed = remove_model_profile(self.config, model_name)
        if removed and result["active_profile_deleted"]:
            set_active_profile_path(self.config, None)
        return result

    def get_cache_cleanup_preview(self) -> dict[str, Any]:
        candidates = []
        cache_root = resolve_repo_path(self.config, self.config["paths"]["cache_dir"])
        tmp_root = resolve_repo_path(self.config, self.config["paths"]["tmp_dir"])
        for root in (cache_root, tmp_root):
            if root.exists():
                candidates.append(str(root))

        legacy_paths = [
            self.repo_root / ".codex_pycache",
            self.repo_root / ".matplotlib-cache",
            self.repo_root / ".pytest_cache",
            self.repo_root / "data" / "cache",
            self.repo_root / "data" / "yfinance_cache",
        ]
        for path in legacy_paths:
            if path.exists():
                candidates.append(str(path))

        for path in self.repo_root.iterdir():
            if path.is_dir() and path.name.startswith("pytest-cache-files-"):
                candidates.append(str(path))

        for path in self.repo_root.rglob("__pycache__"):
            if ".venv" in path.parts or ".git" in path.parts:
                continue
            candidates.append(str(path))

        return {
            "candidate_count": len(candidates),
            "candidates": sorted(dict.fromkeys(candidates)),
        }

    def purge_cache_and_temp(self) -> dict[str, Any]:
        preview = self.get_cache_cleanup_preview()
        deleted_paths: list[str] = []
        for path_text in preview["candidates"]:
            path = Path(path_text)
            if not path.exists():
                continue
            self._ensure_within_repo(path)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            deleted_paths.append(str(path))

        for root_value in (self.config["paths"]["cache_dir"], self.config["paths"]["tmp_dir"]):
            Path(resolve_repo_path(self.config, root_value)).mkdir(parents=True, exist_ok=True)

        return {
            "deleted_count": len(deleted_paths),
            "deleted_paths": deleted_paths,
        }

    def _profile_specific_paths(self, profile_path: str | None) -> dict[str, list[str]]:
        if not profile_path:
            return {"existing_paths": [], "missing_paths": []}

        resolved_profile_path = resolve_repo_path(self.config, profile_path)
        profile_manager = ConfigManager(str(resolved_profile_path))
        profile_config = profile_manager.config
        optuna_dir = profile_config["paths"].get("optuna_dir")
        candidate_paths = [
            resolved_profile_path,
            resolve_repo_path(profile_config, profile_config["paths"]["models_dir"]),
            resolve_repo_path(profile_config, profile_config["paths"]["normalizers_dir"]),
            resolve_repo_path(profile_config, profile_config["paths"]["logs_dir"]),
            resolve_repo_path(profile_config, profile_config["data"]["raw_data_path"]).parent,
            resolve_repo_path(profile_config, profile_config["data"]["processed_data_path"]).parent,
        ]
        if optuna_dir:
            candidate_paths.append(resolve_repo_path(profile_config, optuna_dir))
        existing_paths: list[str] = []
        missing_paths: list[str] = []
        for path in candidate_paths:
            if str(path) in {"", "."}:
                continue
            if path.exists():
                existing_paths.append(str(path))
            else:
                missing_paths.append(str(path))
        return {"existing_paths": existing_paths, "missing_paths": missing_paths}
