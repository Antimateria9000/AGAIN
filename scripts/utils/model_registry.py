from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _registry_path(config: dict) -> Path:
    return Path(config["paths"]["model_registry_path"])


def load_model_registry(config: dict) -> dict[str, Any]:
    registry_path = _registry_path(config)
    if not registry_path.exists():
        return {"active_profile_path": None, "profiles": {}}
    with open(registry_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    payload.setdefault("active_profile_path", None)
    payload.setdefault("profiles", {})
    return payload


def save_model_registry(config: dict, registry: dict[str, Any]) -> Path:
    registry_path = _registry_path(config)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(registry, handle, sort_keys=False, allow_unicode=False)
    return registry_path


def register_model_profile(config: dict, profile_entry: dict[str, Any], set_active: bool = True) -> dict[str, Any]:
    registry = load_model_registry(config)
    entry = deepcopy(profile_entry)
    model_name = str(entry["model_name"])
    registry["profiles"][model_name] = entry
    if set_active:
        registry["active_profile_path"] = entry["profile_path"]
    save_model_registry(config, registry)
    return entry


def list_model_profiles(config: dict) -> list[dict[str, Any]]:
    registry = load_model_registry(config)
    profiles = list(registry.get("profiles", {}).values())
    profiles.sort(key=lambda item: str(item.get("last_trained_at") or item.get("created_at") or ""), reverse=True)
    return profiles


def get_active_profile_path(config: dict) -> str | None:
    registry = load_model_registry(config)
    active_profile = registry.get("active_profile_path")
    return str(active_profile) if active_profile else None


def set_active_profile_path(config: dict, profile_path: str | None) -> None:
    registry = load_model_registry(config)
    registry["active_profile_path"] = str(profile_path) if profile_path else None
    save_model_registry(config, registry)
