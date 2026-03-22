from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import torch

from scripts.utils.artifact_utils import ensure_relative_to, read_metadata, verify_checksum
from scripts.utils.data_schema import metadata_matches_active_schema


@dataclass
class ModelReadinessReport:
    ready: bool
    summary: str
    issues: list[str]


def _schema_matches(config: dict, metadata: dict | None) -> bool:
    return metadata_matches_active_schema(config, metadata)


def _load_checkpoint_metadata(model_path: Path) -> dict | None:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    return checkpoint.get("metadata")


def _load_normalizers_metadata(normalizers_path: Path) -> dict | None:
    with open(normalizers_path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "metadata" in payload:
        return payload.get("metadata")
    return None


def assess_model_readiness(config: dict) -> ModelReadinessReport:
    issues: list[str] = []
    artifacts_required = bool(config.get("artifacts", {}).get("require_hash_validation", True))

    model_path = Path(config["paths"]["models_dir"]) / f"{config['model_name']}.pth"
    normalizers_path = Path(config["paths"]["normalizers_dir"]) / f"{config['model_name']}_normalizers.pkl"
    dataset_path = Path(config["data"]["processed_data_path"])

    try:
        ensure_relative_to(model_path, Path(config["paths"]["models_dir"]))
        if not model_path.exists():
            issues.append(f"Falta el checkpoint canonico: {model_path}")
        else:
            verify_checksum(model_path, required=artifacts_required)
            checkpoint_metadata = _load_checkpoint_metadata(model_path)
            if not checkpoint_metadata:
                issues.append(f"El checkpoint no tiene metadatos internos: {model_path}")
            elif not _schema_matches(config, checkpoint_metadata):
                issues.append(f"El checkpoint no coincide con el esquema activo: {model_path}")
    except Exception as exc:
        issues.append(f"Checkpoint invalido: {model_path} ({exc})")

    try:
        ensure_relative_to(normalizers_path, Path(config["paths"]["normalizers_dir"]))
        if not normalizers_path.exists():
            issues.append(f"Faltan los normalizadores: {normalizers_path}")
        else:
            verify_checksum(normalizers_path, required=artifacts_required)
            normalizers_metadata = _load_normalizers_metadata(normalizers_path)
            if not normalizers_metadata:
                issues.append(f"Los normalizadores no tienen metadatos: {normalizers_path}")
            elif not _schema_matches(config, normalizers_metadata):
                issues.append(f"Los normalizadores no coinciden con el esquema activo: {normalizers_path}")
    except Exception as exc:
        issues.append(f"Normalizadores invalidos: {normalizers_path} ({exc})")

    try:
        ensure_relative_to(dataset_path, Path(config["paths"]["data_dir"]))
        if not dataset_path.exists():
            issues.append(f"Falta el dataset procesado de entrenamiento: {dataset_path}")
        else:
            verify_checksum(dataset_path, required=artifacts_required)
            dataset_metadata = read_metadata(dataset_path)
            if not dataset_metadata:
                issues.append(f"El dataset procesado no tiene metadatos: {dataset_path}")
            elif not _schema_matches(config, dataset_metadata):
                issues.append(f"El dataset procesado no coincide con el esquema activo: {dataset_path}")
    except Exception as exc:
        issues.append(f"Dataset procesado invalido: {dataset_path} ({exc})")

    if issues:
        return ModelReadinessReport(
            ready=False,
            summary="El modelo no esta preparado para inferencia fiable. Falta preparar o reentrenar los artefactos canonicos.",
            issues=issues,
        )

    return ModelReadinessReport(
        ready=True,
        summary="El modelo tiene checkpoint, normalizadores y dataset procesado compatibles.",
        issues=[],
    )
