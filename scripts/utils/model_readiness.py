from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import torch

from scripts.utils.artifact_utils import ensure_relative_to, load_trusted_torch_artifact, read_metadata, verify_checksum
from scripts.utils.data_schema import metadata_matches_active_schema
from scripts.utils.repo_layout import resolve_repo_path


@dataclass
class ModelReadinessReport:
    ready: bool
    summary: str
    issues: list[str]


def _schema_matches(config: dict, metadata: dict | None) -> bool:
    return metadata_matches_active_schema(config, metadata)


def _resolve_dataset_base_dir(config: dict, dataset_path: Path) -> Path:
    candidates = [
        resolve_repo_path(config, config.get("paths", {}).get("artifacts_dir", "artifacts")),
        resolve_repo_path(config, config["paths"]["data_dir"]),
    ]
    for candidate in candidates:
        try:
            ensure_relative_to(dataset_path, candidate)
            return candidate
        except ValueError:
            continue
    raise ValueError(f"La ruta {dataset_path} no cae dentro de roots permitidos")


def _load_checkpoint_metadata(model_path: Path) -> dict | None:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    return checkpoint.get("metadata")


def _load_normalizers_metadata(normalizers_path: Path) -> dict | None:
    with open(normalizers_path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "metadata" in payload:
        return payload.get("metadata")
    return None


def _load_processed_dataset(dataset_path: Path):
    from pytorch_forecasting import TimeSeriesDataSet

    return load_trusted_torch_artifact(dataset_path, trusted_types=[TimeSeriesDataSet])


def _universe_semantically_valid(metadata: dict | None) -> tuple[bool, str | None]:
    if not metadata:
        return True, None
    training_run = dict(metadata.get("training_run") or {})
    integrity = dict(training_run.get("universe_integrity") or {})
    if not integrity:
        return True, None
    decision = str(integrity.get("decision") or "").strip().upper()
    if not decision:
        return True, None
    if decision in {"ABORT", "DEGRADED_FORBIDDEN"} or integrity.get("training_allowed") is False:
        reasons = integrity.get("decision_reasons") or [integrity.get("summary", "Sin detalle")]
        return False, " | ".join(str(reason) for reason in reasons if str(reason))
    return True, integrity.get("summary") if integrity.get("degraded") else None


def assess_model_readiness(config: dict) -> ModelReadinessReport:
    issues: list[str] = []
    warnings: list[str] = []
    artifacts_required = bool(config.get("artifacts", {}).get("require_hash_validation", True))

    model_path = resolve_repo_path(config, config["paths"]["models_dir"]) / f"{config['model_name']}.pth"
    normalizers_path = resolve_repo_path(config, config["paths"]["normalizers_dir"]) / f"{config['model_name']}_normalizers.pkl"
    dataset_path = resolve_repo_path(config, config["data"]["processed_data_path"])

    try:
        ensure_relative_to(model_path, resolve_repo_path(config, config["paths"]["models_dir"]))
        if not model_path.exists():
            issues.append(f"Falta el checkpoint canonico: {model_path}")
        else:
            verify_checksum(model_path, required=artifacts_required)
            checkpoint_metadata = _load_checkpoint_metadata(model_path)
            if not checkpoint_metadata:
                issues.append(f"El checkpoint no tiene metadatos internos: {model_path}")
            elif not _schema_matches(config, checkpoint_metadata):
                issues.append(f"El checkpoint no coincide con el esquema activo: {model_path}")
            else:
                universe_ok, detail = _universe_semantically_valid(checkpoint_metadata)
                if not universe_ok:
                    issues.append(f"El checkpoint proviene de un universo semanticamente invalido: {detail}")
                elif detail:
                    warnings.append(f"El checkpoint proviene de un universo degradado permitido: {detail}")
    except Exception as exc:
        issues.append(f"Checkpoint invalido: {model_path} ({exc})")

    try:
        ensure_relative_to(normalizers_path, resolve_repo_path(config, config["paths"]["normalizers_dir"]))
        if not normalizers_path.exists():
            issues.append(f"Faltan los normalizadores: {normalizers_path}")
        else:
            verify_checksum(normalizers_path, required=artifacts_required)
            normalizers_metadata = _load_normalizers_metadata(normalizers_path)
            if not normalizers_metadata:
                issues.append(f"Los normalizadores no tienen metadatos: {normalizers_path}")
            elif not _schema_matches(config, normalizers_metadata):
                issues.append(f"Los normalizadores no coinciden con el esquema activo: {normalizers_path}")
            else:
                universe_ok, detail = _universe_semantically_valid(normalizers_metadata)
                if not universe_ok:
                    issues.append(f"Los normalizadores provienen de un universo semanticamente invalido: {detail}")
                elif detail:
                    warnings.append(f"Los normalizadores provienen de un universo degradado permitido: {detail}")
    except Exception as exc:
        issues.append(f"Normalizadores invalidos: {normalizers_path} ({exc})")

    try:
        ensure_relative_to(dataset_path, _resolve_dataset_base_dir(config, dataset_path))
        if not dataset_path.exists():
            issues.append(f"Falta el dataset procesado de entrenamiento: {dataset_path}")
        else:
            verify_checksum(dataset_path, required=artifacts_required)
            dataset_metadata = read_metadata(dataset_path)
            if not dataset_metadata:
                issues.append(f"El dataset procesado no tiene metadatos: {dataset_path}")
            elif not _schema_matches(config, dataset_metadata):
                issues.append(f"El dataset procesado no coincide con el esquema activo: {dataset_path}")
            else:
                _load_processed_dataset(dataset_path)
                universe_ok, detail = _universe_semantically_valid(dataset_metadata)
                if not universe_ok:
                    issues.append(f"El dataset procesado proviene de un universo semanticamente invalido: {detail}")
                elif detail:
                    warnings.append(f"El dataset procesado proviene de un universo degradado permitido: {detail}")
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
        summary=(
            "El modelo tiene checkpoint, normalizadores y dataset procesado compatibles."
            if not warnings
            else "El modelo es utilizable, pero fue entrenado con un universo degradado explicitamente permitido."
        ),
        issues=[],
    )
