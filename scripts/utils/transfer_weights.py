from __future__ import annotations

import logging
import pickle
from pathlib import Path

import torch

from scripts.model import CustomTemporalFusionTransformer
from scripts.utils.artifact_utils import ensure_relative_to, verify_checksum, write_checksum, write_metadata
from scripts.utils.data_schema import REQUIRED_NORMALIZER_KEYS, build_artifact_metadata, metadata_matches_active_schema
from scripts.utils.repo_layout import resolve_repo_path

logger = logging.getLogger(__name__)



def _load_normalizer_payload(normalizers_path: Path):
    verify_checksum(normalizers_path, required=False)
    with open(normalizers_path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "normalizers" in payload:
        return payload["normalizers"], payload.get("metadata")
    return payload, None



def transfer_weights(old_checkpoint_path: str, new_model: CustomTemporalFusionTransformer, config: dict, normalizers_path: Path, device: str = "cpu") -> tuple[CustomTemporalFusionTransformer, dict]:
    old_checkpoint_path = resolve_repo_path(config, old_checkpoint_path)
    ensure_relative_to(old_checkpoint_path, resolve_repo_path(config, config["paths"]["models_dir"]))
    verify_checksum(old_checkpoint_path, required=config["artifacts"]["require_hash_validation"])

    old_checkpoint = torch.load(old_checkpoint_path, map_location=device, weights_only=False)
    old_state_dict = old_checkpoint["state_dict"]
    new_state_dict = new_model.state_dict()
    transferred_state_dict = {}
    transferred_keys = 0

    for key in new_state_dict.keys():
        if key in old_state_dict and old_state_dict[key].shape == new_state_dict[key].shape:
            transferred_state_dict[key] = old_state_dict[key]
            transferred_keys += 1
        else:
            transferred_state_dict[key] = new_state_dict[key]

    logger.info("Se han transferido %s de %s tensores", transferred_keys, len(new_state_dict))
    new_model.load_state_dict(transferred_state_dict)

    normalizers_path = Path(normalizers_path)
    ensure_relative_to(normalizers_path, resolve_repo_path(config, config["paths"]["normalizers_dir"]))
    if not normalizers_path.exists():
        raise FileNotFoundError(f"No existe el fichero de normalizadores destino {normalizers_path}")
    verify_checksum(normalizers_path, required=config["artifacts"]["require_hash_validation"])
    destination_normalizers, destination_metadata = _load_normalizer_payload(normalizers_path)
    missing_numeric = set(REQUIRED_NORMALIZER_KEYS) - set(destination_normalizers.keys())
    if missing_numeric:
        raise ValueError(f"Los normalizadores destino no cubren el esquema numerico actual: {sorted(missing_numeric)}")

    if destination_metadata is None or not metadata_matches_active_schema(config, destination_metadata):
        raise ValueError("Los normalizadores destino no son compatibles con la configuracion actual")

    config["paths"]["model_save_path"] = str(resolve_repo_path(config, config["paths"]["models_dir"]) / f"{config['model_name']}.pth")
    transfer_metadata = build_artifact_metadata(
        config,
        extra={
            "transfer_learning": {
                "source_checkpoint": old_checkpoint_path.name,
                "source_model_name": old_checkpoint.get("metadata", {}).get("model_name"),
            }
        },
    )
    checkpoint = {
        "state_dict": new_model.state_dict(),
        "hyperparams": dict(new_model.hparams),
        "metadata": transfer_metadata,
    }
    checkpoint_path = resolve_repo_path(config, config["paths"]["model_save_path"])
    torch.save(checkpoint, checkpoint_path)
    write_metadata(checkpoint_path, checkpoint["metadata"])
    write_checksum(checkpoint_path)
    logger.info("Modelo con pesos transferidos guardado en %s", checkpoint_path)
    return new_model, config
