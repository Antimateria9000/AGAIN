from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path

import torch

from scripts.model import CustomTemporalFusionTransformer
from scripts.utils.artifact_utils import ensure_relative_to, verify_checksum, write_checksum, write_metadata
from scripts.utils.data_schema import REQUIRED_NORMALIZER_KEYS, metadata_matches_active_schema

logger = logging.getLogger(__name__)



def _load_normalizer_payload(normalizers_path: Path):
    verify_checksum(normalizers_path, required=False)
    with open(normalizers_path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "normalizers" in payload:
        return payload["normalizers"], payload.get("metadata")
    return payload, None



def transfer_weights(old_checkpoint_path: str, new_model: CustomTemporalFusionTransformer, config: dict, normalizers_path: Path, device: str = "cpu") -> tuple[CustomTemporalFusionTransformer, dict]:
    old_checkpoint_path = Path(old_checkpoint_path)
    ensure_relative_to(old_checkpoint_path, Path(config["paths"]["models_dir"]))
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

    models_dir = Path(config["paths"]["models_dir"])
    old_normalizers_path = models_dir / "normalizers" / f"{old_checkpoint_path.stem}_normalizers.pkl"
    if not old_normalizers_path.exists():
        raise FileNotFoundError(f"No existe el fichero de normalizadores {old_normalizers_path}")

    old_normalizers, old_metadata = _load_normalizer_payload(old_normalizers_path)
    missing_numeric = set(REQUIRED_NORMALIZER_KEYS) - set(old_normalizers.keys())
    if missing_numeric:
        raise ValueError(f"Los normalizadores origen no cubren el esquema numerico actual: {sorted(missing_numeric)}")

    if old_metadata is not None:
        if not metadata_matches_active_schema(config, old_metadata):
            raise ValueError("Los normalizadores origen no son compatibles con la configuracion actual")

    normalizers_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old_normalizers_path, normalizers_path)
    old_checksum = old_normalizers_path.with_name(f"{old_normalizers_path.name}.sha256")
    if old_checksum.exists():
        shutil.copy2(old_checksum, normalizers_path.with_name(f"{normalizers_path.name}.sha256"))
    else:
        write_checksum(normalizers_path)

    config["paths"]["model_save_path"] = str(Path(config["paths"]["models_dir"]) / f"{config['model_name']}.pth")
    checkpoint = {
        "state_dict": new_model.state_dict(),
        "hyperparams": dict(new_model.hparams),
        "metadata": old_checkpoint.get("metadata"),
    }
    checkpoint_path = Path(config["paths"]["model_save_path"])
    torch.save(checkpoint, checkpoint_path)
    if checkpoint.get("metadata"):
        write_metadata(checkpoint_path, checkpoint["metadata"])
    write_checksum(checkpoint_path)
    logger.info("Modelo con pesos transferidos guardado en %s", checkpoint_path)
    return new_model, config
