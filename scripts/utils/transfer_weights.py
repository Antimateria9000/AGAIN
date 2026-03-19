import logging
import os
import pickle
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.model import CustomTemporalFusionTransformer
from scripts.utils.data_schema import KNOWN_CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_schema_hash

logger = logging.getLogger(__name__)



def _load_normalizer_payload(normalizers_path: Path):
    with open(normalizers_path, 'rb') as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and 'normalizers' in payload:
        return payload['normalizers'], payload.get('metadata')
    return payload, None



def transfer_weights(old_checkpoint_path: str, new_model: CustomTemporalFusionTransformer, config: dict, normalizers_path: Path, device: str = 'cpu') -> tuple[CustomTemporalFusionTransformer, dict]:
    try:
        old_checkpoint = torch.load(old_checkpoint_path, map_location=device, weights_only=False)
        old_state_dict = old_checkpoint['state_dict']
        logger.info(f"Checkpoint origen cargado desde {old_checkpoint_path}")
    except Exception as e:
        logger.error(f"Error al cargar el checkpoint origen: {e}")
        raise

    new_state_dict = new_model.state_dict()
    transferred_state_dict = {}
    transferred_keys = 0
    total_keys = len(new_state_dict)

    for key in new_state_dict.keys():
        if key in old_state_dict and old_state_dict[key].shape == new_state_dict[key].shape:
            transferred_state_dict[key] = old_state_dict[key]
            transferred_keys += 1
        else:
            transferred_state_dict[key] = new_state_dict[key]
            if key in old_state_dict:
                logger.warning(f"Se omite {key} por incompatibilidad de dimensiones")
            else:
                logger.info(f"{key} no existe en el checkpoint origen; se deja la inicializacion nueva")

    logger.info(f"Se han transferido {transferred_keys} de {total_keys} tensores ({(transferred_keys / total_keys) * 100:.2f}%)")
    new_model.load_state_dict(transferred_state_dict)

    models_dir = Path(config['paths']['models_dir'])
    old_normalizers_path = models_dir / 'normalizers' / f"{Path(old_checkpoint_path).stem}_normalizers.pkl"
    if not old_normalizers_path.exists():
        raise FileNotFoundError(f"No existe el fichero de normalizadores {old_normalizers_path}")

    old_normalizers, old_metadata = _load_normalizer_payload(old_normalizers_path)
    required_numeric = set(NUMERIC_FEATURES)
    missing_numeric = required_numeric - set(old_normalizers.keys())
    if missing_numeric:
        raise ValueError(f"Los normalizadores origen no cubren el esquema numerico actual: {sorted(missing_numeric)}")

    if old_metadata is not None:
        expected_hash = build_schema_hash(config, old_metadata.get('numeric_features'), old_metadata.get('known_categoricals', KNOWN_CATEGORICAL_FEATURES))
        if old_metadata.get('schema_hash') != expected_hash:
            raise ValueError('Los normalizadores origen no son compatibles con la configuracion actual')
    else:
        logger.warning('Los normalizadores origen no incluyen metadatos de esquema; se valida solo por claves.')

    normalizers_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old_normalizers_path, normalizers_path)
    logger.info(f"Normalizadores compatibles copiados a {normalizers_path}")

    config['paths']['model_save_path'] = str(Path(config['paths']['models_dir']) / f"{config['model_name']}.pth")
    checkpoint = {
        'state_dict': new_model.state_dict(),
        'hyperparams': dict(new_model.hparams),
        'metadata': old_checkpoint.get('metadata'),
    }
    torch.save(checkpoint, config['paths']['model_save_path'])
    logger.info(f"Modelo con pesos transferidos guardado en {config['paths']['model_save_path']}")
    return new_model, config
