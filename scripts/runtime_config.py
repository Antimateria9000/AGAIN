import logging
import pickle
import os
from pathlib import Path

import yaml

from scripts.utils.artifact_utils import ensure_relative_to, verify_checksum, write_checksum
from scripts.utils.config_validation import apply_runtime_defaults, validate_config_schema

logger = logging.getLogger(__name__)


class ConfigManager:
    def __init__(self, config_path: str | None = None):
        requested_path = Path(config_path or os.environ.get("PREDICTOR_CONFIG_PATH", "config/config.yaml"))
        self.config_path = requested_path
        self.config = self._load_config()
        self._last_normalizers_metadata = None

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            validate_config_schema(config)
            config = apply_runtime_defaults(config)
            config.setdefault("_meta", {})
            config["_meta"]["config_path"] = str(self.config_path)
            logger.info(f"Configuracion cargada desde {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"El archivo de configuracion {self.config_path} no existe")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error al parsear config.yaml: {e}")
            raise

    def get(self, key: str):
        keys = key.split(".")
        value = self.config
        try:
            for single_key in keys:
                value = value[single_key]
            return value
        except (KeyError, TypeError):
            logger.error(f"No existe la clave '{key}' en la configuracion")
            raise KeyError(f"No existe la clave '{key}' en la configuracion")

    def _normalizers_path(self, model_name: str) -> Path:
        return Path(self.get("paths.normalizers_dir")) / f"{model_name}_normalizers.pkl"

    def load_normalizers(self, model_name: str, required: bool = False) -> dict:
        normalizers_path = Path(self.get("paths.normalizers_dir")) / f"{model_name}_normalizers.pkl"
        if not normalizers_path.exists():
            self._last_normalizers_metadata = None
            if required:
                raise FileNotFoundError(f"No existe el archivo de normalizadores {normalizers_path}")
            logger.warning(f"No existe el archivo de normalizadores {normalizers_path}")
            return {}

        ensure_relative_to(normalizers_path, Path(self.get("paths.normalizers_dir")))
        try:
            verify_checksum(normalizers_path, required=self.get("artifacts.require_hash_validation"))
            with open(normalizers_path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and "normalizers" in payload:
                self._last_normalizers_metadata = payload.get("metadata")
                normalizers = payload["normalizers"]
            else:
                self._last_normalizers_metadata = None
                normalizers = payload
            logger.info(f"Normalizadores cargados desde {normalizers_path}")
            return normalizers
        except Exception as e:
            logger.error(f"Error al cargar normalizadores: {e}")
            if required:
                raise
            return {}

    def get_last_normalizers_metadata(self):
        return self._last_normalizers_metadata

    def save_normalizers(self, model_name: str, normalizers: dict, metadata: dict | None = None, overwrite: bool = True):
        normalizers_path = self._normalizers_path(model_name)
        normalizers_path.parent.mkdir(parents=True, exist_ok=True)
        if normalizers_path.exists() and not overwrite:
            logger.warning(f"Los normalizadores de {model_name} ya existen en {normalizers_path}. No se sobrescriben.")
            return

        try:
            with open(normalizers_path, "wb") as f:
                pickle.dump({"normalizers": normalizers, "metadata": metadata or {}}, f)
            write_checksum(normalizers_path)
            logger.info(f"Normalizadores guardados en {normalizers_path}")
        except Exception as e:
            logger.error(f"Error al guardar normalizadores: {e}")
            raise
