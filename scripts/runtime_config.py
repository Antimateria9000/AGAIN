import logging
import pickle
from pathlib import Path

import yaml

from scripts.utils.config_validation import validate_config_schema

logger = logging.getLogger(__name__)


class ConfigManager:
    _instance = None

    def __new__(cls, config_path: str = "config/config.yaml"):
        requested_path = Path(config_path)
        if cls._instance is None or cls._instance.config_path != requested_path:
            cls._instance = super().__new__(cls)
            cls._instance.config_path = requested_path
            cls._instance.config = cls._instance._load_config()
            cls._instance._last_normalizers_metadata = None
        return cls._instance

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            validate_config_schema(config)
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

    def load_normalizers(self, model_name: str) -> dict:
        normalizers_path = Path(self.get("paths.normalizers_dir")) / f"{model_name}_normalizers.pkl"
        if not normalizers_path.exists():
            self._last_normalizers_metadata = None
            logger.warning(f"No existe el archivo de normalizadores {normalizers_path}")
            return {}

        try:
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
            return {}

    def get_last_normalizers_metadata(self):
        return self._last_normalizers_metadata

    def save_normalizers(self, model_name: str, normalizers: dict, metadata: dict | None = None, overwrite: bool = True):
        normalizers_path = Path(self.get("paths.normalizers_dir")) / f"{model_name}_normalizers.pkl"
        normalizers_path.parent.mkdir(parents=True, exist_ok=True)
        if normalizers_path.exists() and not overwrite:
            logger.warning(f"Los normalizadores de {model_name} ya existen en {normalizers_path}. No se sobrescriben.")
            return

        try:
            with open(normalizers_path, "wb") as f:
                pickle.dump({"normalizers": normalizers, "metadata": metadata or {}}, f)
            logger.info(f"Normalizadores guardados en {normalizers_path}")
        except Exception as e:
            logger.error(f"Error al guardar normalizadores: {e}")
