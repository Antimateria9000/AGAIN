from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import torch


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256")


def metadata_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.meta.json")


def write_checksum(path: Path) -> str:
    checksum = sha256_file(path)
    checksum_file = checksum_path(path)
    ensure_parent_dir(checksum_file)
    checksum_file.write_text(f"{checksum}  {path.name}\n", encoding="utf-8")
    return checksum


def verify_checksum(path: Path, required: bool = True) -> str | None:
    checksum_file = checksum_path(path)
    if not checksum_file.exists():
        if required:
            raise FileNotFoundError(f"No existe el checksum de {path}")
        return None

    expected = checksum_file.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(path)
    if expected != actual:
        raise ValueError(f"Checksum invalido para {path}")
    return actual


def write_metadata(path: Path, metadata: dict) -> Path:
    sidecar = metadata_path(path)
    ensure_parent_dir(sidecar)
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar


def read_metadata(path: Path) -> dict | None:
    sidecar = metadata_path(path)
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text(encoding="utf-8"))


def write_json_artifact(path: Path, payload: dict) -> Path:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_json_artifact(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_relative_to(path: Path, base_dir: Path) -> None:
    resolved_path = path.resolve()
    resolved_base = base_dir.resolve()
    if resolved_base not in resolved_path.parents and resolved_path != resolved_base:
        raise ValueError(f"La ruta {resolved_path} no esta dentro de {resolved_base}")


_SAFE_GLOBAL_ALLOWED_PREFIXES = (
    "numpy.",
    "pandas.",
    "pytorch_forecasting.",
    "sklearn.preprocessing.",
)
_SAFE_GLOBAL_ALLOWED_NAMES = {
    "builtins.slice",
}


def _resolve_default_safe_globals() -> list[object]:
    resolved: list[object] = []
    try:
        import numpy as np
    except Exception:
        return resolved

    resolved.extend([np.dtype, np.ndarray])
    dtype_module = getattr(np, "dtypes", None)
    if dtype_module is not None:
        resolved.extend(value for value in vars(dtype_module).values() if isinstance(value, type))
    return resolved


def _resolve_global_reference(reference: str):
    module_name, _, attribute_path = reference.rpartition(".")
    if not module_name or not attribute_path:
        return None
    try:
        current = importlib.import_module(module_name)
    except Exception:
        return None
    for attribute in attribute_path.split("."):
        if not hasattr(current, attribute):
            return None
        current = getattr(current, attribute)
    return current


def _resolve_checkpoint_safe_globals(path: Path) -> list[object]:
    getter = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if getter is None:
        return []

    resolved: list[object] = []
    for reference in getter(path):
        if reference not in _SAFE_GLOBAL_ALLOWED_NAMES and not reference.startswith(_SAFE_GLOBAL_ALLOWED_PREFIXES):
            continue
        global_object = _resolve_global_reference(reference)
        if global_object is not None:
            resolved.append(global_object)
    return resolved


def load_trusted_torch_artifact(path: Path, trusted_types: list[type] | None = None):
    trusted_types = trusted_types or []
    safe_globals_context = getattr(torch.serialization, "safe_globals", None)
    load_kwargs = {"map_location": "cpu", "weights_only": True}
    if safe_globals_context is not None and trusted_types:
        try:
            safe_globals = list(
                dict.fromkeys(
                    [
                        *trusted_types,
                        *_resolve_default_safe_globals(),
                        *_resolve_checkpoint_safe_globals(path),
                    ]
                )
            )
            with safe_globals_context(safe_globals):
                return torch.load(path, **load_kwargs)
        except Exception as exc:
            trusted_names = ", ".join(sorted({trusted_type.__name__ for trusted_type in trusted_types})) or "sin tipos"
            raise ValueError(
                f"No se ha podido cargar el artefacto {path} con una whitelist segura de tipos [{trusted_names}]"
            ) from exc
    try:
        return torch.load(path, **load_kwargs)
    except Exception as exc:
        raise ValueError(f"No se ha podido cargar el artefacto {path} de forma segura") from exc

