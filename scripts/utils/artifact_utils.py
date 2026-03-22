from __future__ import annotations

import hashlib
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


def load_trusted_torch_artifact(path: Path, trusted_types: list[type] | None = None):
    trusted_types = trusted_types or []
    safe_globals_context = getattr(torch.serialization, "safe_globals", None)
    if safe_globals_context is not None and trusted_types:
        try:
            with safe_globals_context(trusted_types):
                return torch.load(path, map_location="cpu")
        except Exception:
            pass
    return torch.load(path, map_location="cpu", weights_only=False)

