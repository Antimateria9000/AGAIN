from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.utils.repo_layout import resolve_repo_path


@dataclass(frozen=True)
class BenchmarkModuleConfig:
    storage_root: Path
    benchmark_version: int = 1

    @classmethod
    def from_again_config(cls, config: dict) -> "BenchmarkModuleConfig":
        storage_root = resolve_repo_path(
            config,
            str(config.get("paths", {}).get("benchmark_storage_dir", "artifacts/benchmarks")),
        )
        return cls(storage_root=storage_root)
