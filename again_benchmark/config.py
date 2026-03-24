from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkModuleConfig:
    storage_root: Path
    benchmark_version: int = 1

    @classmethod
    def from_again_config(cls, config: dict) -> "BenchmarkModuleConfig":
        config_path = Path(str(config.get("_meta", {}).get("config_path", "config/config.yaml"))).resolve()
        repo_root = config_path.parents[1]
        storage_root = Path(str(config.get("paths", {}).get("benchmark_storage_dir", repo_root / "benchmarks"))).resolve()
        return cls(storage_root=storage_root)
