from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from again_econ.errors import BacktestConfigurationError


@dataclass(frozen=True)
class WalkforwardConfig:
    train_size: int
    test_size: int
    step_size: int | None = None

    def __post_init__(self) -> None:
        if self.train_size <= 0:
            raise BacktestConfigurationError("walkforward.train_size debe ser > 0")
        if self.test_size <= 0:
            raise BacktestConfigurationError("walkforward.test_size debe ser > 0")
        resolved_step = self.step_size if self.step_size is not None else self.test_size
        if resolved_step <= 0:
            raise BacktestConfigurationError("walkforward.step_size debe ser > 0")
        object.__setattr__(self, "step_size", resolved_step)


@dataclass(frozen=True)
class SignalConfig:
    long_threshold: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.long_threshold, (int, float)):
            raise BacktestConfigurationError("signal.long_threshold debe ser numerico")


@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 10_000.0
    allocation_fraction: float = 1.0
    slippage_bps: float = 0.0
    commission_rate: float = 0.0
    commission_per_order: float = 0.0
    allow_fractional_shares: bool = False
    bars_per_year: int = 252

    def __post_init__(self) -> None:
        if self.initial_cash <= 0.0:
            raise BacktestConfigurationError("execution.initial_cash debe ser > 0")
        if not 0.0 < self.allocation_fraction <= 1.0:
            raise BacktestConfigurationError("execution.allocation_fraction debe estar en (0, 1]")
        if self.slippage_bps < 0.0:
            raise BacktestConfigurationError("execution.slippage_bps no puede ser negativo")
        if self.commission_rate < 0.0:
            raise BacktestConfigurationError("execution.commission_rate no puede ser negativo")
        if self.commission_per_order < 0.0:
            raise BacktestConfigurationError("execution.commission_per_order no puede ser negativo")
        if self.bars_per_year <= 0:
            raise BacktestConfigurationError("execution.bars_per_year debe ser > 0")


@dataclass(frozen=True)
class BacktestConfig:
    walkforward: WalkforwardConfig
    signal: SignalConfig = SignalConfig()
    execution: ExecutionConfig = ExecutionConfig()
    label: str = "again_econ_v1"

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "BacktestConfig":
        walkforward_section = mapping.get("walkforward") or {}
        signal_section = mapping.get("signal") or {}
        execution_section = mapping.get("execution") or {}
        label = str(mapping.get("label", "again_econ_v1"))
        return cls(
            walkforward=WalkforwardConfig(
                train_size=int(walkforward_section["train_size"]),
                test_size=int(walkforward_section["test_size"]),
                step_size=int(walkforward_section["step_size"])
                if walkforward_section.get("step_size") is not None
                else None,
            ),
            signal=SignalConfig(long_threshold=float(signal_section.get("long_threshold", 0.0))),
            execution=ExecutionConfig(
                initial_cash=float(execution_section.get("initial_cash", 10_000.0)),
                allocation_fraction=float(execution_section.get("allocation_fraction", 1.0)),
                slippage_bps=float(execution_section.get("slippage_bps", 0.0)),
                commission_rate=float(execution_section.get("commission_rate", 0.0)),
                commission_per_order=float(execution_section.get("commission_per_order", 0.0)),
                allow_fractional_shares=bool(execution_section.get("allow_fractional_shares", False)),
                bars_per_year=int(execution_section.get("bars_per_year", 252)),
            ),
            label=label,
        )


def load_backtest_config(path: str | Path) -> BacktestConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise BacktestConfigurationError("El archivo de configuracion economica debe contener un mapping YAML")
    return BacktestConfig.from_mapping(payload)
