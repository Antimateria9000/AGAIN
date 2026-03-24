from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from again_econ.contracts import (
    CapitalCompetitionPolicy,
    PositionSizingPolicy,
    SchedulingPolicy,
    WindowClosePolicy,
)
from again_econ.errors import BacktestConfigurationError


@dataclass(frozen=True)
class WalkforwardConfig:
    train_size: int
    test_size: int
    step_size: int | None = None
    lookahead_bars: int = 1
    execution_lag_bars: int = 1
    close_policy: WindowClosePolicy = WindowClosePolicy.ADMINISTRATIVE_CLOSE_ON_LAST_BAR
    calendar_name: str = "daily_default"

    def __post_init__(self) -> None:
        if self.train_size <= 0:
            raise BacktestConfigurationError("walkforward.train_size debe ser > 0")
        if self.test_size <= 0:
            raise BacktestConfigurationError("walkforward.test_size debe ser > 0")
        resolved_step = self.step_size if self.step_size is not None else self.test_size
        if resolved_step <= 0:
            raise BacktestConfigurationError("walkforward.step_size debe ser > 0")
        if self.lookahead_bars <= 0:
            raise BacktestConfigurationError("walkforward.lookahead_bars debe ser > 0")
        if self.execution_lag_bars <= 0:
            raise BacktestConfigurationError("walkforward.execution_lag_bars debe ser > 0")
        if self.lookahead_bars < self.execution_lag_bars:
            raise BacktestConfigurationError("walkforward.lookahead_bars no puede ser menor que execution_lag_bars")
        if not self.calendar_name:
            raise BacktestConfigurationError("walkforward.calendar_name es obligatorio")
        object.__setattr__(self, "step_size", resolved_step)


@dataclass(frozen=True)
class SignalConfig:
    long_threshold: float = 0.0
    translation_policy_name: str = "long_only_threshold"
    translation_policy_version: str = "v1"

    def __post_init__(self) -> None:
        if not isinstance(self.long_threshold, (int, float)):
            raise BacktestConfigurationError("signal.long_threshold debe ser numerico")
        if not self.translation_policy_name:
            raise BacktestConfigurationError("signal.translation_policy_name es obligatorio")
        if not self.translation_policy_version:
            raise BacktestConfigurationError("signal.translation_policy_version es obligatorio")


@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 10_000.0
    allocation_fraction: float = 1.0
    slippage_bps: float = 0.0
    commission_rate: float = 0.0
    commission_per_order: float = 0.0
    allow_fractional_shares: bool = False
    bars_per_year: int = 252
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.NEXT_AVAILABLE_OPEN
    scheduling_policy_version: str = "v1"
    capital_competition_policy: CapitalCompetitionPolicy = CapitalCompetitionPolicy.INSTRUMENT_ASC
    capital_competition_policy_version: str = "v1"
    sizing_policy: PositionSizingPolicy = PositionSizingPolicy.ALLOCATION_FRACTION_OF_CASH
    sizing_policy_version: str = "v1"

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
        if not self.scheduling_policy_version:
            raise BacktestConfigurationError("execution.scheduling_policy_version es obligatorio")
        if not self.capital_competition_policy_version:
            raise BacktestConfigurationError("execution.capital_competition_policy_version es obligatorio")
        if not self.sizing_policy_version:
            raise BacktestConfigurationError("execution.sizing_policy_version es obligatorio")


@dataclass(frozen=True)
class ProviderConfig:
    name: str = "direct_input"
    version: str = "v1"

    def __post_init__(self) -> None:
        if not self.name:
            raise BacktestConfigurationError("provider.name es obligatorio")
        if not self.version:
            raise BacktestConfigurationError("provider.version es obligatorio")


@dataclass(frozen=True)
class ManifestConfig:
    seed: int | None = None
    command: str | None = None
    code_commit_sha: str | None = None
    artifact_policy_name: str = "references_only"
    artifact_policy_version: str = "v1"

    def __post_init__(self) -> None:
        if self.command is not None and not self.command:
            raise BacktestConfigurationError("manifest.command no puede ser vacio")
        if self.code_commit_sha is not None and not self.code_commit_sha:
            raise BacktestConfigurationError("manifest.code_commit_sha no puede ser vacio")
        if not self.artifact_policy_name:
            raise BacktestConfigurationError("manifest.artifact_policy_name es obligatorio")
        if not self.artifact_policy_version:
            raise BacktestConfigurationError("manifest.artifact_policy_version es obligatorio")


@dataclass(frozen=True)
class BacktestConfig:
    walkforward: WalkforwardConfig
    signal: SignalConfig = SignalConfig()
    execution: ExecutionConfig = ExecutionConfig()
    provider: ProviderConfig = ProviderConfig()
    manifest: ManifestConfig = ManifestConfig()
    label: str = "again_econ_v2"

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "BacktestConfig":
        walkforward_section = mapping.get("walkforward") or {}
        signal_section = mapping.get("signal") or {}
        execution_section = mapping.get("execution") or {}
        provider_section = mapping.get("provider") or {}
        manifest_section = mapping.get("manifest") or {}
        label = str(mapping.get("label", "again_econ_v2"))
        return cls(
            walkforward=WalkforwardConfig(
                train_size=int(walkforward_section["train_size"]),
                test_size=int(walkforward_section["test_size"]),
                step_size=int(walkforward_section["step_size"])
                if walkforward_section.get("step_size") is not None
                else None,
                lookahead_bars=int(walkforward_section.get("lookahead_bars", 1)),
                execution_lag_bars=int(walkforward_section.get("execution_lag_bars", 1)),
                close_policy=WindowClosePolicy(
                    str(
                        walkforward_section.get(
                            "close_policy",
                            WindowClosePolicy.ADMINISTRATIVE_CLOSE_ON_LAST_BAR.value,
                        )
                    )
                ),
                calendar_name=str(walkforward_section.get("calendar_name", "daily_default")),
            ),
            signal=SignalConfig(
                long_threshold=float(signal_section.get("long_threshold", 0.0)),
                translation_policy_name=str(signal_section.get("translation_policy_name", "long_only_threshold")),
                translation_policy_version=str(signal_section.get("translation_policy_version", "v1")),
            ),
            execution=ExecutionConfig(
                initial_cash=float(execution_section.get("initial_cash", 10_000.0)),
                allocation_fraction=float(execution_section.get("allocation_fraction", 1.0)),
                slippage_bps=float(execution_section.get("slippage_bps", 0.0)),
                commission_rate=float(execution_section.get("commission_rate", 0.0)),
                commission_per_order=float(execution_section.get("commission_per_order", 0.0)),
                allow_fractional_shares=bool(execution_section.get("allow_fractional_shares", False)),
                bars_per_year=int(execution_section.get("bars_per_year", 252)),
                scheduling_policy=SchedulingPolicy(
                    str(execution_section.get("scheduling_policy", SchedulingPolicy.NEXT_AVAILABLE_OPEN.value))
                ),
                scheduling_policy_version=str(execution_section.get("scheduling_policy_version", "v1")),
                capital_competition_policy=CapitalCompetitionPolicy(
                    str(
                        execution_section.get(
                            "capital_competition_policy",
                            CapitalCompetitionPolicy.INSTRUMENT_ASC.value,
                        )
                    )
                ),
                capital_competition_policy_version=str(
                    execution_section.get("capital_competition_policy_version", "v1")
                ),
                sizing_policy=PositionSizingPolicy(
                    str(
                        execution_section.get(
                            "sizing_policy",
                            PositionSizingPolicy.ALLOCATION_FRACTION_OF_CASH.value,
                        )
                    )
                ),
                sizing_policy_version=str(execution_section.get("sizing_policy_version", "v1")),
            ),
            provider=ProviderConfig(
                name=str(provider_section.get("name", "direct_input")),
                version=str(provider_section.get("version", "v1")),
            ),
            manifest=ManifestConfig(
                seed=int(manifest_section["seed"]) if manifest_section.get("seed") is not None else None,
                command=str(manifest_section["command"]) if manifest_section.get("command") is not None else None,
                code_commit_sha=str(manifest_section["code_commit_sha"])
                if manifest_section.get("code_commit_sha") is not None
                else None,
                artifact_policy_name=str(manifest_section.get("artifact_policy_name", "references_only")),
                artifact_policy_version=str(manifest_section.get("artifact_policy_version", "v1")),
            ),
            label=label,
        )


def load_backtest_config(path: str | Path) -> BacktestConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise BacktestConfigurationError("El archivo de configuracion economica debe contener un mapping YAML")
    return BacktestConfig.from_mapping(payload)
