from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from typing import Any

from again_econ.errors import ContractValidationError


class TargetKind(str, Enum):
    PRICE = "price"
    RETURN = "return"
    DIRECTION = "direction"
    SCORE = "score"
    SIGNAL = "signal"


class PositionTarget(str, Enum):
    FLAT = "flat"
    LONG = "long"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class MarketBar:
    instrument_id: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ContractValidationError("MarketBar.instrument_id es obligatorio")
        numeric_values = {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }
        for name, value in numeric_values.items():
            if not math.isfinite(float(value)):
                raise ContractValidationError(f"MarketBar.{name} debe ser finito")
        if min(self.open, self.high, self.low, self.close) <= 0.0:
            raise ContractValidationError("Los precios OHLC deben ser > 0")
        if self.volume < 0.0:
            raise ContractValidationError("MarketBar.volume no puede ser negativo")
        if self.high < max(self.open, self.low, self.close):
            raise ContractValidationError("MarketBar.high es incoherente con OHLC")
        if self.low > min(self.open, self.high, self.close):
            raise ContractValidationError("MarketBar.low es incoherente con OHLC")


@dataclass(frozen=True)
class MarketFrame:
    bars: tuple[MarketBar, ...]

    def __post_init__(self) -> None:
        if not self.bars:
            raise ContractValidationError("MarketFrame debe contener al menos una barra")

    def timestamps(self) -> tuple[datetime, ...]:
        return tuple(sorted({bar.timestamp for bar in self.bars}))

    def instruments(self) -> tuple[str, ...]:
        return tuple(sorted({bar.instrument_id for bar in self.bars}))

    def bars_for_instrument(self, instrument_id: str) -> tuple[MarketBar, ...]:
        return tuple(sorted((bar for bar in self.bars if bar.instrument_id == instrument_id), key=lambda bar: bar.timestamp))

    def bars_at(self, timestamp: datetime) -> tuple[MarketBar, ...]:
        return tuple(sorted((bar for bar in self.bars if bar.timestamp == timestamp), key=lambda bar: bar.instrument_id))

    def bar_for(self, instrument_id: str, timestamp: datetime) -> MarketBar | None:
        for bar in self.bars:
            if bar.instrument_id == instrument_id and bar.timestamp == timestamp:
                return bar
        return None

    def slice_between(self, start: datetime, end: datetime) -> "MarketFrame":
        sliced = tuple(bar for bar in self.bars if start <= bar.timestamp <= end)
        if not sliced:
            raise ContractValidationError("El recorte de MarketFrame no contiene barras")
        return MarketFrame(bars=sliced)


@dataclass(frozen=True)
class WalkforwardWindow:
    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass(frozen=True)
class ForecastRecord:
    instrument_id: str
    decision_timestamp: datetime
    available_at: datetime
    target_kind: TargetKind
    value: float
    reference_value: float | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalRecord:
    instrument_id: str
    decision_timestamp: datetime
    available_at: datetime
    target_state: PositionTarget
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScheduledSignal:
    signal: SignalRecord
    execution_timestamp: datetime


@dataclass(frozen=True)
class OrderIntent:
    instrument_id: str
    side: OrderSide
    decision_timestamp: datetime
    execution_timestamp: datetime
    quantity: float


@dataclass(frozen=True)
class FillEvent:
    instrument_id: str
    side: OrderSide
    decision_timestamp: datetime
    execution_timestamp: datetime
    price: float
    quantity: float
    gross_notional: float
    fee: float
    slippage_bps: float


@dataclass(frozen=True)
class TradeRecord:
    instrument_id: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    quantity: float
    entry_fee: float
    exit_fee: float
    gross_pnl: float
    net_pnl: float


@dataclass(frozen=True)
class PositionState:
    instrument_id: str
    quantity: float
    avg_entry_price: float
    entry_timestamp: datetime
    entry_fee: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    timestamp: datetime
    cash: float
    market_value: float
    total_equity: float
    realized_pnl: float
    unrealized_pnl: float
    open_positions: int


@dataclass(frozen=True)
class MetricBundle:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    trade_count: int
    win_rate: float
    exposure_ratio: float


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    label: str
    adapter_name: str
    config_fingerprint: str
    market_fingerprint: str
    input_fingerprint: str
    window_count: int
    input_reference: str | None = None


@dataclass(frozen=True)
class WindowResult:
    window: WalkforwardWindow
    fills: tuple[FillEvent, ...]
    trades: tuple[TradeRecord, ...]
    snapshots: tuple[PortfolioSnapshot, ...]
    metrics: MetricBundle


@dataclass(frozen=True)
class BacktestResult:
    manifest: RunManifest
    windows: tuple[WindowResult, ...]
    summary_metrics: MetricBundle


@dataclass(frozen=True)
class InputBundle:
    adapter_name: str
    forecasts: tuple[ForecastRecord, ...] = ()
    signals: tuple[SignalRecord, ...] = ()
    source_path: str | None = None
