from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any

from again_econ.errors import ContractValidationError


def normalize_naive_utc_timestamp(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ContractValidationError(f"{field_name} debe ser datetime")
    if value.tzinfo is not None and value.utcoffset() is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _is_aware_timestamp(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _ensure_matching_timestamp_awareness(*, fields: dict[str, datetime]) -> None:
    awareness = {_is_aware_timestamp(value) for value in fields.values()}
    if len(awareness) > 1:
        names = ", ".join(sorted(fields))
        raise ContractValidationError(
            f"Los timestamps de {names} deben usar una politica temporal coherente: todos naive UTC o todos aware UTC"
        )


class TargetKind(str, Enum):
    PRICE = "price"
    RETURN = "return"
    DIRECTION = "direction"
    SCORE = "score"


class PositionTarget(str, Enum):
    FLAT = "flat"
    LONG = "long"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class ExecutionReason(str, Enum):
    SIGNAL = "signal"
    WINDOW_CLOSE = "window_close"


class DiscardReason(str, Enum):
    NEXT_OPEN_OUTSIDE_WINDOW = "next_open_outside_window"
    NO_NEXT_OPEN_AVAILABLE = "no_next_open_available"


class BundleProvenanceMode(str, Enum):
    LEGACY_DEGRADED = "legacy_degraded"
    STRICT_V2 = "strict_v2"


@dataclass(frozen=True)
class WindowProvenance:
    window_index: int
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def __post_init__(self) -> None:
        _ensure_matching_timestamp_awareness(
            fields={
                "train_end": self.train_end,
                "test_start": self.test_start,
                "test_end": self.test_end,
            }
        )
        object.__setattr__(self, "train_end", normalize_naive_utc_timestamp(self.train_end, field_name="train_end"))
        object.__setattr__(self, "test_start", normalize_naive_utc_timestamp(self.test_start, field_name="test_start"))
        object.__setattr__(self, "test_end", normalize_naive_utc_timestamp(self.test_end, field_name="test_end"))
        if self.window_index < 0:
            raise ContractValidationError("window_index no puede ser negativo")
        if not (self.train_end < self.test_start <= self.test_end):
            raise ContractValidationError("WindowProvenance requiere train_end < test_start <= test_end")


@dataclass(frozen=True)
class BundleProvenance:
    generated_at: datetime
    model_run_id: str
    data_fingerprint: str
    code_fingerprint: str
    window: WindowProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generated_at",
            normalize_naive_utc_timestamp(self.generated_at, field_name="generated_at"),
        )
        if not self.model_run_id:
            raise ContractValidationError("model_run_id es obligatorio en provenance v2")
        if not self.data_fingerprint:
            raise ContractValidationError("data_fingerprint es obligatorio en provenance v2")
        if not self.code_fingerprint:
            raise ContractValidationError("code_fingerprint es obligatorio en provenance v2")


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
        object.__setattr__(self, "timestamp", normalize_naive_utc_timestamp(self.timestamp, field_name="MarketBar.timestamp"))
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
        normalized = normalize_naive_utc_timestamp(timestamp, field_name="MarketFrame.timestamp")
        return tuple(sorted((bar for bar in self.bars if bar.timestamp == normalized), key=lambda bar: bar.instrument_id))

    def bar_for(self, instrument_id: str, timestamp: datetime) -> MarketBar | None:
        normalized = normalize_naive_utc_timestamp(timestamp, field_name="MarketFrame.timestamp")
        for bar in self.bars:
            if bar.instrument_id == instrument_id and bar.timestamp == normalized:
                return bar
        return None

    def slice_between(self, start: datetime, end: datetime) -> "MarketFrame":
        normalized_start = normalize_naive_utc_timestamp(start, field_name="MarketFrame.slice_start")
        normalized_end = normalize_naive_utc_timestamp(end, field_name="MarketFrame.slice_end")
        sliced = tuple(bar for bar in self.bars if normalized_start <= bar.timestamp <= normalized_end)
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

    def __post_init__(self) -> None:
        _ensure_matching_timestamp_awareness(
            fields={
                "train_start": self.train_start,
                "train_end": self.train_end,
                "test_start": self.test_start,
                "test_end": self.test_end,
            }
        )
        object.__setattr__(self, "train_start", normalize_naive_utc_timestamp(self.train_start, field_name="train_start"))
        object.__setattr__(self, "train_end", normalize_naive_utc_timestamp(self.train_end, field_name="train_end"))
        object.__setattr__(self, "test_start", normalize_naive_utc_timestamp(self.test_start, field_name="test_start"))
        object.__setattr__(self, "test_end", normalize_naive_utc_timestamp(self.test_end, field_name="test_end"))
        if self.index < 0:
            raise ContractValidationError("WalkforwardWindow.index no puede ser negativo")
        if not (self.train_start <= self.train_end < self.test_start <= self.test_end):
            raise ContractValidationError("WalkforwardWindow invalida")


@dataclass(frozen=True)
class ForecastRecord:
    instrument_id: str
    decision_timestamp: datetime
    available_at: datetime
    target_kind: TargetKind
    value: float
    reference_value: float | None = None
    score: float | None = None
    provenance: WindowProvenance | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_matching_timestamp_awareness(
            fields={
                "decision_timestamp": self.decision_timestamp,
                "available_at": self.available_at,
            }
        )
        object.__setattr__(
            self,
            "decision_timestamp",
            normalize_naive_utc_timestamp(self.decision_timestamp, field_name="decision_timestamp"),
        )
        object.__setattr__(self, "available_at", normalize_naive_utc_timestamp(self.available_at, field_name="available_at"))
        if not self.instrument_id:
            raise ContractValidationError("ForecastRecord.instrument_id es obligatorio")
        if not math.isfinite(float(self.value)):
            raise ContractValidationError("ForecastRecord.value debe ser finito")
        if self.reference_value is not None and not math.isfinite(float(self.reference_value)):
            raise ContractValidationError("ForecastRecord.reference_value debe ser finito")
        if self.score is not None and not math.isfinite(float(self.score)):
            raise ContractValidationError("ForecastRecord.score debe ser finito")
        if not isinstance(self.metadata, dict):
            raise ContractValidationError("ForecastRecord.metadata debe ser un dict")


@dataclass(frozen=True)
class SignalRecord:
    instrument_id: str
    decision_timestamp: datetime
    available_at: datetime
    target_state: PositionTarget
    score: float | None = None
    provenance: WindowProvenance | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_matching_timestamp_awareness(
            fields={
                "decision_timestamp": self.decision_timestamp,
                "available_at": self.available_at,
            }
        )
        object.__setattr__(
            self,
            "decision_timestamp",
            normalize_naive_utc_timestamp(self.decision_timestamp, field_name="decision_timestamp"),
        )
        object.__setattr__(self, "available_at", normalize_naive_utc_timestamp(self.available_at, field_name="available_at"))
        if not self.instrument_id:
            raise ContractValidationError("SignalRecord.instrument_id es obligatorio")
        if self.score is not None and not math.isfinite(float(self.score)):
            raise ContractValidationError("SignalRecord.score debe ser finito")
        if not isinstance(self.metadata, dict):
            raise ContractValidationError("SignalRecord.metadata debe ser un dict")


@dataclass(frozen=True)
class ScheduledSignal:
    signal: SignalRecord
    execution_timestamp: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_timestamp",
            normalize_naive_utc_timestamp(self.execution_timestamp, field_name="execution_timestamp"),
        )


@dataclass(frozen=True)
class OrderIntent:
    instrument_id: str
    side: OrderSide
    decision_timestamp: datetime
    execution_timestamp: datetime
    quantity: float
    reason: ExecutionReason = ExecutionReason.SIGNAL

    def __post_init__(self) -> None:
        _ensure_matching_timestamp_awareness(
            fields={
                "decision_timestamp": self.decision_timestamp,
                "execution_timestamp": self.execution_timestamp,
            }
        )
        object.__setattr__(
            self,
            "decision_timestamp",
            normalize_naive_utc_timestamp(self.decision_timestamp, field_name="decision_timestamp"),
        )
        object.__setattr__(
            self,
            "execution_timestamp",
            normalize_naive_utc_timestamp(self.execution_timestamp, field_name="execution_timestamp"),
        )
        if not self.instrument_id:
            raise ContractValidationError("OrderIntent.instrument_id es obligatorio")
        if self.quantity <= 0.0 or not math.isfinite(float(self.quantity)):
            raise ContractValidationError("OrderIntent.quantity debe ser > 0 y finita")


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
    reason: ExecutionReason

    def __post_init__(self) -> None:
        _ensure_matching_timestamp_awareness(
            fields={
                "decision_timestamp": self.decision_timestamp,
                "execution_timestamp": self.execution_timestamp,
            }
        )
        object.__setattr__(
            self,
            "decision_timestamp",
            normalize_naive_utc_timestamp(self.decision_timestamp, field_name="decision_timestamp"),
        )
        object.__setattr__(
            self,
            "execution_timestamp",
            normalize_naive_utc_timestamp(self.execution_timestamp, field_name="execution_timestamp"),
        )
        numeric_values = {
            "price": self.price,
            "quantity": self.quantity,
            "gross_notional": self.gross_notional,
            "fee": self.fee,
            "slippage_bps": self.slippage_bps,
        }
        for name, value in numeric_values.items():
            if not math.isfinite(float(value)):
                raise ContractValidationError(f"FillEvent.{name} debe ser finito")
        if self.price <= 0.0:
            raise ContractValidationError("FillEvent.price debe ser > 0")
        if self.quantity <= 0.0:
            raise ContractValidationError("FillEvent.quantity debe ser > 0")
        if self.gross_notional <= 0.0:
            raise ContractValidationError("FillEvent.gross_notional debe ser > 0")
        if self.fee < 0.0:
            raise ContractValidationError("FillEvent.fee no puede ser negativo")


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
    exit_reason: ExecutionReason

    def __post_init__(self) -> None:
        _ensure_matching_timestamp_awareness(
            fields={
                "entry_timestamp": self.entry_timestamp,
                "exit_timestamp": self.exit_timestamp,
            }
        )
        object.__setattr__(self, "entry_timestamp", normalize_naive_utc_timestamp(self.entry_timestamp, field_name="entry_timestamp"))
        object.__setattr__(self, "exit_timestamp", normalize_naive_utc_timestamp(self.exit_timestamp, field_name="exit_timestamp"))
        if self.quantity <= 0.0 or not math.isfinite(float(self.quantity)):
            raise ContractValidationError("TradeRecord.quantity debe ser > 0 y finita")
        if self.entry_price <= 0.0 or self.exit_price <= 0.0:
            raise ContractValidationError("TradeRecord entry/exit_price deben ser > 0")


@dataclass(frozen=True)
class PositionState:
    instrument_id: str
    quantity: float
    avg_entry_price: float
    entry_timestamp: datetime
    entry_fee: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_timestamp", normalize_naive_utc_timestamp(self.entry_timestamp, field_name="entry_timestamp"))
        if not self.instrument_id:
            raise ContractValidationError("PositionState.instrument_id es obligatorio")
        if self.quantity <= 0.0 or not math.isfinite(float(self.quantity)):
            raise ContractValidationError("PositionState.quantity debe ser > 0 y finita")
        if self.avg_entry_price <= 0.0 or not math.isfinite(float(self.avg_entry_price)):
            raise ContractValidationError("PositionState.avg_entry_price debe ser > 0 y finita")
        if self.entry_fee < 0.0:
            raise ContractValidationError("PositionState.entry_fee no puede ser negativa")


@dataclass(frozen=True)
class PortfolioSnapshot:
    timestamp: datetime
    cash: float
    market_value: float
    total_equity: float
    realized_pnl: float
    unrealized_pnl: float
    open_positions: int
    window_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", normalize_naive_utc_timestamp(self.timestamp, field_name="snapshot.timestamp"))
        if self.open_positions < 0:
            raise ContractValidationError("PortfolioSnapshot.open_positions no puede ser negativo")
        if self.window_index < 0:
            raise ContractValidationError("PortfolioSnapshot.window_index no puede ser negativo")


@dataclass(frozen=True)
class DiscardedSignal:
    instrument_id: str
    decision_timestamp: datetime
    execution_timestamp: datetime | None
    window_index: int
    reason: DiscardReason

    def __post_init__(self) -> None:
        if self.execution_timestamp is not None:
            _ensure_matching_timestamp_awareness(
                fields={
                    "decision_timestamp": self.decision_timestamp,
                    "execution_timestamp": self.execution_timestamp,
                }
            )
        object.__setattr__(
            self,
            "decision_timestamp",
            normalize_naive_utc_timestamp(self.decision_timestamp, field_name="decision_timestamp"),
        )
        if self.execution_timestamp is not None:
            object.__setattr__(
                self,
                "execution_timestamp",
                normalize_naive_utc_timestamp(self.execution_timestamp, field_name="execution_timestamp"),
            )
        if not self.instrument_id:
            raise ContractValidationError("DiscardedSignal.instrument_id es obligatorio")
        if self.window_index < 0:
            raise ContractValidationError("DiscardedSignal.window_index no puede ser negativo")


@dataclass(frozen=True)
class GlobalOOSPoint:
    timestamp: datetime
    window_index: int
    equity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", normalize_naive_utc_timestamp(self.timestamp, field_name="oos.timestamp"))
        if self.window_index < 0:
            raise ContractValidationError("GlobalOOSPoint.window_index no puede ser negativo")
        if self.equity <= 0.0 or not math.isfinite(float(self.equity)):
            raise ContractValidationError("GlobalOOSPoint.equity debe ser > 0 y finita")


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
    bundle_version: int
    provenance_mode: BundleProvenanceMode
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
    discarded_signals: tuple[DiscardedSignal, ...]
    metrics: MetricBundle


@dataclass(frozen=True)
class BacktestResult:
    manifest: RunManifest
    windows: tuple[WindowResult, ...]
    oos_curve: tuple[GlobalOOSPoint, ...]
    summary_metrics: MetricBundle
    window_average_metrics: MetricBundle | None = None


@dataclass(frozen=True)
class InputBundle:
    adapter_name: str
    bundle_version: int
    provenance_mode: BundleProvenanceMode
    forecasts: tuple[ForecastRecord, ...] = ()
    signals: tuple[SignalRecord, ...] = ()
    provenance: BundleProvenance | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not self.adapter_name:
            raise ContractValidationError("InputBundle.adapter_name es obligatorio")
        if self.bundle_version <= 0:
            raise ContractValidationError("InputBundle.bundle_version debe ser > 0")
        if self.forecasts and self.signals:
            raise ContractValidationError("InputBundle no puede mezclar forecasts y signals")
        if not self.forecasts and not self.signals:
            raise ContractValidationError("InputBundle debe contener forecasts o signals")
