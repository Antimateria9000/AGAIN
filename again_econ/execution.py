from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math

from again_econ.config import ExecutionConfig
from again_econ.contracts import (
    CapitalCompetitionPolicy,
    DiscardReason,
    DiscardedSignal,
    ExecutionReason,
    FillEvent,
    MarketBar,
    MarketFrame,
    OrderIntent,
    OrderSide,
    PortfolioSnapshot,
    PositionTarget,
    PositionState,
    ScheduledSignal,
    SignalRecord,
    TradeRecord,
    WindowClosePolicy,
)
from again_econ.errors import ExecutionError, TemporalIntegrityError
from again_econ.validation import ensure_market_timestamp_exists, validate_scheduled_signals, validate_snapshot_invariants


@dataclass
class LedgerState:
    cash: float
    realized_pnl: float = 0.0
    positions: dict[str, PositionState] = field(default_factory=dict)
    fills: list[FillEvent] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    snapshots: list[PortfolioSnapshot] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionResult:
    fills: tuple[FillEvent, ...]
    trades: tuple[TradeRecord, ...]
    snapshots: tuple[PortfolioSnapshot, ...]


class SignalScheduler:
    def __init__(self, market_frame: MarketFrame, *, execution_lag_bars: int = 1) -> None:
        self._market_frame = market_frame
        self._execution_lag_bars = execution_lag_bars

    @staticmethod
    def operational_timestamp(signal: SignalRecord):
        return max(signal.decision_timestamp, signal.available_at)

    def schedule_signal_next_open(self, signal: SignalRecord) -> ScheduledSignal:
        instrument_bars = self._market_frame.bars_for_instrument(signal.instrument_id)
        operational_timestamp = self.operational_timestamp(signal)
        future_bars = [bar for bar in instrument_bars if bar.timestamp > operational_timestamp]
        if len(future_bars) < self._execution_lag_bars:
            raise TemporalIntegrityError(
                f"No existe una apertura posterior suficiente para programar la senal de {signal.instrument_id}"
            )
        execution_bar = future_bars[self._execution_lag_bars - 1]
        ensure_market_timestamp_exists(self._market_frame, signal.instrument_id, execution_bar.timestamp)
        return ScheduledSignal(signal=signal, execution_timestamp=execution_bar.timestamp)

    def schedule_signals_next_open(self, signals) -> tuple[ScheduledSignal, ...]:
        scheduled = tuple(
            self.schedule_signal_next_open(signal)
            for signal in sorted(signals, key=lambda item: (item.decision_timestamp, item.instrument_id))
        )
        validate_scheduled_signals(scheduled)
        return scheduled


def schedule_signal_next_open(signal, market_frame: MarketFrame) -> ScheduledSignal:
    scheduler = SignalScheduler(market_frame, execution_lag_bars=1)
    return scheduler.schedule_signal_next_open(signal)


def schedule_signals_next_open(signals, market_frame: MarketFrame) -> tuple[ScheduledSignal, ...]:
    scheduler = SignalScheduler(market_frame, execution_lag_bars=1)
    return scheduler.schedule_signals_next_open(signals)


def calculate_fee(gross_notional: float, config: ExecutionConfig) -> float:
    return float(config.commission_per_order + (gross_notional * config.commission_rate))


def apply_slippage(price: float, side: OrderSide, config: ExecutionConfig) -> float:
    multiplier = 1.0 + (config.slippage_bps / 10_000.0)
    return float(price * multiplier) if side == OrderSide.BUY else float(price / multiplier)


class FillEngine:
    def __init__(self, config: ExecutionConfig) -> None:
        self._config = config

    def build_fill(
        self,
        *,
        instrument_id: str,
        side: OrderSide,
        decision_timestamp,
        execution_timestamp,
        quantity: float,
        base_price: float,
        reason: ExecutionReason,
    ) -> FillEvent:
        fill_price = apply_slippage(base_price, side, self._config)
        gross_notional = quantity * fill_price
        fee = calculate_fee(gross_notional, self._config)
        return FillEvent(
            instrument_id=instrument_id,
            side=side,
            decision_timestamp=decision_timestamp,
            execution_timestamp=execution_timestamp,
            price=fill_price,
            quantity=quantity,
            gross_notional=gross_notional,
            fee=fee,
            slippage_bps=self._config.slippage_bps,
            reason=reason,
        )


class OrderSizer:
    def __init__(self, config: ExecutionConfig) -> None:
        self._config = config

    def determine_buy_quantity(self, cash: float, bar: MarketBar) -> float:
        budget = cash * self._config.allocation_fraction
        if budget <= self._config.commission_per_order:
            return 0.0
        fill_price = apply_slippage(bar.open, OrderSide.BUY, self._config)
        effective_unit_cost = fill_price * (1.0 + self._config.commission_rate)
        if self._config.allow_fractional_shares:
            quantity = max((budget - self._config.commission_per_order) / effective_unit_cost, 0.0)
            return quantity if quantity > 0.0 else 0.0
        quantity = math.floor(max((budget - self._config.commission_per_order) / effective_unit_cost, 0.0))
        while quantity > 0:
            gross_notional = quantity * fill_price
            total_cost = gross_notional + calculate_fee(gross_notional, self._config)
            if total_cost <= cash + 1e-12:
                return float(quantity)
            quantity -= 1
        return 0.0


class OrderValidator:
    @staticmethod
    def ensure_buy_is_fundable(fill: FillEvent, *, available_cash: float) -> None:
        total_cost = fill.gross_notional + fill.fee
        if total_cost > available_cash + 1e-12:
            raise ExecutionError("La orden de compra no es financiable con el cash disponible")


class PortfolioLedger:
    def __init__(self, *, initial_cash: float, window_index: int) -> None:
        self.state = LedgerState(cash=float(initial_cash))
        self._window_index = window_index

    def _assert_cash_non_negative(self) -> None:
        if self.state.cash < -1e-9:
            raise ExecutionError("El ledger produjo cash negativo")

    def open_position(self, fill: FillEvent) -> None:
        total_cost = fill.gross_notional + fill.fee
        self.state.cash -= total_cost
        self._assert_cash_non_negative()
        self.state.positions[fill.instrument_id] = PositionState(
            instrument_id=fill.instrument_id,
            quantity=fill.quantity,
            avg_entry_price=fill.price,
            entry_timestamp=fill.execution_timestamp,
            entry_fee=fill.fee,
        )
        self.state.fills.append(fill)

    def close_position(self, position: PositionState, fill: FillEvent) -> None:
        self.state.cash += fill.gross_notional - fill.fee
        self.state.fills.append(fill)
        self._realize_trade(position, fill)
        self._assert_cash_non_negative()

    def _realize_trade(self, position: PositionState, fill: FillEvent) -> None:
        gross_pnl = (fill.price - position.avg_entry_price) * position.quantity
        net_pnl = gross_pnl - position.entry_fee - fill.fee
        self.state.realized_pnl += net_pnl
        self.state.trades.append(
            TradeRecord(
                instrument_id=position.instrument_id,
                entry_timestamp=position.entry_timestamp,
                exit_timestamp=fill.execution_timestamp,
                entry_price=position.avg_entry_price,
                exit_price=fill.price,
                quantity=position.quantity,
                entry_fee=position.entry_fee,
                exit_fee=fill.fee,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                exit_reason=fill.reason,
            )
        )

    def append_snapshot(self, market_map: dict[str, MarketBar], timestamp) -> None:
        market_value = 0.0
        unrealized_pnl = 0.0
        for instrument_id, position in self.state.positions.items():
            bar = market_map.get(instrument_id)
            if bar is None:
                raise ExecutionError(f"Falta barra para valorar la posicion abierta de {instrument_id} en {timestamp!s}")
            market_value += position.quantity * bar.close
            unrealized_pnl += (bar.close - position.avg_entry_price) * position.quantity
        total_equity = self.state.cash + market_value
        self.state.snapshots.append(
            PortfolioSnapshot(
                timestamp=timestamp,
                cash=self.state.cash,
                market_value=market_value,
                total_equity=total_equity,
                realized_pnl=self.state.realized_pnl,
                unrealized_pnl=unrealized_pnl,
                open_positions=len(self.state.positions),
                window_index=self._window_index,
            )
        )

    def force_close_open_positions(self, market_map: dict[str, MarketBar], timestamp, fill_engine: FillEngine) -> None:
        for instrument_id in sorted(tuple(self.state.positions)):
            position = self.state.positions.pop(instrument_id)
            bar = market_map.get(instrument_id)
            if bar is None:
                raise ExecutionError(f"Falta barra para clausurar la posicion abierta de {instrument_id} en {timestamp!s}")
            fill = fill_engine.build_fill(
                instrument_id=instrument_id,
                side=OrderSide.SELL,
                decision_timestamp=timestamp,
                execution_timestamp=timestamp,
                quantity=position.quantity,
                base_price=bar.close,
                reason=ExecutionReason.WINDOW_CLOSE,
            )
            self.close_position(position, fill)


def _ranking_key(scheduled: ScheduledSignal, policy: CapitalCompetitionPolicy):
    if policy == CapitalCompetitionPolicy.SCORE_DESC:
        score = scheduled.signal.score if scheduled.signal.score is not None else float("-inf")
        return (-score, scheduled.signal.instrument_id)
    return (scheduled.signal.instrument_id,)


def rank_scheduled_signals(signals: list[ScheduledSignal], config: ExecutionConfig) -> list[ScheduledSignal]:
    return sorted(signals, key=lambda item: _ranking_key(item, config.capital_competition_policy))


def run_window_execution(
    market_frame: MarketFrame,
    scheduled_signals: tuple[ScheduledSignal, ...],
    config: ExecutionConfig,
    *,
    window_index: int,
    close_policy: WindowClosePolicy = WindowClosePolicy.ADMINISTRATIVE_CLOSE_ON_LAST_BAR,
) -> ExecutionResult:
    validate_scheduled_signals(scheduled_signals)
    timestamps = market_frame.timestamps()
    bars_by_timestamp = {timestamp: {bar.instrument_id: bar for bar in market_frame.bars_at(timestamp)} for timestamp in timestamps}
    signals_by_timestamp: dict = defaultdict(list)
    for scheduled in scheduled_signals:
        signals_by_timestamp[scheduled.execution_timestamp].append(scheduled)

    fill_engine = FillEngine(config)
    order_sizer = OrderSizer(config)
    ledger = PortfolioLedger(initial_cash=config.initial_cash, window_index=window_index)
    last_timestamp = timestamps[-1]

    for timestamp in timestamps:
        day_bars = bars_by_timestamp[timestamp]
        for scheduled in rank_scheduled_signals(list(signals_by_timestamp.get(timestamp, [])), config):
            instrument_id = scheduled.signal.instrument_id
            bar = day_bars.get(instrument_id)
            if bar is None:
                raise ExecutionError(f"No existe barra de mercado para ejecutar {instrument_id} en {timestamp!s}")

            if scheduled.signal.target_state == PositionTarget.LONG and instrument_id not in ledger.state.positions:
                quantity = order_sizer.determine_buy_quantity(ledger.state.cash, bar)
                if quantity <= 0.0:
                    continue
                intent = OrderIntent(
                    instrument_id=instrument_id,
                    side=OrderSide.BUY,
                    decision_timestamp=scheduled.signal.decision_timestamp,
                    execution_timestamp=timestamp,
                    quantity=quantity,
                    reason=ExecutionReason.SIGNAL,
                )
                fill = fill_engine.build_fill(
                    instrument_id=intent.instrument_id,
                    side=intent.side,
                    decision_timestamp=intent.decision_timestamp,
                    execution_timestamp=intent.execution_timestamp,
                    quantity=intent.quantity,
                    base_price=bar.open,
                    reason=intent.reason,
                )
                OrderValidator.ensure_buy_is_fundable(fill, available_cash=ledger.state.cash)
                ledger.open_position(fill)
            elif scheduled.signal.target_state == PositionTarget.FLAT and instrument_id in ledger.state.positions:
                position = ledger.state.positions.pop(instrument_id)
                intent = OrderIntent(
                    instrument_id=instrument_id,
                    side=OrderSide.SELL,
                    decision_timestamp=scheduled.signal.decision_timestamp,
                    execution_timestamp=timestamp,
                    quantity=position.quantity,
                    reason=ExecutionReason.SIGNAL,
                )
                fill = fill_engine.build_fill(
                    instrument_id=intent.instrument_id,
                    side=intent.side,
                    decision_timestamp=intent.decision_timestamp,
                    execution_timestamp=intent.execution_timestamp,
                    quantity=intent.quantity,
                    base_price=bar.open,
                    reason=intent.reason,
                )
                ledger.close_position(position, fill)

        if timestamp == last_timestamp and ledger.state.positions:
            if close_policy != WindowClosePolicy.ADMINISTRATIVE_CLOSE_ON_LAST_BAR:
                raise ExecutionError(f"close_policy no soportada: {close_policy}")
            ledger.force_close_open_positions(day_bars, timestamp, fill_engine)

        ledger.append_snapshot(day_bars, timestamp)

    result = ExecutionResult(
        fills=tuple(ledger.state.fills),
        trades=tuple(ledger.state.trades),
        snapshots=tuple(ledger.state.snapshots),
    )
    validate_snapshot_invariants(result.snapshots, close_policy=close_policy)
    return result
