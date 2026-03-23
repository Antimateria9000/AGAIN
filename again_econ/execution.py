from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from again_econ.config import ExecutionConfig
from again_econ.contracts import (
    FillEvent,
    MarketBar,
    MarketFrame,
    OrderIntent,
    OrderSide,
    PortfolioSnapshot,
    PositionState,
    ScheduledSignal,
    TradeRecord,
)
from again_econ.errors import ExecutionError, TemporalIntegrityError
from again_econ.validation import ensure_market_timestamp_exists, validate_scheduled_signals


@dataclass(frozen=True)
class ExecutionResult:
    fills: tuple[FillEvent, ...]
    trades: tuple[TradeRecord, ...]
    snapshots: tuple[PortfolioSnapshot, ...]


def schedule_signal_next_open(signal, market_frame: MarketFrame) -> ScheduledSignal:
    instrument_bars = market_frame.bars_for_instrument(signal.instrument_id)
    for bar in instrument_bars:
        if bar.timestamp > signal.decision_timestamp:
            ensure_market_timestamp_exists(market_frame, signal.instrument_id, bar.timestamp)
            return ScheduledSignal(signal=signal, execution_timestamp=bar.timestamp)
    raise TemporalIntegrityError(
        f"No existe una apertura posterior para programar la senal de {signal.instrument_id} tras {signal.decision_timestamp!s}"
    )


def schedule_signals_next_open(signals, market_frame: MarketFrame) -> tuple[ScheduledSignal, ...]:
    scheduled = tuple(
        schedule_signal_next_open(signal, market_frame)
        for signal in sorted(signals, key=lambda item: (item.decision_timestamp, item.instrument_id))
    )
    validate_scheduled_signals(scheduled)
    return scheduled


def calculate_fee(gross_notional: float, config: ExecutionConfig) -> float:
    return float(config.commission_per_order + (gross_notional * config.commission_rate))


def apply_slippage(price: float, side: OrderSide, config: ExecutionConfig) -> float:
    multiplier = 1.0 + (config.slippage_bps / 10_000.0)
    return float(price * multiplier) if side == OrderSide.BUY else float(price / multiplier)


def _determine_buy_quantity(cash: float, bar: MarketBar, config: ExecutionConfig) -> float:
    budget = cash * config.allocation_fraction
    if budget <= config.commission_per_order:
        return 0.0
    fill_price = apply_slippage(bar.open, OrderSide.BUY, config)
    effective_unit_cost = fill_price * (1.0 + config.commission_rate)
    if config.allow_fractional_shares:
        quantity = max((budget - config.commission_per_order) / effective_unit_cost, 0.0)
        return quantity if quantity > 0.0 else 0.0
    quantity = math.floor(max((budget - config.commission_per_order) / effective_unit_cost, 0.0))
    while quantity > 0:
        gross_notional = quantity * fill_price
        total_cost = gross_notional + calculate_fee(gross_notional, config)
        if total_cost <= cash + 1e-12:
            return float(quantity)
        quantity -= 1
    return 0.0


def _build_buy_fill(instrument_id: str, timestamp, decision_timestamp, quantity: float, bar: MarketBar, config: ExecutionConfig):
    fill_price = apply_slippage(bar.open, OrderSide.BUY, config)
    gross_notional = quantity * fill_price
    fee = calculate_fee(gross_notional, config)
    return FillEvent(
        instrument_id=instrument_id,
        side=OrderSide.BUY,
        decision_timestamp=decision_timestamp,
        execution_timestamp=timestamp,
        price=fill_price,
        quantity=quantity,
        gross_notional=gross_notional,
        fee=fee,
        slippage_bps=config.slippage_bps,
    )


def _build_sell_fill(instrument_id: str, timestamp, decision_timestamp, quantity: float, bar: MarketBar, config: ExecutionConfig):
    fill_price = apply_slippage(bar.open, OrderSide.SELL, config)
    gross_notional = quantity * fill_price
    fee = calculate_fee(gross_notional, config)
    return FillEvent(
        instrument_id=instrument_id,
        side=OrderSide.SELL,
        decision_timestamp=decision_timestamp,
        execution_timestamp=timestamp,
        price=fill_price,
        quantity=quantity,
        gross_notional=gross_notional,
        fee=fee,
        slippage_bps=config.slippage_bps,
    )


def run_window_execution(
    market_frame: MarketFrame,
    scheduled_signals: tuple[ScheduledSignal, ...],
    config: ExecutionConfig,
) -> ExecutionResult:
    validate_scheduled_signals(scheduled_signals)
    timestamps = market_frame.timestamps()
    bars_by_timestamp = {timestamp: {bar.instrument_id: bar for bar in market_frame.bars_at(timestamp)} for timestamp in timestamps}
    signals_by_timestamp: dict = defaultdict(list)
    for scheduled in scheduled_signals:
        signals_by_timestamp[scheduled.execution_timestamp].append(scheduled)
    cash = float(config.initial_cash)
    realized_pnl = 0.0
    positions: dict[str, PositionState] = {}
    fills: list[FillEvent] = []
    trades: list[TradeRecord] = []
    snapshots: list[PortfolioSnapshot] = []

    for timestamp in timestamps:
        day_bars = bars_by_timestamp[timestamp]
        for scheduled in sorted(signals_by_timestamp.get(timestamp, []), key=lambda item: item.signal.instrument_id):
            instrument_id = scheduled.signal.instrument_id
            bar = day_bars.get(instrument_id)
            if bar is None:
                raise ExecutionError(f"No existe barra de mercado para ejecutar {instrument_id} en {timestamp!s}")

            if scheduled.signal.target_state.value == "long" and instrument_id not in positions:
                quantity = _determine_buy_quantity(cash, bar, config)
                if quantity <= 0.0:
                    continue
                intent = OrderIntent(
                    instrument_id=instrument_id,
                    side=OrderSide.BUY,
                    decision_timestamp=scheduled.signal.decision_timestamp,
                    execution_timestamp=timestamp,
                    quantity=quantity,
                )
                fill = _build_buy_fill(intent.instrument_id, intent.execution_timestamp, intent.decision_timestamp, intent.quantity, bar, config)
                total_cost = fill.gross_notional + fill.fee
                if total_cost > cash + 1e-12:
                    raise ExecutionError("El scheduler genero una compra no financiable")
                cash -= total_cost
                positions[instrument_id] = PositionState(
                    instrument_id=instrument_id,
                    quantity=fill.quantity,
                    avg_entry_price=fill.price,
                    entry_timestamp=fill.execution_timestamp,
                    entry_fee=fill.fee,
                )
                fills.append(fill)
            elif scheduled.signal.target_state.value == "flat" and instrument_id in positions:
                position = positions.pop(instrument_id)
                intent = OrderIntent(
                    instrument_id=instrument_id,
                    side=OrderSide.SELL,
                    decision_timestamp=scheduled.signal.decision_timestamp,
                    execution_timestamp=timestamp,
                    quantity=position.quantity,
                )
                fill = _build_sell_fill(intent.instrument_id, intent.execution_timestamp, intent.decision_timestamp, intent.quantity, bar, config)
                cash += fill.gross_notional - fill.fee
                gross_pnl = (fill.price - position.avg_entry_price) * position.quantity
                net_pnl = gross_pnl - position.entry_fee - fill.fee
                realized_pnl += net_pnl
                trades.append(
                    TradeRecord(
                        instrument_id=instrument_id,
                        entry_timestamp=position.entry_timestamp,
                        exit_timestamp=fill.execution_timestamp,
                        entry_price=position.avg_entry_price,
                        exit_price=fill.price,
                        quantity=position.quantity,
                        entry_fee=position.entry_fee,
                        exit_fee=fill.fee,
                        gross_pnl=gross_pnl,
                        net_pnl=net_pnl,
                    )
                )
                fills.append(fill)

        market_value = 0.0
        unrealized_pnl = 0.0
        for instrument_id, position in positions.items():
            bar = day_bars.get(instrument_id)
            if bar is None:
                raise ExecutionError(f"Falta barra para valorar la posicion abierta de {instrument_id} en {timestamp!s}")
            market_value += position.quantity * bar.close
            unrealized_pnl += (bar.close - position.avg_entry_price) * position.quantity
        total_equity = cash + market_value
        snapshots.append(
            PortfolioSnapshot(
                timestamp=timestamp,
                cash=cash,
                market_value=market_value,
                total_equity=total_equity,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                open_positions=len(positions),
            )
        )

    return ExecutionResult(fills=tuple(fills), trades=tuple(trades), snapshots=tuple(snapshots))
