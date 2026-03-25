from __future__ import annotations

from collections import defaultdict

import torch

from again_econ.config import ExecutionConfig
from again_econ.contracts import (
    CapitalCompetitionPolicy,
    DiscardReason,
    DiscardedSignal,
    ExecutionReason,
    OrderSide,
    PortfolioSnapshot,
    PositionTarget,
    ScheduledSignal,
    SignalRecord,
    TradeRecord,
    WindowClosePolicy,
)
from again_econ.errors import ExecutionError
from again_econ.execution import ExecutionResult, FillEngine
from again_econ.gpu.tensor_market import TensorMarketFrame, _timestamp_to_sortable_int
from again_econ.validation import validate_scheduled_signals, validate_snapshot_invariants


def _rank_signals_tensor(
    signals: list[ScheduledSignal],
    policy: CapitalCompetitionPolicy,
    tensor_market: TensorMarketFrame,
) -> list[ScheduledSignal]:
    if len(signals) <= 1:
        return list(signals)
    instrument_indices = torch.tensor(
        [tensor_market.instrument_index[scheduled.signal.instrument_id] for scheduled in signals],
        dtype=torch.int64,
        device=tensor_market.device,
    )
    base_order = torch.argsort(instrument_indices, stable=True)
    if policy == CapitalCompetitionPolicy.SCORE_DESC:
        scores = torch.tensor(
            [
                float(scheduled.signal.score) if scheduled.signal.score is not None else float("-inf")
                for scheduled in signals
            ],
            dtype=torch.float64,
            device=tensor_market.device,
        )
        score_order = torch.argsort(-scores[base_order], stable=True)
        final_order = base_order[score_order]
    else:
        final_order = base_order
    return [signals[int(index)] for index in final_order.detach().cpu().tolist()]


def _determine_buy_quantity_tensor(cash: torch.Tensor, open_price: torch.Tensor, config: ExecutionConfig) -> float:
    budget = cash * float(config.allocation_fraction)
    if float(budget.item()) <= float(config.commission_per_order):
        return 0.0
    fill_price = open_price * (1.0 + (float(config.slippage_bps) / 10_000.0))
    effective_unit_cost = fill_price * (1.0 + float(config.commission_rate))
    if config.allow_fractional_shares:
        quantity = torch.clamp((budget - float(config.commission_per_order)) / effective_unit_cost, min=0.0)
        return float(quantity.item()) if float(quantity.item()) > 0.0 else 0.0

    quantity = torch.floor(torch.clamp((budget - float(config.commission_per_order)) / effective_unit_cost, min=0.0))
    while float(quantity.item()) > 0.0:
        gross_notional = quantity * fill_price
        total_cost = gross_notional + (float(config.commission_per_order) + gross_notional * float(config.commission_rate))
        if float(total_cost.item()) <= float(cash.item()) + 1e-12:
            return float(quantity.item())
        quantity = quantity - 1.0
    return 0.0


def schedule_window_signals_tensor(
    signals: tuple[SignalRecord, ...],
    market_frame,
    window,
    *,
    device: torch.device,
) -> tuple[tuple[ScheduledSignal, ...], tuple[DiscardedSignal, ...], dict[str, int | str]]:
    tensor_market = TensorMarketFrame.from_market_frame(market_frame, device=device)
    ordered_signals = list(sorted(signals, key=lambda item: (item.decision_timestamp, item.instrument_id)))
    resolved: dict[int, ScheduledSignal | DiscardedSignal] = {}
    signals_by_instrument: dict[str, list[tuple[int, SignalRecord]]] = defaultdict(list)
    for index, signal in enumerate(ordered_signals):
        signals_by_instrument[signal.instrument_id].append((index, signal))

    for instrument_id, instrument_signals in signals_by_instrument.items():
        instrument_timestamps = tensor_market.instrument_timestamp_values.get(instrument_id)
        instrument_indices = tensor_market.instrument_timestamp_indices.get(instrument_id)
        if instrument_timestamps is None or instrument_timestamps.numel() == 0:
            for index, signal in instrument_signals:
                resolved[index] = DiscardedSignal(
                    instrument_id=signal.instrument_id,
                    decision_timestamp=signal.decision_timestamp,
                    execution_timestamp=None,
                    available_at=signal.available_at,
                    window_index=window.index,
                    reason=DiscardReason.NO_NEXT_OPEN_AVAILABLE,
                )
            continue

        operational_values = torch.tensor(
            [
                _timestamp_to_sortable_int(max(signal.decision_timestamp, signal.available_at))
                for _, signal in instrument_signals
            ],
            dtype=torch.int64,
            device=device,
        )
        execution_positions = torch.searchsorted(instrument_timestamps, operational_values, right=True)
        execution_positions = execution_positions + int(window.execution_lag_bars) - 1
        valid_mask = execution_positions < instrument_timestamps.numel()

        for local_index, (signal_index, signal) in enumerate(instrument_signals):
            if not bool(valid_mask[local_index].item()):
                resolved[signal_index] = DiscardedSignal(
                    instrument_id=signal.instrument_id,
                    decision_timestamp=signal.decision_timestamp,
                    execution_timestamp=None,
                    available_at=signal.available_at,
                    window_index=window.index,
                    reason=DiscardReason.NO_NEXT_OPEN_AVAILABLE,
                )
                continue

            timestamp_position = int(execution_positions[local_index].item())
            execution_idx = int(instrument_indices[timestamp_position].item())
            execution_timestamp = tensor_market.timestamps[execution_idx]
            if execution_timestamp < window.test_start or execution_timestamp > window.test_end:
                resolved[signal_index] = DiscardedSignal(
                    instrument_id=signal.instrument_id,
                    decision_timestamp=signal.decision_timestamp,
                    execution_timestamp=execution_timestamp,
                    available_at=signal.available_at,
                    window_index=window.index,
                    reason=DiscardReason.NEXT_OPEN_OUTSIDE_WINDOW,
                )
                continue

            resolved[signal_index] = ScheduledSignal(signal=signal, execution_timestamp=execution_timestamp)

    scheduled = tuple(
        item for index in range(len(ordered_signals)) for item in (resolved[index],) if isinstance(item, ScheduledSignal)
    )
    discarded = tuple(
        item for index in range(len(ordered_signals)) for item in (resolved[index],) if isinstance(item, DiscardedSignal)
    )
    return scheduled, discarded, {
        "scheduled_signal_count": len(scheduled),
        "discarded_signal_count": len(discarded),
        "device": str(device),
        "scheduler_backend": "tensor",
    }


def run_window_execution_tensor(
    market_frame,
    scheduled_signals: tuple[ScheduledSignal, ...],
    config: ExecutionConfig,
    *,
    window_index: int,
    close_policy: WindowClosePolicy = WindowClosePolicy.ADMINISTRATIVE_CLOSE_ON_LAST_BAR,
    device: torch.device,
) -> ExecutionResult:
    validate_scheduled_signals(scheduled_signals)
    tensor_market = TensorMarketFrame.from_market_frame(market_frame, device=device)
    fill_engine = FillEngine(config)
    position_quantities = torch.zeros(len(tensor_market.instruments), dtype=torch.float64, device=device)
    avg_entry_prices = torch.zeros(len(tensor_market.instruments), dtype=torch.float64, device=device)
    entry_fees = torch.zeros(len(tensor_market.instruments), dtype=torch.float64, device=device)
    entry_timestamp_indices = torch.full((len(tensor_market.instruments),), -1, dtype=torch.int64, device=device)
    cash = torch.tensor(float(config.initial_cash), dtype=torch.float64, device=device)
    realized_pnl = torch.tensor(0.0, dtype=torch.float64, device=device)
    fills = []
    trades = []
    snapshots = []
    signals_by_timestamp: dict = defaultdict(list)
    for scheduled in scheduled_signals:
        signals_by_timestamp[scheduled.execution_timestamp].append(scheduled)

    last_timestamp_index = len(tensor_market.timestamps) - 1
    for timestamp_index, timestamp in enumerate(tensor_market.timestamps):
        day_mask = tensor_market.availability_mask[timestamp_index]
        day_open = tensor_market.open_prices[timestamp_index]
        day_close = tensor_market.close_prices[timestamp_index]
        ranked_signals = _rank_signals_tensor(
            list(signals_by_timestamp.get(timestamp, [])),
            config.capital_competition_policy,
            tensor_market,
        )
        for scheduled in ranked_signals:
            instrument_id = scheduled.signal.instrument_id
            instrument_index = tensor_market.instrument_index[instrument_id]
            if not bool(day_mask[instrument_index].item()):
                raise ExecutionError(f"No existe barra de mercado para ejecutar {instrument_id} en {timestamp!s}")

            if scheduled.signal.target_state == PositionTarget.LONG and float(position_quantities[instrument_index].item()) <= 0.0:
                quantity = _determine_buy_quantity_tensor(cash, day_open[instrument_index], config)
                if quantity <= 0.0:
                    continue
                fill = fill_engine.build_fill(
                    instrument_id=instrument_id,
                    side=OrderSide.BUY,
                    decision_timestamp=scheduled.signal.decision_timestamp,
                    execution_timestamp=timestamp,
                    quantity=quantity,
                    base_price=float(day_open[instrument_index].item()),
                    reason=ExecutionReason.SIGNAL,
                )
                total_cost = float(fill.gross_notional + fill.fee)
                if total_cost > float(cash.item()) + 1e-12:
                    raise ExecutionError("La orden de compra no es financiable con el cash disponible")
                cash = cash - total_cost
                position_quantities[instrument_index] = float(fill.quantity)
                avg_entry_prices[instrument_index] = float(fill.price)
                entry_fees[instrument_index] = float(fill.fee)
                entry_timestamp_indices[instrument_index] = timestamp_index
                fills.append(fill)
            elif scheduled.signal.target_state == PositionTarget.FLAT and float(position_quantities[instrument_index].item()) > 0.0:
                quantity = float(position_quantities[instrument_index].item())
                entry_price = float(avg_entry_prices[instrument_index].item())
                entry_fee = float(entry_fees[instrument_index].item())
                entry_timestamp_index = int(entry_timestamp_indices[instrument_index].item())
                fill = fill_engine.build_fill(
                    instrument_id=instrument_id,
                    side=OrderSide.SELL,
                    decision_timestamp=scheduled.signal.decision_timestamp,
                    execution_timestamp=timestamp,
                    quantity=quantity,
                    base_price=float(day_open[instrument_index].item()),
                    reason=ExecutionReason.SIGNAL,
                )
                cash = cash + (float(fill.gross_notional) - float(fill.fee))
                gross_pnl = (float(fill.price) - entry_price) * quantity
                net_pnl = gross_pnl - entry_fee - float(fill.fee)
                realized_pnl = realized_pnl + net_pnl
                trades.append(
                    TradeRecord(
                        instrument_id=instrument_id,
                        entry_timestamp=tensor_market.timestamps[entry_timestamp_index],
                        exit_timestamp=timestamp,
                        entry_price=entry_price,
                        exit_price=float(fill.price),
                        quantity=quantity,
                        entry_fee=entry_fee,
                        exit_fee=float(fill.fee),
                        gross_pnl=gross_pnl,
                        net_pnl=net_pnl,
                        exit_reason=ExecutionReason.SIGNAL,
                    )
                )
                fills.append(fill)
                position_quantities[instrument_index] = 0.0
                avg_entry_prices[instrument_index] = 0.0
                entry_fees[instrument_index] = 0.0
                entry_timestamp_indices[instrument_index] = -1

        if timestamp_index == last_timestamp_index and bool(torch.any(position_quantities > 0.0).item()):
            if close_policy != WindowClosePolicy.ADMINISTRATIVE_CLOSE_ON_LAST_BAR:
                raise ExecutionError(f"close_policy no soportada: {close_policy}")
            open_indices = torch.nonzero(position_quantities > 0.0, as_tuple=False).flatten()
            ordered_open_indices = sorted(int(index) for index in open_indices.detach().cpu().tolist())
            for instrument_index in ordered_open_indices:
                if not bool(day_mask[instrument_index].item()):
                    instrument_id = tensor_market.instruments[instrument_index]
                    raise ExecutionError(f"Falta barra para clausurar la posicion abierta de {instrument_id} en {timestamp!s}")
                quantity = float(position_quantities[instrument_index].item())
                entry_price = float(avg_entry_prices[instrument_index].item())
                entry_fee = float(entry_fees[instrument_index].item())
                entry_timestamp_index = int(entry_timestamp_indices[instrument_index].item())
                instrument_id = tensor_market.instruments[instrument_index]
                fill = fill_engine.build_fill(
                    instrument_id=instrument_id,
                    side=OrderSide.SELL,
                    decision_timestamp=timestamp,
                    execution_timestamp=timestamp,
                    quantity=quantity,
                    base_price=float(day_close[instrument_index].item()),
                    reason=ExecutionReason.WINDOW_CLOSE,
                )
                cash = cash + (float(fill.gross_notional) - float(fill.fee))
                gross_pnl = (float(fill.price) - entry_price) * quantity
                net_pnl = gross_pnl - entry_fee - float(fill.fee)
                realized_pnl = realized_pnl + net_pnl
                trades.append(
                    TradeRecord(
                        instrument_id=instrument_id,
                        entry_timestamp=tensor_market.timestamps[entry_timestamp_index],
                        exit_timestamp=timestamp,
                        entry_price=entry_price,
                        exit_price=float(fill.price),
                        quantity=quantity,
                        entry_fee=entry_fee,
                        exit_fee=float(fill.fee),
                        gross_pnl=gross_pnl,
                        net_pnl=net_pnl,
                        exit_reason=ExecutionReason.WINDOW_CLOSE,
                    )
                )
                fills.append(fill)
                position_quantities[instrument_index] = 0.0
                avg_entry_prices[instrument_index] = 0.0
                entry_fees[instrument_index] = 0.0
                entry_timestamp_indices[instrument_index] = -1

        missing_for_open_positions = (position_quantities > 0.0) & (~day_mask)
        if bool(torch.any(missing_for_open_positions).item()):
            first_missing = int(torch.nonzero(missing_for_open_positions, as_tuple=False)[0].item())
            instrument_id = tensor_market.instruments[first_missing]
            raise ExecutionError(f"Falta barra para valorar la posicion abierta de {instrument_id} en {timestamp!s}")

        market_value_tensor = torch.where(day_mask, position_quantities * day_close, torch.zeros_like(day_close)).sum()
        unrealized_pnl_tensor = torch.where(
            day_mask,
            (day_close - avg_entry_prices) * position_quantities,
            torch.zeros_like(day_close),
        ).sum()
        total_equity_tensor = cash + market_value_tensor
        snapshots.append(
            PortfolioSnapshot(
                timestamp=timestamp,
                cash=float(cash.item()),
                market_value=float(market_value_tensor.item()),
                total_equity=float(total_equity_tensor.item()),
                realized_pnl=float(realized_pnl.item()),
                unrealized_pnl=float(unrealized_pnl_tensor.item()),
                open_positions=int(torch.count_nonzero(position_quantities > 0.0).item()),
                window_index=window_index,
            )
        )

    result = ExecutionResult(
        fills=tuple(fills),
        trades=tuple(trades),
        snapshots=tuple(snapshots),
    )
    validate_snapshot_invariants(result.snapshots, close_policy=close_policy)
    return result
