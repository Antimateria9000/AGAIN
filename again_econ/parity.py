from __future__ import annotations

import math


def _compare_float(label: str, left: float, right: float, *, abs_tol: float) -> list[str]:
    return [] if math.isclose(float(left), float(right), abs_tol=abs_tol, rel_tol=0.0) else [f"{label}: {left} != {right}"]


def compare_scheduling_outputs(
    expected_scheduled,
    actual_scheduled,
    expected_discarded,
    actual_discarded,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if len(expected_scheduled) != len(actual_scheduled):
        mismatches.append(f"scheduled_count: {len(expected_scheduled)} != {len(actual_scheduled)}")
    if len(expected_discarded) != len(actual_discarded):
        mismatches.append(f"discarded_count: {len(expected_discarded)} != {len(actual_discarded)}")

    for index, (left, right) in enumerate(zip(expected_scheduled, actual_scheduled, strict=False)):
        if left.signal.instrument_id != right.signal.instrument_id:
            mismatches.append(f"scheduled[{index}].instrument_id")
        if left.signal.decision_timestamp != right.signal.decision_timestamp:
            mismatches.append(f"scheduled[{index}].decision_timestamp")
        if left.signal.available_at != right.signal.available_at:
            mismatches.append(f"scheduled[{index}].available_at")
        if left.execution_timestamp != right.execution_timestamp:
            mismatches.append(f"scheduled[{index}].execution_timestamp")

    for index, (left, right) in enumerate(zip(expected_discarded, actual_discarded, strict=False)):
        if left.instrument_id != right.instrument_id:
            mismatches.append(f"discarded[{index}].instrument_id")
        if left.decision_timestamp != right.decision_timestamp:
            mismatches.append(f"discarded[{index}].decision_timestamp")
        if left.execution_timestamp != right.execution_timestamp:
            mismatches.append(f"discarded[{index}].execution_timestamp")
        if left.available_at != right.available_at:
            mismatches.append(f"discarded[{index}].available_at")
        if left.reason != right.reason:
            mismatches.append(f"discarded[{index}].reason")
    return tuple(mismatches)


def compare_execution_results(expected, actual, *, abs_tol: float = 1e-8) -> tuple[str, ...]:
    mismatches: list[str] = []
    if len(expected.fills) != len(actual.fills):
        mismatches.append(f"fill_count: {len(expected.fills)} != {len(actual.fills)}")
    if len(expected.trades) != len(actual.trades):
        mismatches.append(f"trade_count: {len(expected.trades)} != {len(actual.trades)}")
    if len(expected.snapshots) != len(actual.snapshots):
        mismatches.append(f"snapshot_count: {len(expected.snapshots)} != {len(actual.snapshots)}")

    for index, (left, right) in enumerate(zip(expected.fills, actual.fills, strict=False)):
        if left.instrument_id != right.instrument_id:
            mismatches.append(f"fills[{index}].instrument_id")
        if left.side != right.side:
            mismatches.append(f"fills[{index}].side")
        if left.decision_timestamp != right.decision_timestamp:
            mismatches.append(f"fills[{index}].decision_timestamp")
        if left.execution_timestamp != right.execution_timestamp:
            mismatches.append(f"fills[{index}].execution_timestamp")
        if left.reason != right.reason:
            mismatches.append(f"fills[{index}].reason")
        mismatches.extend(_compare_float(f"fills[{index}].price", left.price, right.price, abs_tol=abs_tol))
        mismatches.extend(_compare_float(f"fills[{index}].quantity", left.quantity, right.quantity, abs_tol=abs_tol))
        mismatches.extend(_compare_float(f"fills[{index}].gross_notional", left.gross_notional, right.gross_notional, abs_tol=abs_tol))
        mismatches.extend(_compare_float(f"fills[{index}].fee", left.fee, right.fee, abs_tol=abs_tol))

    for index, (left, right) in enumerate(zip(expected.trades, actual.trades, strict=False)):
        if left.instrument_id != right.instrument_id:
            mismatches.append(f"trades[{index}].instrument_id")
        if left.entry_timestamp != right.entry_timestamp:
            mismatches.append(f"trades[{index}].entry_timestamp")
        if left.exit_timestamp != right.exit_timestamp:
            mismatches.append(f"trades[{index}].exit_timestamp")
        if left.exit_reason != right.exit_reason:
            mismatches.append(f"trades[{index}].exit_reason")
        for field_name in (
            "entry_price",
            "exit_price",
            "quantity",
            "entry_fee",
            "exit_fee",
            "gross_pnl",
            "net_pnl",
        ):
            mismatches.extend(
                _compare_float(
                    f"trades[{index}].{field_name}",
                    getattr(left, field_name),
                    getattr(right, field_name),
                    abs_tol=abs_tol,
                )
            )

    for index, (left, right) in enumerate(zip(expected.snapshots, actual.snapshots, strict=False)):
        if left.timestamp != right.timestamp:
            mismatches.append(f"snapshots[{index}].timestamp")
        if left.window_index != right.window_index:
            mismatches.append(f"snapshots[{index}].window_index")
        if left.open_positions != right.open_positions:
            mismatches.append(f"snapshots[{index}].open_positions")
        for field_name in ("cash", "market_value", "total_equity", "realized_pnl", "unrealized_pnl"):
            mismatches.extend(
                _compare_float(
                    f"snapshots[{index}].{field_name}",
                    getattr(left, field_name),
                    getattr(right, field_name),
                    abs_tol=abs_tol,
                )
            )
    return tuple(mismatches)


def compare_metric_bundles(expected, actual, *, abs_tol: float = 1e-8) -> tuple[str, ...]:
    mismatches: list[str] = []
    if expected.trade_count != actual.trade_count:
        mismatches.append(f"metrics.trade_count: {expected.trade_count} != {actual.trade_count}")
    for field_name in ("total_return", "annualized_return", "max_drawdown", "sharpe_ratio", "win_rate", "exposure_ratio"):
        mismatches.extend(
            _compare_float(
                f"metrics.{field_name}",
                getattr(expected, field_name),
                getattr(actual, field_name),
                abs_tol=abs_tol,
            )
        )
    return tuple(mismatches)
