from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from again_econ.contracts import BacktestResult


def serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: serialize_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    return value


def build_summary_payload(
    result: BacktestResult,
    *,
    created_at: datetime,
    mode: str,
    preset_name: str,
    methodology_label: str,
    model_name: str,
    config_reference: str | None,
    requested_universe: tuple[str, ...],
    effective_universe: tuple[str, ...],
    market_context: dict[str, Any],
    warnings: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "run_id": result.manifest.run_id,
        "created_at": created_at.isoformat(),
        "mode": mode,
        "preset_name": preset_name,
        "methodology_label": methodology_label,
        "model_name": model_name,
        "config_reference": config_reference,
        "requested_universe": list(requested_universe),
        "effective_universe": list(effective_universe),
        "summary_metrics": serialize_value(result.summary_metrics),
        "window_average_metrics": serialize_value(result.window_average_metrics),
        "market_context": serialize_value(market_context),
        "warnings": list(warnings),
    }


def build_oos_curve_frame(result: BacktestResult) -> pd.DataFrame:
    rows = [
        {
            "timestamp": point.timestamp,
            "window_index": point.window_index,
            "equity": point.equity,
        }
        for point in result.oos_curve
    ]
    return pd.DataFrame(rows, columns=["timestamp", "window_index", "equity"])


def build_window_rows(result: BacktestResult) -> pd.DataFrame:
    rows = []
    for window_result in result.windows:
        rows.append(
            {
                "window_index": window_result.window.index,
                "train_start": window_result.window.train_start,
                "train_end": window_result.window.train_end,
                "test_start": window_result.window.test_start,
                "test_end": window_result.window.test_end,
                "lookahead_bars": window_result.window.lookahead_bars,
                "execution_lag_bars": window_result.window.execution_lag_bars,
                "fill_count": len(window_result.fills),
                "trade_count": len(window_result.trades),
                "discarded_signal_count": len(window_result.discarded_signals),
                **serialize_value(window_result.metrics),
            }
        )
    return pd.DataFrame(rows)


def build_trade_rows(result: BacktestResult) -> pd.DataFrame:
    columns = [
        "instrument_id",
        "entry_timestamp",
        "exit_timestamp",
        "entry_price",
        "exit_price",
        "quantity",
        "entry_fee",
        "exit_fee",
        "gross_pnl",
        "net_pnl",
        "exit_reason",
    ]
    rows = [serialize_value(trade) for window in result.windows for trade in window.trades]
    return pd.DataFrame(rows, columns=columns)


def build_fill_rows(result: BacktestResult) -> pd.DataFrame:
    columns = [
        "instrument_id",
        "side",
        "decision_timestamp",
        "execution_timestamp",
        "price",
        "quantity",
        "gross_notional",
        "fee",
        "slippage_bps",
        "reason",
    ]
    rows = [serialize_value(fill) for window in result.windows for fill in window.fills]
    return pd.DataFrame(rows, columns=columns)


def build_discard_rows(result: BacktestResult) -> pd.DataFrame:
    columns = [
        "instrument_id",
        "decision_timestamp",
        "execution_timestamp",
        "window_index",
        "reason",
        "available_at",
        "detail",
        "metadata",
    ]
    rows = [serialize_value(discard) for window in result.windows for discard in window.discarded_signals]
    return pd.DataFrame(rows, columns=columns)


def build_runs_table_rows(entries: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": entry["run_id"],
            "created_at": entry["created_at"],
            "mode": entry["mode"],
            "preset_name": entry["preset_name"],
            "methodology_label": entry["methodology_label"],
            "label": entry["label"],
            "model_name": entry["model_name"],
            "provider_name": entry["provider_name"],
            "provider_version": entry["provider_version"],
            "config_reference": entry["config_reference"],
            "requested_universe_size": len(entry["requested_universe"]),
            "effective_universe_size": len(entry["effective_universe"]),
            "effective_universe": ", ".join(entry["effective_universe"]),
            "market_source_summary": ", ".join(
                f"{key}:{value}" for key, value in sorted(entry["market_source_summary"].items())
            ),
            "window_count": entry["window_count"],
            "discarded_signal_count": entry["discarded_signal_count"],
            **entry["summary_metrics"],
        }
        for entry in entries
    ]


def build_run_view(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest": bundle["manifest"],
        "summary": bundle["summary"],
        "oos_curve": bundle["oos_curve"],
        "windows": bundle["windows"],
        "trades": bundle["trades"],
        "fills": bundle["fills"],
        "discards": bundle["discards"],
        "market_data": bundle["market_data"],
        "artifact_audit": bundle["artifact_audit"],
    }


def compare_run_views(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_metrics = dict(left["summary"]["summary_metrics"])
    right_metrics = dict(right["summary"]["summary_metrics"])
    shared_metric_names = sorted(set(left_metrics).intersection(right_metrics))
    summary_delta = {
        metric: float(right_metrics[metric]) - float(left_metrics[metric])
        for metric in shared_metric_names
    }

    left_windows = left["windows"].set_index("window_index") if not left["windows"].empty else pd.DataFrame()
    right_windows = right["windows"].set_index("window_index") if not right["windows"].empty else pd.DataFrame()
    if not left_windows.empty and not right_windows.empty:
        merged = left_windows.join(right_windows, lsuffix="_left", rsuffix="_right", how="inner")
        window_deltas = []
        for window_index, row in merged.iterrows():
            delta_row = {"window_index": int(window_index)}
            for metric in ("total_return", "annualized_return", "max_drawdown", "sharpe_ratio", "win_rate"):
                left_key = f"{metric}_left"
                right_key = f"{metric}_right"
                if left_key in row and right_key in row:
                    delta_row[metric] = float(row[right_key]) - float(row[left_key])
            window_deltas.append(delta_row)
    else:
        window_deltas = []

    return {
        "left_run_id": left["manifest"]["run_id"],
        "right_run_id": right["manifest"]["run_id"],
        "summary_delta": summary_delta,
        "window_deltas": window_deltas,
    }
