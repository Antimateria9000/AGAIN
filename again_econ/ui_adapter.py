from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from again_econ.contracts import BacktestResult
from again_econ.reports import (
    build_discard_rows,
    build_fill_rows,
    build_oos_curve_frame,
    build_run_view,
    build_runs_table_rows,
    build_summary_payload,
    build_trade_rows,
    build_window_rows,
    compare_run_views,
    serialize_value,
)
from again_econ.storage import BacktestStorage


class BacktestUIAdapter:
    def __init__(self, storage: BacktestStorage):
        self.storage = storage

    def persist_result(
        self,
        result: BacktestResult,
        *,
        mode: str,
        preset_name: str,
        methodology_label: str,
        model_name: str,
        config_reference: str | None,
        requested_universe: tuple[str, ...],
        effective_universe: tuple[str, ...],
        market_data: pd.DataFrame,
        market_context: dict[str, Any],
        warnings: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        created_at = datetime.now(UTC).replace(tzinfo=None)
        summary = build_summary_payload(
            result,
            created_at=created_at,
            mode=mode,
            preset_name=preset_name,
            methodology_label=methodology_label,
            model_name=model_name,
            config_reference=config_reference,
            requested_universe=requested_universe,
            effective_universe=effective_universe,
            market_context=market_context,
            warnings=warnings,
        )
        stored = self.storage.write_run(
            manifest=serialize_value(result.manifest),
            summary=summary,
            market_data=market_data,
            oos_curve=build_oos_curve_frame(result),
            windows=build_window_rows(result),
            trades=build_trade_rows(result),
            fills=build_fill_rows(result),
            discards=build_discard_rows(result),
        )
        return build_run_view(stored)

    def list_runs(
        self,
        *,
        mode: str | None = None,
        preset_name: str | None = None,
        model_name: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        entries = list(self.storage.list_runs())
        if mode:
            entries = [entry for entry in entries if entry["mode"] == mode]
        if preset_name:
            entries = [entry for entry in entries if preset_name.lower() in entry["preset_name"].lower()]
        if model_name:
            entries = [entry for entry in entries if model_name.lower() in entry["model_name"].lower()]
        if run_id:
            entries = [entry for entry in entries if run_id.lower() in entry["run_id"].lower()]
        return build_runs_table_rows(tuple(entries))

    def load_run_view(self, run_id: str) -> dict[str, Any]:
        return build_run_view(self.storage.load_run(run_id))

    def compare_runs(self, left_run_id: str, right_run_id: str) -> dict[str, Any]:
        left = self.load_run_view(left_run_id)
        right = self.load_run_view(right_run_id)
        return compare_run_views(left, right)
