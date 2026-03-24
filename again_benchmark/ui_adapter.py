from __future__ import annotations

from datetime import datetime

from again_benchmark.comparison import compare_run_bundles
from again_benchmark.contracts import BenchmarkComparisonResult, BenchmarkRunBundle
from again_benchmark.definitions import build_default_definition
from again_benchmark.reports import build_run_view, build_runs_table_rows
from again_benchmark.runner import BenchmarkRunner
from again_benchmark.storage import BenchmarkStorage


class BenchmarkUIAdapter:
    def __init__(self, storage: BenchmarkStorage, runner: BenchmarkRunner):
        self.storage = storage
        self.runner = runner

    def ensure_default_definition(self, config: dict):
        definition = build_default_definition(config)
        self.storage.write_definition(definition)
        return definition

    def list_definitions(self):
        return self.storage.list_definitions()

    def load_definition(self, definition_id: str):
        return self.storage.load_definition(definition_id)

    def run_live(self, definition_id: str, *, as_of_timestamp: datetime) -> BenchmarkRunBundle:
        definition = self.storage.load_definition(definition_id)
        return self.runner.run_live(definition, as_of_timestamp=as_of_timestamp)

    def run_frozen(self, definition_id: str, *, as_of_timestamp: datetime) -> BenchmarkRunBundle:
        definition = self.storage.load_definition(definition_id)
        snapshot_manifest = self.runner.create_frozen_snapshot(definition, as_of_timestamp=as_of_timestamp)
        return self.runner.run_frozen_from_snapshot(definition, snapshot_manifest.snapshot_id)

    def rerun_from_run_id(self, run_id: str) -> BenchmarkRunBundle:
        return self.runner.rerun_from_run_id(run_id)

    def list_runs(
        self,
        *,
        benchmark_id: str | None = None,
        mode: str | None = None,
        model_name: str | None = None,
        run_id: str | None = None,
    ) -> list[dict]:
        entries = list(self.storage.list_runs())
        if benchmark_id:
            entries = [entry for entry in entries if entry.benchmark_id == benchmark_id]
        if mode:
            entries = [entry for entry in entries if entry.mode.value == mode]
        if model_name:
            entries = [entry for entry in entries if model_name.lower() in entry.model_name.lower()]
        if run_id:
            entries = [entry for entry in entries if run_id.lower() in entry.run_id.lower()]
        return build_runs_table_rows(tuple(entries))

    def load_run_view(self, run_id: str) -> dict:
        return build_run_view(self.storage.load_run_bundle(run_id))

    def compare_runs(self, left_run_id: str, right_run_id: str) -> BenchmarkComparisonResult:
        left = self.storage.load_run_bundle(left_run_id)
        right = self.storage.load_run_bundle(right_run_id)
        return compare_run_bundles(left, right)
