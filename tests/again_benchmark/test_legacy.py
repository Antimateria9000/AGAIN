import json
from pathlib import Path

from app.benchmark_store import save_benchmark_payload
from again_benchmark.adapters.legacy_benchmark_bridge import load_legacy_benchmark_rows


def test_legacy_bridge_loads_existing_history_without_becoming_official_mode(tmp_path):
    benchmark_tickers_path = tmp_path / "benchmark_tickers.yaml"
    benchmark_tickers_path.write_text("tickers:\n  usa:\n    AAA: AAA Corp\n", encoding="utf-8")
    db_path = tmp_path / "benchmarks_history.sqlite"
    payload_json = json.dumps(
        {
            "AAA": {
                "historical_dates": ["2024-01-01T00:00:00"],
                "historical_close": [100.0],
                "pred_dates": ["2024-01-02T00:00:00"],
                "predictions": [101.0],
                "historical_pred_close": [102.0],
                "metrics": {"MAPE": 1.0, "MAE": 1.0, "RMSE": 1.0, "DirAcc": 100.0},
                "last_date": "2024-01-01T00:00:00",
            }
        }
    )
    save_benchmark_payload(Path(db_path), "2024-01-10 10:00:00", "legacy_model", payload_json)
    config = {
        "data": {"benchmark_tickers_file": str(benchmark_tickers_path)},
        "paths": {"benchmark_history_db_path": str(db_path)},
    }

    legacy = load_legacy_benchmark_rows(config)

    assert len(legacy["entries"]) == 1
    assert legacy["entries"][0]["Model_Name"] == "legacy_model"
