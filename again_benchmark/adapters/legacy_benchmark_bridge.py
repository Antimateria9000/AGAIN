from __future__ import annotations

def load_legacy_benchmark_rows(config: dict) -> dict:
    from app.benchmark_utils import load_benchmark_history
    from app.config_loader import load_benchmark_tickers

    benchmark_tickers = load_benchmark_tickers(config)
    history_df, entries = load_benchmark_history(config, benchmark_tickers)
    return {
        "entries": entries,
        "history_columns": [tuple(value) if isinstance(value, tuple) else value for value in history_df.columns.tolist()],
        "history_rows": history_df.reset_index(drop=True).to_dict(orient="records"),
    }
