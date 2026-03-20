from __future__ import annotations

import sqlite3
from pathlib import Path

TABLE_NAME = "benchmarks"


def ensure_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                benchmark_date TEXT NOT NULL,
                model_name TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.commit()


def save_benchmark_payload(db_path: Path, benchmark_date: str, model_name: str, payload_json: str) -> None:
    ensure_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            f"INSERT INTO {TABLE_NAME} (benchmark_date, model_name, payload_json) VALUES (?, ?, ?)",
            (benchmark_date, model_name, payload_json),
        )
        connection.commit()


def load_benchmark_rows(db_path: Path) -> list[tuple[int, str, str, str]]:
    ensure_database(db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            f"SELECT id, benchmark_date, model_name, payload_json FROM {TABLE_NAME} ORDER BY benchmark_date DESC, id DESC"
        ).fetchall()
    return rows


def delete_benchmark_row(db_path: Path, row_id: int) -> bool:
    ensure_database(db_path)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(f"DELETE FROM {TABLE_NAME} WHERE id = ?", (row_id,))
        connection.commit()
    return cursor.rowcount > 0

