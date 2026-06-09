#!/usr/bin/env python3
"""Load CSV data into the ai_models SQLite database."""

import contextlib
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from ai_config import load_config

cfg = load_config()
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = Path(cfg.get("db_path", str(DATA_DIR / "ai_models.db")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCHEMA_SQL = """
DROP TABLE IF EXISTS pricing;
DROP TABLE IF EXISTS benchmarks;
DROP TABLE IF EXISTS models;

CREATE TABLE models (
    model_id              TEXT PRIMARY KEY,
    provider              TEXT NOT NULL,
    model_name            TEXT NOT NULL,
    version               TEXT NOT NULL,
    release_date          TEXT NOT NULL,
    context_window_tokens INTEGER NOT NULL CHECK (context_window_tokens > 0),
    tokens_per_second     REAL NOT NULL CHECK (tokens_per_second > 0)
);

CREATE TABLE benchmarks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id        TEXT NOT NULL REFERENCES models(model_id),
    benchmark_name  TEXT NOT NULL CHECK (benchmark_name IN ('MMLU', 'HumanEval', 'MATH', 'GPQA')),
    score           REAL NOT NULL CHECK (score >= 0 AND score <= 100),
    source_url      TEXT NOT NULL,
    recorded_date   TEXT NOT NULL,
    UNIQUE (model_id, benchmark_name, recorded_date)
);

CREATE TABLE pricing (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id            TEXT NOT NULL REFERENCES models(model_id),
    input_cost_per_1m   REAL NOT NULL CHECK (input_cost_per_1m >= 0),
    output_cost_per_1m  REAL NOT NULL CHECK (output_cost_per_1m >= 0),
    currency            TEXT NOT NULL DEFAULT 'USD',
    effective_date      TEXT NOT NULL,
    source_url          TEXT NOT NULL,
    UNIQUE (model_id, effective_date)
);

CREATE INDEX idx_benchmarks_model_id ON benchmarks(model_id);
CREATE INDEX idx_benchmarks_name ON benchmarks(benchmark_name);
CREATE INDEX idx_pricing_model_id ON pricing(model_id);
CREATE INDEX idx_pricing_effective_date ON pricing(effective_date);
CREATE INDEX idx_models_provider ON models(provider);
"""

# Tablas y columnas FK conocidas en tiempo de compilación; no hay
# interpolación de entrada externa, por lo que el uso de f-string es seguro.
_FK_CHECKS: list[tuple[str, str]] = [
    ("benchmarks", "model_id"),
    ("pricing", "model_id"),
]


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection, db_path: Path = DB_PATH) -> None:
    """Create database tables and indexes.

    WARNING: drops and recreates all tables on every call.
    Intended only for initial setup or full reloads.
    """
    logger.info("Creating database schema at %s", db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    logger.info("Schema created successfully")


def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame."""
    logger.info("Reading %s", path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def _to_python(row: tuple) -> tuple:
    """Convert numpy/pandas scalar types to native Python for SQLite."""
    return tuple(value.item() if hasattr(value, "item") else value for value in row)


def insert_models(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert model records from a DataFrame."""
    rows = [_to_python(row) for row in df.to_records(index=False)]
    conn.executemany(
        """
        INSERT INTO models (
            model_id, provider, model_name, version,
            release_date, context_window_tokens, tokens_per_second
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    count = len(rows)
    logger.info("Inserted %d model(s)", count)
    return count


def insert_benchmarks(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert benchmark records from a DataFrame."""
    rows = [_to_python(row) for row in df.to_records(index=False)]
    conn.executemany(
        """
        INSERT INTO benchmarks (
            model_id, benchmark_name, score, source_url, recorded_date
        ) VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    count = len(rows)
    logger.info("Inserted %d benchmark record(s)", count)
    return count


def insert_pricing(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert pricing records from a DataFrame."""
    rows = [_to_python(row) for row in df.to_records(index=False)]
    conn.executemany(
        """
        INSERT INTO pricing (
            model_id, input_cost_per_1m, output_cost_per_1m,
            currency, effective_date, source_url
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    count = len(rows)
    logger.info("Inserted %d pricing record(s)", count)
    return count


def validate_referential_integrity(conn: sqlite3.Connection) -> None:
    """Verify all foreign keys resolve to existing models."""
    cursor = conn.cursor()
    for table, column in _FK_CHECKS:
        cursor.execute(
            f"""
            SELECT COUNT(*) FROM {table} t
            WHERE NOT EXISTS (
                SELECT 1 FROM models m WHERE m.model_id = t.{column}
            )
            """
        )
        orphans = cursor.fetchone()[0]
        if orphans:
            raise ValueError(f"{orphans} orphan row(s) in {table}.{column}")
    logger.info("Referential integrity check passed")


def print_summary(conn: sqlite3.Connection) -> None:
    """Print row counts for each table."""
    cursor = conn.cursor()
    for table in ("models", "benchmarks", "pricing"):
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} row(s)")


def main() -> int:
    """Create schema and load all CSV files into SQLite."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        models_df = load_csv(DATA_DIR / "models.csv")
        benchmarks_df = load_csv(DATA_DIR / "benchmarks.csv")
        pricing_df = load_csv(DATA_DIR / "pricing.csv")

        # contextlib.closing garantiza el cierre explícito de la conexión;
        # el context manager nativo de sqlite3 sólo gestiona transacciones.
        with contextlib.closing(get_connection(DB_PATH)) as conn:
            create_schema(conn, DB_PATH)
            insert_models(conn, models_df)
            insert_benchmarks(conn, benchmarks_df)
            insert_pricing(conn, pricing_df)
            validate_referential_integrity(conn)

            print("\nDatabase loaded successfully:")
            print_summary(conn)

        return 0
    except Exception:
        logger.exception("Failed to load data")
        return 1


if __name__ == "__main__":
    sys.exit(main())