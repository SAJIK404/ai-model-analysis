from pathlib import Path
import sqlite3
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def create_schema(conn: sqlite3.Connection, schema_sql: str) -> None:
    conn.executescript(schema_sql)
    conn.commit()
    logger.info("Schema created via src.ingestion.create_schema")


def insert_table_from_df(conn: sqlite3.Connection, table: str, df: pd.DataFrame, columns: list[str]):
    rows = [tuple(v) for v in df[columns].to_records(index=False)]
    placeholders = ",".join(["?" for _ in columns])
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    conn.executemany(sql, rows)
    conn.commit()
    logger.info("Inserted %d rows into %s", len(rows), table)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)
