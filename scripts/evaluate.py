#!/usr/bin/env python3
"""Run evaluation analyses and produce CSV reports."""
import sys
import contextlib
from pathlib import Path
import logging
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from ai_config import load_config
cfg = load_config()
DB_PATH = Path(cfg.get("db_path", str(PROJECT_ROOT / "data" / "ai_models.db")))
OUTPUT_DIR = PROJECT_ROOT / "output"

from src.processing import (
    fetch_model_summary as fetch_summary_sql,
)
from src.evaluation import write_bootstrap_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}. Run scripts/load_data.py first.")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def main() -> int:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(get_connection()) as conn:
            # fetch SQL tables as DataFrames
            # fetch_model_summary returns joined table; for bootstrap we need raw tables
            # we'll load raw tables directly
            import pandas as pd
            models = pd.read_sql_query("SELECT * FROM models", conn)
            benchmarks = pd.read_sql_query("SELECT * FROM benchmarks", conn)
            pricing = pd.read_sql_query("SELECT * FROM pricing", conn)

            out_csv = OUTPUT_DIR / "evaluation_bootstrap_report.csv"
            write_bootstrap_report(models, benchmarks, pricing, out_csv, n_boot=200, random_state=0)

        print(f"Wrote evaluation report to: {out_csv}")
        return 0
    except Exception:
        logger.exception("Evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
