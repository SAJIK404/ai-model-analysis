#!/usr/bin/env python3
"""Run analytical SQL queries against the ai_models SQLite database."""

import contextlib
import logging
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from ai_config import load_config

cfg = load_config()
DB_PATH = Path(cfg.get("db_path", str(PROJECT_ROOT / "data" / "ai_models.db")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from db.ctes import LATEST_PRICING_CTE

# Precio promedio vigente por modelo (filtra rn = 1 del CTE anterior).
# Depende de _CTE_LATEST_PRICING; siempre debe declararse después de él.
_CTE_CURRENT_PRICING = """
current_pricing AS (
    SELECT model_id, avg_cost_per_1m
    FROM latest_pricing
    WHERE rn = 1
)"""

# Score promedio de benchmarks por modelo (escala 0-100).
_CTE_AVG_BENCHMARKS = """
avg_benchmarks AS (
    SELECT
        model_id,
        AVG(score) AS avg_benchmark_score
    FROM benchmarks
    GROUP BY model_id
)"""


QUERY_VALUE_SCORE = f"""
WITH
{LATEST_PRICING_CTE},
{_CTE_AVG_BENCHMARKS},
{_CTE_CURRENT_PRICING}
SELECT
    m.model_id,
    m.provider,
    m.model_name,
    ROUND(ab.avg_benchmark_score, 2)                                    AS avg_benchmark_score,
    ROUND(cp.avg_cost_per_1m, 4)                                        AS avg_cost_per_1m,
    ROUND(ab.avg_benchmark_score / NULLIF(cp.avg_cost_per_1m, 0), 4)   AS value_score,
    RANK() OVER (
        ORDER BY ab.avg_benchmark_score / NULLIF(cp.avg_cost_per_1m, 0) DESC
    )                                                                   AS value_rank
FROM models m
JOIN avg_benchmarks  ab ON m.model_id = ab.model_id
JOIN current_pricing cp ON m.model_id = cp.model_id
ORDER BY value_score DESC;
"""

QUERY_BEST_PER_PRICE_TIER = f"""
WITH
{LATEST_PRICING_CTE},
{_CTE_AVG_BENCHMARKS},
{_CTE_CURRENT_PRICING},
model_metrics AS (
    SELECT
        m.model_id,
        m.provider,
        m.model_name,
        ab.avg_benchmark_score,
        cp.avg_cost_per_1m,
        ab.avg_benchmark_score / NULLIF(cp.avg_cost_per_1m, 0) AS value_score,
        CASE
            WHEN cp.avg_cost_per_1m <= 1.50 THEN 'budget'
            WHEN cp.avg_cost_per_1m <= 5.00 THEN 'mid'
            ELSE 'premium'
        END AS price_tier
    FROM models m
    JOIN avg_benchmarks  ab ON m.model_id = ab.model_id
    JOIN current_pricing cp ON m.model_id = cp.model_id
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY price_tier
            ORDER BY value_score DESC
        ) AS tier_rank
    FROM model_metrics
)
SELECT
    price_tier,
    model_id,
    provider,
    model_name,
    ROUND(avg_benchmark_score, 2) AS avg_benchmark_score,
    ROUND(avg_cost_per_1m, 4)     AS avg_cost_per_1m,
    ROUND(value_score, 4)         AS value_score
FROM ranked
WHERE tier_rank = 1
ORDER BY
    CASE price_tier WHEN 'budget' THEN 1 WHEN 'mid' THEN 2 ELSE 3 END;
"""

QUERY_BENCHMARK_RANKING_BY_CATEGORY = """
WITH category_scores AS (
    SELECT
        b.model_id,
        m.provider,
        m.model_name,
        -- ELSE 'other' evita NULLs silenciosos si se añaden benchmarks nuevos.
        CASE b.benchmark_name
            WHEN 'HumanEval' THEN 'coding'
            WHEN 'MMLU'      THEN 'reasoning'
            WHEN 'GPQA'      THEN 'reasoning'
            WHEN 'MATH'      THEN 'math'
            ELSE 'other'
        END AS category,
        b.score
    FROM benchmarks b
    JOIN models m ON b.model_id = m.model_id
),
category_avg AS (
    SELECT
        model_id,
        provider,
        model_name,
        category,
        AVG(score) AS category_score
    FROM category_scores
    -- Excluye benchmarks no reconocidos del ranking principal.
    WHERE category <> 'other'
    GROUP BY model_id, provider, model_name, category
),
ranked AS (
    SELECT
        category,
        model_id,
        provider,
        model_name,
        ROUND(category_score, 2) AS category_score,
        RANK() OVER (
            PARTITION BY category
            ORDER BY category_score DESC
        ) AS category_rank
    FROM category_avg
)
SELECT
    category,
    model_id,
    provider,
    model_name,
    category_score,
    category_rank
FROM ranked
ORDER BY
    CASE category WHEN 'coding' THEN 1 WHEN 'reasoning' THEN 2 ELSE 3 END,
    category_rank;
"""

QUERY_PRICE_EVOLUTION = """
WITH pricing_timeline AS (
    SELECT
        m.provider,
        p.model_id,
        p.effective_date,
        (p.input_cost_per_1m + p.output_cost_per_1m) / 2.0 AS avg_cost_per_1m,
        LAG((p.input_cost_per_1m + p.output_cost_per_1m) / 2.0) OVER (
            PARTITION BY p.model_id
            ORDER BY p.effective_date
        ) AS prev_avg_cost,
        ROW_NUMBER() OVER (
            PARTITION BY p.model_id
            ORDER BY p.effective_date
        ) AS price_version
    FROM pricing p
    JOIN models m ON p.model_id = m.model_id
)
SELECT
    provider,
    model_id,
    effective_date,
    ROUND(avg_cost_per_1m, 4) AS avg_cost_per_1m,
    ROUND(
        CASE
            WHEN prev_avg_cost IS NULL OR prev_avg_cost = 0 THEN NULL
            ELSE ((avg_cost_per_1m - prev_avg_cost) / prev_avg_cost) * 100.0
        END,
        2
    ) AS pct_change_from_prev,
    price_version
FROM pricing_timeline
ORDER BY provider, model_id, effective_date;
"""

QUERY_SPEED_VS_COST = f"""
WITH
{LATEST_PRICING_CTE},
{_CTE_AVG_BENCHMARKS},
{_CTE_CURRENT_PRICING}
SELECT
    m.model_id,
    m.provider,
    m.model_name,
    m.tokens_per_second,
    ROUND(cp.avg_cost_per_1m, 4)                                            AS avg_cost_per_1m,
    ROUND(ab.avg_benchmark_score, 2)                                        AS avg_benchmark_score,
    ROUND(m.tokens_per_second / NULLIF(cp.avg_cost_per_1m, 0), 4)          AS speed_per_dollar,
    RANK() OVER (
        ORDER BY m.tokens_per_second / NULLIF(cp.avg_cost_per_1m, 0) DESC
    )                                                                       AS efficiency_rank
FROM models m
JOIN current_pricing cp ON m.model_id = cp.model_id
JOIN avg_benchmarks  ab ON m.model_id = ab.model_id
ORDER BY efficiency_rank;
"""

QUERY_TOP_MODEL_PER_BENCHMARK = """
WITH ranked AS (
    SELECT
        b.benchmark_name,
        b.model_id,
        m.provider,
        m.model_name,
        b.score,
        RANK() OVER (
            PARTITION BY b.benchmark_name
            ORDER BY b.score DESC
        ) AS benchmark_rank
    FROM benchmarks b
    JOIN models m ON b.model_id = m.model_id
)
SELECT
    benchmark_name,
    model_id,
    provider,
    model_name,
    score,
    benchmark_rank
FROM ranked
WHERE benchmark_rank = 1
ORDER BY benchmark_name;
"""

QUERIES: list[tuple[str, str]] = [
    (
        "Query 1: Value Score (avg benchmark / avg cost per 1M tokens)",
        QUERY_VALUE_SCORE,
    ),
    (
        "Query 2: Best Model per Price Tier (budget / mid / premium)",
        QUERY_BEST_PER_PRICE_TIER,
    ),
    (
        "Query 3: Benchmark Ranking per Category (coding / reasoning / math)",
        QUERY_BENCHMARK_RANKING_BY_CATEGORY,
    ),
    (
        "Query 4: Price Evolution Over Time per Provider",
        QUERY_PRICE_EVOLUTION,
    ),
    (
        "Query 5: Speed vs. Cost Scatter Analysis",
        QUERY_SPEED_VS_COST,
    ),
    (
        "Query 6: Top Model per Benchmark Type",
        QUERY_TOP_MODEL_PER_BENCHMARK,
    ),
]


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection with foreign keys enabled."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run scripts/load_data.py first."
        )
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def run_query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame."""
    return pd.read_sql_query(sql, conn)


def print_query_results(title: str, df: pd.DataFrame) -> None:
    """Print a query title and formatted results to stdout."""
    separator = "=" * 72
    print(f"\n{separator}")
    print(title)
    print(separator)
    if df.empty:
        print("(no rows returned)")
    else:
        # max_colwidth=None evita truncamiento en terminales angostos
        # o cuando la salida se redirige a un archivo.
        print(df.to_string(index=False, max_colwidth=None))
    print()


def execute_all_queries(conn: sqlite3.Connection) -> None:
    """Run all analytical queries and print results."""
    for title, sql in QUERIES:
        logger.info("Running: %s", title)
        try:
            df = run_query(conn, sql)
        except Exception:
            # Re-lanza incluyendo el título para que el traceback en main()
            # identifique qué query falló.
            logger.exception("Failed while running: %s", title)
            raise
        print_query_results(title, df)


def main() -> int:
    """Execute all analytical queries against the database."""
    try:
        # contextlib.closing garantiza el cierre explícito de la conexión;
        # el context manager nativo de sqlite3 solo gestiona transacciones.
        with contextlib.closing(get_connection()) as conn:
            execute_all_queries(conn)
        logger.info("All queries completed successfully")
        return 0
    except Exception:
        logger.exception("Query execution failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())