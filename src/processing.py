from pathlib import Path
import sqlite3
import pandas as pd
from db.ctes import LATEST_PRICING_CTE


def fetch_model_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = f"""
    WITH
    {LATEST_PRICING_CTE},
    avg_benchmarks AS (
        SELECT model_id, AVG(score) AS avg_benchmark_score
        FROM benchmarks
        GROUP BY model_id
    ),
    current_pricing AS (
        SELECT model_id, avg_cost_per_1m
        FROM latest_pricing
        WHERE rn = 1
    )
    SELECT
        m.model_id,
        m.provider,
        m.model_name,
        m.tokens_per_second,
        cp.avg_cost_per_1m,
        ab.avg_benchmark_score,
        ab.avg_benchmark_score / NULLIF(cp.avg_cost_per_1m, 0) AS value_score
    FROM models m
    JOIN current_pricing cp ON m.model_id = cp.model_id
    JOIN avg_benchmarks ab ON m.model_id = ab.model_id
    ORDER BY m.model_id;
    """
    return pd.read_sql_query(sql, conn)


def fetch_benchmark_matrix(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = """
    SELECT
        m.model_id,
        m.provider,
        b.benchmark_name,
        b.score
    FROM benchmarks b
    JOIN models m ON b.model_id = m.model_id;
    """
    raw = pd.read_sql_query(sql, conn)
    matrix = raw.pivot(index="model_id", columns="benchmark_name", values="score")
    # attach provider per model_id and ensure a sensible column order
    default_order = ["MMLU", "HumanEval", "MATH", "GPQA"]
    try:
        providers = raw.drop_duplicates("model_id").set_index("model_id")["provider"]
        matrix["provider"] = providers
    except Exception:
        # best-effort: leave as-is if provider mapping fails
        pass
    # ensure columns exist in expected order if possible
    cols = [c for c in default_order if c in matrix.columns]
    # keep provider as the last column (if present)
    if "provider" in matrix.columns:
        cols = cols + ["provider"]
    matrix = matrix.reindex(columns=cols)
    return matrix


def fetch_price_evolution(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = """
    SELECT
        m.provider,
        p.model_id,
        p.effective_date,
        (p.input_cost_per_1m + p.output_cost_per_1m) / 2.0 AS avg_cost_per_1m
    FROM pricing p
    JOIN models m ON p.model_id = m.model_id
    ORDER BY p.effective_date;
    """
    return pd.read_sql_query(sql, conn)


def fetch_category_top3(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = """
    WITH category_scores AS (
        SELECT
            b.model_id,
            m.provider,
            CASE b.benchmark_name
                WHEN 'HumanEval' THEN 'coding'
                WHEN 'MMLU' THEN 'reasoning'
                WHEN 'GPQA' THEN 'reasoning'
                WHEN 'MATH' THEN 'math'
                ELSE 'other'
            END AS category,
            b.score
        FROM benchmarks b
        JOIN models m ON b.model_id = m.model_id
    ),
    category_avg AS (
        SELECT model_id, provider, category, AVG(score) AS score
        FROM category_scores
        WHERE category <> 'other'
        GROUP BY model_id, provider, category
    ),
    ranked AS (
        SELECT
            category,
            model_id,
            provider,
            score,
            RANK() OVER (
                PARTITION BY category
                ORDER BY score DESC
            ) AS category_rank
        FROM category_avg
    )
    SELECT category, model_id, provider, score, category_rank
    FROM ranked
    WHERE category_rank <= 3
    ORDER BY category, category_rank;
    """
    return pd.read_sql_query(sql, conn)


def compute_value_score_from_dfs(models: pd.DataFrame, benchmarks: pd.DataFrame, pricing: pd.DataFrame) -> pd.DataFrame:
    # avg benchmark per model
    ab = benchmarks.groupby('model_id', as_index=False)['score'].mean().rename(columns={'score':'avg_benchmark_score'})
    # latest pricing per model by effective_date
    pricing['effective_date'] = pd.to_datetime(pricing['effective_date'])
    pricing['avg_cost_per_1m'] = (pricing['input_cost_per_1m'] + pricing['output_cost_per_1m'])/2.0
    latest = pricing.sort_values('effective_date').groupby('model_id', as_index=False).last()[['model_id','avg_cost_per_1m']]
    merged = models.merge(latest, on='model_id').merge(ab, on='model_id')
    merged['value_score'] = merged['avg_benchmark_score'] / merged['avg_cost_per_1m'].replace(0, pd.NA)
    return merged


def cap_at_percentile(series: pd.Series, percentile: float = 90) -> pd.Series:
    """Cap values at the given percentile to reduce outlier distortion."""
    cap = series.quantile(percentile / 100.0)
    return series.clip(upper=cap)


def normalize_column(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to 0–1; constant columns map to 0.5."""
    col_min = series.min()
    col_max = series.max()
    if col_max == col_min:
        return pd.Series(0.5, index=series.index)
    return (series - col_min) / (col_max - col_min)
