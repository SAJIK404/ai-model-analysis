#!/usr/bin/env python3
"""Generate visualization charts from the ai_models SQLite database."""

import logging
import sqlite3
import sys
import contextlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from db.ctes import LATEST_PRICING_CTE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "ai_models.db"
OUTPUT_DIR = PROJECT_ROOT / "output"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# One consistent color per provider, reused across all charts.
PROVIDER_COLORS = {
    "OpenAI": "#10A37F",
    "Anthropic": "#D97757",
    "Google": "#4285F4",
    "Meta": "#0668E1",
    "Mistral": "#FF7000",
    "Cohere": "#39594D",
}

BENCHMARK_ORDER = ["MMLU", "HumanEval", "MATH", "GPQA"]
CATEGORY_BENCHMARKS = {
    "coding": "HumanEval",
    "reasoning": "MMLU",
    "math": "MATH",
}




def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection with foreign keys enabled."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run scripts/load_data.py first."
        )
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def provider_color(provider: str) -> str:
    """Return the chart color for a provider."""
    return PROVIDER_COLORS.get(provider, "#888888")


def provider_legend_handles(providers: list[str] | None = None) -> list[Patch]:
    """Build legend patch handles for a given set of providers.

    If `providers` is None, return handles for all known providers.
    """
    if providers is None:
        providers = list(PROVIDER_COLORS.keys())
    providers = [p for p in providers if p in PROVIDER_COLORS]
    return [
        Patch(facecolor=PROVIDER_COLORS[p], edgecolor="black", linewidth=0.5, label=p)
        for p in providers
    ]


def fetch_model_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load model metrics joined with latest pricing and avg benchmark scores."""
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
    """Load benchmark scores as a models × benchmarks matrix."""
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
    matrix = matrix.reindex(columns=BENCHMARK_ORDER)
    providers = raw.drop_duplicates("model_id").set_index("model_id")["provider"]
    matrix["provider"] = providers
    return matrix


def fetch_price_evolution(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load pricing history grouped by provider."""
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
    """Load the top 3 models per benchmark category for grouped bar chart."""
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


def cap_at_percentile(series: pd.Series, percentile: float = 90) -> pd.Series:
    """Cap values at the given percentile to reduce outlier distortion."""
    cap = np.percentile(series, percentile)
    return series.clip(upper=cap)


def normalize_column(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to 0–1; constant columns map to 0.5."""
    col_min = series.min()
    col_max = series.max()
    if col_max == col_min:
        return pd.Series(0.5, index=series.index)
    return (series - col_min) / (col_max - col_min)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a figure to the output directory as PNG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart: %s", path)
    return path





def plot_speed_vs_cost_scatter(summary: pd.DataFrame) -> Path:
    """Scatter plot of throughput vs. average cost, sized by benchmark score."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in summary.iterrows():
        color = provider_color(row["provider"])
        ax.scatter(
            row["avg_cost_per_1m"],
            row["tokens_per_second"],
            s=row["avg_benchmark_score"] * 8,
            c=color,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        ax.annotate(
            row["model_id"],
            (row["avg_cost_per_1m"], row["tokens_per_second"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.set_xlabel("Average Cost per 1M Tokens (USD)")
    ax.set_ylabel("Tokens per Second")
    ax.set_title("Speed vs. Cost (bubble size = avg benchmark score)")
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=provider_legend_handles(sorted(summary["provider"].unique())),
        title="Provider",
        loc="best",
    )

    size_ref = [60, 80, 100]
    size_handles = [
        ax.scatter([], [], s=s * 8, c="gray", alpha=0.5, edgecolors="black", linewidths=0.5)
        for s in size_ref
    ]
    size_legend = ax.legend(
        size_handles,
        [f"Avg score {s}" for s in size_ref],
        title="Bubble Size",
        loc="lower right",
    )
    ax.add_artist(size_legend)
    fig.tight_layout()
    return save_figure(fig, "02_speed_vs_cost_scatter.png")


def plot_value_score_bar(summary: pd.DataFrame) -> Path:
    """Bar chart ranking models by value score."""
    ranked = summary.sort_values("value_score", ascending=True)
    colors = [provider_color(p) for p in ranked["provider"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(ranked["model_id"], ranked["value_score"], color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Value Score (avg benchmark / avg cost per 1M tokens)")
    ax.set_ylabel("Model")
    ax.set_title("Value Score Ranking Across All Models")
    ax.grid(axis="x", alpha=0.3)

    for bar, score in zip(bars, ranked["value_score"]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}",
            va="center",
            fontsize=9,
        )

    ax.legend(handles=provider_legend_handles(sorted(summary["provider"].unique())), title="Provider", loc="lower right")
    fig.tight_layout()
    return save_figure(fig, "03_value_score_ranking.png")


def plot_provider_summary(summary: pd.DataFrame) -> Path:
    """Bar chart: average value score per provider (client-facing summary)."""
    agg = (
        summary.groupby("provider")["value_score"]
        .mean()
        .reset_index()
        .sort_values("value_score", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [provider_color(p) for p in agg["provider"]]
    bars = ax.barh(agg["provider"], agg["value_score"], color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Average Value Score (avg benchmark / avg cost)")
    ax.set_title("Provider-Level Value Summary")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, agg["value_score"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=9)

    fig.tight_layout()
    return save_figure(fig, "01_provider_summary.png")





 

def plot_benchmark_heatmap(matrix: pd.DataFrame) -> Path:
    """Heatmap of benchmark scores (models × benchmarks)."""
    score_cols = BENCHMARK_ORDER
    scores = matrix[score_cols].values
    model_ids = matrix.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(scores, aspect="auto", cmap="YlGnBu", vmin=30, vmax=100)

    ax.set_xticks(range(len(score_cols)))
    ax.set_xticklabels(score_cols)
    ax.set_yticks(range(len(model_ids)))
    ax.set_yticklabels(model_ids)
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Model")
    ax.set_title("Benchmark Scores Heatmap (Models × Benchmarks)")

    for i, model_id in enumerate(model_ids):
        for j, val in enumerate(scores[i]):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Score (0–100)")

    seen = set()
    provider_handles = []
    for model_id in model_ids:
        prov = matrix.loc[model_id, "provider"]
        if prov in seen:
            continue
        seen.add(prov)
        provider_handles.append(
            Line2D(
                [0], [0],
                marker="s",
                color="w",
                markerfacecolor=provider_color(prov),
                markersize=10,
                label=prov,
            )
        )
    ax.legend(handles=provider_handles, title="Provider", loc="upper left", bbox_to_anchor=(1.12, 1.0))
    fig.tight_layout()
    return save_figure(fig, "04_benchmark_heatmap.png")


def plot_best_per_category(top3: pd.DataFrame) -> Path:
    """Grouped bar chart showing top 3 models per benchmark category."""
    categories = ["coding", "reasoning", "math"]
    category_labels = [c.capitalize() for c in categories]
    rank_labels = ["1st", "2nd", "3rd"]
    n_cats = len(categories)
    n_ranks = 3
    bar_width = 0.22
    x = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=(12, 7))

    for rank in range(1, n_ranks + 1):
        offset = (rank - 2) * bar_width
        for cat_idx, category in enumerate(categories):
            row = top3[(top3["category"] == category) & (top3["category_rank"] == rank)]
            if row.empty:
                continue
            row = row.iloc[0]
            bar = ax.bar(
                x[cat_idx] + offset,
                row["score"],
                bar_width,
                color=provider_color(row["provider"]),
                edgecolor="black",
                linewidth=0.5,
                label=rank_labels[rank - 1] if cat_idx == 0 else None,
            )
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                bar[0].get_height() + 1.0,
                row["model_id"].replace("-", "\n"),
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(category_labels)
    ax.set_xlabel("Benchmark Category")
    ax.set_ylabel("Score (0–100)")
    ax.set_title("Top 3 Models per Benchmark Category")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    ax.legend(
        handles=provider_legend_handles(sorted(top3["provider"].unique())),
        title="Provider",
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    return save_figure(fig, "05_best_per_category.png")


def generate_all_charts(conn: sqlite3.Connection) -> list[Path]:
    """Generate and save all visualization charts."""
    summary = fetch_model_summary(conn)
    matrix = fetch_benchmark_matrix(conn)
    pricing = fetch_price_evolution(conn)
    top3 = fetch_category_top3(conn)

    paths = [
        plot_provider_summary(summary),
        plot_speed_vs_cost_scatter(summary),
        plot_value_score_bar(summary),
        # visualization 04 removed by request — provider-level cost distribution is not generated here
        plot_benchmark_heatmap(matrix),
        plot_best_per_category(top3),
    ]
    return paths


def generate_selected_charts(conn: sqlite3.Connection, chart_ids: list[str]) -> list[Path]:
    """Regenerate only the specified chart PNGs (e.g. ['01', '04', '06'])."""
    # Support charts 01–05; warn on unknown IDs but don't fail.
    allowed = {"01", "02", "03", "04", "05"}
    invalid = [c for c in chart_ids if c not in allowed]
    if invalid:
        logger.warning("Ignoring unknown chart ids: %s", invalid)

    chart_ids = [c for c in chart_ids if c in allowed]
    paths: list[Path] = []
    # Fetch common datasets lazily to avoid extra queries
    summary = None
    matrix = None
    pricing = None
    top3 = None

    for cid in chart_ids:
        if cid == "01":
            summary = summary or fetch_model_summary(conn)
            paths.append(plot_provider_summary(summary))
        elif cid == "02":
            summary = summary or fetch_model_summary(conn)
            paths.append(plot_speed_vs_cost_scatter(summary))
        elif cid == "03":
            summary = summary or fetch_model_summary(conn)
            paths.append(plot_value_score_bar(summary))
        elif cid == "04":
            matrix = matrix or fetch_benchmark_matrix(conn)
            paths.append(plot_benchmark_heatmap(matrix))
        elif cid == "05":
            top3 = top3 or fetch_category_top3(conn)
            paths.append(plot_best_per_category(top3))

    return paths


def main() -> int:
    """Generate all charts and save them to output/."""
    try:
        # Use contextlib.closing to ensure the connection is closed explicitly
        with contextlib.closing(get_connection()) as conn:
            paths = generate_all_charts(conn)

        print("\nCharts saved to output/:")
        for path in paths:
            print(f"  {path.name}")
        logger.info("All charts generated successfully")
        return 0
    except Exception:
        logger.exception("Chart generation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
