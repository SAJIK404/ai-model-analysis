#!/usr/bin/env python3
"""Generate visualization charts from the ai_models SQLite database."""

import logging
import sqlite3
import sys
import contextlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from ai_config import load_config

cfg = load_config()
DB_PATH = Path(cfg.get("db_path", str(PROJECT_ROOT / "data" / "ai_models.db")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from db.ctes import LATEST_PRICING_CTE
from src.processing import (
    fetch_model_summary,
    fetch_benchmark_matrix,
    fetch_price_evolution,
    fetch_category_top3,
    compute_value_score_from_dfs,
    cap_at_percentile,
    normalize_column,
)

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








def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a figure to the output directory as PNG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart: %s", path)
    return path


def save_manifest(paths: list[Path]) -> Path:
    """Write a manifest.json to the output directory listing generated files and metadata."""
    import json
    import subprocess
    from datetime import datetime

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "manifest.json"
    commit = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        commit = None

    entries = []
    for p in paths:
        entries.append({
            "file": p.name,
            "path": str(p),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        })

    manifest = {"commit": commit, "files": entries}
    with manifest_path.open("w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest: %s", manifest_path)
    # Also append a run entry to runs.json for lightweight experiment tracking
    runs_path = OUTPUT_DIR / "runs.json"
    run_entry = {
        "commit": commit,
        "files": [p.name for p in paths],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        if runs_path.exists():
            with runs_path.open("r", encoding="utf8") as f:
                runs = json.load(f)
        else:
            runs = []
        runs.append(run_entry)
        with runs_path.open("w", encoding="utf8") as f:
            json.dump(runs, f, indent=2)
        logger.info("Appended run entry to: %s", runs_path)
    except Exception:
        logger.exception("Failed to update runs.json")

    return manifest_path





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
    paths = []
    paths.append(plot_provider_summary(summary))
    paths.append(plot_speed_vs_cost_scatter(summary))
    paths.append(plot_value_score_bar(summary))
    paths.append(plot_benchmark_heatmap(matrix))
    paths.append(plot_best_per_category(top3))

    # write manifest
    try:
        save_manifest(paths)
    except Exception:
        logger.exception("Failed to write manifest")

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
