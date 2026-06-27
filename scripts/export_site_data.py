#!/usr/bin/env python3
"""Export data and chart assets for the static GitHub Pages site."""
import contextlib
import json
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_config import load_config
import query_data

cfg = load_config()
DB_PATH = Path(cfg.get("db_path", str(PROJECT_ROOT / "data" / "ai_models.db")))
OUTPUT_DIR = PROJECT_ROOT / "output"
SITE_DIR = PROJECT_ROOT / "docs"
SITE_IMAGES = SITE_DIR / "images"
SITE_DATA = SITE_DIR / "data.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CHART_FILES = [
    "01_provider_summary.png",
    "02_speed_vs_cost_scatter.png",
    "03_value_score_ranking.png",
    "04_benchmark_heatmap.png",
    "05_best_per_category.png",
]


def _load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf8") as f:
        return json.load(f)


def _df_to_records(df):
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _copy_chart_images():
    SITE_IMAGES.mkdir(parents=True, exist_ok=True)
    copied = []
    for filename in CHART_FILES:
        src = OUTPUT_DIR / filename
        if not src.exists():
            logger.warning("Missing chart image: %s", src)
            continue
        dst = SITE_IMAGES / filename
        shutil.copy2(src, dst)
        copied.append(filename)
    return copied


def _build_payload(metadata, results):
    return {
        "project": {
            "name": "AI Model Benchmark Portfolio",
            "description": "An interactive technical portfolio for AI model value, pricing, and benchmark analysis.",
            "last_run": metadata.get("generated_at") if metadata else None,
            "commit": metadata.get("commit") if metadata else None,
        },
        "charts": [
            {"id": "01", "title": "Provider Value Summary", "file": "images/01_provider_summary.png"},
            {"id": "02", "title": "Speed vs Cost", "file": "images/02_speed_vs_cost_scatter.png"},
            {"id": "03", "title": "Value Score Ranking", "file": "images/03_value_score_ranking.png"},
            {"id": "04", "title": "Benchmark Heatmap", "file": "images/04_benchmark_heatmap.png"},
            {"id": "05", "title": "Best per Category", "file": "images/05_best_per_category.png"},
        ],
        "metrics": {
            "value_score_ranking": _df_to_records(results["value_score"]),
            "best_per_tier": _df_to_records(results["best_per_tier"]),
            "category_rankings": _df_to_records(results["category_rankings"]),
            "price_evolution": _df_to_records(results["price_evolution"]),
            "speed_vs_cost": _df_to_records(results["speed_vs_cost"]),
            "top_model_per_benchmark": _df_to_records(results["top_model_per_benchmark"]),
        },
        "metadata": {
            "manifest": _load_json(OUTPUT_DIR / "manifest.json"),
            "runs": _load_json(OUTPUT_DIR / "runs.json"),
        },
    }


def build_site_data() -> Path:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    with contextlib.closing(query_data.get_connection()) as conn:
        results["value_score"] = query_data.run_query(conn, query_data.QUERY_VALUE_SCORE)
        results["best_per_tier"] = query_data.run_query(conn, query_data.QUERY_BEST_PER_PRICE_TIER)
        results["category_rankings"] = query_data.run_query(conn, query_data.QUERY_BENCHMARK_RANKING_BY_CATEGORY)
        results["price_evolution"] = query_data.run_query(conn, query_data.QUERY_PRICE_EVOLUTION)
        results["speed_vs_cost"] = query_data.run_query(conn, query_data.QUERY_SPEED_VS_COST)
        results["top_model_per_benchmark"] = query_data.run_query(conn, query_data.QUERY_TOP_MODEL_PER_BENCHMARK)

    copied_charts = _copy_chart_images()
    if not copied_charts:
        logger.warning("No chart images were copied to docs/images/. Be sure visualize.py has generated PNGs.")

    manifest_json = _load_json(OUTPUT_DIR / "manifest.json") or {}
    runs_json = _load_json(OUTPUT_DIR / "runs.json") or []
    last_run = runs_json[-1] if isinstance(runs_json, list) and runs_json else {}
    run_metadata = {
        "generated_at": last_run.get("generated_at"),
        "commit": last_run.get("commit") or manifest_json.get("commit"),
    }

    payload = _build_payload(run_metadata, results)
    with SITE_DATA.open("w", encoding="utf8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Exported site data to %s", SITE_DATA)
    return SITE_DATA


def main() -> int:
    try:
        build_site_data()
        return 0
    except Exception:
        logger.exception("Failed to export site data")
        return 1


if __name__ == "__main__":
    sys.exit(main())
