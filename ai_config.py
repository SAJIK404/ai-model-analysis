"""Lightweight configuration loader for the project.

Looks for `conf/config.yaml` at project root. Returns a dict with defaults
when file is absent.
"""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    "db_path": str(PROJECT_ROOT / "data" / "ai_models.db"),
    "output_dir": str(PROJECT_ROOT / "output"),
    "chart_ids": ["01", "02", "03", "04", "05"],
    "benchmark_categories": {
        "coding": "HumanEval",
        "reasoning": ["MMLU", "GPQA"],
        "math": "MATH",
    },
}


def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "conf" / "config.yaml"
    if not cfg_path.exists():
        return DEFAULTS.copy()
    with cfg_path.open("r", encoding="utf8") as f:
        cfg = yaml.safe_load(f) or {}
    out = DEFAULTS.copy()
    out.update(cfg)
    return out
