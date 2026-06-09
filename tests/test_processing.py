import pandas as pd
import numpy as np
from src.processing import compute_value_score_from_dfs


def test_value_score_basic():
    models = pd.DataFrame([
        {"model_id": "m1", "provider": "A", "model_name": "m1", "version": "v1", "release_date": "2023-01-01", "context_window_tokens": 2048, "tokens_per_second": 100},
        {"model_id": "m2", "provider": "B", "model_name": "m2", "version": "v1", "release_date": "2023-01-01", "context_window_tokens": 2048, "tokens_per_second": 200},
    ])
    benchmarks = pd.DataFrame([
        {"model_id": "m1", "benchmark_name": "MMLU", "score": 80},
        {"model_id": "m2", "benchmark_name": "MMLU", "score": 60},
    ])
    pricing = pd.DataFrame([
        {"model_id": "m1", "input_cost_per_1m": 0.5, "output_cost_per_1m": 0.5, "currency": "USD", "effective_date": "2024-01-01", "source_url": "u"},
        {"model_id": "m2", "input_cost_per_1m": 1.0, "output_cost_per_1m": 1.0, "currency": "USD", "effective_date": "2024-01-01", "source_url": "u"},
    ])

    res = compute_value_score_from_dfs(models, benchmarks, pricing)
    # avg benchmark = score, avg cost = (in+out)/2
    m1 = res[res['model_id'] == 'm1'].iloc[0]
    m2 = res[res['model_id'] == 'm2'].iloc[0]
    assert np.isclose(m1['avg_benchmark_score'], 80)
    assert np.isclose(m1['avg_cost_per_1m'], 0.5)
    assert np.isclose(m1['value_score'], 160.0)
    assert np.isclose(m2['value_score'], 60.0)


def test_value_score_div_by_zero():
    models = pd.DataFrame([
        {"model_id": "m1", "provider": "A", "model_name": "m1", "version": "v1", "release_date": "2023-01-01", "context_window_tokens": 2048, "tokens_per_second": 100},
    ])
    benchmarks = pd.DataFrame([
        {"model_id": "m1", "benchmark_name": "MMLU", "score": 50},
    ])
    pricing = pd.DataFrame([
        {"model_id": "m1", "input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0, "currency": "USD", "effective_date": "2024-01-01", "source_url": "u"},
    ])

    res = compute_value_score_from_dfs(models, benchmarks, pricing)
    val = res.iloc[0]['value_score']
    assert pd.isna(val) or val is None
