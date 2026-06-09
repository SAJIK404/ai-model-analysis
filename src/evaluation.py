import pandas as pd
import numpy as np
from typing import Optional
from src.processing import compute_value_score_from_dfs


def spearman_rank_correlation(a: pd.Series, b: pd.Series) -> float:
    """Compute Spearman rank correlation between two series.

    Returns NaN if insufficient data.
    """
    # avoid optional scipy dependency by correlating ranks with Pearson
    return a.rank().corr(b.rank(), method='pearson')


def compare_value_scores(df: pd.DataFrame, col='value_score') -> pd.DataFrame:
    """Return a simple summary of value_score distribution per provider."""
    return df.groupby('provider')[col].agg(['mean', 'median', 'std', 'count']).reset_index()


def bootstrap_value_score_spearman(models: pd.DataFrame, benchmarks: pd.DataFrame, pricing: pd.DataFrame,
                                  n_boot: int = 200, random_state: Optional[int] = None) -> pd.Series:
    """Bootstrap Spearman rank correlations of value_score by resampling benchmark records.

    The function resamples the `benchmarks` DataFrame with replacement and recomputes
    `value_score` using `compute_value_score_from_dfs`. It returns a Series of Spearman
    correlations between the original value_score ranking and each bootstrap sample.
    """
    rng = np.random.default_rng(random_state)

    # baseline value_score
    base = compute_value_score_from_dfs(models.copy(), benchmarks.copy(), pricing.copy())
    base_idx = base.set_index('model_id')['value_score']
    base_idx = base_idx.dropna()

    corrs = []
    n = len(benchmarks)
    for _ in range(n_boot):
        sampled = benchmarks.sample(n=n, replace=True, random_state=rng.integers(0, 2**31 - 1))
        boot_df = compute_value_score_from_dfs(models.copy(), sampled, pricing.copy())
        boot_idx = boot_df.set_index('model_id')['value_score']
        # align to common index
        joined = base_idx.to_frame('base').join(boot_idx.rename('boot'))
        if joined['boot'].dropna().empty:
            corrs.append(np.nan)
            continue
        corr = joined['base'].rank().corr(joined['boot'].rank(), method='pearson')
        corrs.append(corr)

    return pd.Series(corrs)


def write_bootstrap_report(models: pd.DataFrame, benchmarks: pd.DataFrame, pricing: pd.DataFrame,
                           out_csv_path, n_boot: int = 200, random_state: Optional[int] = None):
    """Run bootstrap and write a small CSV summarizing the distribution of Spearman correlations."""
    corrs = bootstrap_value_score_spearman(models, benchmarks, pricing, n_boot=n_boot, random_state=random_state)
    summary = {
        'n_boot': [len(corrs)],
        'mean_spearman': [float(np.nanmean(corrs))],
        'std_spearman': [float(np.nanstd(corrs, ddof=1))],
        'p10': [float(np.nanpercentile(corrs, 10))],
        'p50': [float(np.nanpercentile(corrs, 50))],
        'p90': [float(np.nanpercentile(corrs, 90))],
    }
    df = pd.DataFrame(summary)
    df.to_csv(out_csv_path, index=False)
    return out_csv_path
