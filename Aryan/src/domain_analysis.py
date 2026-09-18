"""Domain-level correlation analysis (4x4 domain heatmap)."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from .config import DOMAIN_PREFIXES, DOMAIN_NAMES, DOMAIN_ORDER


def get_domain_columns(statement_cols: List[str]) -> Dict[str, List[str]]:
    """Group statement columns by domain prefix."""
    domain_cols = {}
    for prefix, name in DOMAIN_PREFIXES.items():
        domain_cols[name] = [c for c in statement_cols if c.split(".")[0][0] == prefix]
    return domain_cols


def compute_domain_correlations(
    corr: pd.DataFrame,
    statement_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute domain-level correlations.
    Returns: (domain_corr_matrix, within_domain_stats)
    """
    domain_cols = get_domain_columns(statement_cols)

    # 4x4 domain correlation matrix
    domains = DOMAIN_NAMES
    domain_corr = pd.DataFrame(index=domains, columns=domains, dtype=float)

    for d1 in domains:
        for d2 in domains:
            cols1 = domain_cols[d1]
            cols2 = domain_cols[d2]
            if d1 == d2:
                # Within-domain: average of all pairwise correlations (excluding diagonal)
                if len(cols1) > 1:
                    pairs = []
                    for i, c1 in enumerate(cols1):
                        for c2 in cols1[i+1:]:
                            pairs.append(corr.loc[c1, c2])
                    domain_corr.loc[d1, d2] = np.nanmean(pairs) if pairs else np.nan
                else:
                    domain_corr.loc[d1, d2] = np.nan
            else:
                # Cross-domain: average of all cross pairs
                pairs = []
                for c1 in cols1:
                    for c2 in cols2:
                        pairs.append(corr.loc[c1, c2])
                domain_corr.loc[d1, d2] = np.nanmean(pairs) if pairs else np.nan

    # Within-domain statistics
    within_stats = []
    for d in domains:
        cols = domain_cols[d]
        if len(cols) > 1:
            pairs = []
            for i, c1 in enumerate(cols):
                for c2 in cols[i+1:]:
                    r = corr.loc[c1, c2]
                    pairs.append(r)
            if pairs:
                within_stats.append({
                    "domain": d,
                    "n_statements": len(cols),
                    "mean_correlation": np.nanmean(pairs),
                    "median_correlation": np.nanmedian(pairs),
                    "std_correlation": np.nanstd(pairs),
                    "min_correlation": np.nanmin(pairs),
                    "max_correlation": np.nanmax(pairs),
                })

    within_df = pd.DataFrame(within_stats)

    return domain_corr, within_df


def compute_statement_means(df_num: pd.DataFrame, statement_cols: List[str]) -> pd.DataFrame:
    """Compute mean agreement per statement, grouped by domain."""
    domain_cols = get_domain_columns(statement_cols)

    means = df_num[statement_cols].mean().rename("mean_score")
    stds = df_num[statement_cols].std().rename("std_score")

    # Add domain info
    domain_map = {}
    for prefix, name in DOMAIN_PREFIXES.items():
        for c in domain_cols[name]:
            domain_map[c] = name

    result = pd.DataFrame({
        "mean": means,
        "std": stds,
        "domain": pd.Series(domain_map)
    })

    return result


def print_domain_summary(domain_corr: pd.DataFrame, within_df: pd.DataFrame) -> None:
    """Pretty print domain analysis."""
    print("=" * 50)
    print("DOMAIN-LEVEL CORRELATION MATRIX (4x4)")
    print("=" * 50)
    print(domain_corr.round(4).to_string())
    print()
    print("=" * 50)
    print("WITHIN-DOMAIN CORRELATION STATISTICS")
    print("=" * 50)
    print(within_df.round(4).to_string(index=False))


if __name__ == "__main__":
    from src import preprocess, compute_correlation_matrix

    df_num, stmt_cols, col_to_domain, validation = preprocess()
    corr = compute_correlation_matrix(df_num)

    domain_corr, within_df = compute_domain_correlations(corr, stmt_cols)
    print_domain_summary(domain_corr, within_df)

    # Also print statement means
    means_df = compute_statement_means(df_num, stmt_cols)
    print()
    print("=" * 50)
    print("STATEMENT MEANS (TOP 5 / BOTTOM 5)")
    print("=" * 50)
    print("Top 5:")
    print(means_df.nlargest(5, "mean")[["mean", "domain"]].to_string())
    print("\nBottom 5:")
    print(means_df.nsmallest(5, "mean")[["mean", "domain"]].to_string())