"""Data loading, cleaning, and validation for survey responses."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List

from .config import (
    RAW_CSV,
    LIKERT_MAPPING,
    MISSING_STRINGS,
    DOMAIN_PREFIXES,
    DOMAIN_ORDER,
    DOMAIN_NAMES,
)


def load_raw_data() -> pd.DataFrame:
    """Load CSV with UTF-8 BOM handling."""
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    return df


def identify_columns(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """
    Identify respondent ID column and opinion statement columns.
    Returns: (statement_columns, column_to_domain_map, domain_array)
    """
    cols = df.columns.tolist()
    id_col = cols[0]  # "id. Response ID"
    statement_cols = cols[1:]

    col_to_domain = {}
    domain_array = []

    for col in statement_cols:
        prefix = col.split(".")[0]  # e.g., "T01"
        domain_char = prefix[0]  # e.g., "T"
        domain = DOMAIN_PREFIXES.get(domain_char, "Unknown")
        col_to_domain[col] = domain
        domain_array.append(domain)

    return statement_cols, col_to_domain, np.array(domain_array)


def encode_likert(df: pd.DataFrame, statement_cols: List[str]) -> pd.DataFrame:
    """Map Likert responses to numeric values (-2..+2), missing -> NaN."""
    df_num = df[statement_cols].copy()

    # Replace missing strings with NaN
    for ms in MISSING_STRINGS:
        df_num = df_num.replace(ms, np.nan)

    # Apply Likert mapping
    df_num = df_num.replace(LIKERT_MAPPING)

    # Ensure numeric
    df_num = df_num.apply(pd.to_numeric, errors="coerce")

    return df_num


def validate_data(df_num: pd.DataFrame, respondent_ids: pd.Series) -> Dict:
    """Run data quality checks and return summary dict."""
    n_respondents, n_statements = df_num.shape
    n_cells = n_respondents * n_statements
    n_missing = df_num.isna().sum().sum()
    missing_pct = n_missing / n_cells * 100

    # Check for duplicate respondents (by response pattern)
    dup_mask = df_num.duplicated(keep=False)
    n_duplicates = dup_mask.sum()

    # Per-statement missing counts
    stmt_missing = df_num.isna().sum().sort_values(ascending=False)

    # Per-respondent missing counts
    resp_missing = df_num.isna().sum(axis=1).sort_values(ascending=False)

    # Unique response values check (should only be -2,-1,0,1,2,NaN)
    all_values = set()
    for col in df_num.columns:
        all_values.update(df_num[col].dropna().unique())
    unexpected = [v for v in all_values if v not in [-2, -1, 0, 1, 2]]

    summary = {
        "n_respondents": n_respondents,
        "n_statements": n_statements,
        "n_cells": n_cells,
        "n_missing": int(n_missing),
        "missing_pct": missing_pct,
        "n_duplicate_respondents": int(n_duplicates),
        "statements_missing_counts": stmt_missing.to_dict(),
        "respondents_missing_counts": resp_missing.to_dict(),
        "unexpected_values": unexpected,
        "respondent_ids": respondent_ids.tolist(),
    }

    return summary


def compute_cronbach_alpha(items: pd.DataFrame) -> float:
    """Compute Cronbach's alpha for a set of items."""
    items = items.dropna()
    if items.shape[0] < 2 or items.shape[1] < 2:
        return np.nan
    k = items.shape[1]
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return float(alpha)


def preprocess() -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
    """
    Full preprocessing pipeline.
    Returns: (df_num, statement_cols, col_to_domain, validation_summary)
    """
    df = load_raw_data()
    respondent_ids = df.iloc[:, 0]

    statement_cols, col_to_domain, _ = identify_columns(df)
    df_num = encode_likert(df, statement_cols)

    validation = validate_data(df_num, respondent_ids)

    # Cronbach's alpha per domain
    cronbach = {}
    for prefix, name in DOMAIN_PREFIXES.items():
        domain_cols = [c for c in statement_cols if c.split(".")[0][0] == prefix]
        if domain_cols:
            cronbach[name] = compute_cronbach_alpha(df_num[domain_cols])

    validation["cronbach_alpha"] = cronbach

    return df_num, statement_cols, col_to_domain, validation


if __name__ == "__main__":
    df_num, stmt_cols, col_to_domain, validation = preprocess()

    print(f"Respondents: {validation['n_respondents']}")
    print(f"Statements: {validation['n_statements']}")
    print(f"Missing: {validation['n_missing']} ({validation['missing_pct']:.2f}%)")
    print(f"Duplicates: {validation['n_duplicate_respondents']}")
    print(f"Unexpected values: {validation['unexpected_values']}")
    print("\nCronbach's alpha per domain:")
    for domain, alpha in validation["cronbach_alpha"].items():
        print(f"  {domain}: {alpha:.4f}")