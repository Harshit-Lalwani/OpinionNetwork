#!/usr/bin/env python
"""
Main pipeline runner for Opinion Network Analysis.
Executes full pipeline and exports all results.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src import (
    preprocess, build_full_pipeline, compute_correlation_matrix,
    FIGURES_DIR, RESULTS_DIR
)
from src.network_metrics import compute_all_metrics
from src.community_detection import detect_communities, get_community_summary, analyze_community_composition
from src.domain_analysis import compute_domain_correlations, compute_statement_means
from src.negative_network import build_negative_network, get_negative_summary
from src.null_models import run_null_models, compare_with_null
from src.sensitivity import run_sensitivity_analysis
from src.visualizations import generate_all_figures


def export_results(
    G: nx.Graph,
    G_neg: nx.Graph,
    corr: pd.DataFrame,
    edge_df: pd.DataFrame,
    metrics: Dict,
    communities: List[set],
    col_to_domain: Dict[str, str],
    validation: Dict,
    domain_corr: pd.DataFrame,
    within_df: pd.DataFrame,
    means_df: pd.DataFrame,
    neg_summary: Dict,
    neg_edge_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> None:
    """Export all results to CSV files."""
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # correlations.csv - full matrix + edge list
    corr.to_csv(RESULTS_DIR / "correlations.csv")
    edge_df.to_csv(RESULTS_DIR / "edge_list.csv", index=False)
    
    # node_metrics.csv
    metrics["nodes"].to_csv(RESULTS_DIR / "node_metrics.csv")
    
    # network_summary.csv
    pd.DataFrame([metrics["global"]]).to_csv(RESULTS_DIR / "network_summary.csv", index=False)
    
    # communities.csv
    comm_map = {n: i for i, comm in enumerate(communities) for n in comm}
    comm_df = pd.DataFrame({
        "node": list(comm_map.keys()),
        "community": list(comm_map.values()),
        "domain": [col_to_domain.get(n, "Unknown") for n in comm_map.keys()],
    })
    comm_df.to_csv(RESULTS_DIR / "communities.csv", index=False)
    
    # domain_correlations.csv
    domain_corr.to_csv(RESULTS_DIR / "domain_correlations.csv")
    within_df.to_csv(RESULTS_DIR / "within_domain_stats.csv", index=False)
    
    # negative_edges.csv
    neg_edge_df.to_csv(RESULTS_DIR / "negative_edges.csv", index=False)
    pd.DataFrame([neg_summary]).to_csv(RESULTS_DIR / "negative_network_summary.csv", index=False)
    
    # null_model_results.csv
    comparison_df.to_csv(RESULTS_DIR / "null_model_results.csv", index=False)
    
    # sensitivity_results.csv
    sensitivity_df.to_csv(RESULTS_DIR / "sensitivity_results.csv", index=False)
    
    # statement_means.csv
    means_df.to_csv(RESULTS_DIR / "statement_means.csv")
    
    # validation_summary.csv
    val_df = pd.DataFrame([{
        k: v for k, v in validation.items() 
        if k not in ["statements_missing_counts", "respondents_missing_counts"]
    }])
    val_df.to_csv(RESULTS_DIR / "validation_summary.csv", index=False)
    
    print(f"\nAll CSV results saved to {RESULTS_DIR}")


def main():
    print("=" * 60)
    print("OPINION NETWORK ANALYSIS - FULL PIPELINE")
    print("=" * 60)
    
    # Phase 1: Preprocessing & Network Construction
    print("\n[1/6] Preprocessing data...")
    df_num, stmt_cols, col_to_domain, validation = preprocess()
    print(f"      {validation['n_respondents']} respondents, {validation['n_statements']} statements")
    print(f"      Missing: {validation['missing_pct']:.1f}%")
    
    print("\n[2/6] Building opinion correlation network...")
    G, corr, edge_df, summary = build_full_pipeline(df_num)
    print(f"      {summary['nodes']} nodes, {summary['edges']} edges")
    print(f"      Connected: {summary['connected_components'] == 1}")
    
    # Phase 2: Core Analysis
    print("\n[3/6] Computing network metrics & communities...")
    metrics = compute_all_metrics(G)
    communities = detect_communities(G)
    comm_summary = get_community_summary(G, communities, col_to_domain)
    domain_corr, within_df = compute_domain_correlations(corr, stmt_cols)
    means_df = compute_statement_means(df_num, stmt_cols)
    composition_df = analyze_community_composition(communities, col_to_domain)
    
    print(f"      {comm_summary['n_communities']} communities, Modularity={comm_summary['modularity']:.3f}")
    print(f"      ARI vs T/E/S/V: {comm_summary['ari_vs_domains']:.3f}")
    
    # Phase 3: Extended Analysis
    print("\n[4/6] Building negative correlation network...")
    G_neg, neg_edge_df = build_negative_network(corr, threshold=-0.25)
    neg_summary = get_negative_summary(G_neg, neg_edge_df)
    print(f"      {neg_summary['edges']} negative edges")
    
    print("\n[5/6] Running null models...")
    er_df, config_df = run_null_models(G, n_null=100, parallel=False)
    comparison_df = compare_with_null(metrics["global"], er_df, config_df)
    
    print("\n[6/6] Sensitivity analysis...")
    sensitivity_df = run_sensitivity_analysis(corr, col_to_domain)
    
    # Export CSVs
    print("\nExporting results...")
    export_results(
        G, G_neg, corr, edge_df, metrics, communities, col_to_domain,
        validation, domain_corr, within_df, means_df,
        neg_summary, neg_edge_df, comparison_df, sensitivity_df
    )
    
    # Generate Figures
    print("\nGenerating figures...")
    generate_all_figures(
        G, G_neg, corr, domain_corr, within_df,
        metrics["nodes"], communities, col_to_domain,
        comparison_df, composition_df
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Figures: {FIGURES_DIR}")
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    import networkx as nx
    main()