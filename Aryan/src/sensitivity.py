"""Sensitivity analysis: top-K robustness check."""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any

from .config import SENSITIVITY_K_VALUES, RANDOM_SEED
from .correlation_network import compute_correlation_matrix, build_topk_network, get_network_summary
from .network_metrics import compute_all_metrics
from .community_detection import detect_communities, get_community_mapping, compute_ari
from .domain_analysis import compute_domain_correlations


def build_network_for_k(corr: pd.DataFrame, k: int) -> Tuple[nx.Graph, pd.DataFrame]:
    """Build top-K network for a given K."""
    G, edge_df = build_topk_network(corr, k=k)
    return G, edge_df


def compute_stability(comm_map_1: Dict[str, int], comm_map_2: Dict[str, int]) -> float:
    """Compute ARI between two community assignments."""
    from sklearn.metrics import adjusted_rand_score
    
    nodes = sorted(set(comm_map_1.keys()) & set(comm_map_2.keys()))
    labels_1 = [comm_map_1[n] for n in nodes]
    labels_2 = [comm_map_2[n] for n in nodes]
    
    return adjusted_rand_score(labels_1, labels_2)


def run_sensitivity_analysis(
    corr: pd.DataFrame,
    col_to_domain: Dict[str, str],
    k_values: List[int] = SENSITIVITY_K_VALUES
) -> pd.DataFrame:
    """
    Run sensitivity analysis across different K values.
    
    Returns DataFrame with metrics for each K.
    """
    results = []
    prev_comm_map = None
    
    for k in k_values:
        print(f"  Analyzing K={k}...")
        G, edge_df = build_network_for_k(corr, k)
        summary = get_network_summary(G)
        metrics = compute_all_metrics(G)
        global_m = metrics["global"]
        
        # Community detection
        communities = detect_communities(G)
        comm_map = get_community_mapping(communities)
        n_communities = len(communities)
        
        # ARI vs previous K
        ari_vs_prev = np.nan
        if prev_comm_map is not None:
            ari_vs_prev = compute_stability(prev_comm_map, comm_map)
        prev_comm_map = comm_map
        
        # ARI vs true domains
        ari_vs_domains = compute_ari(communities, col_to_domain)
        
        # Domain correlation stats
        domain_corr, within_df = compute_domain_correlations(corr, corr.index.tolist())
        
        result = {
            "K": k,
            "nodes": summary["nodes"],
            "edges": summary["edges"],
            "density": summary["density"],
            "avg_degree": summary["avg_degree"],
            "avg_weighted_degree": summary["avg_weighted_degree"],
            "connected_components": summary["connected_components"],
            "largest_component_size": summary["largest_component_size"],
            "avg_clustering": global_m["avg_clustering"],
            "transitivity": global_m["transitivity"],
            "avg_path_length": global_m["avg_path_length"],
            "diameter": global_m["diameter"],
            "modularity": global_m["modularity"],
            "n_communities": n_communities,
            "ari_vs_previous_k": ari_vs_prev,
            "ari_vs_domains": ari_vs_domains,
        }
        
        # Add within-domain correlations
        for _, row in within_df.iterrows():
            result[f"within_{row['domain']}"] = row["mean_correlation"]
        
        results.append(result)
    
    return pd.DataFrame(results)


def print_sensitivity_results(df: pd.DataFrame) -> None:
    """Print sensitivity analysis results."""
    print("=" * 80)
    print("SENSITIVITY ANALYSIS: TOP-K ROBUSTNESS")
    print("=" * 80)
    
    # Key metrics
    key_metrics = [
        "K", "edges", "density", "avg_degree", "avg_weighted_degree",
        "connected_components", "avg_clustering", "modularity",
        "n_communities", "ari_vs_previous_k", "ari_vs_domains"
    ]
    
    print(df[key_metrics].to_string(index=False))
    
    print("\nWithin-domain correlations by K:")
    within_cols = [c for c in df.columns if c.startswith("within_")]
    print(df[["K"] + within_cols].to_string(index=False))


def check_community_stability(
    corr: pd.DataFrame,
    col_to_domain: Dict[str, str],
    k_values: List[int] = SENSITIVITY_K_VALUES
) -> pd.DataFrame:
    """Detailed community composition comparison across K values."""
    community_maps = {}
    
    for k in k_values:
        G, _ = build_network_for_k(corr, k)
        communities = detect_communities(G)
        community_maps[k] = get_community_mapping(communities)
    
    # Compare each K to K=3 (reference)
    ref_k = 3 if 3 in k_values else k_values[1]
    ref_map = community_maps[ref_k]
    
    stability_rows = []
    for k, comm_map in community_maps.items():
        if k == ref_k:
            continue
        ari = compute_stability(ref_map, comm_map)
        stability_rows.append({"K": k, "reference_K": ref_k, "ARI": ari})
    
    return pd.DataFrame(stability_rows)


if __name__ == "__main__":
    from src import preprocess, compute_correlation_matrix
    
    df_num, stmt_cols, col_to_domain, validation = preprocess()
    corr = compute_correlation_matrix(df_num)
    
    print("Running sensitivity analysis...")
    results = run_sensitivity_analysis(corr, col_to_domain)
    print_sensitivity_results(results)
    
    print("\nCommunity stability (vs K=3):")
    stability = check_community_stability(corr, col_to_domain)
    print(stability.to_string(index=False))