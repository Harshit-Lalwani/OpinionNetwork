"""Null model generation and comparison for opinion network."""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import RANDOM_SEED, N_NULL_NETWORKS


def generate_er_graph(n_nodes: int, n_edges: int, seed: int = None) -> nx.Graph:
    """Generate Erdős-Rényi random graph with given nodes and edges."""
    if seed is not None:
        np.random.seed(seed)
    
    # Use G(n, m) model - exactly n_edges
    G = nx.gnm_random_graph(n_nodes, n_edges, seed=seed)
    return G


def generate_configuration_model(G: nx.Graph, seed: int = None) -> nx.Graph:
    """Generate degree-preserving random graph using configuration model."""
    if seed is not None:
        np.random.seed(seed)
    
    degree_sequence = [d for _, d in G.degree()]
    G_null = nx.configuration_model(degree_sequence, seed=seed)
    # Convert to simple graph (remove self-loops, parallel edges)
    G_null = nx.Graph(G_null)
    G_null.remove_edges_from(nx.selfloop_edges(G_null))
    return G_null


def compute_graph_metrics(G: nx.Graph) -> Dict[str, float]:
    """Compute standard graph metrics for comparison."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    if m == 0:
        return {
            "clustering": 0.0,
            "transitivity": 0.0,
            "avg_path_length": np.nan,
            "diameter": np.nan,
            "modularity": 0.0,
            "avg_degree": 0.0,
        }
    
    # Connected components
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    G_giant = G.subgraph(largest_cc).copy()
    
    metrics = {
        "clustering": nx.average_clustering(G, weight="weight") if nx.is_weighted(G) else nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "avg_degree": 2 * m / n,
    }
    
    # Path metrics on giant component
    try:
        metrics["avg_path_length"] = nx.average_shortest_path_length(G_giant)
    except nx.NetworkXError:
        metrics["avg_path_length"] = np.nan
    
    try:
        metrics["diameter"] = nx.diameter(G_giant)
    except nx.NetworkXError:
        metrics["diameter"] = np.nan
    
    # Modularity (Louvain)
    try:
        from networkx.algorithms.community import louvain_communities
        from networkx.algorithms.community.quality import modularity
        comms = louvain_communities(G_giant, weight="weight" if nx.is_weighted(G_giant) else None, seed=RANDOM_SEED)
        metrics["modularity"] = modularity(G_giant, comms, weight="weight" if nx.is_weighted(G_giant) else None)
    except Exception:
        metrics["modularity"] = 0.0
    
    return metrics


def run_null_models(
    G_obs: nx.Graph,
    n_null: int = N_NULL_NETWORKS,
    parallel: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate null networks and compute metrics.
    
    Returns:
        er_results: DataFrame with ER graph metrics
        config_results: DataFrame with configuration model metrics
    """
    n_nodes = G_obs.number_of_nodes()
    n_edges = G_obs.number_of_edges()
    
    # Observed metrics
    obs_metrics = compute_graph_metrics(G_obs)
    print(f"Observed metrics: {obs_metrics}")
    
    er_results = []
    config_results = []
    
    def generate_er(i):
        G = generate_er_graph(n_nodes, n_edges, seed=RANDOM_SEED + i)
        return compute_graph_metrics(G)
    
    def generate_config(i):
        G = generate_configuration_model(G_obs, seed=RANDOM_SEED + i + 1000)
        return compute_graph_metrics(G)
    
    if parallel:
        with ProcessPoolExecutor() as executor:
            er_futures = [executor.submit(generate_er, i) for i in range(n_null)]
            config_futures = [executor.submit(generate_config, i) for i in range(n_null)]
            
            for f in as_completed(er_futures):
                er_results.append(f.result())
            for f in as_completed(config_futures):
                config_results.append(f.result())
    else:
        for i in range(n_null):
            if i % 20 == 0:
                print(f"  Progress: {i}/{n_null}")
            er_results.append(generate_er(i))
            config_results.append(generate_config(i))
    
    er_df = pd.DataFrame(er_results)
    config_df = pd.DataFrame(config_results)
    
    return er_df, config_df


def compare_with_null(
    obs_metrics: Dict,
    er_df: pd.DataFrame,
    config_df: pd.DataFrame
) -> pd.DataFrame:
    """Compare observed metrics against null distributions."""
    comparison = []
    
    # Only compare metrics that exist in null model results
    null_metrics = set(er_df.columns) & set(config_df.columns)
    
    for metric in obs_metrics:
        if metric not in null_metrics:
            continue
        obs_val = obs_metrics[metric]
        if pd.isna(obs_val):
            continue
            
        er_vals = er_df[metric].dropna()
        config_vals = config_df[metric].dropna()
        
        row = {
            "metric": metric,
            "observed": obs_val,
            "er_mean": er_vals.mean() if len(er_vals) > 0 else np.nan,
            "er_std": er_vals.std() if len(er_vals) > 0 else np.nan,
            "er_zscore": (obs_val - er_vals.mean()) / er_vals.std() if len(er_vals) > 0 and er_vals.std() > 0 else np.nan,
            "er_p_gt": (er_vals > obs_val).mean() if len(er_vals) > 0 else np.nan,
            "config_mean": config_vals.mean() if len(config_vals) > 0 else np.nan,
            "config_std": config_vals.std() if len(config_vals) > 0 else np.nan,
            "config_zscore": (obs_val - config_vals.mean()) / config_vals.std() if len(config_vals) > 0 and config_vals.std() > 0 else np.nan,
            "config_p_gt": (config_vals > obs_val).mean() if len(config_vals) > 0 else np.nan,
        }
        comparison.append(row)
    
    return pd.DataFrame(comparison)


def print_null_comparison(comparison_df: pd.DataFrame) -> None:
    """Print null model comparison."""
    print("=" * 80)
    print("NULL MODEL COMPARISON")
    print("=" * 80)
    for _, row in comparison_df.iterrows():
        m = row["metric"]
        print(f"\n{m}:")
        print(f"  Observed: {row['observed']:.4f}")
        print(f"  ER:       mean={row['er_mean']:.4f}, std={row['er_std']:.4f}, z={row['er_zscore']:.2f}, p(gt)={row['er_p_gt']:.4f}")
        print(f"  Config:   mean={row['config_mean']:.4f}, std={row['config_std']:.4f}, z={row['config_zscore']:.2f}, p(gt)={row['config_p_gt']:.4f}")


if __name__ == "__main__":
    from src import preprocess, build_full_pipeline
    
    df_num, stmt_cols, col_to_domain, validation = preprocess()
    G, corr, edge_df, summary = build_full_pipeline(df_num)
    
    print("Running null models (this may take a minute)...")
    er_df, config_df = run_null_models(G, n_null=100, parallel=False)
    
    obs_metrics = compute_graph_metrics(G)
    comparison = compare_with_null(obs_metrics, er_df, config_df)
    print_null_comparison(comparison)