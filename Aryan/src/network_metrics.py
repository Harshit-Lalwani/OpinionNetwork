"""Global and node-level network metrics for opinion network."""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, Optional


def compute_global_metrics(G: nx.Graph) -> Dict[str, Any]:
    """Compute global network metrics."""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if m == 0:
        return {
            "nodes": n,
            "edges": 0,
            "density": 0.0,
            "avg_degree": 0.0,
            "avg_weighted_degree": 0.0,
            "avg_clustering": 0.0,
            "transitivity": 0.0,
            "avg_path_length": np.nan,
            "diameter": np.nan,
            "connected_components": n,
            "largest_component_size": 1,
            "modularity": 0.0,
        }

    degrees = dict(G.degree())
    strengths = dict(G.degree(weight="weight"))

    # Connected components
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    G_giant = G.subgraph(largest_cc).copy()

    # Path-based metrics on giant component only
    try:
        avg_path_length = nx.average_shortest_path_length(G_giant, weight="weight")
    except nx.NetworkXError:
        avg_path_length = np.nan

    try:
        diameter = nx.diameter(G_giant, weight="weight")
    except nx.NetworkXError:
        diameter = np.nan

    # Clustering (weighted)
    try:
        avg_clustering = nx.average_clustering(G, weight="weight")
    except (ZeroDivisionError, ValueError):
        avg_clustering = 0.0

    # Transitivity (global clustering)
    try:
        transitivity = nx.transitivity(G)
    except ZeroDivisionError:
        transitivity = 0.0

    # Modularity (Louvain)
    try:
        from networkx.algorithms.community import louvain_communities
        from networkx.algorithms.community.quality import modularity
        comms = louvain_communities(G_giant, weight="weight", seed=42)
        mod = modularity(G_giant, comms, weight="weight")
    except Exception:
        mod = 0.0

    metrics = {
        "nodes": n,
        "edges": m,
        "density": nx.density(G),
        "avg_degree": 2 * m / n,
        "avg_weighted_degree": np.mean(list(strengths.values())),
        "avg_clustering": avg_clustering,
        "transitivity": transitivity,
        "avg_path_length": avg_path_length,
        "diameter": diameter,
        "connected_components": len(components),
        "largest_component_size": len(largest_cc),
        "modularity": mod,
    }

    return metrics


def compute_node_metrics(G: nx.Graph) -> pd.DataFrame:
    """Compute node-level centrality and structural metrics."""
    n = G.number_of_nodes()

    if n == 0:
        return pd.DataFrame()

    # Basic
    degree = dict(G.degree())
    strength = dict(G.degree(weight="weight"))

    # Clustering
    clustering = nx.clustering(G, weight="weight")

    # Centralities (on giant component for path-based ones)
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    G_giant = G.subgraph(largest_cc).copy()

    # Betweenness centrality (weighted)
    betweenness = nx.betweenness_centrality(G_giant, weight="weight", normalized=True)
    # Pad for nodes not in giant component
    for node in G.nodes():
        if node not in betweenness:
            betweenness[node] = 0.0

    # Closeness centrality (weighted)
    closeness = nx.closeness_centrality(G_giant, distance="weight")
    for node in G.nodes():
        if node not in closeness:
            closeness[node] = 0.0

    # Eigenvector centrality (weighted)
    try:
        eigenvector = nx.eigenvector_centrality(G_giant, weight="weight", max_iter=1000)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        eigenvector = {node: 0.0 for node in G_giant.nodes()}
    for node in G.nodes():
        if node not in eigenvector:
            eigenvector[node] = 0.0

    # Build DataFrame
    df = pd.DataFrame({
        "node": list(G.nodes()),
        "degree": [degree[n] for n in G.nodes()],
        "strength": [strength[n] for n in G.nodes()],
        "clustering": [clustering[n] for n in G.nodes()],
        "betweenness": [betweenness[n] for n in G.nodes()],
        "closeness": [closeness[n] for n in G.nodes()],
        "eigenvector": [eigenvector[n] for n in G.nodes()],
    }).set_index("node")

    return df


def compute_all_metrics(G: nx.Graph) -> Dict[str, Any]:
    """Compute both global and node metrics."""
    global_metrics = compute_global_metrics(G)
    node_metrics = compute_node_metrics(G)
    return {
        "global": global_metrics,
        "nodes": node_metrics,
    }


def print_global_summary(global_metrics: Dict[str, Any]) -> None:
    """Pretty print global metrics."""
    print("=" * 50)
    print("GLOBAL NETWORK METRICS")
    print("=" * 50)
    for k, v in global_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def print_top_nodes(node_metrics: pd.DataFrame, metric: str = "strength", top_n: int = 10) -> None:
    """Print top N nodes by a given metric."""
    if node_metrics.empty:
        print("No node metrics available.")
        return
    sorted_df = node_metrics.sort_values(metric, ascending=False).head(top_n)
    print(f"\nTop {top_n} nodes by {metric}:")
    print(sorted_df[[metric]].to_string())


if __name__ == "__main__":
    from src import preprocess, build_full_pipeline

    df_num, stmt_cols, col_to_domain, validation = preprocess()
    G, corr, edge_df, summary = build_full_pipeline(df_num)

    metrics = compute_all_metrics(G)
    print_global_summary(metrics["global"])
    print_top_nodes(metrics["nodes"], "strength", 10)
    print_top_nodes(metrics["nodes"], "betweenness", 10)
    print_top_nodes(metrics["nodes"], "eigenvector", 10)