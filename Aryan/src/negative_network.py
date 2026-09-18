"""Negative correlation / opinion tension network."""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional

from .config import NEGATIVE_THRESHOLD


def build_negative_network(
    corr: pd.DataFrame,
    threshold: float = NEGATIVE_THRESHOLD,
    max_edges: Optional[int] = None
) -> Tuple[nx.Graph, pd.DataFrame]:
    """
    Build network from negative correlations.
    
    Args:
        corr: Correlation matrix
        threshold: Keep edges with r < threshold (negative)
        max_edges: If set, keep only the strongest |max_edges| negative correlations
        
    Returns:
        G: NetworkX Graph with negative edges (weight = correlation, negative)
        edge_df: DataFrame with node1, node2, weight
    """
    nodes = corr.index.tolist()
    edges = []

    # Collect all negative correlations below threshold
    for i, u in enumerate(nodes):
        for v in nodes[i+1:]:
            r = corr.loc[u, v]
            if pd.notna(r) and r < threshold:
                edges.append((u, v, float(r)))

    # If max_edges specified, keep only strongest negative correlations
    if max_edges is not None and len(edges) > max_edges:
        edges.sort(key=lambda x: x[2])  # Most negative first
        edges = edges[:max_edges]

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    edge_df = pd.DataFrame(edges, columns=["node1", "node2", "weight"])
    
    return G, edge_df


def get_negative_summary(G: nx.Graph, edge_df: pd.DataFrame) -> Dict:
    """Get summary statistics for negative network."""
    if G.number_of_edges() == 0:
        return {
            "nodes": G.number_of_nodes(),
            "edges": 0,
            "components": G.number_of_nodes(),
            "avg_weight": 0.0,
            "min_weight": 0.0,
        }
    
    weights = edge_df["weight"].values
    components = list(nx.connected_components(G))
    
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": len(components),
        "largest_component": max(len(c) for c in components),
        "avg_weight": float(np.mean(weights)),
        "min_weight": float(np.min(weights)),
        "max_weight": float(np.max(weights)),
    }


def print_negative_summary(summary: Dict, edge_df: pd.DataFrame, top_n: int = 15) -> None:
    """Print negative network summary."""
    print("=" * 50)
    print("NEGATIVE CORRELATION NETWORK (Opinion Tensions)")
    print("=" * 50)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    if len(edge_df) > 0:
        print(f"\nTop {top_n} most negative correlations:")
        print(edge_df.nsmallest(top_n, "weight").to_string(index=False))


if __name__ == "__main__":
    from src import preprocess, compute_correlation_matrix

    df_num, stmt_cols, col_to_domain, validation = preprocess()
    corr = compute_correlation_matrix(df_num)

    # Threshold-based
    G_thresh, edge_df_thresh = build_negative_network(corr, threshold=NEGATIVE_THRESHOLD)
    summary_thresh = get_negative_summary(G_thresh, edge_df_thresh)
    print_negative_summary(summary_thresh, edge_df_thresh)
    
    print("\n" + "=" * 50)
    print("TOP 20 NEGATIVE EDGES (by magnitude)")
    print("=" * 50)
    G_top20, edge_df_top20 = build_negative_network(corr, max_edges=20)
    summary_top20 = get_negative_summary(G_top20, edge_df_top20)
    print_negative_summary(summary_top20, edge_df_top20)