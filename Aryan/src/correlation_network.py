"""Build opinion–opinion correlation network with top-K sparsification."""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, Dict, Optional

from .config import TOP_K, CORRELATION_METHOD


def compute_correlation_matrix(df_num: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise correlation matrix (Pearson or Spearman)."""
    if CORRELATION_METHOD == "pearson":
        corr = df_num.corr(method="pearson")
    elif CORRELATION_METHOD == "spearman":
        corr = df_num.corr(method="spearman")
    else:
        raise ValueError(f"Unknown correlation method: {CORRELATION_METHOD}")
    return corr


def build_topk_network(
    corr: pd.DataFrame,
    k: int = TOP_K,
    min_correlation: float = 0.0
) -> Tuple[nx.Graph, pd.DataFrame]:
    """
    Build sparse weighted graph: each node connects to its top-K positive neighbors.
    Edges are symmetrized (undirected) and deduplicated.

    Args:
        corr: N x N correlation matrix (statements x statements)
        k: number of top positive neighbors per node
        min_correlation: minimum correlation to consider (default 0.0 = only positive)

    Returns:
        G: networkx Graph with weighted edges (weight = correlation)
        edge_df: DataFrame with columns [node1, node2, weight, node1_domain, node2_domain]
    """
    nodes = corr.index.tolist()
    n = len(nodes)

    # For each node, find top-K positive correlations (excluding self)
    edges = []

    for i, node in enumerate(nodes):
        # Get correlations for this node, exclude self
        node_corrs = corr.loc[node].drop(node)

        # Keep only positive correlations above threshold
        positive_corrs = node_corrs[node_corrs > min_correlation]

        # Sort descending, take top k
        top_neighbors = positive_corrs.sort_values(ascending=False).head(k)

        for neighbor, weight in top_neighbors.items():
            edges.append((node, neighbor, float(weight)))

    # Create graph and add edges (NetworkX auto-deduplicates)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u, v, w in edges:
        if G.has_edge(u, v):
            # Keep the maximum weight if edge already added from other direction
            existing_w = G[u][v]["weight"]
            G[u][v]["weight"] = max(existing_w, w)
        else:
            G.add_edge(u, v, weight=w)

    # Build edge dataframe
    edge_data = []
    for u, v, d in G.edges(data=True):
        edge_data.append({
            "node1": u,
            "node2": v,
            "weight": d["weight"],
        })

    edge_df = pd.DataFrame(edge_data)

    return G, edge_df


def get_network_summary(G: nx.Graph) -> Dict:
    """Compute basic network summary statistics."""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if m == 0:
        return {
            "nodes": n,
            "edges": 0,
            "density": 0.0,
            "avg_degree": 0.0,
            "avg_weighted_degree": 0.0,
            "connected_components": n,
            "largest_component_size": 1,
        }

    degrees = dict(G.degree())
    strengths = dict(G.degree(weight="weight"))

    # Connected components
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)

    summary = {
        "nodes": n,
        "edges": m,
        "density": nx.density(G),
        "avg_degree": 2 * m / n,
        "avg_weighted_degree": np.mean(list(strengths.values())),
        "connected_components": len(components),
        "largest_component_size": len(largest_cc),
    }

    return summary


def build_full_pipeline(
    df_num: pd.DataFrame,
    k: int = TOP_K
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run full network construction pipeline.
    Returns: (G, corr_matrix, edge_df, summary)
    """
    corr = compute_correlation_matrix(df_num)
    G, edge_df = build_topk_network(corr, k=k)
    summary = get_network_summary(G)

    return G, corr, edge_df, summary


if __name__ == "__main__":
    # Test with sample data
    from data_preprocessing import preprocess

    df_num, stmt_cols, col_to_domain, validation = preprocess()
    G, corr, edge_df, summary = build_full_pipeline(df_num)

    print("Network Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\nTop 10 edges by weight:")
    print(edge_df.nlargest(10, "weight"))