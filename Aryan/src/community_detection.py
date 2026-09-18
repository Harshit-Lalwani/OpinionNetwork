"""Community detection and analysis for opinion network."""

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities
from typing import Dict, List, Tuple, Any

from .config import DOMAIN_PREFIXES, DOMAIN_NAMES


def detect_communities(G: nx.Graph, weight: str = "weight", seed: int = 42) -> List[set]:
    """Run Louvain community detection on weighted graph."""
    if G.number_of_edges() == 0:
        return [{n} for n in G.nodes()]
    communities = louvain_communities(G, weight=weight, seed=seed, resolution=1.0)
    return communities


def get_community_mapping(communities: List[set]) -> Dict[str, int]:
    """Map each node to its community index."""
    mapping = {}
    for i, comm in enumerate(communities):
        for node in comm:
            mapping[node] = i
    return mapping


def compute_modularity(G: nx.Graph, communities: List[set], weight: str = "weight") -> float:
    """Compute modularity of the partition."""
    from networkx.algorithms.community.quality import modularity
    return modularity(G, communities, weight=weight)


def analyze_community_composition(
    communities: List[set],
    col_to_domain: Dict[str, str]
) -> pd.DataFrame:
    """
    Analyze domain composition of each community.
    Returns DataFrame with communities as rows, domains as columns (counts).
    """
    community_domains = {}
    for i, comm in enumerate(communities):
        domain_counts = {d: 0 for d in DOMAIN_NAMES}
        for node in comm:
            domain = col_to_domain.get(node, "Unknown")
            if domain in domain_counts:
                domain_counts[domain] += 1
        community_domains[f"Community {i}"] = domain_counts

    df = pd.DataFrame(community_domains).T
    df.index.name = "community"
    return df


def compute_ari(communities: List[set], col_to_domain: Dict[str, str]) -> float:
    """Compute Adjusted Rand Index vs. a-priori T/E/S/V categories."""
    from sklearn.metrics import adjusted_rand_score

    nodes = []
    true_labels = []
    pred_labels = []

    comm_map = get_community_mapping(communities)

    for node in comm_map:
        nodes.append(node)
        true_labels.append(col_to_domain.get(node, "Unknown"))
        pred_labels.append(comm_map[node])

    # Map domain names to integers
    domain_to_int = {d: i for i, d in enumerate(DOMAIN_NAMES)}
    true_int = [domain_to_int.get(l, -1) for l in true_labels]

    return adjusted_rand_score(true_int, pred_labels)


def get_community_summary(G: nx.Graph, communities: List[set], col_to_domain: Dict[str, str]) -> Dict[str, Any]:
    """Get comprehensive community analysis summary."""
    comm_map = get_community_mapping(communities)
    composition = analyze_community_composition(communities, col_to_domain)
    modularity_score = compute_modularity(G, communities)

    sizes = [len(c) for c in communities]

    summary = {
        "n_communities": len(communities),
        "community_sizes": sizes,
        "modularity": modularity_score,
        "ari_vs_domains": compute_ari(communities, col_to_domain),
        "composition": composition,
        "community_mapping": comm_map,
    }

    return summary


def print_community_summary(summary: Dict[str, Any]) -> None:
    """Pretty print community summary."""
    print("=" * 50)
    print("COMMUNITY DETECTION RESULTS")
    print("=" * 50)
    print(f"Number of communities: {summary['n_communities']}")
    print(f"Community sizes: {summary['community_sizes']}")
    print(f"Modularity: {summary['modularity']:.4f}")
    print(f"ARI vs T/E/S/V domains: {summary['ari_vs_domains']:.4f}")
    print()
    print("Domain composition (counts):")
    print(summary["composition"].to_string())


if __name__ == "__main__":
    from src import preprocess, build_full_pipeline

    df_num, stmt_cols, col_to_domain, validation = preprocess()
    G, corr, edge_df, summary = build_full_pipeline(df_num)

    communities = detect_communities(G)
    comm_summary = get_community_summary(G, communities, col_to_domain)
    print_community_summary(comm_summary)