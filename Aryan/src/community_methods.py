"""Compare multiple community detection methods on the opinion network."""

import pandas as pd
import networkx as nx
from networkx.algorithms.community import (
    louvain_communities,
    greedy_modularity_communities,
    label_propagation_communities,
)
from sklearn.metrics import adjusted_rand_score

from .community_detection import get_community_mapping, compute_modularity, compute_ari
from .config import RANDOM_SEED


def run_all_methods(G: nx.Graph, seed: int = RANDOM_SEED) -> dict:
    """Run Louvain, greedy modularity, and label propagation on the same graph."""
    return {
        "louvain": louvain_communities(G, weight="weight", seed=seed, resolution=1.0),
        "greedy_modularity": greedy_modularity_communities(G, weight="weight"),
        "label_propagation": list(label_propagation_communities(G)),
    }


def compare_methods(G: nx.Graph, col_to_domain: dict, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Summary table: n_communities, modularity, ARI vs domains, for each method."""
    results = run_all_methods(G, seed=seed)
    rows = []
    for name, communities in results.items():
        rows.append({
            "method": name,
            "n_communities": len(communities),
            "sizes": sorted([len(c) for c in communities], reverse=True),
            "modularity": compute_modularity(G, communities),
            "ari_vs_domains": compute_ari(communities, col_to_domain),
        })
    return pd.DataFrame(rows), results


def pairwise_method_ari(results: dict) -> pd.DataFrame:
    """ARI between every pair of methods' partitions (agreement between methods)."""
    names = list(results.keys())
    labels = {}
    for name in names:
        mapping = get_community_mapping(results[name])
        labels[name] = mapping

    nodes = list(labels[names[0]].keys())
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            true = [labels[a][n] for n in nodes]
            pred = [labels[b][n] for n in nodes]
            rows.append({"method_a": a, "method_b": b, "ari": adjusted_rand_score(true, pred)})
    return pd.DataFrame(rows)


def export_community_members(
    communities: list, col_to_domain: dict, method_name: str
) -> pd.DataFrame:
    """Full statement text per community, for manual/qualitative interpretation."""
    rows = []
    for i, comm in enumerate(sorted(communities, key=len, reverse=True)):
        for node in sorted(comm):
            rows.append({
                "method": method_name,
                "community": i,
                "size": len(comm),
                "domain": col_to_domain.get(node, "Unknown"),
                "statement": node,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from .config import RESULTS_DIR
    from . import preprocess, build_full_pipeline

    df_num, stmt_cols, col_to_domain, validation = preprocess()
    G, corr, edge_df, summary = build_full_pipeline(df_num)

    comparison_df, results = compare_methods(G, col_to_domain)
    print(comparison_df.to_string())
    comparison_df.to_csv(RESULTS_DIR / "community_methods_comparison.csv", index=False)

    pairwise_df = pairwise_method_ari(results)
    print()
    print(pairwise_df.to_string())
    pairwise_df.to_csv(RESULTS_DIR / "community_methods_pairwise_ari.csv", index=False)

    members_df = pd.concat(
        [export_community_members(results[name], col_to_domain, name) for name in results],
        ignore_index=True,
    )
    members_df.to_csv(RESULTS_DIR / "community_members_by_method.csv", index=False)
    print(f"\nWrote community_methods_comparison.csv, community_methods_pairwise_ari.csv, "
          f"community_members_by_method.csv to {RESULTS_DIR}")
