"""Apples-to-apples null-model comparison for the statement network.

The pipeline in Aryan/src/null_models.py compares a WEIGHTED observed path
length/diameter (correlation used as a distance) against UNWEIGHTED nulls, which
is not a valid comparison. Here every metric is computed unweighted on both
sides, except modularity which is weighted on both sides.
"""
import sys
import numpy as np
import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity

sys.path.insert(0, "/root/DPCN/A1/Aryan")
from src.data_preprocessing import preprocess  # noqa: E402
from src.correlation_network import compute_correlation_matrix, build_topk_network  # noqa: E402

SEED, N_NULL = 42, 500


def metrics(G, weighted_mod=True):
    giant = G.subgraph(max(nx.connected_components(G), key=len))
    comms = louvain_communities(giant, weight="weight" if weighted_mod else None, seed=SEED)
    return {
        "transitivity": nx.transitivity(G),
        "avg_clustering": nx.average_clustering(G),
        "avg_path_length": nx.average_shortest_path_length(giant),
        "diameter": nx.diameter(giant),
        "modularity": modularity(giant, comms, weight="weight" if weighted_mod else None),
    }


def main():
    df_num, _, _, _ = preprocess()
    corr = compute_correlation_matrix(df_num)
    G, _ = build_topk_network(corr)
    obs = metrics(G)

    rng = np.random.default_rng(SEED)
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    deg_seq = [d for _, d in G.degree()]
    er, cfg = [], []
    for i in range(N_NULL):
        # ER: same n and m; configuration: same degree sequence.
        # Observed edge weights are reshuffled onto the null edges so that
        # weighted modularity is comparable rather than computed on 0/1 edges.
        g_er = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges(), seed=int(rng.integers(1e9)))
        g_cf = nx.Graph(nx.configuration_model(deg_seq, seed=int(rng.integers(1e9))))
        g_cf.remove_edges_from(nx.selfloop_edges(g_cf))
        for g in (g_er, g_cf):
            w = rng.permutation(weights)
            for (u, v), ww in zip(g.edges(), w):
                g[u][v]["weight"] = float(ww)
        er.append(metrics(g_er))
        cfg.append(metrics(g_cf))

    rows = []
    for name, nulls in [("ER", pd.DataFrame(er)), ("Configuration", pd.DataFrame(cfg))]:
        for m, o in obs.items():
            vals = nulls[m].dropna()
            sd = vals.std()
            # two-sided empirical p, +1 correction so p is never exactly 0
            p = (min((vals >= o).sum(), (vals <= o).sum()) * 2 + 1) / (len(vals) + 1)
            rows.append({"null": name, "metric": m, "observed": o, "null_mean": vals.mean(),
                         "null_sd": sd, "z": (o - vals.mean()) / sd if sd > 0 else np.nan,
                         "p_two_sided": min(p, 1.0)})
    out = pd.DataFrame(rows)
    out.to_csv("/root/DPCN/A1/report/null_model_corrected.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
