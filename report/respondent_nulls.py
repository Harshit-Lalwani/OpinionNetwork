"""Null-model test for the RESPONDENT similarity network (the report's primary network).

Tests whether the Louvain partition of the r>=0.4 respondent network is more
modular than degree-preserving chance. Unweighted topology on both sides for
structural metrics; weights reshuffled onto nulls for weighted modularity.
"""
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity

SEED, N_NULL, THRESHOLD = 42, 500, 0.4
LIKERT = {"Strongly Disagree": -2, "Disagree": -1, "Neutral": 0, "Agree": 1, "Strongly Agree": 2}


def build():
    df = pd.read_csv("/root/DPCN/A1/Survey_Results_UC.csv", encoding="utf-8-sig")
    df = df.rename(columns={"id. Response ID": "respondent_id"}).set_index("respondent_id")
    num = df.replace({"": np.nan, "No Comments": np.nan}).replace(LIKERT).apply(pd.to_numeric)
    sim = num.T.corr()
    G = nx.Graph()
    G.add_nodes_from(num.index)
    for i, a in enumerate(num.index):
        for b in num.index[i + 1:]:
            r = sim.loc[a, b]
            if pd.notna(r) and r >= THRESHOLD:
                G.add_edge(a, b, weight=float(r))
    return G


def stats(G):
    giant = G.subgraph(max(nx.connected_components(G), key=len))
    comms = louvain_communities(G, weight="weight", seed=SEED)
    return {
        "modularity": modularity(G, comms, weight="weight"),
        "transitivity": nx.transitivity(G),
        "avg_clustering": nx.average_clustering(G),
        "avg_path_length": nx.average_shortest_path_length(giant),
    }


def main():
    G = build()
    obs = stats(G)
    print(f"Observed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    rng = np.random.default_rng(SEED)
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    deg_seq = [d for _, d in G.degree()]
    nulls = []
    for _ in range(N_NULL):
        g = nx.Graph(nx.configuration_model(deg_seq, seed=int(rng.integers(1e9))))
        g.remove_edges_from(nx.selfloop_edges(g))
        w = rng.permutation(weights)
        for (u, v), ww in zip(g.edges(), w):
            g[u][v]["weight"] = float(ww)
        nulls.append(stats(g))

    nulls = pd.DataFrame(nulls)
    rows = []
    for m, o in obs.items():
        v = nulls[m].dropna()
        p = (min((v >= o).sum(), (v <= o).sum()) * 2 + 1) / (len(v) + 1)
        rows.append({"metric": m, "observed": o, "null_mean": v.mean(), "null_sd": v.std(),
                     "z": (o - v.mean()) / v.std(), "p_two_sided": min(p, 1.0)})
    out = pd.DataFrame(rows)
    out.to_csv("/root/DPCN/A1/report/respondent_null_model.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
