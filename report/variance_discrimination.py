"""Does a statement's ability to separate communities follow from its variance?

Links the two halves of the report: S/V statements are consensual (low variance),
so they cannot discriminate between respondent communities even in principle.
Computes per-item std vs ANOVA eta-squared across the 4 respondent communities.
"""
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import f_oneway, spearmanr, pearsonr
from networkx.algorithms.community import louvain_communities

LIKERT = {"Strongly Disagree": -2, "Disagree": -1, "Neutral": 0, "Agree": 1, "Strongly Agree": 2}
DOMAIN = {"T": "Technology", "E": "Education", "S": "Ethics/Society", "V": "Environment"}

df = pd.read_csv("/root/DPCN/A1/Survey_Results_UC.csv", encoding="utf-8-sig")
df = df.rename(columns={"id. Response ID": "respondent_id"}).set_index("respondent_id")
num = df.replace({"": np.nan, "No Comments": np.nan}).replace(LIKERT).apply(pd.to_numeric)

sim = num.T.corr()
G = nx.Graph(); G.add_nodes_from(num.index)
for i, a in enumerate(num.index):
    for b in num.index[i + 1:]:
        r = sim.loc[a, b]
        if pd.notna(r) and r >= 0.4:
            G.add_edge(a, b, weight=float(r))
comms = [c for c in louvain_communities(G, weight="weight", seed=0) if len(c) >= 10]
comms = sorted(comms, key=len, reverse=True)
print("communities:", [len(c) for c in comms])

rows = []
for stmt in num.columns:
    groups = [num.loc[list(c), stmt].dropna() for c in comms]
    groups = [g for g in groups if len(g) > 1]
    F, p = f_oneway(*groups)
    allv = pd.concat(groups); gm = allv.mean()
    ss_t = ((allv - gm) ** 2).sum()
    ss_b = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    rows.append({"statement": stmt, "domain": DOMAIN[stmt[0]], "std": num[stmt].std(),
                 "mean": num[stmt].mean(), "eta_sq": ss_b / ss_t if ss_t else np.nan})
res = pd.DataFrame(rows)
res.to_csv("variance_vs_discrimination.csv", index=False)

r_p, p_p = pearsonr(res["std"], res["eta_sq"])
r_s, p_s = spearmanr(res["std"], res["eta_sq"])
print(f"\nstd vs eta^2:  Pearson r={r_p:.3f} (p={p_p:.2g})   Spearman rho={r_s:.3f} (p={p_s:.2g})")

print("\nPer-domain means:")
print(res.groupby("domain")[["std", "mean", "eta_sq"]].agg(["mean", "max"]).round(3).to_string())

print("\nDomain composition of the top-10 separating statements:")
print(res.nlargest(10, "eta_sq")["domain"].value_counts().to_string())
print("\nDomain composition of the 10 highest-variance statements:")
print(res.nlargest(10, "std")["domain"].value_counts().to_string())
print("\nDomain composition of the 10 LOWEST-variance (consensus) statements:")
print(res.nsmallest(10, "std")["domain"].value_counts().to_string())
print("\nMean agreement (mean score) by domain:")
print(res.groupby("domain")["mean"].mean().round(3).sort_values(ascending=False).to_string())
