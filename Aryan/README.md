# Opinion Network Analysis — DPCN Assignment 1

## Project Overview
Constructs an **opinion–opinion correlation network** from a 96×60 Likert survey (Technology, Education, Ethics/Society, Environment). Survey statements are nodes; edges represent strong positive correlations between responses.

**Key Finding**: Network communities (Louvain, modularity=0.44) **do not align** with survey categories (ARI=0.06) — opinions organize by latent structure, not designer categories.

---

## Pipeline

```
Raw CSV → Preprocess → Correlation Matrix → Top-3 Sparsification → Network Analysis
                                                              ↓
                                                    ┌──────────┴──────────┐
                                                    ▼                     ▼
                                           Core Metrics            Extended Analysis
                                           ├─ Global/Local        ├─ Negative Network
                                           ├─ Centrality          ├─ Null Models (ER + Config)
                                           └─ Communities         └─ Sensitivity (K=2/3/5)
```

---

## Outputs

### Figures (`figures/`)
| File | Description |
|------|-------------|
| `correlation_matrix.png` | 60×60 Pearson heatmap with domain boundaries |
| `opinion_network.png` | Main network: nodes=domain-colored, size=strength |
| `community_network.png` | Same layout, nodes=community-colored (6 communities) |
| `domain_heatmap.png` | 4×4 domain mean correlation matrix |
| `centrality.png` | Top-15 by betweenness (bridges) and strength (hubs) |
| `negative_network.png` | 12 tension edges (r < -0.25) |
| `null_model_comparison.png` | Observed vs ER/Config null (clustering, transitivity, modularity) |
| `domain_vs_community.png` | Community composition by domain (counts + proportions) |

### Results (`results/`)
| File | Description |
|------|-------------|
| `correlations.csv` | Full 60×60 correlation matrix |
| `edge_list.csv` | 136 edges (node1, node2, weight) |
| `node_metrics.csv` | 60×7: degree, strength, clustering, betweenness, closeness, eigenvector |
| `network_summary.csv` | Global: density=0.077, avg_degree=4.53, clustering=0.092, path=2.87, modularity=0.44 |
| `communities.csv` | Node → community + domain mapping |
| `domain_correlations.csv` | 4×4 domain matrix |
| `within_domain_stats.csv` | Per-domain mean/median/std/min/max correlation |
| `statement_means.csv` | Mean agreement per statement (E15=1.71 top, E03=-0.94 bottom) |
| `negative_edges.csv` | 12 tension edges (E03–E10=-0.44 strongest) |
| `null_model_results.csv` | Z-scores/p-values vs ER & Config (100 reps each) |
| `sensitivity_results.csv` | K=2/3/5 comparison (edges, modularity, ARI, community stability) |
| `validation_summary.csv` | Data quality: 9.4% missing, 5 duplicates, Cronbach α per domain |

---

## Quick Start

```bash
cd Aryan
pip install -r requirements.txt  # pandas, numpy, scipy, networkx, matplotlib, seaborn, scikit-learn
python run_pipeline.py
```

Outputs → `figures/` (PNG, 300 DPI) + `results/` (CSV)

---

## Key Numbers for Report

| Metric | Value |
|--------|-------|
| Respondents / Statements | 96 / 60 |
| Network: Nodes / Edges | 60 / 136 |
| Density / Avg Degree | 0.077 / 4.53 |
| Connected Components | 1 (fully connected) |
| Clustering / Transitivity | 0.092 / 0.122 |
| Avg Path Length / Diameter | 2.87 / 5 |
| Modularity (Louvain) | 0.441 |
| Communities / ARI vs T/E/S/V | 6 / 0.059 |
| Top Strength Nodes | V07, E10, V10, S14, S09 |
| Top Betweenness Nodes | E10, T01, E08, S09, T08 |
| Negative Edges (r < -0.25) | 12 (strongest: E03–E10 = -0.44) |
| Within-Domain Mean Corr | Env=0.334, Ethics=0.259, Edu=0.118, Tech=0.103 |
| Cross-Domain Max | Ethics↔Env = 0.243 |
| Null Model (Clustering) | Obs > ER (z=0.58), Obs > Config (z=1.08) |
| Null Model (Transitivity) | Obs >> ER (z=2.96), Obs >> Config (z=2.92) |
| Null Model (Modularity) | Obs > ER (z=2.72), Obs > Config (z=1.14) |

---

## Dependencies
```
pandas>=2.0
numpy>=1.24
scipy>=1.10
networkx>=3.2
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
```

---

## Report Integration
1. Copy `figures/*.png` to Overleaf
2. Use `results/*.csv` for tables (copy-paste or `siunitx` + `csvsimple`)
3. Follow assignment section order: Team → GitHub → Dataset → Pipeline → Analysis → Results → Contributions