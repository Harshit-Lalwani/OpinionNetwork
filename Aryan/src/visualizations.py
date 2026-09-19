"""Publication-quality visualizations for opinion network."""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    FIGURES_DIR, DOMAIN_COLORS, DOMAIN_NAMES, DOMAIN_ORDER, DOMAIN_PREFIXES,
    NODE_SIZE_MIN, NODE_SIZE_MAX, EDGE_WIDTH_MIN, EDGE_WIDTH_MAX,
    FONT_SIZE, TITLE_FONT_SIZE, FIGURE_DPI, FIGURE_FORMAT,
    CORR_MATRIX_FIG, OPINION_NETWORK_FIG, COMMUNITY_NETWORK_FIG,
    DOMAIN_HEATMAP_FIG, CENTRALITY_FIG, NEGATIVE_NETWORK_FIG,
    NULL_MODEL_FIG, DOMAIN_VS_COMMUNITY_FIG,
)


def setup_style():
    """Set consistent matplotlib/seaborn style."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': TITLE_FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 1,
        'ytick.labelsize': FONT_SIZE - 1,
        'legend.fontsize': FONT_SIZE - 1,
        'figure.dpi': FIGURE_DPI,
        'savefig.dpi': FIGURE_DPI,
        'savefig.bbox': 'tight',
        'savefig.format': FIGURE_FORMAT,
    })


def get_node_colors(G: nx.Graph, col_to_domain: Dict[str, str]) -> List[str]:
    """Get color for each node based on domain."""
    return [DOMAIN_COLORS.get(col_to_domain.get(n, "Unknown"), "#999999") for n in G.nodes()]


def get_node_sizes(G: nx.Graph, metric: str = "strength") -> List[float]:
    """Get node sizes proportional to metric."""
    if metric == "strength":
        vals = dict(G.degree(weight="weight"))
    elif metric == "degree":
        vals = dict(G.degree())
    elif metric == "betweenness":
        # Compute on giant component
        components = list(nx.connected_components(G))
        if len(components) > 1:
            G_giant = G.subgraph(max(components, key=len)).copy()
            vals = nx.betweenness_centrality(G_giant, weight="weight")
            # Pad
            for n in G.nodes():
                if n not in vals:
                    vals[n] = 0.0
        else:
            vals = nx.betweenness_centrality(G, weight="weight")
    else:
        vals = dict(G.degree(weight="weight"))
    
    vmin, vmax = min(vals.values()), max(vals.values())
    if vmax == vmin:
        return [NODE_SIZE_MIN] * G.number_of_nodes()
    
    return [NODE_SIZE_MIN + (NODE_SIZE_MAX - NODE_SIZE_MIN) * (vals[n] - vmin) / (vmax - vmin) for n in G.nodes()]


def get_edge_widths(G: nx.Graph) -> List[float]:
    """Get edge widths proportional to weight."""
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    if not weights:
        return []
    wmin, wmax = min(weights), max(weights)
    if wmax == wmin:
        return [EDGE_WIDTH_MIN] * len(weights)
    return [EDGE_WIDTH_MIN + (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN) * (w - wmin) / (wmax - wmin) for w in weights]


# ============================================================
# FIGURE 1: Correlation Matrix Heatmap
# ============================================================
def plot_correlation_matrix(
    corr: pd.DataFrame,
    col_to_domain: Dict[str, str],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot 60x60 correlation heatmap with domain annotations."""
    setup_style()
    
    # Short labels for statements (T01, E02, etc.)
    short_labels = [c.split(".")[0] for c in corr.index]
    
    # Domain boundaries for separator lines
    domain_bounds = {}
    start = 0
    for prefix in DOMAIN_ORDER:
        count = sum(1 for c in corr.index if c.split(".")[0][0] == prefix)
        domain_bounds[prefix] = (start, start + count)
        start += count
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Heatmap
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    
    # Domain separators
    for prefix, (s, e) in domain_bounds.items():
        ax.axhline(e - 0.5, color="white", linewidth=1.5)
        ax.axvline(e - 0.5, color="white", linewidth=1.5)
    
    # Labels
    ax.set_xticks(range(len(short_labels)))
    ax.set_yticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(short_labels, fontsize=6)
    
    # Domain labels on top and left
    for prefix, (s, e) in domain_bounds.items():
        mid = (s + e) / 2
        ax.text(mid, -2, DOMAIN_PREFIXES[prefix], ha="center", va="bottom", 
                fontweight="bold", fontsize=9, color=DOMAIN_COLORS[DOMAIN_PREFIXES[prefix]])
        ax.text(-2, mid, DOMAIN_PREFIXES[prefix], ha="right", va="center",
                fontweight="bold", fontsize=9, color=DOMAIN_COLORS[DOMAIN_PREFIXES[prefix]], rotation=90)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    cbar.ax.tick_params(labelsize=8)
    
    ax.set_title("Statement–Statement Correlation Matrix", pad=20)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


# ============================================================
# FIGURE 2: Main Opinion Network
# ============================================================
def plot_opinion_network(
    G: nx.Graph,
    col_to_domain: Dict[str, str],
    save_path: Optional[Path] = None,
    label_top_n: int = 15
) -> plt.Figure:
    """Plot main opinion network colored by domain, sized by strength."""
    setup_style()
    
    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.4, iterations=100)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Edges
    edge_widths = get_edge_widths(G)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color="gray", ax=ax)
    
    # Nodes by domain
    for prefix, name in DOMAIN_PREFIXES.items():
        nodes = [n for n in G.nodes() if col_to_domain.get(n, "").startswith(prefix)]
        if nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes,
                node_color=DOMAIN_COLORS[name],
                node_size=[get_node_sizes(G)[list(G.nodes()).index(n)] for n in nodes],
                label=name, ax=ax
            )
    
    # Labels for top strength nodes
    strengths = dict(G.degree(weight="weight"))
    top_nodes = sorted(strengths.items(), key=lambda x: x[1], reverse=True)[:label_top_n]
    top_labels = {n: n.split(".")[0] for n, _ in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=top_labels, font_size=7, font_weight="bold", ax=ax)
    
    ax.set_title("Opinion Correlation Network (Top-3 per Node)", pad=15)
    ax.legend(scatterpoints=1, markerscale=1.5, fontsize=9, loc="upper left")
    ax.axis("off")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


# ============================================================
# FIGURE 3: Community-colored Network
# ============================================================
def plot_community_network(
    G: nx.Graph,
    communities: List[set],
    save_path: Optional[Path] = None,
    label_top_n: int = 15
) -> plt.Figure:
    """Plot network colored by Louvain community."""
    setup_style()
    
    comm_map = {n: i for i, comm in enumerate(communities) for n in comm}
    n_comms = len(communities)
    
    # Color palette for communities
    comm_palette = sns.color_palette("tab20", n_comms)
    node_colors = [comm_palette[comm_map[n]] for n in G.nodes()]
    
    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.4, iterations=100)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    edge_widths = get_edge_widths(G)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.2, edge_color="gray", ax=ax)
    
    for i, comm in enumerate(communities):
        nx.draw_networkx_nodes(
            G, pos, nodelist=list(comm),
            node_color=[comm_palette[i]],
            node_size=[get_node_sizes(G)[list(G.nodes()).index(n)] for n in comm],
            label=f"Community {i} (n={len(comm)})", ax=ax
        )
    
    # Labels
    strengths = dict(G.degree(weight="weight"))
    top_nodes = sorted(strengths.items(), key=lambda x: x[1], reverse=True)[:label_top_n]
    top_labels = {n: n.split(".")[0] for n, _ in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=top_labels, font_size=7, font_weight="bold", ax=ax)
    
    ax.set_title(f"Opinion Network – Louvain Communities ({n_comms} communities)", pad=15)
    ax.legend(scatterpoints=1, markerscale=1.2, fontsize=8, loc="upper left", ncol=2)
    ax.axis("off")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


# ============================================================
# FIGURE 4: Domain Correlation Heatmap (4x4)
# ============================================================
def plot_domain_heatmap(
    domain_corr: pd.DataFrame,
    within_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot 4x4 domain correlation heatmap with annotations."""
    setup_style()
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Annotate with values
    annot = domain_corr.round(3).astype(str)
    
    im = ax.imshow(domain_corr.values, cmap="RdBu_r", vmin=-0.1, vmax=0.5, aspect="auto")
    
    # Add text annotations
    for i in range(len(domain_corr)):
        for j in range(len(domain_corr)):
            val = domain_corr.iloc[i, j]
            color = "white" if abs(val) > 0.25 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", 
                    color=color, fontweight="bold", fontsize=11)
    
    ax.set_xticks(range(len(domain_corr.columns)))
    ax.set_yticks(range(len(domain_corr.index)))
    ax.set_xticklabels(domain_corr.columns, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(domain_corr.index, fontsize=10)
    
    # Domain colors on axis
    for i, label in enumerate(ax.get_yticklabels()):
        domain = label.get_text()
        label.set_color(DOMAIN_COLORS.get(domain, "black"))
        label.set_fontweight("bold")
    for i, label in enumerate(ax.get_xticklabels()):
        domain = label.get_text()
        label.set_color(DOMAIN_COLORS.get(domain, "black"))
        label.set_fontweight("bold")
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Mean Pearson r")
    cbar.ax.tick_params(labelsize=8)
    
    ax.set_title("Domain–Domain Mean Correlation", pad=15)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


# ============================================================
# FIGURE 5: Centrality / Bridge Opinions
# ============================================================
def plot_centrality(
    node_metrics: pd.DataFrame,
    col_to_domain: Dict[str, str],
    save_path: Optional[Path] = None,
    top_n: int = 15
) -> plt.Figure:
    """Horizontal bar chart of top nodes by betweenness and strength."""
    setup_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Betweenness
    top_betw = node_metrics.nlargest(top_n, "betweenness")
    colors_betw = [DOMAIN_COLORS.get(col_to_domain.get(n, "Unknown"), "#999999") for n in top_betw.index]
    axes[0].barh(range(len(top_betw)), top_betw["betweenness"].values, color=colors_betw, edgecolor="black", linewidth=0.5)
    axes[0].set_yticks(range(len(top_betw)))
    axes[0].set_yticklabels([n.split(".")[0] for n in top_betw.index], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Betweenness Centrality")
    axes[0].set_title(f"Top {top_n} Bridge Opinions (Betweenness)")
    
    # Strength
    top_str = node_metrics.nlargest(top_n, "strength")
    colors_str = [DOMAIN_COLORS.get(col_to_domain.get(n, "Unknown"), "#999999") for n in top_str.index]
    axes[1].barh(range(len(top_str)), top_str["strength"].values, color=colors_str, edgecolor="black", linewidth=0.5)
    axes[1].set_yticks(range(len(top_str)))
    axes[1].set_yticklabels([n.split(".")[0] for n in top_str.index], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Strength (Weighted Degree)")
    axes[1].set_title(f"Top {top_n} Central Opinions (Strength)")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=DOMAIN_COLORS[d], label=d) for d in DOMAIN_NAMES]
    axes[1].legend(handles=legend_elements, loc="lower right", fontsize=8, title="Domain")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


# ============================================================
# FIGURE 6: Negative Correlation Network
# ============================================================
def plot_negative_network(
    G_neg: nx.Graph,
    col_to_domain: Dict[str, str],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot negative correlation (tension) network."""
    setup_style()
    
    if G_neg.number_of_edges() == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No negative edges above threshold", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig
    
    pos = nx.spring_layout(G_neg, weight="weight", seed=42, k=0.8, iterations=100)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Edges - width by absolute weight
    edges = list(G_neg.edges(data=True))
    widths = [abs(d["weight"]) * 5 for _, _, d in edges]
    colors = ["red" if d["weight"] < 0 else "blue" for _, _, d in edges]
    nx.draw_networkx_edges(G_neg, pos, width=widths, alpha=0.6, edge_color=colors, ax=ax)
    
    # Nodes by domain
    for prefix, name in DOMAIN_PREFIXES.items():
        nodes = [n for n in G_neg.nodes() if col_to_domain.get(n, "").startswith(prefix)]
        if nodes:
            nx.draw_networkx_nodes(
                G_neg, pos, nodelist=nodes,
                node_color=DOMAIN_COLORS[name],
                node_size=400, label=name, ax=ax
            )
    
    # Labels - all nodes in negative network
    nx.draw_networkx_labels(G_neg, pos, labels={n: n.split(".")[0] for n in G_neg.nodes()}, 
                            font_size=8, font_weight="bold", ax=ax)
    
    ax.set_title(f"Opinion Tension Network (r < -0.25, {G_neg.number_of_edges()} edges)", pad=15)
    ax.legend(scatterpoints=1, fontsize=9, loc="upper left")
    ax.axis("off")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


# ============================================================
# FIGURE 7: Null Model Comparison
# ============================================================
def plot_null_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Boxplot/histogram comparing observed vs null model metrics."""
    setup_style()
    
    metrics = comparison_df["metric"].tolist()
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), squeeze=False)
    
    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        row = comparison_df[comparison_df["metric"] == metric].iloc[0]
        
        # We need the raw null distributions - this is a simplified version
        # showing observed vs mean ± std
        obs = row["observed"]
        er_mean, er_std = row["er_mean"], row["er_std"]
        config_mean, config_std = row["config_mean"], row["config_std"]
        
        x_pos = [0, 1, 2]
        means = [obs, er_mean, config_mean]
        stds = [0, er_std, config_std]
        labels = ["Observed", "ER Null", "Config Null"]
        colors = ["black", "steelblue", "darkorange"]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor="black")
        bars[0].set_hatch("///")
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(True, axis="y", alpha=0.3)
    
    plt.suptitle("Observed vs Null Model Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


# ============================================================
# FIGURE 8: Domain vs Community Table
# ============================================================
def plot_domain_vs_community(
    composition_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot domain vs community composition as annotated heatmap/table."""
    setup_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize by community size for proportion view
    prop_df = composition_df.div(composition_df.sum(axis=1), axis=0)
    
    im = ax.imshow(prop_df.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    
    # Annotate with counts
    for i in range(len(composition_df)):
        for j in range(len(composition_df.columns)):
            count = composition_df.iloc[i, j]
            prop = prop_df.iloc[i, j]
            color = "white" if prop > 0.5 else "black"
            ax.text(j, i, f"{int(count)}\n({prop:.0%})", ha="center", va="center",
                    color=color, fontweight="bold", fontsize=9)
    
    ax.set_xticks(range(len(composition_df.columns)))
    ax.set_yticks(range(len(composition_df.index)))
    ax.set_xticklabels(composition_df.columns, fontsize=9)
    ax.set_yticklabels([f"Comm {idx}" for idx in composition_df.index], fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Proportion of Community")
    cbar.ax.tick_params(labelsize=8)
    
    ax.set_title("Community Composition by Domain", pad=15)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig


# ============================================================
# MASTER FUNCTION: Generate All Figures
# ============================================================
def generate_all_figures(
    G: nx.Graph,
    G_neg: nx.Graph,
    corr: pd.DataFrame,
    domain_corr: pd.DataFrame,
    within_df: pd.DataFrame,
    node_metrics: pd.DataFrame,
    communities: List[set],
    col_to_domain: Dict[str, str],
    comparison_df: Optional[pd.DataFrame] = None,
    composition_df: Optional[pd.DataFrame] = None,
) -> Dict[str, plt.Figure]:
    """Generate all 8 figures and save to figures/."""
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    print("Generating Figure 1: Correlation Matrix...")
    figures["correlation_matrix"] = plot_correlation_matrix(
        corr, col_to_domain, FIGURES_DIR / CORR_MATRIX_FIG
    )
    
    print("Generating Figure 2: Opinion Network...")
    figures["opinion_network"] = plot_opinion_network(
        G, col_to_domain, FIGURES_DIR / OPINION_NETWORK_FIG
    )
    
    print("Generating Figure 3: Community Network...")
    figures["community_network"] = plot_community_network(
        G, communities, FIGURES_DIR / COMMUNITY_NETWORK_FIG
    )
    
    print("Generating Figure 4: Domain Heatmap...")
    figures["domain_heatmap"] = plot_domain_heatmap(
        domain_corr, within_df, FIGURES_DIR / DOMAIN_HEATMAP_FIG
    )
    
    print("Generating Figure 5: Centrality...")
    figures["centrality"] = plot_centrality(
        node_metrics, col_to_domain, FIGURES_DIR / CENTRALITY_FIG
    )
    
    print("Generating Figure 6: Negative Network...")
    figures["negative_network"] = plot_negative_network(
        G_neg, col_to_domain, FIGURES_DIR / NEGATIVE_NETWORK_FIG
    )
    
    if comparison_df is not None:
        print("Generating Figure 7: Null Model Comparison...")
        figures["null_model"] = plot_null_model_comparison(
            comparison_df, FIGURES_DIR / NULL_MODEL_FIG
        )
    
    if composition_df is not None:
        print("Generating Figure 8: Domain vs Community...")
        figures["domain_vs_community"] = plot_domain_vs_community(
            composition_df, FIGURES_DIR / DOMAIN_VS_COMMUNITY_FIG
        )
    
    print(f"\nAll figures saved to {FIGURES_DIR}")
    return figures


if __name__ == "__main__":
    from src import preprocess, build_full_pipeline, compute_correlation_matrix
    from src.network_metrics import compute_all_metrics
    from src.community_detection import detect_communities, analyze_community_composition
    from src.domain_analysis import compute_domain_correlations
    from src.negative_network import build_negative_network
    from src.null_models import run_null_models, compare_with_null
    
    df_num, stmt_cols, col_to_domain, validation = preprocess()
    G, corr, edge_df, summary = build_full_pipeline(df_num)
    G_neg, edge_df_neg = build_negative_network(corr, threshold=-0.25)
    
    metrics = compute_all_metrics(G)
    communities = detect_communities(G)
    domain_corr, within_df = compute_domain_correlations(corr, stmt_cols)
    composition_df = analyze_community_composition(communities, col_to_domain)
    
    # Quick null model for plotting
    er_df, config_df = run_null_models(G, n_null=50, parallel=False)
    comparison_df = compare_with_null(compute_all_metrics(G)["global"], er_df, config_df)
    
    generate_all_figures(
        G, G_neg, corr, domain_corr, within_df,
        metrics["nodes"], communities, col_to_domain,
        comparison_df, composition_df
    )