"""Centralized configuration for Opinion Network Analysis."""

from pathlib import Path

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]  # Aryan/
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "Survey_Results_UC.csv"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

# ============================================================
# LIKERT ENCODING
# ============================================================
LIKERT_MAPPING = {
    "Strongly Disagree": -2,
    "Disagree": -1,
    "Neutral": 0,
    "Agree": 1,
    "Strongly Agree": 2,
}

MISSING_STRINGS = ["", "No Comments", "no comments", "NO COMMENTS"]

# ============================================================
# DOMAIN DEFINITIONS
# ============================================================
DOMAIN_PREFIXES = {
    "T": "Technology",
    "E": "Education",
    "S": "Ethics/Society",
    "V": "Environment",
}

DOMAIN_ORDER = ["T", "E", "S", "V"]
DOMAIN_NAMES = [DOMAIN_PREFIXES[p] for p in DOMAIN_ORDER]

DOMAIN_COLORS = {
    "Technology": "#1f77b4",
    "Education": "#ff7f0e",
    "Ethics/Society": "#2ca02c",
    "Environment": "#d62728",
}

# ============================================================
# NETWORK CONSTRUCTION PARAMETERS
# ============================================================
TOP_K = 3                    # top-K positive correlations per node
NEGATIVE_THRESHOLD = -0.25   # threshold for negative-correlation network
CORRELATION_METHOD = "pearson"  # "pearson" or "spearman"

# ============================================================
# NULL MODEL PARAMETERS
# ============================================================
RANDOM_SEED = 42
N_NULL_NETWORKS = 100

# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================
SENSITIVITY_K_VALUES = [2, 3, 5]

# ============================================================
# VISUALIZATION DEFAULTS
# ============================================================
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
NODE_SIZE_MIN = 100
NODE_SIZE_MAX = 1200
EDGE_WIDTH_MIN = 0.5
EDGE_WIDTH_MAX = 4.0
FONT_SIZE = 10
TITLE_FONT_SIZE = 12

# ============================================================
# OUTPUT FILENAMES
# ============================================================
CORRELATIONS_CSV = "correlations.csv"
NODE_METRICS_CSV = "node_metrics.csv"
COMMUNITIES_CSV = "communities.csv"
NETWORK_SUMMARY_CSV = "network_summary.csv"
DOMAIN_CORRELATIONS_CSV = "domain_correlations.csv"
NEGATIVE_EDGES_CSV = "negative_edges.csv"
NULL_MODEL_RESULTS_CSV = "null_model_results.csv"

# Figure filenames
CORR_MATRIX_FIG = "correlation_matrix.png"
OPINION_NETWORK_FIG = "opinion_network.png"
COMMUNITY_NETWORK_FIG = "community_network.png"
DOMAIN_HEATMAP_FIG = "domain_heatmap.png"
CENTRALITY_FIG = "centrality.png"
NEGATIVE_NETWORK_FIG = "negative_network.png"
NULL_MODEL_FIG = "null_model_comparison.png"
DOMAIN_VS_COMMUNITY_FIG = "domain_vs_community.png"