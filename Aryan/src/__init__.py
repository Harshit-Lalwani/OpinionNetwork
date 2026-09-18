"""Opinion Network Analysis Package."""

from .config import *
from .data_preprocessing import (
    preprocess, load_raw_data, encode_likert, validate_data,
    compute_cronbach_alpha, identify_columns
)
from .correlation_network import (
    compute_correlation_matrix,
    build_topk_network,
    build_full_pipeline,
    get_network_summary,
)
from .network_metrics import (
    compute_global_metrics,
    compute_node_metrics,
    compute_all_metrics,
    print_global_summary,
    print_top_nodes,
)
from .community_detection import (
    detect_communities,
    get_community_mapping,
    compute_modularity,
    analyze_community_composition,
    compute_ari,
    get_community_summary,
    print_community_summary,
)
from .domain_analysis import (
    get_domain_columns,
    compute_domain_correlations,
    compute_statement_means,
    print_domain_summary,
)
from .negative_network import (
    build_negative_network,
    get_negative_summary,
    print_negative_summary,
)
from .null_models import (
    generate_er_graph,
    generate_configuration_model,
    compute_graph_metrics,
    run_null_models,
    compare_with_null,
    print_null_comparison,
)
from .sensitivity import (
    build_network_for_k,
    compute_stability,
    run_sensitivity_analysis,
    print_sensitivity_results,
    check_community_stability,
)
from .visualizations import (
    setup_style,
    plot_correlation_matrix,
    plot_opinion_network,
    plot_community_network,
    plot_domain_heatmap,
    plot_centrality,
    plot_negative_network,
    plot_null_model_comparison,
    plot_domain_vs_community,
    generate_all_figures,
)

__all__ = [
    # config
    "ROOT", "DATA_DIR", "RAW_CSV", "FIGURES_DIR", "RESULTS_DIR",
    "LIKERT_MAPPING", "MISSING_STRINGS",
    "DOMAIN_PREFIXES", "DOMAIN_ORDER", "DOMAIN_NAMES", "DOMAIN_COLORS",
    "TOP_K", "NEGATIVE_THRESHOLD", "CORRELATION_METHOD",
    "RANDOM_SEED", "N_NULL_NETWORKS", "SENSITIVITY_K_VALUES",
    # data_preprocessing
    "preprocess", "load_raw_data", "encode_likert", "validate_data",
    "compute_cronbach_alpha", "identify_columns",
    # correlation_network
    "compute_correlation_matrix", "build_topk_network",
    "build_full_pipeline", "get_network_summary",
    # network_metrics
    "compute_global_metrics", "compute_node_metrics", "compute_all_metrics",
    "print_global_summary", "print_top_nodes",
    # community_detection
    "detect_communities", "get_community_mapping", "compute_modularity",
    "analyze_community_composition", "compute_ari",
    "get_community_summary", "print_community_summary",
    # domain_analysis
    "get_domain_columns", "compute_domain_correlations",
    "compute_statement_means", "print_domain_summary",
    # negative_network
    "build_negative_network", "get_negative_summary", "print_negative_summary",
    # null_models
    "generate_er_graph", "generate_configuration_model", "compute_graph_metrics",
    "run_null_models", "compare_with_null", "print_null_comparison",
    # sensitivity
    "build_network_for_k", "compute_stability", "run_sensitivity_analysis",
    "print_sensitivity_results", "check_community_stability",
    # visualizations
    "setup_style", "plot_correlation_matrix", "plot_opinion_network",
    "plot_community_network", "plot_domain_heatmap", "plot_centrality",
    "plot_negative_network", "plot_null_model_comparison",
    "plot_domain_vs_community", "generate_all_figures",
]