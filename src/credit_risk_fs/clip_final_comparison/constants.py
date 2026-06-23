from __future__ import annotations

from pathlib import Path

OUTPUT_ROOT = Path("results/clip_final_comparison")
STATE_PATH = OUTPUT_ROOT / "pipeline_state.json"
LOG_PATH = OUTPUT_ROOT / "pipeline_execution.log"
LOCK_PATH = OUTPUT_ROOT / ".pipeline.lock"

DATASETS = ("homecredit", "lendingclub_v2")
MODELS = ("lr", "catboost")
MODEL_BUDGETS = {"lr": 20, "catboost": 40}
POOL_MULTIPLIERS = (2, 5, 10)
SCREENING_METHODS = (
    "clip_v2",
    "variance",
    "correlation_filter",
    "text_similarity",
    "statistics_only",
)
RANDOM_SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
CLIP_V2_SEEDS = (11, 22, 33, 44, 55)
SELECTED_CLIP_V2_SEED = 55
CORRELATION_FILTER_THRESHOLD = 0.95
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 2701
CENTRAL_POOL_MULTIPLIER = 5
FULL_V2_STATISTICAL_SCHEMA = (
    "missing_rate",
    "unique_ratio",
    "concentration_share",
    "signed_log_mean",
    "log_standard_deviation",
    "clipped_skewness",
    "normalized_entropy",
    "is_numeric",
    "is_categorical",
    "is_binary",
    "numeric_stats_valid",
    "skewness_valid",
    "entropy_valid",
)

ABLATIONS = {
    "full_v2": None,
    "text_only": "drop_all_statistical_fields",
    "statistics_only": "drop_text_branch",
    "missingness_only": ["missing_rate"],
    "without_location_scale": ["signed_log_mean", "log_standard_deviation"],
    "without_shape_diversity": ["concentration_share", "clipped_skewness", "normalized_entropy"],
    "without_type_validity": [
        "is_numeric",
        "is_categorical",
        "is_binary",
        "numeric_stats_valid",
        "skewness_valid",
        "entropy_valid",
    ],
}

STAGES = (
    "preflight",
    "seed_artifact_validation",
    "screening_scores",
    "candidate_pools",
    "core_candidate_pool_runs",
    "random_repetitions",
    "seed_score_generation",
    "seed_downstream",
    "ablation_schema_build",
    "ablation_contrastive_data",
    "ablation_training",
    "ablation_checkpoint_selection",
    "ablation_score_generation",
    "ablation_downstream",
    "temporal_cutoffs",
    "temporal_runs",
    "aggregate_rebuild",
    "metric_recomputation",
    "paired_uncertainty",
    "candidate_pool_diagnostics",
    "final_analysis",
    "plots",
    "tests",
    "final_audit",
)

FINAL_ANALYSIS_TABLES = (
    "master_results.csv",
    "candidate_pool_comparison.csv",
    "pool_size_sensitivity.csv",
    "random_baseline_distribution.csv",
    "seed_downstream_robustness.csv",
    "representation_ablations.csv",
    "temporal_cutoff_results.csv",
    "paired_uncertainty.csv",
    "runtime_comparison.csv",
    "feature_stability.csv",
    "semantic_coverage.csv",
    "claim_evidence_matrix.csv",
    "limitations_register.csv",
    "plot_manifest.csv",
    "experiment_summary.md",
)

PLOT_SPECS = (
    ("oot_auc_by_method_pool_size", "OOT AUC by screening method and pool size"),
    ("clip_v2_advantage", "CLIP-v2 advantage relative to each screener"),
    ("random_baseline_distribution", "Random-baseline distribution"),
    ("pool_size_sensitivity", "Pool-size sensitivity"),
    ("full_mrmr_vs_screened_mrmr", "Full-mRMR versus screened-mRMR"),
    ("representation_seed_robustness", "Downstream representation-seed robustness"),
    ("seed_feature_overlap", "Selected-feature overlap across seeds"),
    ("grouped_ablation_performance", "Grouped ablation performance"),
    ("temporal_cutoff_performance", "Temporal-cutoff performance"),
    ("paired_uncertainty_intervals", "Paired uncertainty intervals"),
    ("runtime_vs_auc", "Runtime versus OOT AUC"),
    ("semantic_coverage_vs_auc", "Semantic coverage versus OOT AUC"),
    ("candidate_pool_overlap_heatmap", "Candidate-pool overlap heatmap"),
    ("feature_selection_semantic_map", "Feature-selection semantic map"),
)
