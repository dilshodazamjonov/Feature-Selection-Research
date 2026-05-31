# Final Report Notebook Protocol

## Purpose

Final dataset notebooks are narrative research reports, not coding workspaces. Their job is to present the dataset-specific review clearly enough for professor or research supervision, using already-generated experiment artifacts rather than exploratory scratch code.

## Required Design Rule

- Notebook logic must stay thin.
- Functions live in `src/credit_risk_fs/reporting/notebook_report.py`.
- Public report helpers are exported from `src/credit_risk_fs/reporting/__init__.py`.
- Dataset notebooks live under `Notebooks/<dataset>/notebooks/final_report.ipynb`.
- Notebooks must not define helper functions.
- Notebooks must not manually read many CSV files in scattered cells.
- Notebooks should only import reporting helpers, load prepared tables and plots, display them, and add concise interpretation.
- Do not create a lowercase root-level `notebooks/` directory for these final reports.

## Reproducibility Contract

- Final dataset reports should be reproducible from aggregate CSV artifacts under `results/<dataset>/`.
- The primary inputs are:
  - `final_comparison_table.csv`
  - `feature_stability_table.csv`
  - `feature_drift_table.csv`
  - `semantic_coverage_table.csv`
  - `paired_fold_comparisons.csv`
  - `llm_call_summary.csv`
  - `matrix_runs.csv`
  - `failed_runs.csv`
- Post-run analysis inputs may also be used when present:
  - `analysis/feature_level_drift/feature_level_psi_by_run.csv`
  - `analysis/feature_level_drift/psi_distribution_by_pipeline.csv`
  - `analysis/feature_level_drift/high_psi_features_by_pipeline.csv`
  - `analysis/feature_level_drift/llm_then_mrmr_drift_source_breakdown.csv`
  - `analysis/semantic_redundancy/semantic_coverage_redundancy_by_pipeline.csv`
  - `results/cross_dataset/analysis/stability_significance/llm_stability_diagnosis.csv`
  - `results/cross_dataset/analysis/stability_significance/paired_fold_significance_tests.csv`
- If an aggregate table is missing, the reporting wrapper may fall back to per-run artifacts when available, but it must not invent values.
- If a section is incomplete, the notebook must surface that clearly.
- Final-report plots must be saved only under `results/<dataset>/final_report/plots/`.
- Each plot directory must include `plot_manifest.csv` with columns `plot_file`, `source_table`, `rows_used`, `columns_used`, `purpose`, `status`, and `skip_reason`.
- Skip no-information plots when data is missing, empty, constant, or lacks enough categories for comparison.

## Mandatory Report Content

Each final dataset notebook should keep the same high-level structure:

1. Setup and imports
2. Research question and dataset role
3. Dataset snapshot
4. DEV/OOT split rationale
5. Experiment matrix overview
6. Topline performance comparison
7. Stability review
8. Paired fold significance
9. Drift and robustness review
10. Semantic coverage and redundancy review
11. Efficiency tradeoff
12. Best runs deep dive
13. Leakage and temporal governance
14. Failure cases and surprises
15. Conclusions for the dataset
16. Future CLIP-style validation placeholder
17. Saved final-report plots and manifest
18. Next actions

## DEV/OOT Split Rule

The DEV/OOT split rationale is mandatory in every final notebook.

The report must state:

- The split is time-based, not random.
- DEV is the older window used for CV, feature selection, and model training.
- OOT is the newer holdout used only for final evaluation.
- Window choice should be justified by both observation counts and target-rate behavior over time.
- OOT metrics and OOT bad-rate behavior must not be used to tune selectors or hyperparameters.

## Reporting Tone

- Be concrete and analytical.
- Avoid scratchpad code and debugging cells.
- Avoid repeating the same conclusion in every section.
- Do not over-claim marginal gains.
- Do not say LLM replaces statistical selectors unless the evidence clearly supports that claim.
- Prefer disciplined wording such as: `LLM screening is useful as a first-stage helper`.
- Treat small AUC differences cautiously unless paired-fold tests support them.
- Report whether drift, stability, and semantic evidence agree or conflict.

## Future CLIP-Style Placeholder

The final notebooks should reserve a future-work section for CLIP-style semantic-statistical validation, but they should not implement that method yet.

That placeholder should explain:

- This is not image CLIP.
- The intended future method is a dual-encoder or contrastive alignment between feature text or metadata and empirical feature behavior.
- The method must be evaluated under the same DEV/OOT protocol.
- OOT metrics must not be used for CLIP-style training or feature selection.
- The comparison target is an alternative first-stage screener, not a replacement for the whole pipeline.

## Required Wrapper Surface

The final notebooks should call the wrapper functions rather than implementing their own data loading or transformations:

- `load_dataset_report_inputs`
- `load_dataset_snapshot`
- `load_split_summary`
- `load_time_bucket_summary`
- `load_final_comparison`
- `load_stability_table`
- `load_drift_table`
- `load_semantic_coverage_table`
- `load_best_runs`
- `load_run_artifacts`
- `plot_dev_oot_split_diagnostics`
- `plot_observation_count_by_time`
- `plot_bad_rate_by_time`
- `plot_metric_leaderboard`
- `plot_stability_vs_performance`
- `plot_drift_vs_performance`
- `plot_semantic_coverage`
- `plot_runtime_tradeoff`
- `summarize_split_rationale`
- `summarize_dataset_findings`
- `summarize_clip_validation_placeholder`
