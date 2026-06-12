# Credit Feature Selection Research

This repository is a research framework for testing whether LLM-based metadata screening is useful as a first-stage feature-selection helper for credit scoring.

Main hypothesis:
- LLMs can use column metadata and semantic descriptions to produce a useful first-stage screening of candidate features.
- Statistical selectors and hybrid selectors can then refine that screened set.
- The value claim is research-oriented feature selection support, not end-to-end automated credit decisioning.

Datasets:
- `Home Credit` is the primary dataset.
- `LendingClub` is the external validation dataset.

Selectors compared:
- `mrmr`
- `boruta`
- `pca`
- `domain_rule_baseline`
- `llm`
- `llm_then_mrmr`
- `llm_then_boruta`
- `stable_core_llm_fill`

Evaluation models:
- `Logistic Regression`
- `CatBoost`

Out of scope:
- no stacking
- no calibration modules
- no production scoring or deployment logic
- no claim that the predictive models are production PD engines

## Research Framing

The LLM in this project is a metadata screener, not the final model and not a production underwriter.

That means:
- the LLM sees feature metadata and descriptions
- the LLM produces a ranked or screened candidate set
- downstream selectors and models evaluate whether that screening is useful
- the same fold-specific LLM ranking can be reused across downstream selectors and models for fairness and reproducibility

## Canonical Structure

The canonical codebase now lives under `src/credit_risk_fs/`. The structure below is the intended working layout for the refactored repository.

```text
credit_feature_selection_research/
|-- src/
|   `-- credit_risk_fs/
|       |-- __init__.py
|       |-- data/
|       |   |-- __init__.py
|       |   |-- loaders.py
|       |   |-- schemas.py
|       |   |-- dataset_registry.py
|       |   |-- homecredit.py
|       |   `-- lendingclub.py
|       |-- preprocessing/
|       |   |-- __init__.py
|       |   |-- cleaning.py
|       |   |-- missingness.py
|       |   |-- encoding.py
|       |   |-- temporal_split.py
|       |   |-- leakage.py
|       |   |-- labeling.py
|       |   |-- homecredit.py
|       |   `-- lendingclub.py
|       |-- feature_engineering/
|       |   |-- __init__.py
|       |   |-- base.py
|       |   |-- homecredit/
|       |   |   |-- __init__.py
|       |   |   |-- bureau.py
|       |   |   |-- previous_application.py
|       |   |   |-- installments.py
|       |   |   |-- pos_cash.py
|       |   |   |-- credit_card.py
|       |   |   `-- assemble.py
|       |   `-- lendingclub/
|       |       |-- __init__.py
|       |       |-- application.py
|       |       `-- assemble.py
|       |-- feature_metadata/
|       |   |-- __init__.py
|       |   |-- builder.py
|       |   |-- semantic_groups.py
|       |   |-- descriptions.py
|       |   |-- metadata_schema.py
|       |   |-- homecredit.py
|       |   `-- lendingclub.py
|       |-- selectors/
|       |   |-- __init__.py
|       |   |-- base.py
|       |   |-- registry.py
|       |   |-- mrmr.py
|       |   |-- boruta.py
|       |   |-- pca.py
|       |   |-- domain_rule_baseline.py
|       |   |-- llm_screening.py
|       |   |-- llm_then_stat.py
|       |   `-- stable_core_llm_fill.py
|       |-- models/
|       |   |-- __init__.py
|       |   |-- registry.py
|       |   |-- logistic_regression.py
|       |   |-- catboost_model.py
|       |   `-- training.py
|       |-- evaluation/
|       |   |-- __init__.py
|       |   |-- metrics.py
|       |   |-- stability.py
|       |   |-- drift.py
|       |   |-- semantic_coverage.py
|       |   |-- redundancy.py
|       |   |-- aggregation.py
|       |   `-- plotting.py
|       |-- experiments/
|       |   |-- __init__.py
|       |   |-- config.py
|       |   |-- matrix.py
|       |   |-- runner.py
|       |   |-- tracking.py
|       |   |-- single_run.py
|       |   `-- compare.py
|       |-- pipelines/
|       |   |-- __init__.py
|       |   |-- common.py
|       |   |-- dataset_adapter.py
|       |   |-- homecredit_pipeline.py
|       |   |-- lendingclub_pipeline.py
|       |   `-- run_pipeline.py
|       `-- utils/
|           |-- __init__.py
|           |-- io.py
|           |-- logging.py
|           |-- hashing.py
|           |-- paths.py
|           `-- serialization.py
|-- configs/
|   |-- base.yaml
|   |-- datasets/
|   |   |-- homecredit.yaml
|   |   `-- lendingclub.yaml
|   |-- selectors/
|   |   |-- llm.yaml
|   |   |-- mrmr.yaml
|   |   |-- boruta.yaml
|   |   |-- pca.yaml
|   |   `-- hybrids.yaml
|   |-- models/
|   |   |-- lr.yaml
|   |   `-- catboost.yaml
|   `-- experiments/
|       |-- homecredit_matrix.yaml
|       `-- lendingclub_matrix.yaml
|-- data/
|   |-- homecredit/
|   |   |-- raw/
|   |   |-- interim/
|   |   |-- processed/
|   |   `-- metadata/
|   |       |-- columns_description.csv
|   |       |-- raw_schema_snapshot.json
|   |       `-- leakage_columns.yaml
|   `-- lendingclub/
|       |-- raw/
|       |-- interim/
|       |-- processed/
|       `-- metadata/
|           |-- columns_description.csv
|           |-- raw_schema_snapshot.json
|           `-- leakage_columns.yaml
|-- Notebooks/
|   |-- homecredit/
|   `-- lendingclub/
|-- reports/
|   |-- cross_dataset_v2_analysis.md
|   |-- homecredit_report.md
|   `-- lendingclub_report.md
|-- scripts/
|   |-- prepare_homecredit.py
|   |-- prepare_lendingclub.py
|   |-- run_matrix.py
|   |-- run_single.py
|   |-- aggregate_results.py
|   |-- make_plots.py
|   `-- check_setup.py
|-- results/
|   |-- homecredit/
|   |   |-- matrix_runs.csv
|   |   |-- final_comparison_table.csv
|   |   |-- paired_fold_comparisons.csv
|   |   |-- llm_call_summary.csv
|   |   |-- feature_stability_table.csv
|   |   |-- feature_drift_table.csv
|   |   |-- semantic_coverage_table.csv
|   |   |-- plot_reports/
|   |   `-- runs/
|   |-- lendingclub/
|   |   |-- matrix_runs.csv
|   |   |-- final_comparison_table.csv
|   |   |-- paired_fold_comparisons.csv
|   |   |-- llm_call_summary.csv
|   |   |-- feature_stability_table.csv
|   |   |-- feature_drift_table.csv
|   |   |-- semantic_coverage_table.csv
|   |   |-- plot_reports/
|   |   `-- runs/
|   `-- cross_dataset/
|       |-- final_comparison_table.csv
|       |-- paired_fold_comparisons.csv
|       |-- semantic_coverage_table.csv
|       |-- method_rank_summary.csv
|       `-- plot_reports/
|-- artifacts/
|   |-- llm_cache/
|   `-- prompts/
|       `-- llm_screening/
|           |-- stability_expert_v1.txt
|           |-- stability_expert_v2.txt
|           `-- stability_expert_v3.txt
|-- reports/
|   |-- homecredit_report.md
|   |-- lendingclub_report.md
|   `-- cross_dataset_summary.md
|-- tests/
|   |-- fixtures/
|   |   |-- homecredit/
|   |   `-- lendingclub/
|   |-- data/
|   |-- preprocessing/
|   |-- feature_engineering/
|   |-- feature_metadata/
|   |-- selectors/
|   |-- evaluation/
|   |-- experiments/
|   `-- regression/
|-- docs/
|   |-- project_structure.md
|   |-- reproducibility.md
|   |-- leakage_policy.md
|   |-- experiment_protocol.md
|   |-- llm_prompt_protocol.md
|   `-- dataset_notes/
|       |-- homecredit.md
|       `-- lendingclub.md
|-- .env.example
|-- .gitignore
|-- pyproject.toml
|-- README.md
`-- run_experiment.py
```

## Compatibility And Historical Directories

The repo still contains some legacy top-level directories because the refactor was done conservatively and backward compatibility was preserved.

Examples:
- `Preprocessing/`
- `feature_selection/`
- `Models/`
- `evaluation/`
- `experiments/`
- `pipelines/`
- `training/`
- `utils/`
- `data/inputs/`
- `results_full_run/`

These are not the canonical implementation paths anymore. The real code should be treated as living under `src/credit_risk_fs/`. Legacy module trees are compatibility shims unless a specific historical artifact is being inspected.

## Dataset Layout

Home Credit:
- raw multi-table data belongs in `data/homecredit/raw/`
- metadata files belong in `data/homecredit/metadata/`
- processed or intermediate outputs belong in `data/homecredit/interim/` and `data/homecredit/processed/`

LendingClub:
- raw Kaggle files belong in `data/lendingclub/raw/`
- the preparation step produces `data/lendingclub/processed/application_train.csv`
- metadata files belong in `data/lendingclub/metadata/`

LendingClub preparation is single-table and includes:
- target construction into `TARGET`
- temporal proxy construction into `recent_decision`
- removal of major post-outcome leakage columns
- removal of major policy/leakage fields used after origination

## Setup

Use the project environment:

```powershell
uv sync
```

If you already have the checked-in virtual environment, activate it first:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then validate the repository:

```powershell
python scripts/check_setup.py --dataset homecredit
python scripts/check_setup.py --dataset lendingclub
```

## Data Preparation

Prepare Home Credit:

```powershell
python scripts/prepare_homecredit.py
```

Prepare LendingClub:

```powershell
python scripts/prepare_lendingclub.py
```

Expected raw LendingClub file:
- `data/lendingclub/raw/accepted_2007_to_2018Q4.csv`

Expected processed LendingClub file after preparation:
- `data/lendingclub/processed/application_train.csv`

LendingClub preparation also writes audit artifacts under `data/lendingclub/metadata/`:
- `target_definition.md`
- `leakage_columns.yaml`
- `label_distribution.csv`
- `issue_date_target_distribution.csv`

## Main Commands

Single experiment:

```powershell
python scripts/run_single.py --dataset homecredit --model lr --selector mrmr
python scripts/run_single.py --dataset homecredit --model catboost --selector llm_then_mrmr
python scripts/run_single.py --dataset lendingclub --model lr --selector mrmr
python scripts/run_single.py --dataset lendingclub --model lr --selector llm_then_mrmr
```

Matrix dry-run:

```powershell
python scripts/run_matrix.py --dataset homecredit --dry-run
python scripts/run_matrix.py --dataset lendingclub --dry-run
```

Full matrix:

```powershell
python scripts/run_matrix.py --dataset homecredit
python scripts/run_matrix.py --dataset lendingclub
```

Convenience wrapper:

```powershell
python run_experiment.py --dataset homecredit
python run_experiment.py --dataset lendingclub
python run_experiment.py --dataset all
```

Aggregation:

```powershell
python scripts/aggregate_results.py --dataset homecredit
python scripts/aggregate_results.py --dataset lendingclub
```

Plots:

```powershell
python scripts/make_plots.py --dataset homecredit
python scripts/make_plots.py --dataset lendingclub
```

Run both datasets in sequence:

```powershell
foreach ($ds in @('homecredit','lendingclub')) {
  python scripts/run_matrix.py --dataset $ds
  python scripts/aggregate_results.py --dataset $ds
  python scripts/make_plots.py --dataset $ds
}
```

## Expected Results Layout

Each dataset result root is expected to keep:
- `matrix_runs.csv`
- `final_comparison_table.csv`
- `paired_fold_comparisons.csv`
- `llm_call_summary.csv`
- `feature_stability_table.csv`
- `feature_drift_table.csv`
- `semantic_coverage_table.csv`
- `plot_reports/`
- `runs/`

Each run folder under `results/<dataset>/runs/<run_id>/` is expected to keep:
- `features/`
- `models/`
- `results/`
- `llm_responses/`
- `feature_rankings/`
- `selected_feature_sets/`

Cross-dataset outputs belong under `results/cross_dataset/`.

## Research Metrics Preserved

The framework preserves or supports:
- OOT AUC
- OOT Gini
- OOT KS
- Lift@10
- selected-feature PSI
- model-score PSI
- Nogueira stability
- pairwise Jaccard stability
- semantic coverage by group
- redundancy analysis
- runtime summaries
- LLM call, token, and cache summaries

## LLM Reproducibility

Reusable LLM artifacts:
- `artifacts/llm_cache/`
- `artifacts/prompts/llm_screening/`

Per-run LLM-related artifacts include:
- prompt text or prompt-linked output
- prompt hash
- metadata signature or metadata hash
- raw LLM response payload when available
- parsed ranked or selected features
- final selected feature sets

This is designed so the LLM stage is inspectable and repeatable across folds, methods, and datasets.

## Methodological Notes

Important interpretation rules:
- the LLM is a first-stage screening helper
- LR and CatBoost are evaluation vehicles only
- Home Credit and LendingClub should follow the same core experiment protocol
- leakage policy is mandatory, especially for LendingClub
- behavior should not be changed silently across refactors

For the paper, the safest framing is:
- LLM metadata screening is compared with statistical selectors and hybrid selectors
- Home Credit is the main benchmark
- LendingClub is external validation
- the claim is about feature-selection utility, not deployment readiness

## Documentation

Project documentation is organized under `docs/`:
- `docs/project_structure.md`
- `docs/reproducibility.md`
- `docs/leakage_policy.md`
- `docs/experiment_protocol.md`
- `docs/llm_prompt_protocol.md`
- `docs/dataset_notes/homecredit.md`
- `docs/dataset_notes/lendingclub.md`

Generated research summaries belong under `reports/`.

## Current Defaults

Home Credit remains the default dataset in compatibility entrypoints.

The refactor was intentionally conservative:
- core experiment behavior was preserved where possible
- legacy import paths were kept as shims
- the canonical implementation is the `credit_risk_fs` package under `src/`
