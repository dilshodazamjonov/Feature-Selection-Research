# Reproducibility

This project is set up so the full experiment matrix can be rerun from one
config file and audited from run manifests.

## 1. Install

```bash
uv sync
```

Or activate the existing virtual environment before running plain `python`
commands:

```bash
.venv\Scripts\activate
```

## 2. Data Placement

Place the Home Credit CSVs under:

```text
data/inputs/
```

At minimum, the setup validator expects:

```text
data/inputs/application_train.csv
data/inputs/previous_application.csv
data/inputs/bureau.csv
data/HomeCredit_columns_description.csv
```

## 3. Config Setup

Edit:

```text
config.yaml
```

Important reproducibility fields:

```yaml
results_dir: results
random_seed: 42
feature_budgets:
  lr: 20
  catboost: 40
data_dir: data/inputs
description_path: data/HomeCredit_columns_description.csv
llm:
  shared_ranking_enabled: true
  ranking_budget: 40
```

LLM runs require:

```bash
OPENAI_API_KEY=your_key_here
```

## 4. Validate Setup

```bash
python scripts/check_research_setup.py
```

With uv:

```bash
uv run python scripts/check_research_setup.py
```

## 5. Run Full Matrix

Inspect the planned matrix first:

```bash
uv run python main.py --dry-run
```

Then run:

```bash
uv run python main.py
```

The matrix is explicitly defined in `experiments/matrix.py`:

```python
MODELS = ["lr", "catboost"]
STAT_SELECTORS = ["mrmr", "boruta", "pca"]
HYBRID_SELECTORS = ["mrmr", "boruta"]
```

Completed runs are skipped on rerun. To intentionally rerun completed entries:

```bash
python main.py --force
```

## 6. Generate Plots

```bash
uv run python plots.py --all results/
```

This writes plot reports under:

```text
results/plot_reports/all/
```

## 7. Reproduce Final Tables

```bash
uv run python scripts/aggregate_results.py results/
```

Outputs:

```text
results/final_comparison_table.csv
results/paired_fold_comparisons.csv
results/llm_call_summary.csv
results/failed_runs.csv
```

## Audit Artifacts

Matrix-level files:

```text
results/matrix_runs.csv
results/final_comparison_table.csv
results/paired_fold_comparisons.csv
results/llm_call_summary.csv
results/failed_runs.csv
results/_llm_rankings_cache/
```

Every matrix run writes:

```text
results/<model>/<bucket>/<run_id>/_SUCCESS
results/<model>/<bucket>/<run_id>/run_manifest.json
results/<model>/<bucket>/<run_id>/run.log
results/<model>/<bucket>/<run_id>/leakage_report.json
results/<model>/<bucket>/<run_id>/data_split_manifest.json
results/<model>/<bucket>/<run_id>/features/final_selected_features.csv
results/<model>/<bucket>/<run_id>/features/fold_selected_features.csv
results/<model>/<bucket>/<run_id>/features/selection_frequency.csv
results/<model>/<bucket>/<run_id>/features/feature_stability_metrics.csv
results/<model>/<bucket>/<run_id>/models/final_model.model
results/<model>/<bucket>/<run_id>/models/final_preprocessor.pkl
results/<model>/<bucket>/<run_id>/models/final_model_metadata.json
results/<model>/<bucket>/<run_id>/results/experiment_summary.csv
results/<model>/<bucket>/<run_id>/results/cv_results.csv
results/<model>/<bucket>/<run_id>/results/oot_test_results.csv
results/<model>/<bucket>/<run_id>/results/oot_predictions.csv
results/<model>/<bucket>/<run_id>/results/selected_feature_psi.csv
results/<model>/<bucket>/<run_id>/results/model_score_psi.csv
results/<model>/<bucket>/<run_id>/results/credit_risk_utility.csv
results/<model>/<bucket>/<run_id>/results/runtime_summary.csv
```

LLM and hybrid runs additionally write:

```text
results/<model>/<bucket>/<run_id>/features/llm_rankings_summary.csv
```

Hybrid runs additionally write:

```text
results/<model>/<bucket>/<run_id>/features/llm_hybrid_trace.csv
```

`fold_selected_features.csv` includes `fold_id`, `selector`, `feature_name`,
`rank`, and `score` when a selector exposes a score.

## Shared LLM Ranking

The LLM now produces one shared fold-local ranked list of up to 40 raw features.

For each CV fold:

- feature metadata is built from the training fold only
- the LLM ranks up to 40 features once
- the ranking is cached using scope, fold id, metadata signature, and shared LLM config hash
- LLM-only and hybrid runs reuse the cached ranking

For final OOT evaluation:

- feature metadata is built from full DEV only
- the final DEV ranking is used only for final model fitting and OOT evaluation
- OOT is never used in metadata, prompts, preprocessing fit, feature selection, or training

Model-specific budgets:

- Logistic Regression uses top 20 LLM-ranked features
- CatBoost uses top 40 LLM-ranked features

The same budgets are applied to statistical and hybrid selectors where possible:

- LR mRMR/Boruta/PCA/LLM/hybrid: budget 20
- CatBoost mRMR/Boruta/PCA/LLM/hybrid: budget 40

Runtime-oriented defaults:

- Boruta uses `max_iter=15`
- CatBoost RFE refinement is disabled by default
- CatBoost model training uses `iterations=1500`

This reduces worst-case LLM calls from separate per-model/per-selector calls to
one top-40 ranking per fold scope, plus one final DEV ranking, with cache reuse.

## Metric Definitions

- **Nogueira Stability Index**: main fold-level stability score over selected feature indicators. Higher means selected sets are more consistent across folds.
- **Kuncheva Stability Index**: pairwise stability score best suited to fixed-size selected sets such as LR top-20 and CatBoost top-40.
- **Mean Pairwise Jaccard Similarity**: average overlap divided by union across all pairs of CV fold selected-feature sets.
- **Selection Frequency**: number of folds in which a feature was selected divided by total CV folds.
- **Rank Stability**: pairwise Spearman and Kendall rank correlation over selector rankings across CV folds.
- **Feature PSI**: DEV-vs-OOT population stability index for each final selected feature.
- **Model Score PSI**: PSI between DEV predicted scores and OOT predicted scores.
- **Feature Reduction Ratio**: `1 - selected_feature_count / total_candidate_feature_count`.
- **Lift@10**: bad rate in the top 10% highest-risk OOT predictions divided by the overall OOT bad rate.
- **Bad-rate Capture@10**: share of all OOT bad cases captured in the top 10% highest-risk predictions.

## Exact Run Order

```bash
uv run python scripts/check_research_setup.py
uv run python main.py --dry-run
uv run python main.py
uv run python scripts/aggregate_results.py results/
uv run python plots.py --all results/
```
