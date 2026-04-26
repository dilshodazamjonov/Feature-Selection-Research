# Project Structure

This repository is organized around a reproducible credit-risk experiment matrix.
The user-facing flow is:

```bash
python scripts/check_research_setup.py
python main.py
python scripts/aggregate_results.py results/
python plots.py --all results/
```

`main.py` is now the full matrix runner, not just a sequential wrapper over a
single configured model/selector choice.

## Top-Level Layout

```text
Research/
|-- config.yaml                  # project-wide defaults, seeds, paths, model params
|-- data/                        # raw input files and dataset descriptions
|-- docs/                        # project documentation and reproducibility notes
|-- evaluation/                  # metrics, plotting helpers, stability calculations
|-- experiments/                 # experiment matrix, config, tracking, orchestration
|-- feature_selection/           # statistical, LLM, and hybrid selectors
|-- main.py                      # full experiment matrix entrypoint
|-- Models/                      # model wrappers and model bundle factory
|-- pipelines/                   # shared data prep, leakage checks, final fit logic
|-- plots.py                     # standalone plot generation wrapper
|-- Preprocessing/               # loading, feature engineering, preprocessing
|-- results/                     # generated run outputs, ignored by git
|-- scripts/                     # validation, aggregation, and compatibility CLIs
|-- tests/                       # automated regression and research guardrail tests
|-- training/                    # fold processing and temporal CV training
|-- utils/                       # metadata and logging helpers
|-- pyproject.toml
`-- uv.lock
```

## Core Directories

### `experiments/`

Owns orchestration and run metadata.

Important files:

- `matrix.py`: explicitly defines the full research matrix.
- `run_all.py`: runs the matrix with stable output folders and resume behavior.
- `tracking.py`: run ids, manifests, config hashes, git commit capture, data fingerprinting.
- `config.py`: config loading, default config, config hashing, seed propagation.
- `common.py`: shared helpers used by the older script-specific experiment entrypoints.
- `statistical_baselines.py`, `llm_vs_statistical.py`, `hybrid_comparison.py`: compatibility runners for individual research slices.

The matrix constants are intentionally hard-coded in `experiments/matrix.py`:

```python
MODELS = ["lr", "catboost"]
STAT_SELECTORS = ["mrmr", "boruta", "pca"]
HYBRID_SELECTORS = ["mrmr", "boruta"]
```

This prevents a hidden config mistake from silently omitting a major run.

### `pipelines/`

Owns shared end-to-end experiment logic.

Important files:

- `common.py`: data loading, feature engineering merge, temporal split, leakage checks, CV launch, final DEV fit, OOT evaluation.
- `comparison.py`: summary rows, feature overlap, comparison utilities.

`pipelines/common.py` also writes:

- `leakage_report.json`
- `data_split_manifest.json`
- `features/final_selected_features.csv`
- `results/oot_predictions.csv`
- `results/oot_test_results.csv`
- final model artifacts under `models/`

### `training/`

Owns fold-level execution and temporal CV.

Important files:

- `cv_utils.py`: grouped time-series split.
- `kfold_trainer.py`: fold orchestration, OOF metrics, stability summaries.
- `fold.py`: per-fold preprocessing, feature selection, model training, fold artifacts.

Fold selected features are saved as one combined file:

```text
features/fold_selected_features.csv
```

Each file includes:

- `fold_id`
- `selector`
- `feature_name`
- `feature`
- `rank`
- `score` when available

### `feature_selection/`

Owns selector implementations.

Important files:

- `mrmr.py`: mRMR/RF-importance selector.
- `boruta_rfe.py`: Boruta with optional RFE; RFE is disabled by default for the full matrix runtime.
- `pca.py`: PCA selector with fixed random state.
- `llm_selector.py`: metadata-only, fold-local LLM feature selector.
- `hybrid.py`: LLM preselection followed by statistical selector.
- `missing_filter.py`: missingness guardrail used by LLM selection.

The LLM selector receives training-fold metadata only. It does not receive OOT
data or raw row-level training records.

### `Models/`

Owns model wrappers and the factory used by training.

Important files:

- `utils.py`: model and selector factories.
- `logistic_regression_model.py`
- `catboost_model.py`
- `random_forest_model.py`

The active matrix uses:

- `lr`
- `catboost`

Random seeds are propagated into model kwargs from `config.yaml`.

### `Preprocessing/`

Owns raw tabular preparation.

Important files:

- `data_process.py`: CSV loading and Home Credit sentinel cleanup.
- `feature_engineering.py`: engineered aggregate feature tables and temporal proxy.
- `preprocessing.py`: train-only fitted numeric and categorical preprocessing.

### `evaluation/`

Owns evaluation helpers and reporting internals.

Important files:

- `metrics.py`: AUC, Gini, KS, thresholds, confusion metrics.
- `stability_scores.py`: PSI and feature stability helpers.
- `feature_utils.py`: feature artifacts, importance extraction, selected feature CSVs.
- `plotting.py`: plot generation from completed runs.

### `scripts/`

Owns user-facing utility commands.

Primary commands:

- `check_research_setup.py`: validates config, data, selectors, models, paths, and LLM key.
- `aggregate_results.py`: creates `final_comparison_table.csv` and `paired_fold_comparisons.csv`.
- `run_all_experiments.py`: compatibility wrapper for `experiments.run_all`.

Older slice-specific wrappers are still available:

- `run_statistical_comparison.py`
- `run_llm_vs_statistical.py`
- `run_hybrid_comparison.py`
- `run_single_experiment.py`

### `tests/`

Owns automated checks.

Current coverage includes:

- temporal split behavior
- no OOT/leakage columns reaching model features
- selector validity
- config parsing
- tiny end-to-end pipeline run
- LLM selector cache isolation
- hybrid selector behavior
- comparison summary behavior

## Output Layout

The full matrix writes stable, resumable outputs under `results/`:

```text
results/
|-- matrix_runs.csv
|-- final_comparison_table.csv
|-- paired_fold_comparisons.csv
|-- llm_call_summary.csv
|-- failed_runs.csv
|-- _llm_rankings_cache/
|-- lr/
|   |-- statistical/
|   |-- llm/
|   |-- hybrid_mrmr/
|   `-- hybrid_boruta/
`-- catboost/
    |-- statistical/
    |-- llm/
    |-- hybrid_mrmr/
    `-- hybrid_boruta/
```

Each atomic run folder contains:

```text
<run_id>/
|-- _SUCCESS
|-- run_manifest.json
|-- run.log
|-- leakage_report.json
|-- data_split_manifest.json
|-- features/
|   |-- final_selected_features.csv
|   |-- fold_selected_features.csv
|   |-- selection_frequency.csv
|   |-- feature_stability_metrics.csv
|   |-- llm_rankings_summary.csv      # LLM/hybrid only
|   `-- llm_hybrid_trace.csv          # hybrid only
|-- models/
|   |-- final_model.model
|   |-- final_preprocessor.pkl
|   `-- final_model_metadata.json
`-- results/
    |-- experiment_summary.csv
    |-- cv_results.csv
    |-- oot_test_results.csv
    |-- oot_predictions.csv
    |-- selected_feature_psi.csv
    |-- model_score_psi.csv
    |-- credit_risk_utility.csv
    `-- runtime_summary.csv
```

Resume behavior is marker-based. If `_SUCCESS` and required artifacts exist,
`main.py` skips that run unless `--force` is passed.

## Recommended Mental Model

- `config.yaml`: declare paths, seeds, and model parameters.
- `experiments/matrix.py`: declare what must be run.
- `main.py`: execute the full matrix.
- `pipelines/`: enforce temporal split, leakage checks, and final evaluation.
- `training/`: run fold-local preprocessing, selection, and training.
- `results/`: store the auditable outputs.
- `scripts/aggregate_results.py`: turn completed runs into final comparison tables.
- `plots.py --all results/`: generate post-training plots.

## Legacy Naming Note

The repo still uses legacy folder names `Models/` and `Preprocessing/`.
They are intentionally left as-is for now to avoid a broad import refactor right
before full experiment execution.

If a cleanup is needed later, do it as a dedicated refactor:

- `Models/` -> `models/`
- `Preprocessing/` -> `preprocessing/`
