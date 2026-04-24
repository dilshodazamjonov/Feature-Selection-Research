# Project Structure

This repository is organized around a shared experiment core with thin CLI entrypoints.

## Top-Level Layout

```text
Research/
├── config.yaml              # project-wide defaults for models and runs
├── data/                    # raw input files and dataset descriptions
├── docs/                    # project documentation
├── experiments/             # orchestration for each research question
├── feature_selection/       # statistical, LLM, and hybrid selectors
├── main.py                  # run all three experiment parts in sequence
├── Models/                  # model wrappers and model bundle factory
├── pipelines/               # shared data prep and experiment comparison logic
├── plots.py                 # standalone comparison plotting entrypoint
├── Preprocessing/           # loading, feature engineering, preprocessing
├── scripts/                 # canonical CLI wrappers
├── tests/                   # automated tests
├── training/                # fold processing and temporal CV training
├── utils/                   # metadata and logging helpers
└── pyproject.toml
```

## Directory Roles

### `experiments/`

Owns experiment orchestration only:

- `statistical_baselines.py`
- `llm_vs_statistical.py`
- `hybrid_comparison.py`
- `single_experiment.py`
- `run_all.py`
- `common.py`
- `config.py`

These modules compose configs, prepare shared data, launch runs, and write summaries.

### `pipelines/`

Owns shared experiment plumbing:

- `common.py`: data loading, feature-table merge, temporal split, final fit
- `comparison.py`: overlap, summary, and comparison utilities

### `Preprocessing/`

Owns raw tabular preparation:

- dataset loading
- feature engineering
- preprocessing

### `feature_selection/`

Owns selection logic:

- `mrmr.py`
- `boruta_rfe.py`
- `pca.py`
- `llm_selector.py`
- `hybrid.py`
- `missing_filter.py`

### `training/`

Owns fold execution and temporal CV logic:

- `fold.py`
- `kfold_trainer.py`
- `cv_utils.py`

### `scripts/`

Holds the CLI wrappers:

- `run_statistical_comparison.py`
- `run_llm_vs_statistical.py`
- `run_hybrid_comparison.py`
- `run_single_experiment.py`
- `run_all_experiments.py`

## Recommended Mental Model

- `Preprocessing/`: build the dataset
- `feature_selection/`: choose features
- `training/`: train and validate
- `pipelines/`: shared end-to-end experiment logic
- `experiments/`: research-question-specific orchestration
- `scripts/`: user-facing commands
- `plots.py`: standalone comparison reporting
- `main.py`: overnight all-in-one launcher

## Why Some Legacy Names Remain

The repo still uses legacy folder names like `Models/` and `Preprocessing/`.
They are not ideal stylistically, but they were kept to avoid a risky full-package rename while experiments are in active use.

If you want a deeper cleanup later, the next safe step would be renaming:

- `Models/` -> `models/`
- `Preprocessing/` -> `preprocessing/`

That would require a broad import update across the repo, so it is best done as a dedicated refactor.
