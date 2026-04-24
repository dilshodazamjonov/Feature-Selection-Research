# Research Pipeline

This repository runs temporal credit-risk experiments on the Home Credit dataset.
The codebase is organized around a shared training core plus canonical CLI
entrypoints under `scripts/`.

The current experiment design follows this research structure:

1. Statistical baselines
2. LLM vs statistical selectors
3. Hybrid `LLM -> statistical`

## Architecture

The repo is now split by responsibility:

- `scripts/`: canonical CLI wrappers
- `experiments/`: experiment orchestration and CLI-facing runners
- `pipelines/`: shared data preparation and comparison utilities
- `Preprocessing/`: raw data loading, feature engineering, preprocessing
- `feature_selection/`: selectors and selector-specific helpers
- `training/`: fold execution and temporal CV training
- `evaluation/`: metrics and summary calculations
- `Models/`: model factory and model-specific training helpers
- `utils/`: logging and feature metadata helpers

The important design rule is that experiment scripts should stay thin.
They parse arguments and hand off to `experiments/`, while the shared
training logic stays in `pipelines/` and `training/`.

The full directory guide is in [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

## Config

The project now supports a root-level [config.yaml](</d:/python projects/Research/config.yaml>) so you can
change the default model and experiment settings in one place.

The main field for model switching is:

```yaml
model_selector: lr
```

Change it to:

```yaml
model_selector: catboost
```

or:

```yaml
model_selector: rf
```

You can also edit model-specific parameters under:

```yaml
model_params:
  lr: ...
  rf: ...
  catboost: ...
```

All scripts read `config.yaml` by default. You can also pass a custom file:

```bash
python scripts/run_statistical_comparison.py --config my_config.yaml
```

## Data Split

Experiments use the engineered descendant of
`previous_application.DAYS_DECISION` as the temporal split column.

Default windows:

- `DEV`: `-600 <= recent_decision < -240`
- `OOT`: `-240 <= recent_decision <= 0`
- rows older than `-600` are dropped

The split column is resolved from the engineered feature set using the first
available match from:

- `recent_decision`
- `PREV_recent_decision_MAX`
- `DAYS_DECISION`

## LLM Selector Rules

The LLM selector is treated as a domain-level selector, not a row-level model.

Inside each fold, on the training slice only, the LLM pipeline does:

1. `95%` missing-rate filter
2. IV filter
3. metadata build
4. LLM feature selection
5. model fit

The LLM only receives summarized metadata, not raw training rows.
Metadata includes values such as:

- `dtype`
- `non_null_count`
- `missing_rate`
- `mean`
- `min`
- `max`
- `std`
- `var`
- `unique_count` for non-numeric fields

## Experiment Entry Points

### Part 1: Statistical Baselines

Compares statistical selectors such as `mrmr`, `boruta`, and `pca`.

```bash
python scripts/run_statistical_comparison.py
```

To switch model just for one run:

```bash
python scripts/run_statistical_comparison.py --model-selector catboost
```

Useful options:

```bash
python scripts/run_statistical_comparison.py ^
  --selectors mrmr boruta pca ^
  --model-selector lr ^
  --n-splits 5 ^
  --dev-start-day -600 ^
  --oot-start-day -240 ^
  --oot-end-day 0
```

### Part 2: LLM vs Statistical

Runs the LLM selector and compares it against one or more statistical baselines.
Outputs include performance summaries and feature-overlap tables.

```bash
python scripts/run_llm_vs_statistical.py
```

To switch model just for one run:

```bash
python scripts/run_llm_vs_statistical.py --model-selector catboost
```

Useful options:

```bash
python scripts/run_llm_vs_statistical.py ^
  --stat-selectors mrmr boruta ^
  --model-selector lr ^
  --llm-model gpt-4.1-mini ^
  --llm-max-features 50 ^
  --n-splits 5
```

### Part 3: Hybrid

Runs a hybrid selector where the LLM narrows the raw engineered features and a
statistical selector performs the downstream refinement.

```bash
python scripts/run_hybrid_comparison.py
```

To switch model just for one run:

```bash
python scripts/run_hybrid_comparison.py --model-selector catboost
```

Useful options:

```bash
python scripts/run_hybrid_comparison.py ^
  --stat-selector mrmr ^
  --model-selector lr ^
  --llm-model gpt-4.1-mini ^
  --llm-max-features 50 ^
  --n-splits 5
```

## Single Experiment Entry Point

Use the single-experiment wrapper when you want to run one selector/model
configuration directly.

Example:

```bash
python scripts/run_single_experiment.py --selector llm --model-selector lr
```

## Common CLI Options

All experiment scripts support the shared temporal and data arguments:

```bash
--config config.yaml
--data-dir data/inputs
--description-path data/HomeCredit_columns_description.csv
--model-selector lr
--n-splits 5
--dev-start-day -600
--oot-start-day -240
--oot-end-day 0
--cv-gap-groups 1
```

LLM-based scripts also support:

```bash
--llm-model gpt-4.1-mini
--llm-max-features 50
--llm-cache-dir outputs/llm_selector_cache
```

You can inspect any CLI surface with:

```bash
python scripts/run_statistical_comparison.py --help
python scripts/run_llm_vs_statistical.py --help
python scripts/run_hybrid_comparison.py --help
python scripts/run_single_experiment.py --help
```

## Output Layout

Each run creates its own timestamped folder:

- `outputs/statistical_comparison/run_statistical_comparison_.../`
- `outputs/llm_vs_statistical/run_llm_vs_statistical_.../`
- `outputs/hybrid_comparison/run_hybrid_comparison_.../`
- `outputs/single_experiment/run_single_experiment_.../`

Typical contents:

- `run_manifest.json`
- `experiments/`
- `feature_overlap/` for comparison runs
- summary `.csv` files at the run root

Each experiment directory contains the underlying fold artifacts, selected
features, models, and result files produced by the shared training pipeline.

Each experiment now also writes a dedicated fold-stability summary:

- `results/stability_confidence_summary.csv`

That file contains fold-based stability statistics only for:

- `gini`
- `ks`
- `psi_feature_mean`
- `psi_feature_max`
- `psi_model`
- `jaccard_similarity`

For each metric it includes just:

- `value`
- `ci95_lower`
- `ci95_upper`

## Separate Comparison Plot Tool

Use [plots.py](</d:/python projects/Research/plots.py>) to
build comparison plots from finished experiment folders. This is separate from the
training pipeline and is meant for reporting and stability analysis.

It can generate plots such as:

- `gini_over_time.png`
- `auc_over_time.png`
- `ks_over_time.png`
- `psi_feature_mean_over_time.png`
- `psi_feature_max_over_time.png`
- `psi_model_over_time.png`
- `jaccard_similarity_over_time.png`
- `selected_features_over_time.png`
- `oot_gini_comparison.png`
- `oot_auc_comparison.png`
- `oot_ks_comparison.png`

You can compare specific experiment folders directly:

```bash
python plots.py ^
  --experiment "lr_mrmr=outputs/statistical_comparison/run_a/experiments/lr_mrmr_..." ^
  --experiment "catboost_mrmr=outputs/statistical_comparison/run_b/experiments/catboost_mrmr_..." ^
  --output-dir outputs/plot_reports/lr_vs_catboost_mrmr
```

Or point it at whole run folders:

```bash
python plots.py ^
  --run-dir outputs/statistical_comparison/run_statistical_comparison_2026-04-24_14-52-16 ^
  --output-dir outputs/plot_reports/statistical_run_plots
```

The script also writes:

- `monthly_metric_table.csv`
- `oot_metric_table.csv`

## Overnight Run

Use [main.py](</d:/python projects/Research/main.py>) when you want the whole research sequence to run overnight.

```bash
python main.py
```

That runs:

1. Statistical baselines
2. LLM vs statistical
3. Hybrid comparison

You can override the model for the whole overnight run:

```bash
python main.py --model-selector catboost
```

## Environment

Recommended setup:

```bash
uv sync
```

If you added the latest changes to an existing environment, run `uv sync` again so
plot dependencies such as `matplotlib` are installed.

or, if you are using the existing virtual environment:

```bash
.venv\\Scripts\\activate
```

The LLM experiments require an OpenAI API key in `.env`:

```bash
OPENAI_API_KEY=your_key_here
```

## Reliability Guardrails

- Day-based sentinel values such as `365243` are converted to missing values at load time.
- Temporal CV uses grouped time splits so the same time group cannot land in both train and validation.
- Threshold-based metrics use thresholds learned on training probabilities before being applied to validation or OOT data.
- LLM metadata is built from fold-local training data only.
- LLM cache entries are isolated per run folder.
- Numeric preprocessing now forces finite outputs so downstream selectors such as Boruta receive model-ready data.

## Notes

- `boruta` can be slow on the full engineered dataset.
- `pca` is useful for performance comparison, but it is not directly comparable to subset selectors for raw-feature overlap.
- Full end-to-end experiment runs can take a while on the full Home Credit feature set.

## Tests And Checks

Helpful checks:

```bash
python -m py_compile main.py plots.py scripts/run_all_experiments.py scripts/run_statistical_comparison.py scripts/run_llm_vs_statistical.py scripts/run_hybrid_comparison.py scripts/run_single_experiment.py
python main.py --help
python plots.py --help
python scripts/run_all_experiments.py --help
python scripts/run_statistical_comparison.py --help
python scripts/run_llm_vs_statistical.py --help
python scripts/run_hybrid_comparison.py --help
python scripts/run_single_experiment.py --help
```

If `pytest` is available cleanly in your environment:

```bash
pytest
```
