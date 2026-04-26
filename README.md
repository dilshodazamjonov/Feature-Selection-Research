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
For exact reproducibility commands, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Reproducibility Quickstart

```bash
uv sync
uv run python scripts/check_research_setup.py
uv run python main.py
uv run python scripts/aggregate_results.py results/
uv run python plots.py --all results/
```

If you prefer an activated environment, use the same commands without `uv run`.

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
  --llm-max-features 40 ^
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
  --llm-max-features 40 ^
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
--llm-max-features 40
--llm-ranking-budget 40
--llm-cache-dir outputs/llm_selector_cache
```

The full matrix uses a shared fold-local LLM top-40 ranking cache. LR consumes
the top 20 ranked features, while CatBoost consumes the top 40. CV fold rankings
are created from each fold's training subset only; the final DEV ranking is used
only for final OOT evaluation.

You can inspect any CLI surface with:

```bash
python scripts/run_statistical_comparison.py --help
python scripts/run_llm_vs_statistical.py --help
python scripts/run_hybrid_comparison.py --help
python scripts/run_single_experiment.py --help
```

## Output Layout

`main.py` uses stable, resumable folders under `results/`:

- `results/lr/statistical/<run_id>/`
- `results/lr/llm/<run_id>/`
- `results/lr/hybrid_mrmr/<run_id>/`
- `results/lr/hybrid_boruta/<run_id>/`
- `results/catboost/statistical/<run_id>/`
- `results/catboost/llm/<run_id>/`
- `results/catboost/hybrid_mrmr/<run_id>/`
- `results/catboost/hybrid_boruta/<run_id>/`

Typical contents:

- `run_manifest.json`
- `run.log`
- `leakage_report.json`
- `data_split_manifest.json`
- `features/`
- `models/`
- `results/`

Each experiment directory now keeps only the final research artifacts by
default:

- `features/final_selected_features.csv`
- `features/fold_selected_features.csv`
- `features/selection_frequency.csv`
- `features/feature_stability_metrics.csv`
- `features/llm_rankings_summary.csv` for LLM/hybrid runs
- `features/llm_hybrid_trace.csv` for hybrid runs
- `models/final_model.model`
- `models/final_preprocessor.pkl`
- `models/final_model_metadata.json`
- `results/experiment_summary.csv`
- `results/cv_results.csv`
- `results/oot_test_results.csv`
- `results/oot_predictions.csv`
- `results/selected_feature_psi.csv`
- `results/model_score_psi.csv`
- `results/credit_risk_utility.csv`
- `results/runtime_summary.csv`

## Separate Comparison Plot Tool

Use [plots.py](</d:/python projects/Research/plots.py>) to
build comparison plots from finished experiment folders. This is separate from the
training pipeline and is meant for reporting and stability analysis.

It can generate plots such as:

- `oot_performance_comparison.png`
- `stability_comparison.png`
- `performance_vs_stability.png`
- `feature_count_vs_gini.png`
- `selected_feature_psi_comparison.png`
- `model_score_psi_comparison.png`
- `lift_at_10_comparison.png`
- `monthly_gini_trend.png`
- `monthly_psi_trend.png`
- `monthly_lift_trend.png`

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
- `stability_metric_table.csv`

## Full Matrix Run

Use [main.py](</d:/python projects/Research/main.py>) when you want the whole research matrix to run overnight.

```bash
python main.py
```

That automatically runs:

- LR + `mrmr`, `boruta`, `pca`
- LR + LLM
- LR + `LLM -> mRMR`
- LR + `LLM -> Boruta`
- CatBoost + `mrmr`, `boruta`, `pca`
- CatBoost + LLM
- CatBoost + `LLM -> mRMR`
- CatBoost + `LLM -> Boruta`

Completed entries are reused automatically. To rerun completed entries:

```bash
python main.py --force
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
- LLM ranking cache entries are shared through `results/_llm_rankings_cache/`, while each run writes its own `features/llm_rankings_summary.csv` audit copy.
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
