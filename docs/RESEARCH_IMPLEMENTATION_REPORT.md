# Research Implementation Report

## Topic

**Evaluating Feature Selection in Credit Scoring: A Comparison of Statistical,
LLM-Based, and Hybrid Approaches.**

Working hypothesis:

**LLMs can approximate domain-driven feature selection and complement
statistical methods.**

## Executive Status

**Implementation status: ready for full matrix execution**

**Research execution status: pending full-data run**

The codebase now supports the full intended experiment matrix automatically.
The earlier limitation where the main runner could execute only one configured
hybrid setting has been removed.

The remaining work is empirical, not architectural:

- run the full matrix on the real data
- aggregate completed runs
- generate plots
- interpret whether LLM and hybrid selectors improve performance, stability, or parsimony

## Current Full Matrix

The full matrix is explicitly defined in `experiments/matrix.py`.

```python
MODELS = ["lr", "catboost"]
STAT_SELECTORS = ["mrmr", "boruta", "pca"]
HYBRID_SELECTORS = ["mrmr", "boruta"]
```

This produces the required runs:

- LR + mRMR
- LR + Boruta
- LR + PCA
- LR + LLM
- LR + LLM -> mRMR
- LR + LLM -> Boruta
- CatBoost + mRMR
- CatBoost + Boruta
- CatBoost + PCA
- CatBoost + LLM
- CatBoost + LLM -> mRMR
- CatBoost + LLM -> Boruta

This is the minimum defensible matrix for comparing selector families across a
linear model and a stronger nonlinear model.

## What Is Implemented

### 1. Full matrix runner

Implemented in:

- [main.py](../main.py)
- [experiments/run_all.py](../experiments/run_all.py)
- [experiments/matrix.py](../experiments/matrix.py)
- [experiments/tracking.py](../experiments/tracking.py)

Current behavior:

- runs the full LR/CatBoost selector matrix
- writes stable run folders under `results/`
- skips completed runs by default
- reruns completed entries only with `--force`
- supports dry-run inspection with `python main.py --dry-run`
- writes `results/matrix_runs.csv`
- writes `results/llm_call_summary.csv`
- writes `results/failed_runs.csv`

This removes hidden config omissions from the main research execution path.

### 2. Shared temporal experiment framework

Implemented in:

- [pipelines/common.py](../pipelines/common.py)
- [training/kfold_trainer.py](../training/kfold_trainer.py)
- [training/fold.py](../training/fold.py)
- [training/cv_utils.py](../training/cv_utils.py)

Current behavior:

- data loading and feature engineering are centralized
- temporal split is based on the resolved engineered time column
- default split windows are `DEV: -600 <= time < -240` and `OOT: -240 <= time <= 0`
- older rows are dropped
- grouped time-series CV is used inside DEV
- preprocessing is fitted on training folds only
- selectors are fitted on training folds only
- final model is refit on full DEV and evaluated on OOT
- OOT predictions are saved for bootstrap CI and downstream analysis

This matches the core research need for realistic temporal credit-risk
evaluation.

### 3. Statistical baselines

Implemented in:

- [feature_selection/mrmr.py](../feature_selection/mrmr.py)
- [feature_selection/boruta_rfe.py](../feature_selection/boruta_rfe.py)
- [feature_selection/pca.py](../feature_selection/pca.py)
- [experiments/statistical_baselines.py](../experiments/statistical_baselines.py)

Supported statistical selectors:

- `mrmr`
- `boruta`
- `pca`

These provide the non-LLM comparison anchors.

### 4. LLM selector

Implemented in:

- [feature_selection/llm_selector.py](../feature_selection/llm_selector.py)

Current LLM behavior:

- runs fold-locally on training data only
- produces a shared ranked list of up to 40 raw features per fold/scope
- reuses the cached top-40 ranking across LLM-only and hybrid runs
- LR consumes the top 20 ranked features
- CatBoost consumes the top 40 ranked features
- applies missing-rate filtering before LLM selection
- applies IV filtering before LLM selection
- sends summarized feature metadata only
- does not send raw row-level records
- does not use OOT data
- caches selections by training metadata signature

This supports the claim that the LLM acts as a domain-guided selector rather
than a predictive model trained on row data.

### 5. Hybrid LLM -> statistical selectors

Implemented in:

- [feature_selection/hybrid.py](../feature_selection/hybrid.py)

Current behavior:

- LLM preselects raw engineered features first
- LR hybrids use the top 20 LLM-ranked raw features as the candidate pool
- CatBoost hybrids use the top 40 LLM-ranked raw features as the candidate pool
- preprocessing happens after raw-feature LLM selection
- mRMR or Boruta then refines the LLM candidate set
- Boruta uses `max_iter=15` and RFE is disabled by default to avoid repeated CatBoost fits
- both `LLM -> mRMR` and `LLM -> Boruta` are now included in the full matrix

This directly tests whether LLM preselection helps downstream statistical
feature selection.

### 6. Reproducibility and audit tracking

Implemented in:

- [experiments/tracking.py](../experiments/tracking.py)
- [experiments/config.py](../experiments/config.py)
- [utils/logging_config.py](../utils/logging_config.py)

Each atomic run writes:

- `run_manifest.json`
- `run.log`
- `leakage_report.json`
- `data_split_manifest.json`
- `_SUCCESS` marker after completion
- `features/fold_selected_features.csv`
- final DEV-trained selected feature file
- OOT prediction file
- OOT metric file
- final model, fitted preprocessor, and model metadata files

The manifest records:

- `run_id`
- timestamp
- model
- selector
- experiment type: `statistical`, `llm`, or `hybrid`
- exact effective config
- config hash
- data path
- data fingerprint based on file names, sizes, and modification times
- random seed
- git commit hash
- output folder
- shared LLM ranking settings
- actual LLM calls made and cache hits where applicable

This is important for defending which settings produced which results.

### 7. Fixed random seeds

Implemented in:

- [config.yaml](../config.yaml)
- [experiments/config.py](../experiments/config.py)
- [pipelines/common.py](../pipelines/common.py)
- [training/kfold_trainer.py](../training/kfold_trainer.py)
- model and selector factories

Current behavior:

- `random_seed` is read from config
- Python and NumPy seeds are set at run and fold-orchestration level
- model kwargs receive `random_state`
- selector kwargs receive `random_state` where supported
- CatBoost file-writing side effects are disabled by default
- CatBoost matrix training uses `iterations=1500`
- LR selectors use a feature budget of 20
- CatBoost selectors use a feature budget of 40

This does not magically remove all possible nondeterminism from third-party
libraries, but the project now controls the major stochastic sources.

### 8. Leakage checks

Implemented in:

- [pipelines/common.py](../pipelines/common.py)
- [tests/test_research_matrix.py](../tests/test_research_matrix.py)

Current checks:

- target column is excluded from model features
- configured future/time-leaking columns are excluded
- OOT window is strictly after DEV window
- OOT data is not used in feature selection
- LLM metadata scope is recorded as training-fold-only in CV and DEV-only for final fit
- preprocessing fit scope is recorded as training-fold-only in CV and DEV-only for final fit

Each run writes `leakage_report.json`.

### 9. Setup validation command

Implemented in:

- [scripts/check_research_setup.py](../scripts/check_research_setup.py)

Command:

```bash
python scripts/check_research_setup.py
```

It checks:

- config exists
- data path exists
- required columns exist
- temporal source columns exist
- matrix constants are valid
- model selectors are valid
- feature selectors are valid
- description file exists
- output folder is writable
- `OPENAI_API_KEY` exists if LLM is enabled

### 10. Result aggregation and statistical comparison support

Implemented in:

- [scripts/aggregate_results.py](../scripts/aggregate_results.py)
- [pipelines/comparison.py](../pipelines/comparison.py)

Command:

```bash
python scripts/aggregate_results.py results/
```

Outputs:

- `results/final_comparison_table.csv`
- `results/paired_fold_comparisons.csv`
- `results/llm_call_summary.csv`
- `results/failed_runs.csv`

The final comparison table includes:

- model
- selector
- experiment type
- OOT Gini
- OOT KS
- OOT AUC
- OOT bootstrap CI columns when OOT predictions exist
- PSI
- selected feature count
- stability score
- runtime
- config hash
- output folder

The paired comparison file compares candidate runs against the mRMR baseline
within each model over CV folds.

### 11. Plot generation after training

Implemented in:

- [plots.py](../plots.py)
- [evaluation/plotting.py](../evaluation/plotting.py)

Command:

```bash
python plots.py --all results/
```

Training and plotting remain separate. This is intentional: the training jobs
produce artifacts, and reporting can be regenerated after runs finish.

The default plot report is intentionally small and paper-oriented:

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

### 12. Minimal tests

Implemented in:

- [tests/test_research_matrix.py](../tests/test_research_matrix.py)
- existing tests under `tests/`

Covered by tests:

- explicit matrix contains expected runs
- config hash changes by matrix entry
- config parser reads new reproducibility fields
- temporal validation stays ordered
- no leakage columns reach model features
- tiny dummy pipeline run succeeds
- selector and comparison behavior remains compatible

## Verification Status Before Full Data Run

The final pre-run smoke check passed:

```bash
python main.py --dry-run
python -m py_compile ...
uv run python scripts/check_research_setup.py
uv run pytest
```

Observed status:

- dry-run showed all 12 matrix entries
- setup validation passed
- full test suite passed: `32 passed`

Important practical note:

`uv` emitted warnings about stale `pandas` and `numpy` `.dist-info` records in
the local virtual environment while still completing validation successfully.
That is an environment hygiene warning, not a failing project check.

## What Is Still Pending

The code is ready, but the empirical study is not complete until the full
matrix finishes on the real data.

Pending research tasks:

- run `python main.py` or `uv run python main.py`
- monitor long-running Boruta/CatBoost/LLM jobs
- aggregate results
- generate plots
- interpret final tables
- write the final academic narrative

Runtime risks that only the full run can fully expose:

- LLM API rate limits, latency, or cost
- Boruta runtime on the full engineered feature set
- CatBoost runtime or memory pressure
- unexpected data-specific edge cases in rare folds

## Does It Meet The Topic Requirements?

**Yes for implementation readiness.**

The implementation now supports:

- statistical-only baselines
- LLM-only selection
- hybrid LLM -> statistical selection
- LR and CatBoost model families
- temporal DEV/OOT evaluation
- fold stability tracking
- OOT performance tracking
- feature overlap and selected-feature artifacts
- final model and fitted preprocessor artifacts
- run manifests and config hashes
- setup validation
- result aggregation and plots

**Not yet yes for completed research evidence.**

The evidence exists only after the full matrix runs and the outputs are
aggregated.

## Recommended Execution Order

Run the setup check:

```bash
uv run python scripts/check_research_setup.py
```

Inspect the planned matrix:

```bash
uv run python main.py --dry-run
```

Run the matrix:

```bash
uv run python main.py
```

Aggregate results:

```bash
uv run python scripts/aggregate_results.py results/
```

Generate plots:

```bash
uv run python plots.py --all results/
```

## Overall Verdict

The project is now in a strong research-grade position for the full empirical
run.

The major reproducibility risks that existed before have been addressed:

- hidden matrix omissions
- scattered outputs
- shallow manifests
- unclear seeds
- missing setup validation
- missing resume behavior
- confusing fold-vs-final feature lists
- lack of leakage reporting
- lack of final aggregation command

The honest remaining caveat is runtime validation: the full experiment matrix
still has to complete successfully on the real data before final claims can be
made.
