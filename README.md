# Credit Feature Selection Research

Research framework for comparing credit-risk feature-selection methods, including LLM metadata screening and hybrid statistical selectors.

The current research line uses:

- `homecredit`
- `lendingclub_v2`

The old `lendingclub` dataset/config/result line is legacy and kept only for reproducibility checks.

## Current Scope

The project evaluates whether an LLM can act as a metadata-only first-stage feature screener. The LLM is not the final model and is not a production underwriter.

Compared selectors:

- `mrmr`
- `boruta`
- `pca`
- `domain_rule_baseline`
- `llm`
- `llm_then_mrmr`
- `llm_then_boruta`
- `stable_core_llm_fill`

Evaluation models:

- Logistic Regression
- CatBoost

Out of scope:

- production scoring
- credit decision automation
- calibration/stacking/deployment
- CLIP model training until readiness validation passes

## Canonical Structure

```text
src/credit_risk_fs/
  data/                 dataset configs, loaders, schema helpers
  preprocessing/        cleaning, leakage handling, temporal splitting
  feature_engineering/  dataset-specific feature construction
  feature_metadata/     descriptions, semantic groups, metadata builders
  selectors/            statistical, LLM, and hybrid selectors
  clip/                 CLIP-readiness manifests, leakage policy, validators
  models/               LR and CatBoost wrappers
  evaluation/           metrics, stability, drift, plotting, aggregation
  experiments/          config handling, matrix execution, run tracking
  pipelines/            shared experiment pipeline and dataset adapters
  utils/                IO, logging, hashing, path helpers

configs/
  datasets/
    homecredit.yaml
    lendingclub_v2.yaml
    legacy/lendingclub.yaml
  experiments/
    homecredit_matrix.yaml
    lendingclub_v2_matrix.yaml
    legacy/lendingclub_matrix.yaml
  clip/readiness.yaml

scripts/
  run_matrix.py
  aggregate_results.py
  make_plots.py
  build_clip_readiness_feature_evidence.py
  validate_clip_readiness.py

results/
  homecredit/
  lendingclub_v2/
  cross_dataset_v2/
  lendingclub/          legacy

reports/
  archive/              old v1 and stale cross-dataset reports

Notebooks/
  archive/              exploratory notebooks retained for reference
```

Legacy top-level folders such as `Models/`, `Preprocessing/`, `feature_selection/`, `training/`, `utils/`, `evaluation/`, `experiments/`, and `pipelines/` are compatibility shims or historical modules. Treat `src/credit_risk_fs/` as the canonical implementation.

## Setup

```powershell
uv sync
```

Validate the environment:

```powershell
uv run python scripts/check_research_setup.py
uv run python scripts/check_setup.py --dataset homecredit
uv run python scripts/check_setup.py --dataset lendingclub_v2
```

## Current Pipeline

Do not use old `lendingclub` for new experiments.

```powershell
uv run python scripts/run_matrix.py --dataset homecredit
uv run python scripts/aggregate_results.py --dataset homecredit
uv run python scripts/make_plots.py --dataset homecredit

uv run python scripts/run_matrix.py --dataset lendingclub_v2
uv run python scripts/aggregate_results.py --dataset lendingclub_v2
uv run python scripts/make_plots.py --dataset lendingclub_v2
```

Cross-dataset and CLIP-readiness artifacts are generated from completed results:

```powershell
uv run python scripts/analyze_cross_dataset_v2.py
uv run python scripts/build_clip_readiness_feature_evidence.py
uv run python scripts/validate_clip_readiness.py
```

## CLIP Readiness

CLIP training inputs are restricted to DEV-only evidence tables:

```text
results/homecredit/analysis/clip_readiness/dev_only_clip_training_evidence.csv
results/lendingclub_v2/analysis/clip_readiness/dev_only_clip_training_evidence.csv
```

Do not train CLIP from:

```text
feature_level_evidence_for_clip.csv
```

That file intentionally includes evaluation/support evidence such as OOT/PSI fields. The validator enforces that DEV-only CLIP training tables do not contain target, split, OOT, PSI, post-origination, or leakage-excluded columns.

Run before any future CLIP implementation or training:

```powershell
uv run python scripts/validate_clip_readiness.py
```

## Main Reports

Current top-level reports:

- `reports/cross_dataset_v2_analysis.md`
- `reports/clip_readiness_feature_evidence_report.md`
- `reports/homecredit_report.md`
- `reports/lendingclub_v2_final_pre_matrix_approval.md`
- `reports/lendingclub_v2_metadata_quality_audit.md`
- `reports/lendingclub_v2_pre_matrix_sanity_report.md`

Older reports are kept under `reports/archive/`.

## Documentation

- `docs/PROJECT_STRUCTURE.md`
- `docs/current_pipeline.md`
- `docs/leakage_policy.md`
- `docs/REPRODUCIBILITY.md`
- `docs/dataset_notes/homecredit.md`
- `docs/dataset_notes/lendingclub_v2.md`
- `docs/dataset_notes/lendingclub.md` for legacy v1 context

## Current Defaults

Home Credit remains the compatibility default in older entrypoints. New cross-dataset analysis and CLIP-readiness work should use `homecredit` and `lendingclub_v2` only.
