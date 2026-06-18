# Project Structure

The canonical implementation lives under `src/credit_risk_fs/`. Legacy top-level folders are retained for compatibility and historical inspection.

## Active Source Layout

- `src/credit_risk_fs/data`: dataset configs, loaders, registry helpers
- `src/credit_risk_fs/preprocessing`: cleaning, missingness, leakage, temporal split, labels
- `src/credit_risk_fs/feature_engineering`: dataset-specific feature construction
- `src/credit_risk_fs/feature_metadata`: descriptions, semantic groups, metadata builders
- `src/credit_risk_fs/selectors`: statistical, LLM, and hybrid selectors
- `src/credit_risk_fs/clip`: CLIP-readiness manifests, text helpers, and leakage policy
- `src/credit_risk_fs/models`: LR and CatBoost wrappers
- `src/credit_risk_fs/evaluation`: metrics, stability, drift, aggregation, plotting
- `src/credit_risk_fs/experiments`: matrix execution, config handling, run tracking
- `src/credit_risk_fs/pipelines`: dataset adapters and shared experiment pipeline
- `src/credit_risk_fs/utils`: IO, logging, hashing, serialization, paths

## Active Datasets

- `data/homecredit`: active Home Credit data, processed frames, and metadata
- `data/lendingclub_v2`: active LendingClub v2 data, processed frame, metadata, leakage review
- `data/lendingclub`: legacy LendingClub v1 line retained for reproduction only

Current configs:

- `configs/datasets/homecredit.yaml`
- `configs/datasets/lendingclub_v2.yaml`
- `configs/experiments/homecredit_matrix.yaml`
- `configs/experiments/lendingclub_v2_matrix.yaml`
- `configs/clip/readiness.yaml`

Legacy config copies:

- `configs/datasets/legacy/lendingclub.yaml`
- `configs/experiments/legacy/lendingclub_matrix.yaml`

The root-level legacy configs may remain for backward-compatible tests and scripts, but new analysis should not use them.

## Results

- `results/homecredit`: active Home Credit matrix and aggregate outputs
- `results/lendingclub_v2`: active LendingClub v2 matrix and aggregate outputs
- `results/cross_dataset_v2`: active cross-dataset outputs
- `results/lendingclub`: legacy v1 outputs retained for inspection

Each run folder is expected to contain:

- `features/`
- `models/`
- `results/`
- `llm_responses/`
- `feature_rankings/`
- `selected_feature_sets/`

## Reports And Notebooks

Current markdown deliverables stay directly under `reports/`.

Historical or stale reports are under:

- `reports/archive/`

Exploratory notebooks are retained under:

- `Notebooks/archive/`

Notebook outputs are not the canonical final deliverables. The reproducible scripts and markdown reports are the canonical research artifacts.

## CLIP-Readiness Boundary

The only CLIP training inputs approved by the current readiness policy are:

- `results/homecredit/analysis/clip_readiness/dev_only_clip_training_evidence.csv`
- `results/lendingclub_v2/analysis/clip_readiness/dev_only_clip_training_evidence.csv`

The broader `feature_level_evidence_for_clip.csv` files are planning/evaluation evidence only and may include OOT or PSI support fields.

Before any future CLIP training work:

```powershell
uv run python scripts/validate_clip_readiness.py
```

## Compatibility

The following top-level folders are compatibility or historical paths, not the preferred implementation surface:

- `Models/`
- `Preprocessing/`
- `feature_selection/`
- `training/`
- `utils/`
- `evaluation/`
- `experiments/`
- `pipelines/`

Do not remove them casually; some tests and old scripts still verify compatibility behavior.
