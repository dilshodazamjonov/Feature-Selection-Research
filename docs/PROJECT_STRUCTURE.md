# Project Structure

The refactored repository is organized around a source package under `src/credit_risk_fs/`.

## Source Package

- `data/`: dataset configs, registries, loaders, and schema helpers
- `preprocessing/`: cleaning, leakage handling, temporal splitting, and labeling helpers
- `feature_engineering/`: dataset-specific feature construction
- `feature_metadata/`: description parsing, metadata generation, semantic grouping
- `selectors/`: statistical, LLM, and hybrid feature selectors
- `models/`: LR and CatBoost wrappers plus CV helpers
- `evaluation/`: metrics, stability, drift, semantic coverage, redundancy, plotting, aggregation
- `experiments/`: config handling, experiment matrix, run tracking, pairwise comparison
- `pipelines/`: shared experiment pipeline plus dataset adapters
- `utils/`: logging, paths, hashing, IO, serialization

## Data

- `data/homecredit/raw`: raw Home Credit CSVs
- `data/homecredit/metadata`: description file, schema snapshot, leakage policy
- `data/lendingclub/raw`: raw LendingClub CSV
- `data/lendingclub/processed`: single-table prepared file for modeling
- `data/lendingclub/metadata`: description file, schema snapshot, leakage policy

## Results

- `results/homecredit`: Home Credit experiment outputs
- `results/lendingclub`: LendingClub experiment outputs
- `results/cross_dataset`: combined summaries when both datasets are available

Each run folder is expected to contain:
- `features/`
- `models/`
- `results/`
- `llm_responses/`
- `feature_rankings/`
- `selected_feature_sets/`

## Compatibility

Legacy top-level modules such as `Preprocessing/`, `feature_selection/`, `Models/`, `evaluation/`, `experiments/`, `pipelines/`, and `training/` are kept as thin compatibility shims while the real code lives under `src/credit_risk_fs/`.
