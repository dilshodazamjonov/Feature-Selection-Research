# LendingClub v2 Notes

`lendingclub_v2` is the active LendingClub line for current cross-dataset analysis and CLIP-readiness work.

## Data Layout

- Raw/source input is derived from the legacy safe LendingClub prepared frame.
- Current processed data: `data/lendingclub_v2/processed/application_train.csv`
- Current metadata: `data/lendingclub_v2/metadata/`
- Current results: `results/lendingclub_v2/`

## Metadata And Leakage Review

Important metadata artifacts:

- `data/lendingclub_v2/metadata/columns_description.csv`
- `data/lendingclub_v2/metadata/feature_inventory.csv`
- `data/lendingclub_v2/metadata/semantic_group_distribution.csv`
- `data/lendingclub_v2/metadata/feature_sanity_check.csv`
- `data/lendingclub_v2/metadata/missingness_summary.csv`
- `data/lendingclub_v2/metadata/leakage_review.csv`

The v2 line is intended to provide richer engineered features and complete descriptions while preserving the safe application-time/leakage-screened policy.

## Current Commands

```powershell
uv run python scripts/check_setup.py --dataset lendingclub_v2
uv run python scripts/run_matrix.py --dataset lendingclub_v2
uv run python scripts/aggregate_results.py --dataset lendingclub_v2
uv run python scripts/make_plots.py --dataset lendingclub_v2
```

For CLIP-readiness:

```powershell
uv run python scripts/build_clip_readiness_feature_evidence.py
uv run python scripts/validate_clip_readiness.py
```

## Legacy Boundary

Old `lendingclub` remains in the repo for historical reproduction only. New CLIP work should use `lendingclub_v2`.
