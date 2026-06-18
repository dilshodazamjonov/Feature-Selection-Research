# Current Pipeline

Use this path for current research work.

## Active Datasets

- `homecredit`
- `lendingclub_v2`

Do not use old `lendingclub` for new CLIP or cross-dataset analysis. It is retained as a legacy reproduction path.

## Full Experiment Flow

```powershell
uv run python scripts/run_matrix.py --dataset homecredit
uv run python scripts/aggregate_results.py --dataset homecredit
uv run python scripts/make_plots.py --dataset homecredit

uv run python scripts/run_matrix.py --dataset lendingclub_v2
uv run python scripts/aggregate_results.py --dataset lendingclub_v2
uv run python scripts/make_plots.py --dataset lendingclub_v2
```

## Cross-Dataset Analysis

```powershell
uv run python scripts/analyze_cross_dataset_v2.py
```

Primary output:

- `reports/cross_dataset_v2_analysis.md`

## CLIP Readiness

Generate readiness evidence:

```powershell
uv run python scripts/build_clip_readiness_feature_evidence.py
```

Validate readiness before future CLIP implementation/training:

```powershell
uv run python scripts/validate_clip_readiness.py
```

Training-safe CLIP evidence:

- `results/homecredit/analysis/clip_readiness/dev_only_clip_training_evidence.csv`
- `results/lendingclub_v2/analysis/clip_readiness/dev_only_clip_training_evidence.csv`

Evaluation/planning-only evidence:

- `results/homecredit/analysis/clip_readiness/feature_level_evidence_for_clip.csv`
- `results/lendingclub_v2/analysis/clip_readiness/feature_level_evidence_for_clip.csv`

The planning evidence can include OOT/PSI diagnostics. It must not be used as CLIP training input.

## Safety Rules

- CLIP training inputs must be DEV-only.
- OOT labels, OOT summaries, PSI fields, target columns, split columns, IDs, and post-origination fields are forbidden training inputs.
- Features marked non-safe in leakage review must stay blocked.
- `lendingclub_v2` is the active external dataset; old `lendingclub` is legacy.
