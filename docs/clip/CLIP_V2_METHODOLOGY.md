# CLIP-v2 Methodology

CLIP-v2 keeps the CLIP-v1 semantic side fixed and replaces the missingness-only statistical branch with a compact target-free distributional view.

## Frozen CLIP-v1 Reference

CLIP-v1 is frozen under `results/clip_versions/v1/` and tagged locally as `clip-v1-frozen`. It aligned deterministic feature text embeddings with one DEV statistical field: `missing_rate_dev`.

CLIP-v2 must not overwrite `results/clip/`, CLIP-v1 checkpoints, CLIP-v1 score caches, CLIP-v1 predictions, or CLIP-v1 final reports.

## Statistical View

The CLIP-v2 statistical vector has exactly 13 dimensions:

1. `missing_rate`
2. `unique_ratio`
3. `concentration_share`
4. `signed_log_mean`
5. `log_standard_deviation`
6. `clipped_skewness`
7. `normalized_entropy`
8. `is_numeric`
9. `is_categorical`
10. `is_binary`
11. `numeric_stats_valid`
12. `skewness_valid`
13. `entropy_valid`

The first seven descriptors are robust-scaled. Type and validity indicators are passed through unchanged.

## Boundaries

Descriptor computation uses approved DEV feature columns only. It must not use OOT rows, OOT distributions, target labels, PSI, prediction outputs, post-origination fields, LLM ranks, or stable-core membership.

The scaler is fitted only on Home Credit training-split feature vectors. Home Credit validation, Home Credit full ranking candidates, and LendingClub v2 are transformed with the frozen Home Credit-fitted scaler. LendingClub v2 remains external validation only and must not refit preprocessing or influence checkpoint selection.

## Type Resolution

CLIP-v2 uses audited metadata before pandas dtype. The resolver records the original dtype, metadata type, resolved type, resolution rule, and ambiguity warning. This prevents integer-coded categories and binary flags from being silently treated as continuous numeric variables.

## Output Isolation

CLIP-v2 outputs belong under:

```text
configs/clip_v2/
results/clip_v2/
reports/clip_v2_*
```

The v2 scripts default to planning or dry-run behavior. Expensive stages require explicit execution flags and should be run manually from `RUN.md`.
