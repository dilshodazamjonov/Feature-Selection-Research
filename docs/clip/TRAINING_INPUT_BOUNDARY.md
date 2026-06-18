# CLIP Training Input Boundary

This document defines the dry-run input boundary for future CLIP-style feature-selection research. It does not approve model training by itself.

## Allowed Training Evidence

Only these DEV-only files may be used as CLIP training evidence:

- `results/homecredit/analysis/clip_readiness/dev_only_clip_training_evidence.csv`
- `results/lendingclub_v2/analysis/clip_readiness/dev_only_clip_training_evidence.csv`

`homecredit` is the only trainer-fit dataset. `lendingclub_v2` is external-validation evidence only.

## Forbidden Training Evidence

Do not train from:

- `feature_level_evidence_for_clip.csv`
- any file under `results/lendingclub/`
- any file under `results_v1/`
- OOT predictions, OOT metrics, or OOT summaries
- PSI, score PSI, or feature PSI fields
- targets, labels, split/fold indicators, IDs
- post-origination outcome fields
- payment, recovery, settlement, or hardship fields
- rows marked unsafe or blocked by leakage review

`feature_level_evidence_for_clip.csv` is planning/evaluation evidence. It may contain OOT/PSI diagnostics and therefore is forbidden as CLIP training input.

## Dataset Roles

- `homecredit`: `train`
- `lendingclub_v2`: `external_validation`
- `lendingclub`: legacy only; forbidden for new CLIP work

## Field Roles

- `text_input`: text used by future encoders, such as `clip_training_text`, `description`, `semantic_group`, and `source_table`
- `statistical_input`: DEV-only numeric evidence such as `missing_rate_dev`, `iv_score_if_available`, `mrmr_selection_frequency`, and `boruta_selection_frequency`
- `anchor_only`: feature identity fields used for joining and traceability
- `metadata_only`: audit or context fields not used as model inputs
- `evaluation_only`: fields reserved for downstream evaluation, not training
- `supervision_only`: future supervised labels, currently empty
- `forbidden`: target, OOT, PSI, split, ID, or post-outcome fields

## LLM Rank Policy

LLM rank fields are metadata-only in the main CLIP dry-run manifest. They are not training inputs unless a later, separately reviewed experiment explicitly changes that policy.

## Stable-Core Policy

Stable-core fields are metadata-only in the main CLIP dry-run manifest. They are not training inputs under the current boundary.

## OOT And PSI Policy

OOT and PSI fields are forbidden as CLIP training inputs. They may be used only as evaluation/support diagnostics outside the training input table.

## Dry-Run Process

Validate readiness:

```powershell
uv run python scripts/validate_clip_readiness.py
```

Build the dry-run manifest:

```powershell
uv run python scripts/build_clip_training_manifest.py --dry-run
```

Expected outputs:

- `results/clip/dry_run/training_manifest.json`
- `results/clip/dry_run/training_features.csv`
- `results/clip/dry_run/external_validation_features.csv`
- `results/clip/dry_run/blocked_features.csv`
- `results/clip/dry_run/schema_audit.json`
- `results/clip/dry_run/field_role_manifest.csv`
- `results/clip/dry_run/source_hashes.json`

The dry-run builder never trains a model, loads an encoder, creates contrastive pairs, calls an LLM, or integrates a selector into the experiment matrix.

## Required Gate Before Training

Before any future model training starts:

- readiness validation must pass
- dry-run manifest validation must pass
- source hashes must be recorded
- blocked feature reasons must be preserved
- old LendingClub must be absent
- OOT/PSI/target/split/ID fields must be absent from training inputs
