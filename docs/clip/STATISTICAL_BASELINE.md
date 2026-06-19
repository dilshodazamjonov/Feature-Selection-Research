# CLIP Statistical Baseline

This baseline is the DEV-only statistical-vector baseline with a Home Credit training-split stable-core anchor.

It is not contrastive training, not neural training, not matrix integration, and not predictive model evaluation. It ranks feature metadata vectors by similarity to a Home Credit training-split anchor.

## Inputs

Required Prompt 1 and Prompt 2 artifacts:

- `results/clip/dry_run/training_manifest.json`
- `results/clip/dry_run/training_features.csv`
- `results/clip/dry_run/external_validation_features.csv`
- `results/clip/dry_run/field_role_manifest.csv`
- `results/clip/dry_run/source_hashes.json`
- `results/clip/text_baseline/homecredit_group_split.csv`
- `results/clip/text_baseline/group_split_audit.json`
- `results/clip/text_baseline/homecredit_anchor_features.csv`
- `results/clip/text_baseline/text_anchor_manifest.json`
- `results/clip/text_baseline/homecredit_feature_text.csv`
- `results/clip/text_baseline/lendingclub_v2_feature_text.csv`

The builder validates source hashes and stops if required artifacts are missing or inconsistent.

## Statistical Fields

Main statistical vector field:

- `missing_rate_dev`: target-free DEV missingness statistic, shared by Home Credit and LendingClub v2.

Optional ablation fields, excluded from the main vector:

- `iv_score_if_available`: DEV-only target-aware univariate statistic.
- `bootstrap_selection_frequency_if_available`: DEV-only resampling evidence, available in source evidence.
- `mrmr_selection_frequency`: algorithm-derived selector frequency.
- `boruta_selection_frequency`: algorithm-derived selector frequency.

Forbidden from the main view:

- LLM ranks and LLM selection decisions.
- Stable-core membership.
- OOT, PSI, target, label, prediction, fold, split, ID, payment, recovery, settlement, hardship, charged-off, and post-origination fields.
- Legacy `lendingclub` artifacts.

The main default intentionally uses only `missing_rate_dev`. This is conservative and leakage-safe, but the resulting vector is one-dimensional, so cosine rankings are a diagnostic baseline rather than a rich statistical representation.

## Fit Boundary

Preprocessing is fit only on Home Credit features assigned to the Prompt 2 `train` split.

Fitted on Home Credit train split only:

- Median imputation values.
- Scaling parameters.
- Clipping thresholds, if clipping is enabled.

Transform-only:

- Home Credit validation split.
- LendingClub v2 external validation features.

LendingClub v2 is never used to fit preprocessing, thresholds, anchors, or hyperparameters.

## Split Reuse

The baseline reuses:

`results/clip/text_baseline/homecredit_group_split.csv`

The builder verifies that each Home Credit feature appears once and that no group overlaps train and validation.

The reused split includes canonical feature-family metadata:

- `canonical_feature_family`
- `family_resolution_source`
- `family_resolution_rule`
- `family_member_count`

These fields are copied into the statistical-vector metadata so downstream
contrastive-pair artifacts can exclude negatives from the same resolved family.
Validation features never affect imputation, scaling, anchor construction, or
any fitted statistical state.

## Anchor Policy

Stable-core membership is anchor-only. It is never included in the statistical vector.

The statistical anchor is built from stable-core Home Credit features that are also in the Home Credit training split. Home Credit validation features do not build the anchor. LendingClub v2 does not build or modify an anchor.

## Outputs

Full run outputs:

- `results/clip/statistical_baseline/statistical_field_inventory.csv`
- `results/clip/statistical_baseline/statistical_field_inventory.json`
- `results/clip/statistical_baseline/statistical_preprocessor.json`
- `results/clip/statistical_baseline/statistical_preprocessor.joblib`
- `results/clip/statistical_baseline/statistical_feature_order.json`
- `results/clip/statistical_baseline/statistical_preprocessing_audit.json`
- `results/clip/statistical_baseline/homecredit_statistical_vectors.parquet`
- `results/clip/statistical_baseline/lendingclub_v2_statistical_vectors.parquet`
- `results/clip/statistical_baseline/homecredit_statistical_anchor_features.csv`
- `results/clip/statistical_baseline/statistical_anchor_manifest.json`
- `results/clip/statistical_baseline/homecredit_statistical_only_ranking.csv`
- `results/clip/statistical_baseline/lendingclub_v2_statistical_only_ranking.csv`
- `results/clip/statistical_baseline/statistical_baseline_summary.json`

Dry-run outputs are written under:

`results/clip/statistical_baseline/dry_run/`

Dry-run does not overwrite the full-run preprocessor, vectors, rankings, or summary.

## Commands

```powershell
uv run python scripts/build_clip_statistical_baseline.py --config configs/clip/statistical_baseline.yaml --dry-run
uv run python scripts/build_clip_statistical_baseline.py --config configs/clip/statistical_baseline.yaml
```

## Limitations

The current shared, leakage-safe main field set is intentionally narrow. The one-dimensional default is useful for boundary validation and as a conservative diagnostic baseline, not as evidence of predictive performance. Rank correlation with text-only ranking is reported only as a diagnostic and must not be interpreted as model performance.

Future contrastive work must remain separate and must not reuse LendingClub v2 to fit state.
