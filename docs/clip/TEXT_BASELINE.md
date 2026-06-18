# CLIP Text-Only Baseline

This baseline measures whether frozen pretrained text embeddings alone can produce a useful semantic feature ranking. It is not the final CLIP model.

## Scope

The text-only baseline:

- builds one deterministic text string per feature
- encodes the text with a frozen sentence-transformer
- caches embeddings with reproducible keys
- builds a Home Credit anchor centroid from stable-core features
- ranks Home Credit and LendingClub v2 features by cosine similarity to that Home Credit anchor

It does not:

- train or fine-tune an encoder
- implement a statistical encoder
- create contrastive pairs
- run LR or CatBoost
- call an LLM
- integrate with the experiment matrix

## Text Template

Template version: `feature_text_v1`

```text
Feature: {feature_name}. Description: {description}. Semantic group: {semantic_group}. Source or formula: {source_formula}.
```

Allowed text fields:

- feature name
- description
- semantic group
- source table or formula

Forbidden text fields:

- LLM rank
- mRMR/Boruta/bootstrap frequencies
- stable-core membership
- PSI or OOT fields
- target/label fields
- IDs, folds, splits
- model performance metrics

Missing descriptions, semantic groups, or source fields are recorded. The default pipeline does not fabricate missing descriptions.

## Frozen Encoder Policy

Default encoder:

- `sentence-transformers/all-MiniLM-L6-v2`
- revision: `main`

The encoder is used in eval mode with gradients disabled. If the configured model cannot be loaded, the full baseline fails clearly. It must not substitute random embeddings.

## Cache Design

Each embedding cache key depends on:

- dataset
- feature name
- exact feature-text hash
- encoder model name
- encoder revision
- normalization flag
- text template version

Embedding rows include explicit dataset, feature name, text hash, encoder metadata, cache key, and stable numeric embedding columns.

## Group-Aware Split

Home Credit feature-level split uses deterministic groups in this priority:

1. derived base feature family
2. source table
3. semantic group
4. feature-name fallback

The split is for Home Credit feature-level experiments only. LendingClub v2 is never used to fit or tune the split.

## Anchor Construction

The baseline uses `stable_core_membership` only as an anchor-only field.

Stable-core membership:

- may select Home Credit anchor features
- must not enter feature text
- must not enter embeddings
- must not use LendingClub v2

The Home Credit anchor centroid is applied unchanged to LendingClub v2.

## External Validation Protocol

`homecredit` is the training/internal-validation dataset.

`lendingclub_v2` is external validation only. It is encoded with the same frozen encoder and scored against the unchanged Home Credit anchor centroid.

Legacy `lendingclub` is forbidden.

## Commands

Dry-run:

```powershell
uv run python scripts/build_clip_text_baseline.py --config configs/clip/text_baseline.yaml --dry-run
```

Full run when the pretrained encoder and Parquet engine are available:

```powershell
uv run python scripts/build_clip_text_baseline.py --config configs/clip/text_baseline.yaml
```

## Outputs

Text artifacts:

- `results/clip/text_baseline/homecredit_feature_text.csv`
- `results/clip/text_baseline/lendingclub_v2_feature_text.csv`

Embedding artifacts:

- `results/clip/text_baseline/homecredit_text_embeddings.parquet`
- `results/clip/text_baseline/lendingclub_v2_text_embeddings.parquet`
- `results/clip/text_baseline/embedding_cache_manifest.json`
- `results/clip/text_baseline/text_embedding_audit.json`

Split and ranking artifacts:

- `results/clip/text_baseline/homecredit_group_split.csv`
- `results/clip/text_baseline/group_split_audit.json`
- `results/clip/text_baseline/homecredit_text_only_ranking.csv`
- `results/clip/text_baseline/lendingclub_v2_text_only_ranking.csv`
- `results/clip/text_baseline/homecredit_anchor_features.csv`
- `results/clip/text_baseline/text_anchor_manifest.json`
- `results/clip/text_baseline/text_baseline_summary.json`

## Limitation

Cosine similarity to a text-anchor centroid is only a semantic diagnostic. It does not prove predictive utility and does not replace matrix evaluation.
