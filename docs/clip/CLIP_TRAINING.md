# CLIP-Style Semantic-Statistical Contrastive Feature Encoder

This prompt trains a small CLIP-style semantic-statistical contrastive feature encoder. It is not standard image-text CLIP.

## Architecture

Inputs are fixed artifacts from Prompts 1-4:

- Text view: frozen sentence-transformer feature-metadata embeddings.
- Statistical view: approved DEV-only statistical vectors.

Default model:

- Text projection: `384 -> 64 -> 32`
- Statistical projection for the current one-dimensional view: `1 -> 16 -> 32`
- L2-normalized projected outputs
- Fixed temperature: `0.07`
- Trainable parameters: `27,296`

The raw sentence-transformer is not reloaded or fine-tuned during contrastive training.

## Statistical-View Scope

The current statistical vector dimension is `1`, with field:

- `missing_rate_dev`

Therefore:

```text
statistical_view_scope = missingness_only
```

This is an architectural proof of concept. The trained model primarily aligns feature semantics with DEV missingness behavior. It must not be described as learning broad statistical feature quality.

## Loss And Negatives

Training uses symmetric masked InfoNCE:

- text-to-statistical direction
- statistical-to-text direction
- positive same-feature pairs
- in-batch negatives
- Prompt 4 false-negative exclusions

Masked pairs do not contribute as negatives in either direction.

No cross-dataset negatives, validation negatives in training, explicit hard-negative mining, OOT fields, PSI fields, targets, labels, predictions, IDs, LLM ranks, or stable-core inputs are used.

## Data Boundaries

Home Credit training split:

- optimizer updates only

Home Credit validation split:

- validation loss
- retrieval metrics
- early stopping
- checkpoint selection

LendingClub v2:

- external representation application only after model selection
- never used for loss, early stopping, checkpoint selection, seed choice, architecture choice, temperature choice, threshold choice, or training duration

## Seed Protocol

Configured seeds:

```text
11, 22, 33, 44, 55
```

Each seed is trained independently and retained under `results/clip/training/seeds/seed_*`.

Selection rule:

```text
lowest Home Credit validation loss
```

This rule is declared before applying the selected checkpoint to LendingClub v2.

## Checkpoints

Each checkpoint manifest records:

- checkpoint hash
- seed and epoch
- validation criterion and value
- architecture
- parameter count
- temperature
- source, split, text-embedding, statistical-preprocessor, pair-manifest, and negative-policy hashes
- Git commit
- statistical-view scope

Checkpoint loading fails if upstream artifact hashes differ.

## Metrics And Collapse Checks

Training records:

- contrastive loss
- positive-pair cosine similarity
- allowed-negative cosine similarity
- positive-minus-negative margin
- Recall@1, Recall@5, Recall@10
- MRR
- gradient norm
- temperature

Collapse checks report:

- near-zero variance
- identical projected embeddings
- high or uniform pairwise cosine
- representation norm violations
- NaN or infinite loss through training metrics

The one-dimensional statistical branch may warn about repeated projected values because many features share identical missingness statistics. This warning is part of the missingness-only limitation.

## Learned Ranking

After selecting the checkpoint:

1. Freeze the selected checkpoint.
2. Build normalized joint representations as the average of projected text and projected statistical embeddings.
3. Build a Home Credit training-split stable-core anchor.
4. Score Home Credit features.
5. Apply the same frozen model and unchanged Home Credit anchor to LendingClub v2.

No LendingClub v2 anchor is created.

## Outputs

Main outputs are under:

`results/clip/training/`

Key files:

- `training_manifest.json`
- `training_summary.csv`
- `training_summary.json`
- `seed_comparison.csv`
- `model_selection_manifest.json`
- `representation_audit.json`
- `collapse_audit.csv`
- `retrieval_metrics.csv`
- `learned_anchor_manifest.json`
- `homecredit_learned_scores.csv`
- `lendingclub_v2_learned_scores.csv`
- `homecredit_joint_embeddings.parquet`
- `lendingclub_v2_joint_embeddings.parquet`
- `seeds/seed_*/best_checkpoint.pt`

Dry-run writes only under `results/clip/training/dry_run/`.

Smoke-test writes only under `results/clip/training/smoke_test/`.

## Limitations

No LR, CatBoost, OOT AUC, experiment matrix, selector registry integration, or downstream feature-selection evaluation is run in this prompt.

The learned encoder can be compared only on representation-level diagnostics here. Feature-selection superiority has not been established.
