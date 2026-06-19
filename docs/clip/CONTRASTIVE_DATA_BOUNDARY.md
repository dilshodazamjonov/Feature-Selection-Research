# CLIP Contrastive Data Boundary

This boundary creates deterministic text-statistical positive-pair indexes for future CLIP-style training. It does not train a model.

## Two Views

Each observation represents one feature with two frozen views:

- Text view: the Prompt 2 frozen sentence-transformer embedding.
- Statistical view: the Prompt 3 transformed statistical vector.

The pair files store row IDs, hashes, dimensions, and metadata only. They do not duplicate full embedding arrays.

## Positive Pairs

A positive pair is:

- text embedding for feature A
- statistical vector for feature A

Pairs must align on:

- `dataset`
- `feature_name`
- `source_manifest_hash`
- text hash
- statistical vector hash
- split
- group key

Pair roles:

- `train_positive`: Home Credit training split only.
- `validation_positive`: Home Credit validation split only.
- `external_validation_positive`: LendingClub v2 external validation only.

## Split Boundaries

The builder reuses the Prompt 2 Home Credit group-aware split:

`results/clip/text_baseline/homecredit_group_split.csv`

It does not create a new split. It checks group overlap and records a split hash in `split_manifest.json` and `contrastive_tensor_schema.json`.

Home Credit validation pairs are not allowed in training batches. LendingClub v2 is external only and is never used for training, tuning, negative sampling, normalizers, projections, thresholds, or hyperparameters.

Legacy `lendingclub` is forbidden.

The split and pair files carry canonical feature-family metadata from Prompt 2.
This is why engineered variants such as `REGION_RATING_CLIENT` and
`REGION_RATING_CLIENT_W_CITY` cannot cross the train/validation boundary and
cannot be treated as valid training negatives for each other.

## Negative Policy

The future trainer should primarily use in-batch negatives within Home Credit training only.

Excluded as negatives:

- same feature
- same canonical feature family
- exact normalized feature-text duplicate
- near-duplicate text by cosine similarity over frozen normalized text embeddings
- duplicate formula/source-table metadata
- duplicated statistical vector
- any cross-dataset candidate

Default near-duplicate threshold: `0.95`.

Exact text duplicates and embedding-cosine near duplicates are reported as
separate exclusion reasons. The threshold is evaluated exactly over Home Credit
training embeddings only; Home Credit validation and LendingClub v2 rows never
enter the negative-candidate mask.

Threshold sensitivity diagnostics are generated for:

- `0.90`
- `0.95`
- `0.97`
- `0.99`

Hard-negative mining is disabled initially. No explicit hard-negative training set is generated.

The false-negative and threshold audits are stored in:

`results/clip/contrastive_data/negative_candidate_audit.csv`
`results/clip/contrastive_data/near_duplicate_text_audit.csv`
`results/clip/contrastive_data/near_duplicate_threshold_sensitivity.csv`

## Tensor Schema

The tensor schema records:

- text embedding dimension
- statistical vector dimension
- dtype
- text normalization state
- statistical feature order
- padding policy
- missing-value policy
- text encoder identity and revision
- statistical preprocessor hash
- source manifest hash
- split hash

No tensor contains stable-core membership, LLM rank, target labels, OOT metrics, PSI, IDs, model performance, or post-origination outcomes.

## Dataset Class

`ContrastiveFeatureDataset` lazily retrieves frozen text embeddings and statistical vectors from the source parquet files using pair indexes. It verifies hashes at retrieval time and returns tensors plus separate feature metadata.

It refuses to create a training dataset from LendingClub v2 pairs.

## Outputs

Full-run outputs:

- `results/clip/contrastive_data/homecredit_train_positive_pairs.parquet`
- `results/clip/contrastive_data/homecredit_validation_positive_pairs.parquet`
- `results/clip/contrastive_data/lendingclub_v2_external_pairs.parquet`
- `results/clip/contrastive_data/contrastive_tensor_schema.json`
- `results/clip/contrastive_data/contrastive_pair_manifest.json`
- `results/clip/contrastive_data/split_manifest.json`
- `results/clip/contrastive_data/negative_policy_manifest.json`
- `results/clip/contrastive_data/negative_exclusion_pairs.parquet`
- `results/clip/contrastive_data/negative_candidate_audit.csv`
- `results/clip/contrastive_data/near_duplicate_text_audit.csv`
- `results/clip/contrastive_data/near_duplicate_threshold_sensitivity.csv`
- `results/clip/contrastive_data/pair_quality_audit.csv`
- `results/clip/contrastive_data/pair_quality_audit.json`

Dry-run outputs are written under:

`results/clip/contrastive_data/dry_run/`

Dry-run does not overwrite full-run pair parquet files.

## No Training

This prompt creates the data boundary only. It does not implement epochs, optimizer creation, projection heads, backpropagation, checkpoints, training logs, experiment-matrix integration, LR reruns, or CatBoost reruns.
