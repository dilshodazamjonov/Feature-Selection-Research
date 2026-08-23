# Stability 2024 CLIP Preparation v1

This package builder prepares the frozen Home Credit Model Stability 2024 feature representation for later CLIP experiments. It does not train CLIP, fit a selector or classifier, score OOT, or run a downstream experiment.

## Locked inputs and boundary

The feature universe is the authenticated ordered `predictor_columns` list in `outputs/prompt_16_homecredit_model_stability_2024/matrix_v1/metadata.json`. The builder requires exactly 1,959 unique predictors and the ordered-name SHA-256 `882e958aacfb0076ed7291ea8eee86e87b4d1b2d91ed8ad1d9ac7c896eb2681a`. Lineage must map those identities one-to-one.

Every matrix-value read is projected to approved predictor columns plus `date_decision` when the anchor needs time. The reader applies the frozen DEV predicate before exposing rows:

```text
date_decision < 2020-02-26
```

It requires 1,221,743 returned DEV rows on every scan. `target`, identifiers, and other non-predictors cannot be requested. Returned dates are checked again, and any OOT row fails the run.

The JSON protocol pins the authenticated matrix manifest, metadata, lineage, split-membership evidence, feature definitions, original protocol lock, and the historical CLIP implementation files by SHA-256. A changed or missing input fails closed.

## Deterministic semantics and text

Semantics use repository evidence in this order:

1. authenticated `feature_definitions.csv` descriptions;
2. authenticated matrix lineage and source family;
3. the exact recorded aggregation operation;
4. a transparent structural lineage template.

No LLM output is authoritative. The 10 generated row-count fields use the structural fallback because they do not have a source variable. Text is rendered by the existing `feature_text_v1` implementation without changing its template.

## Identity and representation split

`identity_equivalence_v2` admits only documented aliases, documented identity transforms, or exact aligned DEV duplicates. The Stability v1 orchestrator has no separate documented alias file, so its non-singleton groups come from exact DEV duplicates. Hashes identify candidates only. Each candidate relation is then checked for actual equality of canonicalized values and missing-value positions over all DEV rows.

The representation split is across feature identities, not credit rows. It reuses the corrected deterministic group split with seed 42 and target ratio 80/20. An equivalence group cannot cross TRAIN and validation. Only representation-TRAIN feature identities fit the statistical preprocessor or compete for the source anchor.

## Statistical representation

The raw descriptor schema is the existing `compact_target_free_v2` 13-field order. Raw values use DEV predictor values only. The Stability-source preprocessor reuses `RobustStatisticalPreprocessorV2`:

- median/IQR fit on representation-TRAIN feature vectors only;
- the first seven continuous descriptors clipped to `[-8, 8]`;
- the final six validity/type indicators unchanged;
- all 1,959 identities transformed after fitting.

The raw descriptors remain the frozen transfer input for future Home Credit→Stability and LendingClub→Stability work. Those directions must apply the already-frozen source preprocessor without refitting it on Stability.

## Corrected temporal source anchor

The Stability adapter generalizes the active `lendingclub_dev_temporal_stable_core_v1` implementation only for dataset identity and calendar-time boundaries. The locked method is:

- all Stability DEV rows divided into four equal-duration, left-closed/right-open windows;
- candidates limited to representation-TRAIN identities;
- first-window-fitted buckets frozen for all later windows;
- numeric reference quantile bins with unique edges and infinite endpoints;
- categorical reference levels subject to minimum count, with `OTHER` and `MISSING` buckets;
- the existing adjacent-window PSI formula with epsilon `1e-6`;
- maximum missing-rate difference computed as maximum minus minimum across the four windows;
- eligibility at adjacent PSI `<= 0.10`, missing-rate difference `<= 0.05`, and at least 100 non-missing values per window;
- deterministic order by maximum adjacent PSI, maximum missing-rate difference, then `feature_id`;
- identity-group de-duplication and exactly 23 members, otherwise `BLOCKED`.

No target, OOT value, external data, model output, or selector rank participates.

## Frozen text embeddings and pairs

The orchestrator performs inference with frozen `sentence-transformers/all-MiniLM-L6-v2`, revision `main`. It requires 384 float32 dimensions and row-wise L2 normalization. Loading failure, trainable parameters, training mode, a different model/revision, or a different dimension fails without fallback.

The final Parquet pair table binds each feature identity to its equivalence group, representation split, rendered-text hash, 384-dimensional frozen text embedding, raw-descriptor hash, and 13-dimensional Stability-scaled statistical vector. It contains no target, OOT values, AUC, feature importance, selector rank, or classifier output.

## Output package

The manual run publishes atomically to:

```text
outputs/prompt_16_homecredit_model_stability_2024/clip_preparation_v1/
```

It creates the requested metadata, text, pairing, statistics, anchor, pair, provenance, methodology-lock, and validation artifacts. `manifests/sha256_manifest.csv` hashes every other generated artifact. It necessarily excludes its own bytes because a file cannot contain its final self-hash; rerun verification separately validates the manifest schema, its complete expected coverage, every recorded byte size/hash, the methodology/configuration hash, and the PASS report.

The default rerun behavior verifies and reuses a complete matching package. It does not overwrite incomplete, incompatible, or modified output. The explicit `--rebuild` flag moves the old directory to a timestamped backup before rebuilding, so the operation remains recoverable.

After stage 8, the builder also maintains a deterministic, SHA-256-verified work checkpoint beside the final output directory. If anchor construction, MiniLM inference, pair construction, or final validation fails, the same command resumes after the last completed stage instead of recomputing exact-duplicate hashes and raw descriptors. A work checkpoint with a different configuration or a changed artifact fails closed. The checkpoint is removed immediately before successful atomic publication and never enters the final artifact manifest.

## Manual execution

From the repository root:

```powershell
python scripts/prepare_stability_clip_inputs.py --config configs/protocols/homecredit_model_stability_2024_v2/clip_stability_preparation_v1.json
```

The script prints `[1/12]` through `[12/12]` progress. Matrix scans and MiniLM inference happen only during this manual run. Preparation completes only after every validation contract passes and the atomically published package passes a second hash/idempotency verification.
