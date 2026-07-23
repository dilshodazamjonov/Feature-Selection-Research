# Cross-dataset voting integration and resource pilot v1

## Outcome

**CONDITIONAL_PASS.** All nine bounded integration gaps pass, the frozen matrix expands exactly, and all four real K=200 first-fold pilots completed through the common registered lifecycle. Pilot outputs are diagnostic, single-fold, and ineligible for research or paired inference. The remaining condition is a resource-capacity review: conservative LendingClub-v2 largest-fold/full-DEV projections do not clear every unchanged memory guardrail.

## Boundary and architecture

The work covered Home Credit and LendingClub v2, LR and CatBoost, K=200, seed 42, and the canonical first DEV fold only. It did not run a research ID, reference, K=100/K=300 sensitivity, five-fold OOF, full-DEV refit, OOT score, paired p-value, API, CLIP, embedding, SHAP, or GPU workload.

The implementation extends the existing runner, selector registry, frozen rank aggregator, tracking layout, atomic publisher, checkpoint manager, resource supervisor, prediction writer, and repository validator. No competing execution or result architecture was introduced.

## Nine integration gaps

1. Fold-local adapter: both voters, aggregation, top-200 projection, CatBoost RFE, final-model fit, and held-out scoring use the canonical training/validation boundary.
2. Long voter schema: original and normalized names, ranks/scores, selector/seed, training/input hashes, presence, and artifact metadata are preserved with one canonical vote per voter.
3. RFE: the standalone registered CatBoost-backed selector is CPU-only, <=4 threads, emits a trace, and fails unless it returns exactly 20/40 features.
4. Matrix runner: pure dry expansion returns 12 voting plus 4 rerun-required references, 80 future folds, 16 future final fits, 4 primary and 8 sensitivity comparisons.
5. Projection manifest: row, voter, aggregation, RFE, final-model, and evaluation projections are ordered, hashed, and reject implicit all-column requests.
6. Lifecycle: pilot artifacts use the common atomic/checkpoint/resource/tracking path; completed runs are immutable and controlled-stop attempts remain preserved.
7. Prediction contract: deterministic synthetic complete OOF/OOT checks exist; real pilots are `single_dev_fold_pilot`, research/comparison false.
8. Inference wiring: four reference configurations and 4+8 paired comparison definitions are validated but were not executed and no p-value was calculated.
9. Effective models: fitted LR/CatBoost parameters, training-only preprocessing, class order, CPU/thread use, RFE configuration, and exact feature budgets are asserted and saved.

CatBoost records `early_stopping_rounds=150`, but validation targets are intentionally excluded from fit, so no eval set is supplied and early stopping is inactive; all 1,500 configured iterations run. LR records sklearn's L2 deprecation bridge as `penalty=deprecated, l1_ratio=0.0`, resolved explicitly as effective L2.

## Deterministic expansion and tests

The frozen matrix dry expansion produced exactly 16 IDs in frozen order: 12 voting, 4 rerun-required references, 80 future DEV folds, 16 future final fits, 4 primary comparisons, and 8 sensitivity comparisons. Active run-index rows and directories for those 16 IDs remain zero.

Focused validation: 132 passed. Full suite: 480 passed, 31 skipped, 107 warnings. The repository validator, compileall, and `git diff --check` pass.

## Pilot results (diagnostic only)

Wall time below is cumulative across preserved attempts; it includes explicit identical resumes where a bounded integration defect was fixed. No performance metric is reported.

| Pilot | Status | Train | Validation | Universe | K | Final | Wall s | Peak RSS GiB | Min RAM GiB | GPU B | Stop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cdv1-pilot-001-homecredit-voting-k200-lr-s42-f0 | completed | 16242 | 16720 | 529 | 200 | 20 | 469.9 | 2.08 | 22.46 | 0 |  |
| cdv1-pilot-002-homecredit-voting-k200-catboost-s42-f0 | completed | 16242 | 16720 | 529 | 200 | 40 | 349.8 | 2.09 | 22.17 | 0 |  |
| cdv1-pilot-003-lendingclub-v2-voting-k200-lr-s42-f0 | completed | 83283 | 114267 | 675 | 200 | 20 | 834.1 | 12.60 | 12.25 | 0 |  |
| cdv1-pilot-004-lendingclub-v2-voting-k200-catboost-s42-f0 | completed | 83283 | 114267 | 675 | 200 | 40 | 809.7 | 12.59 | 12.12 | 0 |  |

Pilot 1 preserves three failed attempts before completion (schema filename normalization, Home Credit train-ID intersection, and sklearn's L2 deprecation representation); its validated selection checkpoint was reused for the final LR fit. Pilot 3 preserves one pre-selection identity-type failure. Every correction was bounded, tested, and resumed under the identical stable pilot ID.

## Leakage, projection, and OOT proof

Every fold manifest reports zero train/validation identity overlap. Each ranking row carries training-identity and training-identity-target hashes; both voters contain exactly 529 or 675 unique canonical votes, while RFE is absent from the voter set. Candidate artifacts contain exactly 200 ordered features, RFE consumes only that set, and final models consume exactly 20 or 40 features. Prediction identity/target hashes match the held-out canonical fold and class order is `[0, 1]` with class 1 meaning greater default risk.

Every data-access log has `opened_oot_paths=[]`, `retained_oot_rows=0`, `load_oot=false`, and zero implicit all-column requests. Temporally mixed source CSVs were chunk-scanned with explicit projections and filters; no distinct OOT artifact path was opened or retained.

## Artifacts and integrity

- `cdv1-pilot-001-homecredit-voting-k200-lr-s42-f0`
  - voter_rankings: `results/runs/homecredit/cdv1-pilot-001-homecredit-voting-k200-lr-s42-f0/voter_rankings.csv` — `5a9f575b6c57d45d77affae05be481c987a23b64d8e933b56787ae9ef7a1843d` (736,095 bytes)
  - aggregate_ranking: `results/runs/homecredit/cdv1-pilot-001-homecredit-voting-k200-lr-s42-f0/aggregate_ranking.csv` — `221387c1a1c9e71a3a4933e016884c33ef5af3ae40ecfb6dd31d5995f60553f7` (93,419 bytes)
  - candidate_features: `results/runs/homecredit/cdv1-pilot-001-homecredit-voting-k200-lr-s42-f0/candidate_features.csv` — `3629940f3207b3f631be106c5e715606c24b5732987d64a776b5fea8ee986404` (9,376 bytes)
  - selected_features: `results/runs/homecredit/cdv1-pilot-001-homecredit-voting-k200-lr-s42-f0/selected_features.csv` — `f24b15e6570cd49808bd8644110506415bf5344eb243ce313759d01be05ea6db` (1,021 bytes)
  - predictions_dev: `results/runs/homecredit/cdv1-pilot-001-homecredit-voting-k200-lr-s42-f0/predictions_dev.csv` — `0a94158fe5101472c163679c3f1bb05ee482149c35a8f55fbc4461de16612da6` (3,087,288 bytes)
  - effective_model_config: `results/runs/homecredit/cdv1-pilot-001-homecredit-voting-k200-lr-s42-f0/effective_model_config.json` — `0f464070dca643ac15470656d45541570d89d0c1871c54b7a6b481ffb0999738` (1,861 bytes)
  - resource_usage: `results/runs/homecredit/cdv1-pilot-001-homecredit-voting-k200-lr-s42-f0/resource_usage.json` — `252fe48fd824a814d7f49be87d3656773c05167fd53c50e9dfb20134affe8b18` (56,393 bytes)
  - checkpoint: `results/runs/homecredit/cdv1-pilot-001-homecredit-voting-k200-lr-s42-f0/checkpoint.json` — `6dd903b7fa9c16ffe2b9c230c549cb2a56b38f9527a43573c7fb12650843bb97` (25,086 bytes)
- `cdv1-pilot-002-homecredit-voting-k200-catboost-s42-f0`
  - voter_rankings: `results/runs/homecredit/cdv1-pilot-002-homecredit-voting-k200-catboost-s42-f0/voter_rankings.csv` — `5a9f575b6c57d45d77affae05be481c987a23b64d8e933b56787ae9ef7a1843d` (736,095 bytes)
  - aggregate_ranking: `results/runs/homecredit/cdv1-pilot-002-homecredit-voting-k200-catboost-s42-f0/aggregate_ranking.csv` — `221387c1a1c9e71a3a4933e016884c33ef5af3ae40ecfb6dd31d5995f60553f7` (93,419 bytes)
  - candidate_features: `results/runs/homecredit/cdv1-pilot-002-homecredit-voting-k200-catboost-s42-f0/candidate_features.csv` — `3629940f3207b3f631be106c5e715606c24b5732987d64a776b5fea8ee986404` (9,376 bytes)
  - selected_features: `results/runs/homecredit/cdv1-pilot-002-homecredit-voting-k200-catboost-s42-f0/selected_features.csv` — `769064fda073d5ef5360e5098395696f1f20df16d65301d9bd42f1a0579aa92b` (2,240 bytes)
  - predictions_dev: `results/runs/homecredit/cdv1-pilot-002-homecredit-voting-k200-catboost-s42-f0/predictions_dev.csv` — `920dee6f0c35454696c6e8378320b2f92426d626a6a8188466c26bfe08f2f790` (3,293,251 bytes)
  - effective_model_config: `results/runs/homecredit/cdv1-pilot-002-homecredit-voting-k200-catboost-s42-f0/effective_model_config.json` — `1893a5b238e0d95ac1e6b3c797d1b62707505d7b0d1be853c1125bb7799a31a0` (2,521 bytes)
  - resource_usage: `results/runs/homecredit/cdv1-pilot-002-homecredit-voting-k200-catboost-s42-f0/resource_usage.json` — `a44ad54ad4b89312ca0088bb5790da0056346a72df453d604e3502e8043c97a0` (163,346 bytes)
  - checkpoint: `results/runs/homecredit/cdv1-pilot-002-homecredit-voting-k200-catboost-s42-f0/checkpoint.json` — `973c02be5ece83a2364d69cd920fa4d5b431059d291106c1b036a48e101ce2e3` (21,415 bytes)
- `cdv1-pilot-003-lendingclub-v2-voting-k200-lr-s42-f0`
  - voter_rankings: `results/runs/lendingclub_v2/cdv1-pilot-003-lendingclub-v2-voting-k200-lr-s42-f0/voter_rankings.csv` — `f272e5e3b557ebf9ab44df069ed44777cc955490efb39b8b2ede8f8c8d9a6dcf` (940,532 bytes)
  - aggregate_ranking: `results/runs/lendingclub_v2/cdv1-pilot-003-lendingclub-v2-voting-k200-lr-s42-f0/aggregate_ranking.csv` — `fb66c0ce0c5e15d567d5676a183fab2657c72a1600358e3cd6f3fad53bb2ccd8` (121,549 bytes)
  - candidate_features: `results/runs/lendingclub_v2/cdv1-pilot-003-lendingclub-v2-voting-k200-lr-s42-f0/candidate_features.csv` — `e7b84bbe598581ef1b10b1666228487ebaade96a79a6e5923f6c91437ad10488` (9,951 bytes)
  - selected_features: `results/runs/lendingclub_v2/cdv1-pilot-003-lendingclub-v2-voting-k200-lr-s42-f0/selected_features.csv` — `9bd6344b43b0dd53e847c804403bb3ddce827e8d5b6bea06922b8715168b9afc` (1,120 bytes)
  - predictions_dev: `results/runs/lendingclub_v2/cdv1-pilot-003-lendingclub-v2-voting-k200-lr-s42-f0/predictions_dev.csv` — `b89a8df8b62b07d911b84832f3833d910fd2bba0c4669db660c03477c73c7b67` (22,308,629 bytes)
  - effective_model_config: `results/runs/lendingclub_v2/cdv1-pilot-003-lendingclub-v2-voting-k200-lr-s42-f0/effective_model_config.json` — `201701920a28b75ece65cb255db1b8f2763595a528d2f03385c624c81a8017e9` (1,858 bytes)
  - resource_usage: `results/runs/lendingclub_v2/cdv1-pilot-003-lendingclub-v2-voting-k200-lr-s42-f0/resource_usage.json` — `4d699d43ee0d55be7df8c30820325aff0c385bbe6ba447b05e3be63506b3247c` (355,553 bytes)
  - checkpoint: `results/runs/lendingclub_v2/cdv1-pilot-003-lendingclub-v2-voting-k200-lr-s42-f0/checkpoint.json` — `3b2c0ebe42d2f5eb50ef1001b8f5500392cb0d89ca5f815b84c79ebe0a6ce61d` (22,624 bytes)
- `cdv1-pilot-004-lendingclub-v2-voting-k200-catboost-s42-f0`
  - voter_rankings: `results/runs/lendingclub_v2/cdv1-pilot-004-lendingclub-v2-voting-k200-catboost-s42-f0/voter_rankings.csv` — `f272e5e3b557ebf9ab44df069ed44777cc955490efb39b8b2ede8f8c8d9a6dcf` (940,532 bytes)
  - aggregate_ranking: `results/runs/lendingclub_v2/cdv1-pilot-004-lendingclub-v2-voting-k200-catboost-s42-f0/aggregate_ranking.csv` — `fb66c0ce0c5e15d567d5676a183fab2657c72a1600358e3cd6f3fad53bb2ccd8` (121,549 bytes)
  - candidate_features: `results/runs/lendingclub_v2/cdv1-pilot-004-lendingclub-v2-voting-k200-catboost-s42-f0/candidate_features.csv` — `e7b84bbe598581ef1b10b1666228487ebaade96a79a6e5923f6c91437ad10488` (9,951 bytes)
  - selected_features: `results/runs/lendingclub_v2/cdv1-pilot-004-lendingclub-v2-voting-k200-catboost-s42-f0/selected_features.csv` — `affd406de2e5acf2e305f195585eb108a50244bae12c228eb516d8b29264b40d` (2,419 bytes)
  - predictions_dev: `results/runs/lendingclub_v2/cdv1-pilot-004-lendingclub-v2-voting-k200-catboost-s42-f0/predictions_dev.csv` — `3f84ea5cd169ae28f313b634bf1db158a4441ccdbcc8c74a49c2ba15ecc843e6` (23,687,169 bytes)
  - effective_model_config: `results/runs/lendingclub_v2/cdv1-pilot-004-lendingclub-v2-voting-k200-catboost-s42-f0/effective_model_config.json` — `7667f5c92bd0afa917f899a8b5816152a29b6b642a8780dd5ef1a94640b3ddc2` (2,559 bytes)
  - resource_usage: `results/runs/lendingclub_v2/cdv1-pilot-004-lendingclub-v2-voting-k200-catboost-s42-f0/resource_usage.json` — `8ccaff66102d65f4b902989a620bfc66b82af0587615b3a80b726930116ecab2` (375,843 bytes)
  - checkpoint: `results/runs/lendingclub_v2/cdv1-pilot-004-lendingclub-v2-voting-k200-catboost-s42-f0/checkpoint.json` — `71b8a669dbcefa14c86f0e653c0921cb656a274d5352c98e9c219cfb64ed2b32` (21,440 bytes)

The complete artifact inventory and hashes are in `cleanup/audits/cross_dataset_voting_integration_pilot/pilot_manifest.json`; stage samples are in `pilot_stage_resources.csv`.

## Resource amplification and future projections

- `cdv1-pilot-001-homecredit-voting-k200-lr-s42-f0`: observed RSS/input amplification 0.84x; RSS/dense-float32 lower-bound 4.82x; final artifacts 4,221,757 bytes; five-fold upper runtime 0.65 h; full-DEV linear runtime upper bound 0.80 h; capacity `fits_unchanged_policy`.
- `cdv1-pilot-002-homecredit-voting-k200-catboost-s42-f0`: observed RSS/input amplification 0.84x; RSS/dense-float32 lower-bound 4.83x; final artifacts 4,532,736 bytes; five-fold upper runtime 0.49 h; full-DEV linear runtime upper bound 0.59 h; capacity `fits_unchanged_policy`.
- `cdv1-pilot-003-lendingclub-v2-voting-k200-lr-s42-f0`: observed RSS/input amplification 3.21x; RSS/dense-float32 lower-bound 5.62x; final artifacts 24,090,665 bytes; five-fold upper runtime 1.16 h; full-DEV linear runtime upper bound 1.67 h; capacity `capacity_review_required`.
- `cdv1-pilot-004-lendingclub-v2-voting-k200-catboost-s42-f0`: observed RSS/input amplification 3.21x; RSS/dense-float32 lower-bound 5.62x; final artifacts 25,490,525 bytes; five-fold upper runtime 1.12 h; full-DEV linear runtime upper bound 1.62 h; capacity `capacity_review_required`.

Direct facts are the measured one-second process-tree samples, full projected DEV dimensions, finalized bytes, and exact fold counts. Runtime upper bounds repeat the cumulative pilot five times or scale linearly by full-DEV/train rows. Memory projections add to the observed peak (a) float64 training-encoding plus Boruta-shadow growth and (b) retained mixed-frame slice growth using observed bytes per row; folds remain sequential and the frozen 1.35 safety factor is applied separately. These are conservative arithmetic projections, not measurements of later folds.

Home Credit clears the unchanged policy. LendingClub v2 peaked near 12.60 GiB RSS with about 12.12 GiB system RAM available on the first fold. The conservative largest-fold/full-DEV projections cross the 8 GiB available-RAM floor and/or the 28 GiB process ceiling after the 1.35 safety factor. Disk is not limiting: finalized artifacts are far below preflight estimates and hundreds of GiB remained free, though free-space deltas include unrelated OS/background effects and are not treated as attributable writes.

## Limitations and gate

These pilots cover one fold only and cannot establish OOF/OOT performance, stability, statistical significance, or research findings. Later-fold memory scaling is estimated, not observed. CatBoost early stopping is configured but inactive without validation-target leakage. Pandas reported mixed-type inference warnings for explicit LendingClub categorical projections; canonical encoding and hashes still passed.

**CONDITIONAL_PASS:** before Prompt 5 may authorize any frozen research run, approve a separately versioned memory-safe execution refinement and validate the largest LendingClub fold/full-DEV shape under the unchanged policy. The likely non-scientific target is eliminating avoidable full-frame/slice copies and releasing fold-local arrays between sequential folds; rows, features, selectors, seeds, models, and limits must remain unchanged.

Frozen hashes match before and after. The legacy bundle remains 359 files and 110,084,164 bytes with zero added, removed, or changed. Active results contain exactly the four authorized completed pilot IDs, no active lock, and no partial file.
