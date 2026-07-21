# Foundation protocol freeze

## Decision

The Prompt 1.1 foundation gate is **PASS** as of 2026-07-21. The machine-readable sources are [credit_scoring_extension_v1.yaml](../../configs/protocols/credit_scoring_extension_v1.yaml), [row_alignment_contract_v1.json](../../configs/protocols/row_alignment_contract_v1.json), and [cross_dataset_rank_voting_v1.yaml](../../configs/protocols/cross_dataset_rank_voting_v1.yaml). This freeze authorizes no experiment by itself.

No model, selector, prediction, voting, SHAP, embedding, CLIP, or external-API job was executed. OOT outcomes were used only to verify the already-fixed target/split counts and row/target alignment; they were not used to choose any design element.

## Result boundaries

The active output root is `D:\python projects\Research\results`; the immutable historical evidence root is `D:\ResearchFindings\results`. Historical reads require `CREDIT_RISK_LEGACY_RESULTS_ROOT`. Canonical root separation and a write barrier are implemented in [result_paths.py](../../src/credit_risk_fs/experiments/result_paths.py).

The frozen historical inventory contains 359 files and 110,084,164 bytes. Its manifest SHA-256 is `31ca80f026dc9169f14e3c518b49cd4ee0ad71074264daef4cf660263e1f2417`. The final preservation comparison is recorded in [validation_summary.json](../../cleanup/audits/foundation_protocol_freeze/validation_summary.json).

## Dataset and row contracts

Home Credit retains the Prompt 1 contract: `SK_ID_CURR`, `TARGET=1` for payment difficulty, DEV `[-600,-240)` with 99,092 rows/7,859 positives, and OOT `[-240,0]` with 120,053 rows/10,688 positives. IDs are unique, non-missing, and disjoint.

LendingClub now uses the original raw `id`, preserved canonically as `loan_id` in an ignored sidecar beside the processed matrix. It is not a generated row number or feature hash. The raw artifact is `data/lendingclub/raw/accepted_2007_to_2018Q4.csv`, SHA-256 `3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a`. The current processed matrix SHA-256 is `ce9583dcc8ed2394cca3456c1c5528a2b08b659a04a02823b65ddc31d95267ae`; the sidecar SHA-256 is `f634343eb897a92aeb63d7e20ff341bc483c130b3351454031d0f191d83ca6b7`.

Generation streams only raw `id`, `loan_status`, and `issue_d`, and processed `TARGET`, `recent_decision`, and `issue_d`. It validates every retained row's target, time, and issue month. Thirty-three embedded policy-total footer rows were identified as non-records only because their IDs were non-decimal and both status and issue date were empty. The remaining 2,260,668 raw loan IDs were non-missing and globally unique; 1,348,099 finalized loans were retained. The job took 265.639 seconds, peaked at 1,116,401,664 bytes RSS, used one process, and wrote 65,832,644 bytes.

Canonical LendingClub order is `(recent_decision integer days ASC, NFC-normalized canonical decimal loan_id string ASC)`. Missing time or identity fails. DEV and OOT contracts are:

| Split | Rows | Positives (rate) | Equal-time groups / max size | Ordered ID SHA-256 | Ordered ID+target SHA-256 |
|---|---:|---:|---:|---|---|
| DEV | 598,649 | 116,966 (0.1953832713) | 24 / 42,986 | `4d4cd7973f00eb946fef0a6bb09e61fe6d2b9be92892786f352446660c68818e` | `1373baf30fc16b022d1d9059e400df65b5d3bb3f9ae76e3397b3172578f91590` |
| OOT | 293,105 | 68,252 (0.2328585319) | 12 / 48,938 | `86840e88a94f78f328d62e36754f14377c1765a31fb3bc73cbb3f7b2d45f8092` | `9787d44d278d7965b0a966f19717dec1d9718ea8dfb6b96aa531d0f52d0a53e2` |

Both splits have zero missing IDs, zero duplicate IDs, and zero DEV/OOT overlap. [lendingclub_identity.py](../../src/credit_risk_fs/experiments/lendingclub_identity.py) validates sidecar schema/hash/order/target/time before attaching `loan_id`; `loan_id` is then removed from model features. [training.py](../../src/credit_risk_fs/models/training.py) uses the stable secondary key before grouped temporal splitting.

## Prospective voting protocol

The historical Home Credit voting claim remains unverifiable and is not reconstructed. `cross_dataset_rank_voting_v1` is a new prospective protocol applied identically to Home Credit and LendingClub.

The two voters are:

- `rf_corr_mrmr`: `RandomForestRelevanceMRMRSelector`, specifically RF impurity relevance with correlation redundancy, not mutual-information mRMR.
- `boruta`: complete deterministic `BorutaSelector.feature_ranking_`, ordered by Boruta rank and feature name.

Both are supervised and fitted on each DEV training fold only. Corrected CLIP is excluded because provenance is absent. The cached API-backed LLM ranking is excluded because no corresponding verified LendingClub ranking exists. The Home-Credit domain rule is excluded because it lacks a verified cross-dataset conceptual equivalent. RFE is excluded as a voter because it is the distinct downstream reducer.

For an eligible universe of size `N`, voter rank `r` maps to `1-(r-1)/max(N-1,1)`; a missing feature receives zero. Equal weights are averaged. Ties resolve by voter presence descending, best individual rank ascending, then NFC/casefold feature name ascending. Unknown, duplicate, non-finite, target, identity, time-split, and leakage entries fail.

Candidate pools are 100, 200, and 300; pool 200 is primary, while 100 and 300 are sensitivity conditions. Fold-local CatBoost-backed `RFESelector` reduces each pool to 20 LR or 40 CatBoost features. Seed 42 is frozen. OOT is locked to final evaluation and cannot choose any component.

## Statistical inference

[paired_inference.py](../../src/credit_risk_fs/evaluation/paired_inference.py) freezes identity-aligned applicant/loan-level inference. Higher scores mean greater class-1 default risk. AUC is primary; Gini is `2*AUC-1` and is not separately tested. KS is the maximum absolute positive/negative empirical-CDF separation, with the smallest score winning threshold ties. Lift@10 uses `ceil(0.10*n)` highest scores and stable ID ascending at score ties.

Each dataset-model pair is one primary family with three voting-pool comparisons against the same-budget `rf_corr_mrmr` reference: pool 100, 200, and 300. The four families are Home Credit×LR, Home Credit×CatBoost, LendingClub×LR, and LendingClub×CatBoost. Paired two-sided DeLong p-values are Holm-adjusted within each family of three only.

Paired stratified bootstrap samples positives and negatives separately while sharing sampled indices between methods. It makes exactly 2,000 attempts with seed 20260721; failed attempts are counted, no unlimited replacement occurs, and at least 1,900 valid attempts are required. It reports 95% percentile intervals for paired AUC, KS, and Lift@10 differences. Pool-to-pool contrasts, PSI, Jaccard, Kuncheva, runtime, RAM, and GPU are descriptive.

## Historical CLIP integration profile

The 31 historical CLIP tests keep their scientific assertions. They now require one explicit `clip_complete_v1` profile with five artifact groups: readiness, text baseline, v2 statistical view, corrected contrastive data, and corrected training. With no root or with the current incomplete root, they skip as `required_external_evidence_unavailable`. If a root declares the complete profile, every declared artifact and SHA-256 is checked before the tests execute; a missing file, hash mismatch, or scientific assertion fails.

Missing historical CLIP evidence limits historical CLIP claims but does not block the new protocol because CLIP is excluded. The complete node mapping is [legacy_test_mapping.csv](../../cleanup/audits/foundation_protocol_freeze/legacy_test_mapping.csv).

## Handoff

Prompt 2 may begin only after this validation remains green. It must preserve the sidecar/row contract, voter registry, pool and final budgets, statistical families, bootstrap seed, and OOT lock. It must not relabel the prospective protocol as historical replication or restore CLIP without complete provenance and a new protocol version.
